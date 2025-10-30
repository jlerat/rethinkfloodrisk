#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------


import sys
import os
import re
import json
import math
from itertools import combinations as comb

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal as mvn
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from floodstan import marginals, freqplots
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
marginal_name = "GEV"

design_aris = [5, 10, 20, 50, 100, 500]

ari_ref = 100

nsamples = 10000

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs" / "fit"
fout.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
stations = datahub.get_stations()

truepeaks = datahub.get_truepeaks().filter(regex="^2", axis=1)

nstations = truepeaks.shape[1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
marginal = marginals.factory(marginal_name)

# Setup data and initial parameters
cases = np.zeros_like(truepeaks, dtype=int)
theta = {}
aeps = {}

# Fit LH moment
for stationid, values in truepeaks.items():
    marginal.fit_lh_moments(values)
    theta[f"marginal_{stationid}"] = marginal.params.tolist()
    aeps[stationid] = marginal.ppf(1 - 1./np.array(design_aris))

aeps = pd.DataFrame(aeps, index=design_aris)

rk = truepeaks.rank()
corr0 = ((rk / rk.max()).corr()).values
eig, P = np.linalg.eig(corr0)
eig = np.maximum(eig, eig.max() * 1e-5)
corr1 = P@np.diag(eig)@P.T
d = (1 / np.sqrt(np.diag(corr1)))[:, None]
corr2 = (d * corr1 * d.T)
theta["corr"] = corr2.tolist()

# simulate
rv = mvn(cov=corr2)
u = rv.rvs(size=nsamples)
p = norm.cdf(u)
x = np.zeros_like(p)
for ista, stationid in enumerate(truepeaks):
    marginal.params = theta[f"marginal_{stationid}"]
    x[:, ista] = marginal.ppf(p[:, ista])

plt.close("all")

nplots = nstations * (nstations - 1) // 2
nr = int(math.sqrt(nplots))
nc = int(nplots / nr) + 1
w = 2
fig, axs = plt.subplots(ncols=nc, nrows=nr,
                        figsize=(nc * w, nr * w),
                        layout="constrained")
for iplot, (i1, i2) in enumerate(comb(np.arange(nstations), 2)):
    ax = axs.flat[iplot]
    xy = truepeaks.iloc[:, [i1, i2]]
    xy = xy.loc[xy.notnull().all(axis=1)].values
    putils.bivarnplot(ax, xy)

    ax.set(xlabel="", ylabel="")
    if iplot % nc != 0:
        ax.set_yticks([])
    if iplot < nc * (nr - 1):
        ax.set_xticks([])

    txt = f"X={truepeaks.columns[i1]}\n"\
          + f"Y={truepeaks.columns[i2]}"
    ax.text(0.98, 0.02, txt,
            va="bottom", ha="right",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize="small")

axs.flat[-2].axis("off")

ax = axs.flat[-1]
rho = 0.8
xy = mvn(cov=[[1, rho], [rho, 1]]).rvs(size=len(truepeaks))
putils.bivarnplot(ax, xy)
ax.set(xlabel="", ylabel="", yticks=[])
txt = f"Random normal $ρ$={rho:0.2f}"
ax.text(0.98, 0.02, txt,
        va="bottom", ha="right",
        transform=ax.transAxes)

fp = fout / "standard_normal_bivariate.png"
fig.savefig(fp)

#
above = (x - aeps.loc[ari_ref].values[None, :] > 0).astype(int)

uref = norm.ppf(1 - 1./ari_ref)
above_all = np.all(above > 0, axis=1)
th = rv.cdf(-uref * np.ones(nstations))
LOGGER.info(f"ALL freq = {above_all.sum() / nsamples:2.2e} / theory={th:2.2e}")

above_any = np.any(above > 0, axis=1)
th = 1 - rv.cdf(uref * np.ones(nstations))
LOGGER.info(f"ANY freq = {above_any.sum() / nsamples:2.2e} / theory={th:2.2e}")

nplots = nstations
w, h = 10, 4
fig, axs = plt.subplots(ncols=2, nrows=nplots,
                        figsize=(w, h * nplots),
                        layout="constrained")
for ista in range(nplots):
    ax = axs[ista, 0]
    xx = truepeaks.iloc[:, ista].values
    ax.plot(xx)

    stationid = int(truepeaks.columns[ista])
    name = stations.NAME[stationid]
    title = f"{name} - {stationid}"
    ax.set_title(title, x=0.01, y=0.95, fontweight="bold",
                 va="top", ha="left")

    ax = axs[ista, 1]
    ptype = "gumbel"
    freqplots.plot_data(ax, xx, ptype)

    marginal.params = theta[f"marginal_{stationid}"]
    freqplots.plot_marginal_cdf(ax, marginal, ptype, Tmax=500)

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)
    x0, x1 = ax.get_xlim()
    xlim = (0., x1)
    ax.set(xlim=xlim)

fp = fout / "ffa_plots.png"
fig.savefig(fp)

LOGGER.completed()

