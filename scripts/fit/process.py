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
from pandas.plotting import scatter_matrix
from scipy.stats import norm, multivariate_normal as mvn
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from floodstan import marginals, freqplots
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
marginal_name = "Gumbel"

design_aris = np.logspace(math.log10(5), 3., 30)

ari_ref = 100

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs" / "copulafit"
fimg = froot / "images" / "copulafit"
fimg.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
stations = datahub.get_stations()
potpeaks = datahub.get_potpeaks().filter(regex="_PEAK", axis=1)
potpeaks.columns = potpeaks.columns.to_series().str.replace("_PEAK", "")
nstations = potpeaks.shape[1]

fs = fout / "copulafit_samples.zip"
df = pd.read_csv(fs, skiprows=15)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
marginal = marginals.factory(marginal_name)

LOGGER.info("Bivariate plots")
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
    xy = potpeaks.iloc[:, [i1, i2]]
    xy = xy.loc[xy.notnull().all(axis=1)].values
    putils.bivarnplot(ax, xy)

    ax.set(xlabel="", ylabel="")
    if iplot % nc != 0:
        ax.set_yticks([])
    if iplot < nc * (nr - 1):
        ax.set_xticks([])

    txt = f"X={potpeaks.columns[i1]}\n"\
          + f"Y={potpeaks.columns[i2]}"
    ax.text(0.98, 0.02, txt,
            va="bottom", ha="right",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize="small")

axs.flat[-2].axis("off")

ax = axs.flat[-1]
rho = 0.8
xy = mvn(cov=[[1, rho], [rho, 1]]).rvs(size=len(potpeaks))
putils.bivarnplot(ax, xy)
ax.set(xlabel="", ylabel="", yticks=[])
txt = f"Random normal $ρ$={rho:0.2f}"
ax.text(0.98, 0.02, txt,
        va="bottom", ha="right",
        transform=ax.transAxes)

fp = fimg / "standard_normal_bivariate.png"
fig.savefig(fp)

LOGGER.info("MCMC plots")

cols = df.columns.to_series()
mosaic = [cols.filter(regex="ylocn\\[[1-2]\\]").tolist(),
          cols.filter(regex="ylogscale\\[[1-2]\\]").tolist(),
          cols.filter(regex="yshape1\\[[1-2]\\]").tolist(),
          cols.filter(regex="wlat_cens\\[[1-2]\\]").tolist(),
          cols.filter(regex="wlat_miss\\[[1-2]\\]").tolist()]
nrows = len(mosaic)
ncols = len(mosaic[0])
w, h = 6, 2
plt.close("all")
fig = plt.figure(figsize=(w * ncols, h * nrows),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic, sharex=True)
for aname, ax in axs.items():
    pname = aname
    ddf = pd.pivot_table(df,
                         index="iter__",
                         columns="chain__",
                         values=pname)
    ddf.iloc[-200:, :3].plot(ax=ax, legend=False)

    title = pname
    ax.set_title(title, x=0.05, y=0.93,
                 fontweight="bold",
                 va="top", ha="left")

fp = fimg / "mcmc_traces.png"
fig.savefig(fp)

plt.close("all")
pnames = [m for mm in mosaic for m in mm]
fig, ax = plt.subplots(figsize=(13, 13),
                 layout="constrained")
scatter_matrix(df.loc[:, pnames], diagonal="kde",
               alpha=0.05, ax=ax)
fp = fimg / "mcmc_corr.png"
fig.savefig(fp)


LOGGER.info("Frequency plots")
smp = df.filter(regex="yrnd", axis=1)

nplots = nstations
w, h = 10, 4
plt.close("all")
fig, axs = plt.subplots(ncols=2, nrows=nplots,
                        figsize=(w, h * nplots),
                        layout="constrained")
for ista in range(nplots):
    ax = axs[ista, 0]
    xx = potpeaks.iloc[:, ista].values
    ax.plot(xx)

    stationid = int(potpeaks.columns[ista])
    name = stations.NAME[stationid]
    title = f"{name} - {stationid}"
    ax.set_title(title, x=0.01, y=0.95, fontweight="bold",
                 va="top", ha="left")

    ax = axs[ista, 1]
    ptype = "gumbel"
    freqplots.plot_data(ax, xx, ptype)

    p = 1 - 1./np.array(design_aris)
    quantiles = smp.iloc[:, ista].quantile(p)
    quantiles.index = design_aris
    quantiles.name = "stan"
    quantiles = pd.DataFrame(quantiles)
    freqplots.plot_marginal_quantiles(ax, design_aris,
                                      quantiles, ptype,
                                      center_column="stan")

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)
    x0, x1 = ax.get_xlim()
    xlim = (0., x1)
    ax.set(xlim=xlim)

fp = fimg / "ffa_plots.png"
fig.savefig(fp)

LOGGER.completed()
