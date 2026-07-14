#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-11-14 22:00:13.975234
## Comment : Explore correlation matrix prior
##
## ------------------------------


import sys
import os
import re
import json
from itertools import combinations
import math
import argparse
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, percentileofscore
from scipy.stats import norm
from scipy.stats import t as student
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt
import matplotlib.pyplot as plt


from hydrodiy.io import csv, iutils
from hydrodiy.stat import sutils
from hydrodiy.plot import putils

from floodstan import marginals, copulas, freqplots

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ams, _, _, _ = datahub.get_ams_concat()
ams = ams.iloc[:, :4]
nsta = ams.shape[1]

# Gumbel copula utils
psi = lambda s, theta: np.exp(-s**(1. / theta))
psi_inv = lambda t, theta: (-np.log(t))**theta

plt.close("all")
ntot = nsta * (nsta - 1) // 2
ncols = 3
nrows = ntot // ncols + (ntot % ncols != 0)

aw, ah = 4, 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                        figsize=(aw * ncols, ah * nrows),
                        layout="tight")
for iax, (i, j) in enumerate(combinations(range(nsta), 2)):
    iok = ams.iloc[:, [i, j]].notnull().all(axis=1)
    amsx = ams.iloc[iok, i]
    amsy = ams.iloc[iok, j]

    sidx = amsx.name
    sidy = amsy.name
    LOGGER.info(f"{sidx} / {sidy} (#{iax + 1})")

    # Copula model
    gevx = marginals.GEV()
    gevx.fit_lh_moments(amsx)

    gevy = marginals.GEV()
    gevy.fit_lh_moments(amsy)

    tau = kendalltau(amsx, amsy).statistic
    #tau = 0.94
    cop = copulas.GumbelCopula()
    cop.rho = tau

    # Sample and compute sum
    nsamples = 20000
    uv = cop.sample(nsamples)
    x = gevx.ppf(uv[:, 0])
    y = gevy.ppf(uv[:, 1])
    z = x + y

    nbnds = 50
    bnds = np.empty((2, nbnds, 3))
    cst = 0.3
    eps = 1e-2
    x0 = gevx.ppf(eps)
    x1 = gevx.ppf(1 - eps)
    cdfs = np.zeros(nbnds)

    for ibnd in range(nbnds):
        ppos = (ibnd + 1 - cst) / (nbnds + 1  - 2 *cst)
        s = np.percentile(z, ppos * 100)
        cdfs[ibnd] = ppos
        bnds[:, ibnd, 0] = s

        # quantile sum
        zq = gevx.ppf(ppos) + gevy.ppf(ppos)
        bnds[0, ibnd, 1] = np.nan
        bnds[0, ibnd, 2] = percentileofscore(z, zq) * 1e-2

        ofun = lambda xx: -gevx.cdf(xx) * gevy.cdf(s - xx)
        opt = minimize_scalar(ofun, bounds=[x0, x1], method="bounded")
        bnds[1, ibnd, 1] = -opt.fun

        ofun = lambda xx: - (1 - gevx.cdf(xx)) * (1 - gevy.cdf(s - xx))
        opt = minimize_scalar(ofun, bounds=[x0, x1], method="bounded")
        bnds[1, ibnd, 2] = 1 + opt.fun

    # plot
    ax = axs.flat[iax]

    aris = np.logspace(0.8, 2, 100)
    quantiles = pd.DataFrame({"mean": np.percentile(z, 100 * (1 - 1./aris))})
    ptype = "gumbel"
    freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      label="Sum X+Y", color="tab:blue")

    aris = 1. / (1 - bnds[1, :, 1])
    quantiles = pd.DataFrame({"mean": bnds[1, :, 0]})
    freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      label="Lower", color="0.5")

    aris = 1. / (1 - bnds[1, :, 2])
    freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      label="Upper", color="0.3")

    aris = 1. / (1 - bnds[0, :, 2])
    freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      label="Sum Qtl", color="0.3", ls="--")

    retp = [5, 10, 100]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, return_periods=retp)

    xl = "Gumbel variate [-]" if iax // ncols == nrows - 1 else ""
    yl = "Sum [m3.s-1]" if iax % ncols == 0 else ""
    Rx = amsx.mean() / (amsx + amsy).mean()
    title = f"{sidx} / {sidy} Rx={Rx:0.2f} τ={tau:0.2f}"
    ax.set_xlim((1, 6))
    ax.set(xlabel=xl, ylabel=yl, title=title)

    if iax == 0:
        ax.legend()

plt.show()




LOGGER.completed()

