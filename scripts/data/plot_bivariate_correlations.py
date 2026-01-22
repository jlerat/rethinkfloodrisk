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
from scipy.stats import chi2
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

ax_width = 3

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fimg = froot / "images" / "bivariate_correlations"
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
ams, ams_times = datahub.get_ams_concat()
nstations = ams.shape[1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Bivariate plots")
plt.close("all")

nplots = nstations * (nstations - 1) // 2
nr = int(math.sqrt(nplots))
nc = int(nplots / nr) + 1
fig, axs = plt.subplots(ncols=nc, nrows=nr,
                        figsize=(nc * ax_width, nr * ax_width),
                        layout="constrained")
wy = ams.index

for iplot, (i1, i2) in enumerate(comb(np.arange(nstations), 2)):
    sidx = ams.columns[i1]
    sidy = ams.columns[i2]

    LOGGER.info(f"Plotting X={sidx} Y={sidy}")

    # Get data
    xy = ams.iloc[:, [i1, i2]]
    isok = xy.notnull().all(axis=1)
    times = ams_times.loc[isok].iloc[:, [i1, i2]]
    xy = xy.loc[isok].values

    # plot
    ax = axs.flat[iplot]
    unorm, rho, _, _, _ = putils.bivarnplot(ax, xy)

    # Mahalanobis distance
    covi = np.linalg.inv(np.cov(unorm.T))
    maha = np.einsum("ij,jk,ik->i", unorm, covi, unorm)

    # Plot outliers
    pdf = chi2.pdf(maha, df=2)
    eps = 5e-2
    outliers = np.where((pdf < eps) | (pdf > 1 - eps))[0]
    if len(outliers) > 0:
        LOGGER.info(f"Found {len(outliers)} outliers", ntab=1)
        for idx in outliers:
            x, y = unorm[idx]
            delta = abs(times.iloc[idx].diff().iloc[-1].days)

            ax.plot(x, y, "o", color="tab:red")
            txt = f"{wy[idx]} $Δ${delta}"
            ax.text(x, y, txt,
                    color="tab:red",
                    va="bottom", ha="center",
                    fontsize="small", fontweight="bold")

    ax.set(xlabel="", ylabel="")
    if iplot % nc != 0:
        ax.set_yticks([])
    if iplot < nc * (nr - 1):
        ax.set_xticks([])

    txt = f"X={sidx}\nY={sidy}"
    ax.text(0.98, 0.02, txt,
            va="bottom", ha="right",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize="small")

for i in [-1, -2]:
    axs.flat[i].axis("off")

# Random normal data with rho = 0.8 to compare
#ax = axs.flat[-1]
#rho = 0.8
#xy = mvn(cov=[[1, rho], [rho, 1]]).rvs(size=len(potpeaks))
#putils.bivarnplot(ax, xy)
#ax.set(xlabel="", ylabel="", yticks=[])
#txt = f"Random normal $ρ$={rho:0.2f}"
#ax.text(0.98, 0.02, txt,
#        va="bottom", ha="right",
#        transform=ax.transAxes)

fp = fimg / "standard_normal_bivariate.png"
fig.savefig(fp)

LOGGER.completed()
