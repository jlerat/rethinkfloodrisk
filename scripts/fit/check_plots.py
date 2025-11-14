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
from itertools import combinations as combs

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import norm, multivariate_normal as mvn
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from floodstan import marginals, freqplots, report
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt

nv = 8

nx = nv * (nv + 1) // 2
nsmp = 1000
Z = np.zeros((nsmp, nv, nv))
for i in range(nsmp):
    x = np.random.normal(scale=0.9, size=nx)
    X = np.zeros((nv, nv))
    X[np.tril_indices(nv)] = x
    Y = np.exp(X)
    Y[np.triu_indices(nv, 1)] = 0
    Ys = np.sqrt((Y**2).sum(axis=1))[:, None]
    Y = Y / Ys
    Z[i] = Y @ Y.T

plt.close("all")
bins = np.linspace(0, 1, 30)
plt.hist(Z[:, 0, 1], bins=bins, facecolor="0.8", edgecolor="0.2")
plt.show()

sys.exit()


# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs"
fimg = froot / "images" / "copulafit"
fimg.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename, contextual=True)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
stations = datahub.get_stations()
potpeaks = datahub.get_potpeaks().filter(regex="_PEAK", axis=1)
potpeaks.columns = potpeaks.columns.to_series().str.replace("_PEAK", "")
nstations = potpeaks.shape[1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

for ftask in fout.glob("*TASK*"):
    # Setup folders
    taskid = int(re.sub("^.*TASK", "", ftask.stem))
    fimg_task = fimg / f"copulafit_TASK{taskid}"
    fimg_task.mkdir(exist_ok=True)
    LOGGER.context = f"TASK{taskid}"

    # Get data
    LOGGER.info("Load diagnostic")
    fd = ftask / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    LOGGER.info(f"pcens={diag['pcensor']} - period={diag['exclude']}",
                nret=1)

    for imet, me in enumerate(report.STAN_DIAGNOSTIC_VARIABLES):
        txt = diag[me][:100]
        LOGGER.info(f"[diag] {me:12s} : {txt}",
                    nret=imet == 0, ntab=1)

    LOGGER.info("Load samples", nret=1, ntab=1)
    fs = ftask / f"copulafit_samples_TASK{taskid}.zip"
    df = pd.read_csv(fs, skiprows=15)

    LOGGER.info("MCMC traces plot", ntab=1)
    cols = df.columns.to_series()
    mosaic = [cols.filter(regex="ylocn\\[[1-2]\\]").tolist(),
              cols.filter(regex="ylogscale\\[[1-2]\\]").tolist(),
              cols.filter(regex="yshape1\\[[1-2]\\]").tolist(),
              cols.filter(regex="wlat_cens\\[[1-2]\\]").tolist(),
              cols.filter(regex="wlat_miss\\[[1-2]\\]").tolist()]
    mosaic = [m for m in mosaic if len(m) > 0]
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

    fp = fimg_task / f"mcmc_traces_TASK{taskid}.png"
    fig.savefig(fp)

    LOGGER.info("MCMC param distribution", ntab=1)
    pini = df.columns.to_series().filter(regex="^y(locn|shape|logsc)|L_cor").values
    def select_parameters(pini):
        pnames = []
        for pn in pini:
            if re.search("L_cor", pn):
                i1, i2 = [int(i) for i in re.sub(".*\\[|\\]", "", pn).split(",")]
                if i2 < i1:
                    pnames.append(pn)
            else:
                pnames.append(pn)
        return pnames
    pnames = select_parameters(pini)
    nparams = len(pnames)
    LOGGER.info(f"-> {nparams} parameters plotted", ntab=1)
    ncols = min(8, nparams)
    nrows = nparams // ncols + int(nparams % ncols > 0)
    w = 3
    plt.close("all")
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=(w * ncols, w * nrows),
                            layout="constrained")

    for iax, ax in enumerate(axs.flat):
        if iax >= nparams:
            ax.axis("off")
            continue

        x = df.loc[:, pnames[iax]]
        bins = np.linspace(x.min(), x.max(), 30)
        ax.hist(x, bins=bins, edgecolor="0.2", facecolor="0.8")
        title = f"{pnames[iax]} mean={x.mean():0.2f}"
        ax.set_title(title, x=0.05, y=0.95,
                     va="top", ha="left", fontweight="bold")
        ax.set(yticks=[])

    fp = fimg_task / f"mcmc_hist_TASK{taskid}.png"
    fig.savefig(fp)

    LOGGER.info("MCMC param correlation", ntab=1)
    plt.close("all")
    pnames = df.columns.to_series().filter(regex="^y(locn|shape|logsc)").values
    nparams = len(pnames)
    ncombs = nparams * (nparams - 1) // 2
    LOGGER.info(f"-> {ncombs} parameter pairs plotted", ntab=1)
    ncols = min(16, ncombs)
    nrows = ncombs // ncols + int(ncombs % ncols > 0)
    w = 2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=(w * ncols, w * nrows),
                            layout="constrained")

    for iax, (i1, i2) in enumerate(combs(range(nparams), 2)):
        ax = axs.flat[iax]

        # Plot correlation every 5 samples
        # to save time (total is 50,000 samples)
        pn1, pn2 = pnames[[i1, i2]]
        x1 = df.loc[:, pn1].iloc[::5]
        x2 = df.loc[:, pn2].iloc[::5]

        ax.plot(x1, x2, ".", alpha=0.01)

        txt = f"X={pn1}\nY={pn2}"
        ax.text(0.98, 0.02, txt,
                transform=ax.transAxes,
                va="bottom", ha="right",
                fontsize="x-small")
        ax.set(xticks=[], yticks=[])

    for iax in range(ncombs, ncols * nrows):
        ax = axs.flat[iax]
        ax.axis("off")

    fp = fimg_task / f"mcmc_corr_TASK{taskid}.png"
    fig.savefig(fp)

LOGGER.completed()
