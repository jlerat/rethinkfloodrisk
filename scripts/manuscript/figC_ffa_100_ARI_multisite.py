#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
import re
import math
import argparse
import json
import time
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import ticker

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot
from pyrethink import datahub
from floodstan import marginals

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA 100 ARI",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

awidth = 6
aheight = 3
fdpi = 300
ngrid = 40

eep_target_plot = 0.95

# Pcensor = 0.3 period=ALL
taskid = 2

sta1 = "203002"
sta2 = "203014"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs"

basename = source_file.stem
fimg = froot / "images" / "manuscript" / basename
fimg.mkdir(exist_ok=True, parents=True)
for f in fimg.glob("*.png"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
stations = datahub.get_stations()

LOGGER.info(f"Load data TASK {taskid}")
fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_data_TASK{taskid}.json"
with fd.open("r") as fo:
    data = json.load(fo)
stationids = data["stationids"]
nstations = len(stationids)
ista1 = stationids.index(sta1)
ista2 = stationids.index(sta2)

fe = fimg / "expected_parameters.json"
if not fe.exists():
    LOGGER.info(f"Load samples TASK {taskid}")
    fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_samples_TASK{taskid}.zip"
    samples = pd.read_csv(fs, skiprows=15)
    ylocs = samples.filter(regex="ylocn", axis=1).mean()
    ylogscales = samples.filter(regex="ylogsca", axis=1).mean()
    yshape1 = samples.filter(regex="yshape1", axis=1).mean()
    L_cor = samples.filter(regex="L_cor", axis=1).mean()
    expected = {
        "ylocs": ylocs.to_dict(),
        "ylogscales": ylogscales.to_dict(),
        "yshape1": yshape1.to_dict(),
        "L_cor": L_cor.to_dict()
        }
    with fe.open("w") as fo:
        json.dump(expected, fo, indent=4)

else:
    with fe.open("r") as fo:
        expected = json.load(fo)
    ylocs = pd.Series(expected["ylocs"])
    ylogscales = pd.Series(expected["ylogscales"])
    yshape1 = pd.Series(expected["yshape1"])
    L_cor = pd.Series(expected["L_cor"])

L_cor = L_cor.values.reshape((nstations, nstations)).T
cor = L_cor @ L_cor.T
k = np.arange(nstations)
cor[k, k] = 1.

LOGGER.info(f"Load mvnprocess TASK {taskid}")
fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
df, comment = csv.read_csv(fs)

groups = set([cn for cn in df.columns.str.replace("_.*", "", regex=True)
              if cn not in ["", "mvn"]])

#eep_target = float(comment["eep_target"])

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
gev = marginals.GEV()

plt.close("all")
mosaic = [["diagram", stat] for stat in ["pall", "pany"]]
nrows = len(mosaic)
ncols = len(mosaic[0])
fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                 layout="constrained")

kw = dict(wspace=0.4, hspace=0.1, left=0.3, bottom=0.3)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
axs["diagram"].remove()
axs["diagram"] = fig.add_subplot(1, 2, 1, projection="3d")

cols = ["tab:blue", "tab:orange"]

for iax, (aname, ax) in enumerate(axs.items()):
    if aname == "diagram":
        pa, pb = 0.0, 0.995

        xx, zz, marg = {}, {}, {}
        xthresh, xlims = {}, {}
        for ista in [ista1, ista2]:
            gev.params = [ylocs.iloc[ista], ylogscales.iloc[ista],
                          yshape1.iloc[ista]]
            xa, xb = gev.ppf([pa, pb])
            xa = max(xa, 0.)
            xlims[ista] = (xa, xb)
            xx[ista] = np.linspace(xa, xb, ngrid)
            zz[ista] = norm.ppf(gev.cdf(xx[ista]))
            marg[ista] = gev.pdf(xx[ista])
            xthresh[ista] = gev.ppf(eep_target_plot)

        XX1, XX2 = np.meshgrid(xx[ista1], xx[ista2])
        ZZ1, ZZ2 = np.meshgrid(zz[ista1], zz[ista2])
        ZZ = np.dstack((ZZ1, ZZ2))

        ii = [ista1, ista2]
        rv = mvn(cov=cor[ii][:, ii])
        PP = rv.pdf(ZZ)
        ppmax = np.nanmax(PP)

        # Bivariate pdf
        kwargs = dict(cmap="viridis",
                      linewidth=0.0,
                      antialiased=False,
                      alpha=0.4)
        surf = ax.plot_surface(XX1, XX2, PP, **kwargs)

        # Pall integral
        xt1 = xthresh[ista1]
        xt2 = xthresh[ista2]
        ii = (XX1 >= xt1) & (XX2 >= xt2)
        PP[~ii] = np.nan
        kwargs["alpha"] = 1.
        surf = ax.plot_surface(XX1, XX2, PP, **kwargs)

        # Marginals
        y0, y1 = ax.get_ylim()
        x0, x1 = ax.get_xlim()

        z = marg[ista1]
        z *= ppmax / z.max()
        d = z * 0 + y1
        ax.plot(xx[ista1], d, z, "k-", lw=1, zorder=2)

        z = marg[ista2]
        z *= ppmax / z.max()
        d = z * 0 + x1
        ax.plot(d, xx[ista2], z, "k-", lw=1, zorder=2)

        ax.set_xlim((x0, x1))
        ax.set_ylim((y0, y1))

        elev = 50
        azim = -140
        roll = 0.
        ax.view_init(elev, azim, roll)
        ax.set_proj_type("ortho")
        ax.zaxis.set_major_locator(ticker.MaxNLocator(4))

        xlab = f"Peak flow {sta1} [m3.s-1]"
        ylab = f"Peak flow {sta2} [m3.s-1]"
        zlab = "Pr [-]"
        ax.set(xlabel=xlab, ylabel=ylab, zlabel=zlab)
        continue

    stat = aname

    x0, x1 = (-8, -2.7) if stat == "pall" else (-1.7, -1.2)
    bins = np.logspace(x0, x1, 50)

    for ig, gname in enumerate(groups):
        sel = df.loc[:, f"{gname}_log10_{stat}"]
        se = 10**sel
        lab = f"Site group {gname}\n(mean={se.mean():0.2e})"
        ax.hist(se, bins=bins, edgecolor="0.5",
                facecolor=cols[ig],
                alpha=0.6, label=lab)

    title = f"({letters[iax]}) Statistic {stat}"
    xlab = "Event Exceedance Probability [-]"
    ylab = "Sample count [-]"
    ax.set(title=title, xlabel=xlab, xscale="log",
           ylabel=ylab)
    ax.legend(fontsize="small")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

fp = fimg / f"{basename}.png"
fig.savefig(fp)

LOGGER.completed()
