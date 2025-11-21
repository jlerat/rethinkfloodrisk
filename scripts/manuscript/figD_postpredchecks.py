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
import argparse
import json
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from floodstan.report import STAN_DIAGNOSTIC_VARIABLES as SDV
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot posterior predictive checks",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--clear", help="Clear figures",
                    action="store_true", default=False)
parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                    type=float, default=-1.)
args = parser.parse_args()

clear = args.clear
rho_min = args.rho_min

awidth = 6
aheight = 5
fdpi = 300

# Define list of postpred checks to plot
variables = {
    "univ": ["lcoeffvar2", "lskewness2", "lkurtosis2"],
    "biv": ["kendalltau", "kendalltauhigh"]
    }


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
if clear:
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

# Select fit task with
postpred = {}
for fold in fout.glob("copulafit_TASK*"):
    taskid = int(re.sub(".*TASK", "", fold.stem))

    fd = fold / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    exclude = diag["exclude"]
    if exclude != "NONE":
        continue

    pcensor = diag["pcensor"]
    rm = diag["rho_min"]
    if rm != rho_min:
        continue

    LOGGER.info(f"Load data from TASK {taskid}", nret=1)
    LOGGER.info(f"pcensor = {pcensor}", ntab=1)
    LOGGER.info(f"exclude = {exclude}", ntab=1)
    LOGGER.info(f"rho_min = {rho_min}", ntab=1)

    for vn in SDV:
        LOGGER.info(f"{vn}: {diag[vn][:20]}", ntab=1)

    fd = fold / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        data = json.load(fo)

    stationids = data["stationids"]

    pp = {}
    for ppt in ["univ", "biv"]:
        fp = f"postprocess_postpredchecks_{ppt}_TASK{taskid}.csv"
        fp = fold / fp
        if not fp.exists():
            continue
        df = pd.read_csv(fp, skiprows=15)
        df.columns = ["VARIABLE"] + df.columns[1:].tolist()
        pp[ppt] = df

    if len(pp) == 2:
        postpred[(exclude, pcensor, rho_min)] = pp

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ncols = 3
varnames = [f"{ppt}_{vn}" for ppt in variables for vn in variables[ppt]]
nv = len(varnames)
nrows = nv // ncols + int(nv % ncols > 0)
mosaic = [[varnames[ncols * ir + ic] if ncols * ir + ic < nv else "."
          for ic in range(ncols)] for ir in range(nrows)]

arrowprops = dict(arrowstyle="wedge", facecolor="0.6", edgecolor="none")
paeff = pe.withStroke(linewidth=4, foreground="w")

for (exclude, pcensor, rho_min), pp in postpred.items():
    mess = f"Plot ppchecks pcensor={pcensor} exclude={exclude} rho_min={rho_min}"
    LOGGER.info(mess, nret=1)

    plt.close("all")
    fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)

    for aname, ax in axs.items():
        ppt, varname = aname.split("_")
        df = pp[ppt]
        df = df.loc[df.VARIABLE == varname].filter(regex="pvalue\\[", axis=1)

        if ppt == "univ":
            df.columns = stationids
            df.squeeze().plot(ax=ax, kind="barh")

            m = (df - 0.5).abs().mean().mean()
            ax.text(0.5, 0.5, f"mean diff\n{m:0.2f}",
                    transform=ax.transAxes,
                    va="center", ha="center",
                    fontweight="bold", fontsize="x-large",
                    path_effects=[paeff])
        else:
            #bins = np.concatenate([[0], np.linspace(0.05, 0.95, 5), [1]])
            #df.squeeze().plot(ax=ax, kind="hist", bins=bins,
            #                  edgecolor="0.2", facecolor="0.8")
            obj = putils.ecdfplot(ax, df.T)
            obj = obj[df.index[0]]

            idx = obj["index"]
            x = obj["values"]
            y = obj["position"]
            out = {
                "low": x < 0.05,
                "high": x > 0.95
                }

            for name, ipb in out.items():
                npb = ipb.sum()
                if npb == 0:
                    continue

                xpb, ypb, idxp = x[ipb], y[ipb], idx[ipb]
                col = "tab:red"
                for cnt, (xx, yy, ii) in enumerate(zip(xpb, ypb, idxp)):
                    ax.plot(xx, yy, "o", color=col)
                    i1 = int(re.sub(".*\\[|,.*", "", ii)) - 1
                    sta1 = stationids[i1]
                    i2 = int(re.sub(".*,|\\].*", "", ii)) - 1
                    sta2 = stationids[i2]
                    txt = f"{sta1}\n{sta2}"

                    xt = 0.2 if name == "low" else 0.8
                    ha = "left" if name == "low" else "right"
                    yt = np.linspace(0, 1,  2 * npb)[1 + cnt]
                    ax.annotate(txt, xy=(xx, yy),
                                xytext=(xt, yt),
                                textcoords="axes fraction",
                                va="bottom", ha=ha,
                                arrowprops=arrowprops)

        for x in [0.05, 0.95]:
            putils.line(ax, 0, 1, x, 0, "r--")

        putils.line(ax, 0, 1, 0.5, 0, "k-", lw=2)

        title = f"{ppt} post pred checks / {varname}"
        xlab = "check pvalue [-]"
        ax.set(title=title, xlabel=xlab, xlim=(0, 1))

    ftitle = f"Pcensor={pcensor}  Exclude={exclude} rho_min={rho_min}"
    fig.suptitle(ftitle, fontweight="bold")

    fp = fimg / f"{basename}_pcens{pcensor}_exclude{exclude}_rhomin{rho_min}.png"
    fig.savefig(fp)

LOGGER.completed()
