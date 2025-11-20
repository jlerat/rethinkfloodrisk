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

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from floodstan.report import STAN_DIAGNOSTIC_VARIABLES as SDV

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot obs probs",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--clear", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                    type=float, default=0.5)
parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                    type=float, default=-1.)
args = parser.parse_args()

clear = args.clear
pcensor = args.pcensor
rho_min = args.rho_min

awidth = 4
aheight = 3
fdpi = 100 # 300

exclude = "NONE"

#events = ["2022-02-27"]

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

fopm = fout / "copulafit_options.json"
opm = hyruns.OptionManager.from_file(fopm)
taskids = opm.search(pcensor=f"{pcensor:0.1f}",
                     rho_min=f"{rho_min:0.1f}",
                     exclude=exclude)

data = {}
for taskid in taskids:
    # Select fit task with
    fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    ex = diag["exclude"]
    pc = diag["pcensor"]
    rm = diag["rho_min"]

    mess = f"Load report TASK {taskid} exclude={ex} pcensor={pc} rho_min={rm}"
    LOGGER.info(mess)

    for vn in SDV:
        LOGGER.info(f"{vn}: {diag[vn][:50]}", ntab=1)

    fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
    mvnproc, comment = csv.read_csv(fs)

    fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        stan_data = json.load(fo)
    stationids = stan_data["stationids"]
    nstations = len(stationids)

    groups = mvnproc.columns.str.replace("_.*", "", regex=True).unique()
    groups = [g for g in groups if g not in ["", "mvn"]]
    LOGGER.info(f"groups: {groups}", ntab=1)

    events = mvnproc.columns.to_series()\
        .filter(regex="obs_log10")\
        .str.replace(".*aep_", "", regex=True)\
        .unique().tolist()
    LOGGER.info(f"nb events: {len(events)}", ntab=1)

    data[(ex, pc, rm)] = {
        "mvnproc": mvnproc,
        "stationids": stationids,
        "groups": groups,
        "events": events
        }

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
paef = pe.withStroke(linewidth=4,
                     foreground="w")
ncols = 3

for key, dd in data.items():
    exclude, pcensor, rho_min = key
    mess = f"Plotting exclude={exclude} pcensor={pcensor}"\
           + f" rho_min={rho_min}"
    LOGGER.info(mess)

    # Get data
    mvnproc = dd["mvnproc"]
    stationids = dd["stationids"]
    groups = dd["groups"]
    events = dd["events"]

    # Configure plot
    ng = len(groups)
    nrows = ng // ncols + int(ng % ncols > 0)
    mosaic = [[groups[ncols * ir + ic] if ncols * ir + ic < ng else "."
               for ic in range(ncols)] for ir in range(nrows)]

    for event in events:
        LOGGER.info(f"Plotting {event}", ntab=1)
        plt.close("all")
        fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                         layout="constrained")
        axs = fig.subplot_mosaic(mosaic)
        for aname, ax in axs.items():
            grp = aname
            LOGGER.info(f"Group {grp}", ntab=2)

            cn = f"{grp}_obs_log10aep_{event}"
            # value -> %
            aep = 10**(mvnproc.loc[:, cn] + 2)
            prob = aep

            x0 = round(math.log10(max(1e-9, prob.quantile(0.001))), 2)
            x1 = round(math.log10(prob.max()), 2)
            bins = np.logspace(x0, x1, 50)

            ax.hist(prob, bins=bins, facecolor="0.8", edgecolor="0.2")

            m = prob.mean()
            y0, y1 = ax.get_ylim()
            xy = (m, (y0 + y1) / 2)
            ax.annotate(f"Mean\n{m:0.2f}%", xy, (0, 5),
                        xycoords="data", textcoords="offset points",
                        fontweight="bold", va="bottom", ha="center",
                        fontsize=15,
                        path_effects=[paef])
            ax.plot([m, m], [y0, y1], "k-", lw=2)

            title = f"{grp[1:]}"
            xlab = "AEP [%]"
            ax.set(xscale="log", title=title,
                   xlabel=xlab, ylim=(y0, y1))

        ftitle = f"Event {event}\n"
        fig.suptitle(ftitle, fontsize=20, fontweight="bold")

        LOGGER.info("Saving to disk", ntab=1)
        fp = f"{basename}_{event}"\
             + f"_pcensor{pcensor}_rhomin{rho_min}.png"
        fp = fimg / fp
        fig.savefig(fp)

LOGGER.completed()
