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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot obs probs",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

awidth = 4
aheight = 3
fdpi = 100 # 300

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

# Select fit task with
for fold in fout.glob("copulafit_TASK*"):
    lf = [f for f in fold.glob("*.zip")
          if re.search("mvnprocess", f.stem)]
    if len(lf) > 0:
        taskid = int(re.sub(".*TASK", "", fold.stem))
        break

stations = datahub.get_stations()

LOGGER.info(f"Load data TASK {taskid}")
fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_data_TASK{taskid}.json"
with fd.open("r") as fo:
    data = json.load(fo)
stationids = data["stationids"]
nstations = len(stationids)

LOGGER.info(f"Load mvnprocess TASK {taskid}")
fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
df, comment = csv.read_csv(fs)

groups = df.columns.str.replace("_.*", "", regex=True).unique()
groups = [g for g in groups if g not in ["", "mvn"]]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
events = set([re.sub(".*obs_eep_", "", cn) for cn in df.columns
              if re.search("obs_eep", cn)])

paef = pe.withStroke(linewidth=4,
                     foreground="w")
ncols = 3
ng = len(groups)
nrows = ng // ncols + int(ng % ncols > 0)
mosaic = [[groups[ncols * ir + ic] if ncols * ir + ic < ng else "."
           for ic in range(ncols)] for ir in range(nrows)]

for event in events:
    LOGGER.info(f"Plotting {event}", nret=1)
    plt.close("all")
    fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)
    for aname, ax in axs.items():
        grp = aname
        LOGGER.info(f"Group {grp}", ntab=1)

        se = df.loc[:, f"{grp}_obs_eep_{event}"]
        x0 = round(math.log10(max(1e-10, se.quantile(0.001))), 2)
        x1 = round(math.log10(se.max()), 2)
        bins = np.logspace(x0, x1, 50)

        ax.hist(se, bins=bins, facecolor="0.8", edgecolor="0.2")

        m = se.mean()
        y0, y1 = ax.get_ylim()
        xy = (m, (y0 + y1) / 2)
        ax.annotate(f"Mean\n{m:0.1e}", xy, (0, 5),
                    xycoords="data", textcoords="offset points",
                    fontweight="bold", va="bottom", ha="center",
                    fontsize=15,
                    path_effects=[paef])
        ax.plot([m, m], [y0, y1], "k-", lw=2)

        title = f"{grp[1:]}"
        xlab = "Probability [-]"
        ax.set(xscale="log", title=title,
               xlabel=xlab, ylim=(y0, y1))

    ftitle = f"Event {event}\n"
    fig.suptitle(ftitle, fontsize=20, fontweight="bold")

    LOGGER.info("Saving to disk", ntab=1)
    fp = fimg / f"{basename}_{event}.png"
    fig.savefig(fp)

LOGGER.completed()
