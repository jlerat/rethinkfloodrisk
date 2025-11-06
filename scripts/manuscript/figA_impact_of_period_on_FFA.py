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
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from floodstan import freqplots
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

awidth = 6
aheight = 5
fdpi = 300

ptype = "gumbel"

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

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

stations = datahub.get_stations()

ffa = {}
data = {}
for ftask in fout.glob("*TASK*"):
    # Setup folders
    taskid = int(re.sub("^.*TASK", "", ftask.stem))

    # Get data
    fd = ftask / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    if diag["pcensor"] != 0.3 or diag["timeperiod"] == "PRE2008":
        continue

    period = diag["timeperiod"]

    LOGGER.info(f"Load report TASK {taskid} period={period}")
    fr = ftask / f"postprocess_report_TASK{taskid}.csv"
    df, _ = csv.read_csv(fr, index_col=0)
    ffa[period] = df

    fd = ftask / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        d = json.load(fo)
        data[period] = np.array(d["y"])

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
for istation, (stationid, sinfo) in enumerate(stations.iterrows()):
    LOGGER.info(f"Plotting {istation + 1} ({stationid})")
    plt.close("all")
    mosaic = [[per for per in data.keys()]]
    nrows = len(mosaic)
    ncols = len(mosaic[0])
    fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic, sharey=True)

    for iax, (aname, ax) in enumerate(axs.items()):
        period = aname

        # Plot data
        peaks = data[period][:, istation]
        freqplots.plot_data(ax, peaks, ptype)

        # Plot FFA
        df = ffa[period]
        quantiles = df.filter(regex=f"DESIGN.*\\[{istation + 1}\\]", axis=0)
        aris = quantiles.index.to_series().str\
                .replace(".*ERI|\\[.*", "", regex=True).astype(float).values

        freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                          center_column="POSTERIOR_PREDICTIVE",
                                          q0_column="5%",
                                          q1_column="95%",
                                          alpha=0.3,
                                          facecolor="tab:blue",
                                          edgecolor="k")

        retp = [5, 10, 100, 500]
        aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

        title = f"({letters[iax]}) {period}"
        ylab = "Peak flow [m3.s-1]" if iax == 0 else ""
        ax.set(title=title, ylabel=ylab)
        ax.grid(axis="y")

    ftitle = f"{sinfo.NAME} ({stationid})"
    fig.suptitle(ftitle, fontweight="bold")

    fp = fimg / f"{basename}_station{istation + 1}.png"
    fig.savefig(fp, dpi=fdpi)

LOGGER.completed()
