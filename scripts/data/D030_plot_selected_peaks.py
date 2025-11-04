#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-20 16:27:36.472013
## Comment : Collect streamflow data
##
## ------------------------------

import sys
import os
import re
import json
import math
import argparse
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Collect streamflow data",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

version = datahub.DATA_VERSION

select_window = 3
plot_window = 31

axwidth = 5
axheight = 3
fdpi = 200

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata = froot / "data"

fimg = froot / "images" / "potpeaks"
fimg.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
stations = datahub.get_stations()
potpeaks = datahub.get_potpeaks()
qthresh = datahub.get_potpeaks_thresh()

if debug:
    potpeaks = potpeaks.loc["1974"]

daily = {}

for stationid in stations.index:
    fd = fdata / "dailymax" / f"dailymax_streamflow_{stationid}_v{version}.csv"
    df, _ = csv.read_csv(fd, index_col="DAY", parse_dates=True)
    daily[stationid] = df.filter(regex="MAX", axis=1).squeeze()

daily = pd.DataFrame(daily)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ncols = 3
nrows = 3
stationids = potpeaks.columns.to_series()\
    .filter(regex="_PEAK").str.replace("_PEAK", "").astype(int).tolist()
stationids = stationids + ["."] * (ncols * nrows - len(stationids))
mosaic = [[stationids[ncols * irow + icol] for icol in range(ncols)]
          for irow in range(nrows)]

Tqt = 5
quantiles = potpeaks.filter(regex="_PEAK").quantile(1 - 1/Tqt)
quantiles.index = quantiles.index.to_series().str.replace("_PEAK", "")

fpdf = fimg / "potpeaks.pdf"
with PdfPages(fpdf) as pdf:
    for ievent, (day, event) in enumerate(potpeaks.iterrows()):
        LOGGER.info(f"Event {ievent + 1} / {potpeaks.shape[0]}")
        plt.close("all")
        fig = plt.figure(figsize=(axwidth * ncols, axheight * nrows),
                         layout="constrained")
        axs = fig.subplot_mosaic(mosaic, sharex=True)

        offset = pd.DateOffset(days=select_window//2 + 1)
        start_select = day - offset
        end_select = day + offset

        offset = pd.DateOffset(days=plot_window//2 + 1)
        start_plot = day - offset
        end_plot = day + offset

        for aname, ax in axs.items():
            stationid = aname
            se = daily.loc[start_select:end_select, stationid]
            se_plot = daily.loc[start_plot:end_plot, stationid]

            name = re.sub(".*At ", "", stations.NAME[stationid])
            title = f"{stationid} {name}"
            ax.set_title(title, x=0.05, y=0.95,
                         fontweight="bold", va="top",
                         fontsize="small",
                         ha="left")

            if se.isnull().all():
                ax.set(xticks=[], yticks=[])
                continue

            se_plot.plot(ax=ax, label="")

            y0, y1 = ax.get_ylim()
            y0 = 0
            qq = quantiles[str(stationid)]
            qt = qthresh[str(stationid)]
            y1 = np.max([y1, qq * 1.1, qt * 1.1])

            sem = se.loc[[se.idxmax()]]
            ax.plot([sem.index] * 2, [y0, y1], "--",
                    color="tab:red", label="Qmax day")

            v1 = sem.squeeze()
            v2 = event[f"{stationid}_PEAK"]
            if abs(v1 - v2) > 1e-10:
                line = se * 0 + v2
                line.plot(ax=ax, color="purple",
                          label="Qmax event")

            line = se_plot * 0 + qq
            line.plot(ax=ax, color="0.5", lw=0.9, label=f"Quantile {Tqt}")

            line = se_plot * 0 + qt
            line.plot(ax=ax, color="0.5", lw=0.9, ls="--", label=f"POT thresh")

            ax.set(xlabel="", ylim=(y0, y1))
            ax.legend(fontsize="small", framealpha=0.)

        ftitle = f"{day.strftime('%b %y')} flood"
        fig.suptitle(ftitle, fontweight="bold", fontsize="large")

        pdf.savefig(fig)

LOGGER.completed()

