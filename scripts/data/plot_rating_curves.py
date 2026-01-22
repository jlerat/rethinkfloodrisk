#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-20 16:27:36.472013
## Comment : Collect streamflow data
##
## ------------------------------

import sys
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
parser = argparse.ArgumentParser(description="Plot rating curves",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

axwidth = 5
axheight = 5
fdpi = 200

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata = froot / "data"

fimg = froot / "images" / "rating_curves"
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

if debug:
    stations = stations.iloc[:1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
nstations = len(stations)
for istation, (stationid, sinfo) in enumerate(stations.iterrows()):
    LOGGER.info(f"Plotting {stationid} ({istation + 1}/{nstations})")

    rcs, _ = datahub.get_rating_curves(stationid)
    times = pd.DatetimeIndex(rcs.keys()).sort_values().astype(str)
    last = times[-1]

    plt.close("all")
    fig, axs = plt.subplots(ncols=2,
                           figsize=(2 * axwidth, axheight),
                           layout="constrained")

    colors = putils.cmap2colors(len(rcs) + 1, "magma")[:-1][::-1]

    for irc, (time, rc) in enumerate(rcs.items()):
        x = rc.loc[:, "STREAMFLOW[m3_s-1]"]
        y = rc.loc[:, "WATERLEVEL[m]"]

        ipos = (x > 1e-2) & (y > -1)
        x = x[ipos]
        y = y[ipos]

        lw = 1
        ls = "-"

        for iax, ax in enumerate(axs):
            ax.plot(x, y, "-", color=colors[irc], lw=lw, ls=ls, label=time)

            if iax == 0 and time == last:
                # Lin interp close to the top of the rating curve
                ihigh = y > y.max() - 1
                xx = x[ihigh]
                X = np.column_stack([np.ones(ihigh.sum()), xx])
                theta, _, _, _ = np.linalg.lstsq(X, y[ihigh], rcond=1e-6)

                xx0 = xx.values[[0, -1]]
                a, b = theta
                lab = f"h = {a:+0.2e} {b:+0.2e}" + r"$×$q"
                ax.plot(xx0, a + b * xx0, "--",
                        label=lab, lw=1.5, color="tab:green")

    for iax, ax in enumerate(axs):
        if iax == 0:
            ax.legend(loc=4, fontsize="x-small")
        else:
            ax.set_xscale("asinh")

    ftitle = f"{sinfo.NAME} {stationid}"
    fig.suptitle(ftitle, fontsize="large",
                 fontweight="bold")

    fp = fimg / f"rating_curve_{stationid}.png"
    fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

