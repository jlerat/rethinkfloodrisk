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
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug
excludes = ["NONE", "2022-02-27"]
pcens = [0.5]

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
    if exclude not in excludes:
        continue

    pcensor = diag["pcensor"]
    if pcensor not in pcens:
        continue

    LOGGER.info(f"Load data from TASK {taskid}", nret=1)
    LOGGER.info(f"pcensor = {pcensor}", ntab=1)
    LOGGER.info(f"exclude = {exclude}", ntab=1)
    for vn in SDV:
        LOGGER.info(f"{vn}: {diag[vn][:20]}", ntab=1)

    fd = fold / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        data = json.load(fo)

    stationids = data["stationids"]

    postpred[(exclude, pcensor)] = {}
    for ppt in ["univ", "biv"]:
        fp = f"postprocess_postpredchecks_{ppt}_TASK{taskid}.csv"
        fp = fold / fp
        df = pd.read_csv(fp, skiprows=15)
        df.columns = ["VARIABLE"] + df.columns[1:].tolist()
        postpred[(exclude, pcensor)][ppt] = df

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ncols = 3
varnames = [f"{ppt}_{vn}" for ppt in variables for vn in variables[ppt]]
nv = len(varnames)
nrows = nv // ncols + int(nv % ncols > 0)
mosaic = [[varnames[ncols * ir + ic] if ncols * ir + ic < nv else "."
          for ic in range(ncols)] for ir in range(nrows)]

for (exclude, pcensor), pp in postpred.items():
    LOGGER.info(f"Plot ppchecks pcensor={pcensor} exclude={exclude}",
                nret=1)

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
        else:
            bins = np.concatenate([[0], np.linspace(0.05, 0.95, 5), [1]])
            df.squeeze().plot(ax=ax, kind="hist", bins=bins,
                              edgecolor="0.2", facecolor="0.8")

        for x in [0.05, 0.95]:
            putils.line(ax, 0, 1, x, 0, "r--")

        title = f"{ppt} post pred checks / {varname}"
        xlab = "check pvalue [-]"
        ax.set(title=title, xlabel=xlab, xlim=(0, 1))

    ftitle = f"Pcensor={pcensor}  Exclude={exclude}"
    fp = fimg / f"{basename}_pcens{pcensor}_exclude{exclude}.png"
    fig.savefig(fp)

LOGGER.completed()
