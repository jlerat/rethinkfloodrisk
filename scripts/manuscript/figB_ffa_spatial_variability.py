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
from hydrodiy.plot import putils, violinplot
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA spatial variability",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

awidth = 6
aheight = 5
fdpi = 300

stationid_target = "203002"
eep_target = 1 - 1e-2

# Task pcensor=0.3, period=ALL
taskid = 2

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

LOGGER.info(f"Load report TASK {taskid}")
fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
df, comment = csv.read_csv(fs)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Plot violin")
plt.close("all")
fig, ax = plt.subplots(figsize=(awidth, aheight),
                       layout="constrained")

ee = (1 - df.filter(regex="smp", axis=1)) * 100
cols = ee.columns.to_series().str.replace(".*_cond[0-9]{6}_|_smp_cdf", "", regex=True)
ee.columns = cols

vm = violinplot.Violin(ee, number_format="0.1f")
vm.draw()

ax.set(xlabel="Station", ylabel="Event Exceedance Probability [%]")

fp = fimg / f"{basename}.png"
fig.savefig(fp)

LOGGER.completed()
