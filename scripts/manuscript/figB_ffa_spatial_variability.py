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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils, violinplot

from floodstan.report import STAN_DIAGNOSTIC_VARIABLES as SDV

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA spatial variability",
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

awidth = 7
aheight = 5
fdpi = 300

stationid_target = "203002"

exclude = "NONE"

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

fopm = fout / "copulafit_options.json"
opm = hyruns.OptionManager.from_file(fopm)

taskid = opm.search(pcensor=f"{pcensor:0.1f}",
                    exclude=exclude,
                    rho_min=f"{rho_min:0.1f}")[0]
mess = f"Load report TASK {taskid} exclude={exclude}"\
       + f" pcensor={pcensor} rho_min={rho_min}"
LOGGER.info(mess)

# Select fit task with
fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_diagnostic_TASK{taskid}.json"
with fd.open("r") as fo:
    diag = json.load(fo)

for vn in SDV:
    LOGGER.info(f"{vn}: {diag[vn][:50]}", ntab=1)

fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
mvnproc, comment = csv.read_csv(fs)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
aep_targets = mvnproc.columns.to_series()\
        .filter(regex="GALL_log10pall_aeptarget")\
        .str.replace(".*get_p", "", regex=True)\
        .str.replace("_", ".").astype(float).values

for aep_target in aep_targets:
    LOGGER.info(f"Plot violin rho_min={rho_min} p={aep_target}")

    plt.close("all")
    fig, ax = plt.subplots(figsize=(awidth, aheight),
                           layout="constrained")

    etxt = re.sub("\\.", "_", f"{aep_target:0.02f}")
    aep = (1 - mvnproc.filter(regex=f".*p{etxt}_.*_smp", axis=1)) * 100
    cols = aep.columns.to_series().str.replace(f".*_p{etxt}_|_smp_cdf", "", regex=True)
    aep.columns = cols

    vm = violinplot.Violin(aep, number_format="0.1f")
    vm.draw()

    ax.set(xlabel="Station", ylabel="Annual Exceedance Probability [%]")

    fp = fimg / f"{basename}_pcensor{pcensor}_rhomin{rho_min}_p{aep_target:0.02f}.png"
    fig.savefig(fp)

LOGGER.completed()
