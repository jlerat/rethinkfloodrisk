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
parser.add_argument("-v", "--version", help="version",
                    type=int, required=True)
parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                    type=float, default=0.3)
parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                    type=float, default=-1.)
parser.add_argument("-cp", "--copula", help="Copula parameter",
                    type=str, default="^0|2.01")
args = parser.parse_args()

version = args.version
pcensor = args.pcensor
rho_min = args.rho_min
copula = args.copula
has_clusters = False

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

fout = froot / "outputs" / f"copulafit_v{version}"

basename = source_file.stem
fimg = froot / "images" / "manuscript" / basename
fimg.mkdir(exist_ok=True, parents=True)
clear = True
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

taskids = opm.search(pcensor=f"{pcensor:0.1f}",
                     exclude=exclude,
                     copula=copula,
                     has_clusters=has_clusters,
                     rho_min=f"{rho_min:0.1f}")
data = {}
for taskid in taskids:
    fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    pc = diag["task_pcensor"]
    ex = diag["task_exclude"]
    rm = diag["task_rho_min"]
    cop = diag["task_copula"]
    hc = diag["task_has_clusters"]
    config = (pc, ex, rm, cop, hc)
    otxt = f"PC{pc}_RM{rm}_C{cop}_HC{hc}"
    mess = f"Load report TASK {taskid} {otxt}"
    LOGGER.info(mess, nret=1)

    fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
    mvnproc, comment = csv.read_csv(fs)
    data[config] = mvnproc

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
for config, mvnproc in data.items():
    pcensor, exclude, rho_min, copula, has_clusters = config
    otxt = f"PC{pcensor}_RM{rho_min}_C{copula}_HC{has_clusters}"
    LOGGER.info(f"Plotting {otxt}", nret=1)

    #aep_targets = mvnproc.columns.to_series()\
    #        .filter(regex="GALL_log10pall_aeptarget")\
    #        .str.replace(".*get_p", "", regex=True)\
    #        .str.replace("_", ".").astype(float).values
    aep_targets = [0.99]

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

        fn = f"{basename}_PC{pcensor}_RM{rho_min}_C{copula}"\
             + f"_HC{has_clusters}_AEP{aep_target}_v{version}.png"
        fp = fimg / fn
        fig.savefig(fp)

LOGGER.completed()
