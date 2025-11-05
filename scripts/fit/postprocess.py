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
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils
from pyrethink import report
from pyrethink import postpredchecks as ppc

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="copula model result post-processing",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=-1)
args = parser.parse_args()
taskid = args.taskid
debug = taskid < 0

design_eris = [1.1, 1.2, 1.4, 1.6, 1.8,
               2, 5, 10, 20, 50, 70, 100, 150,
               200, 300, 500, 700, 1000]

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / f"copulafit_TASK{taskid}"

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / basename / f"{basename}_TASK{taskid}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
LOGGER = iutils.get_logger(basename, flog=flog, console=debug,
                           contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")
LOGGER = iutils.get_logger(basename)

if taskid < 0:
    fout = froot / "logs" / "copulafit" / "outputs"

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
fs = fout / f"copulafit_samples_TASK{taskid}.zip"
samples = pd.read_csv(fs, skiprows=15)

fd = fout / f"copulafit_data_TASK{taskid}.json"
with fd.open("r") as fo:
    stan_data = json.load(fo)

yobs = np.array(stan_data["y"])

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Computing report")
stat, df = report.ffa_report(samples,
                             design_eris=design_eris)

LOGGER.info("Store report")
fr = fout / f"{basename}_report_TASK{taskid}.csv"
csv.write_csv(stat, fr, "Stat report",
              source_file, write_index=True,
              compress=False, lineterminator="\n")

LOGGER.info("Computing posterior predictive checks")
ppu, ppb, data = ppc.posterior_predictive_checks(yobs, samples)

LOGGER.info("Store posterior predictive checks")
fu = fout / f"{basename}_postpredchecks_univ_TASK{taskid}.csv"
csv.write_csv(ppu, fu, "Univariate post pred checks",
              source_file, write_index=True,
              compress=False, lineterminator="\n")

fb = fout / f"{basename}_postpredchecks_biv_TASK{taskid}.csv"
csv.write_csv(ppb, fb, "Bivariate post pred checks",
              source_file, write_index=True,
              compress=False, lineterminator="\n")



LOGGER.completed()
