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

from hydrodiy.io import csv, iutils, hyruns
from pyrethink import report
from pyrethink import postpredchecks as ppc

import importlib
importlib.reload(report)
importlib.reload(ppc)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="copula model result post-processing",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=-1)
parser.add_argument("-v", "--version", help="version",
                    type=str, required=True)
args = parser.parse_args()
version = args.version
taskid = args.taskid
debug = taskid < 0
taskid = max(0, taskid)

design_eris = [1.1, 1.2, 1.4, 1.6, 1.8,
               2, 5, 10, 20, 50, 70, 100, 150,
               200, 300, 500, 700, 1000]

# Options for this job depends on copula fit options.
# hence they are defined in the data section

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
basename = source_file.stem
froot = source_file.parent.parent.parent

# Get options
fopm = froot / "outputs" / f"copulafit_v{version}" / "copulafit_options.json"
opm_fit = hyruns.OptionManager.from_file(fopm)
fit_taskids = np.arange(opm_fit.ntasks)


opm = hyruns.OptionManager()
opm.from_cartesian_product(fit_taskid=fit_taskids,
                           jobid=[0, 1])
task = opm.get_task(taskid)
fit_taskid = task.fit_taskid
jobid = task.jobid

copula = opm_fit.get_task(fit_taskid).copula

fout = froot / "outputs" / f"copulafit_v{version}" / f"copulafit_TASK{fit_taskid}"

if debug:
    fit_taskid = -1
    jobid = 0
    fout = froot / "logs" / "copulafit" / f"copulafit_v{version}"

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / basename / f"{basename}_TASK{taskid}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
LOGGER = iutils.get_logger(basename, flog=flog, console=debug,
                           contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")
task.log(LOGGER)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

fsmp = fout / f"copulafit_samples_TASK{fit_taskid}.zip"
tsmp = fsmp.stat().st_mtime
samples = pd.read_csv(fsmp, skiprows=15)

fd = fout / f"copulafit_data_TASK{fit_taskid}.json"
with fd.open("r") as fo:
    stan_data = json.load(fo)

yobs = np.array(stan_data["y"])
probs = np.array(stan_data["partitions_probabilities"])

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
if jobid == 0:
    # Check time stamp to avoid redoing
    frep = fout / f"{basename}_report_TASK{fit_taskid}.csv"
    compute = True
    if frep.exists():
        trep = frep.stat().st_mtime
        if trep > tsmp:
            compute = False

    if compute:
        LOGGER.info("Computing report")
        stat, df = report.ffa_report(samples,
                                     design_eris=design_eris,
                                     logger=LOGGER)

        LOGGER.info("Store report")
        csv.write_csv(stat, frep, "Stat report",
                      source_file, write_index=True,
                      compress=False, lineterminator="\n")

    else:
        LOGGER.info("Report already available. Skip.")

elif jobid == 1:
    funiv = fout / f"{basename}_postpredchecks_univ_TASK{fit_taskid}.csv"
    compute = True
    if funiv.exists():
        tuniv = funiv.stat().st_mtime
        if tuniv > tsmp:
            compute = False

    if compute:
        LOGGER.info("Computing posterior predictive checks")
        ppu, ppb, data = ppc.posterior_predictive_checks(yobs, samples,
                                                         copula,
                                                         probs,
                                                         logger=LOGGER)

        LOGGER.info("Store posterior predictive checks")
        csv.write_csv(ppu, funiv, "Univariate post pred checks",
                      source_file, write_index=True,
                      compress=False, lineterminator="\n")

        fbiv = fout / f"{basename}_postpredchecks_biv_TASK{fit_taskid}.csv"
        csv.write_csv(ppb, fbiv, "Bivariate post pred checks",
                      source_file, write_index=True,
                      compress=False, lineterminator="\n")
    else:
        LOGGER.info("Posterior predictive checks already available. Skip.")

LOGGER.completed()
