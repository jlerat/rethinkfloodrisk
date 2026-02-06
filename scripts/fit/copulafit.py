#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model
##
## ------------------------------

import sys
import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

from floodstan import marginals
from floodstan import report
from floodstan.sample import get_logger

from pyrethink import datahub
from pyrethink import sample
from pyrethink import mv_censored_no_missing_sampling

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fit copula model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--version", help="version",
                    type=int, required=True)
parser.add_argument("-ns", "--nsamples", help="Number of MCMC samples",
                    type=int, default=50000)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-p", "--progress", help="Show progress",
                    action="store_true", default=False)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=-1)
args = parser.parse_args()

debug = args.debug
version = args.version
taskid = args.taskid

# Configure stan
if debug:
    stan_nwarm = 200
    stan_nchains = 3
    stan_nsamples = 200
else:
    stan_nwarm = 10000
    stan_nchains = 10
    stan_nsamples = args.nsamples

stan_progress = args.progress

stan_seed = 5446

stan_args = {} #"adapt_delta": 0.9}

# Runner
opm = hyruns.OptionManager(stan_nwarm=stan_nwarm,
                           stan_nchains=stan_nchains,
                           stan_nsamples=stan_nsamples)

pcensors = [0., 0.3]

copulas = [0, 2.5, 3., 3.5, 4.]

excludes = ["NONE",
            "2021",
            "2007",
            "2016"]

rho_mins = [-1.]

has_clusters_all = [False, True]

opm.from_cartesian_product(pcensor=pcensors,
                           exclude=excludes,
                           copula=copulas,
                           has_clusters=has_clusters_all,
                           rho_min=rho_mins)

# Load task
task = opm.get_task(max(0, taskid))
pcensor = task.pcensor
exclude = task.exclude
rho_min = task.rho_min
copula = task.copula
has_clusters = task.has_clusters


if debug:
    pcensor = 0.3
    exclude = "2007"
    copula = 2.5
    has_clusters = True

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

basename = source_file.stem
fout = froot / "outputs" / f"copulafit_v{version}" / f"{basename}_TASK{taskid}"
fout.mkdir(exist_ok=True, parents=True)

fopm = fout.parent / f"{basename}_options.json"
opm.save(fopm)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
flog = froot / "logs" / basename / f"{basename}_TASK{taskid}.log"
flog.parent.mkdir(exist_ok=True, parents=True)

LOGGER = iutils.get_logger(basename, flog=flog, console=debug,
                           contextual=True)
LOGGER.info(f"number of tasks: {opm.ntasks}", nret=1)
LOGGER.context = f"TASK{taskid}"
LOGGER.log_dict(vars(args), "Command line arguments")

task.log(LOGGER)

if debug:
    fout = flog.parent / f"copulafit_v{version}"
    fout.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

ams, times, dows, stations = datahub.get_ams_concat()
censors = datahub.get_censors(pcensor)

# Exclude time period
if exclude != "NONE":
    iok = ams.index != int(exclude)
    ams = ams.loc[iok]
    dows = dows.loc[iok]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Configure stan sampler")
LOGGER.info(f"nwarm    = {stan_nwarm}", ntab=1, nret=1)
LOGGER.info(f"nchains  = {stan_nchains}", ntab=1)
LOGGER.info(f"nsamples = {stan_nsamples}", ntab=1)

sv = sample.StanSamplingMultivariate(ams, dows,
                                     copula=copula,
                                     censors=censors,
                                     rho_min=rho_min,
                                     rho_max=1.)
stan_data = sv.to_dict()
LOGGER.info(f"nobs    = {stan_data['Nobs']}", ntab=1, nret=1)
LOGGER.info(f"ncens   = {stan_data['Ncens']}", ntab=1)
LOGGER.info(f"nmiss   = {stan_data['Nmiss']}", ntab=1)
LOGGER.info(f"rho_min = {stan_data['rho_min']}", ntab=1)
LOGGER.info(f"rho_max = {stan_data['rho_max']}", ntab=1)

pcensors = (ams - censors < 0).sum() / ams.notnull().sum()
for ipn, (pname, pcensor) in enumerate(pcensors.items()):
    stationid = re.sub("_PEAK", "", pname)
    LOGGER.info(f"Prob censor {stationid} = {pcensor:0.2f}", nret=int(ipn==0))

stan_inits = sv.initial_parameters

# Clean stan folder
fout_stan = fout / "stan"
fout_stan.mkdir(exist_ok=True)
for f in fout_stan.glob("*.*"):
    f.unlink()

# Stan arguments
kw = dict(data=stan_data,
          seed=stan_seed,
          iter_sampling=stan_nsamples // stan_nchains,
          output_dir=fout_stan,
          chains=stan_nchains,
          parallel_chains=stan_nchains,
          iter_warmup=stan_nwarm,
          show_progress=stan_progress,
          inits=stan_inits)
kw.update(stan_args)

LOGGER.info("Start sampling", nret=1)
smp = mv_censored_no_missing_sampling(**kw)

LOGGER.info("Process samples and save to disk", nret=1)
df = smp.draws_pd()

diag = report.process_stan_diagnostic(smp.diagnose())

# Report stan diagnostic
for me in report.STAN_DIAGNOSTIC_VARIABLES:
    LOGGER.info(f"Stan diagnostic {me}: {diag[me]}")

diag["version"] = version
diag["stan_nchains"] = stan_nchains
diag["stan_nwarm"] = stan_nwarm
diag["taskid"] = taskid
task_opt = {f"task_{k}": v for k, v in task.to_dict()["options"].items()}
diag.update(task_opt)

fd = fout / f"{basename}_samples_TASK{taskid}.csv"
csv.write_csv(df, fd, f"STAN samples for task {taskid}",
              source_file, compress=True)

fd = fout / f"{basename}_diagnostic_TASK{taskid}.json"
with fd.open("w") as fo:
    json.dump(diag, fo, indent=4)

# Store data with additional info
stan_data["pcensors"] = pcensors.to_dict()
stan_data["ams_time"] = ams.index.tolist()
stan_data["stationids"] = ams.columns.tolist()
stan_data.update(task_opt)

fdd = fout / f"{basename}_data_TASK{taskid}.json"
for n in ["y", "idx_cens", "idx_obs", "idx_miss", "censors",
          "clusters", "clusters_counts",
          "partitions_id"]:
    stan_data[n] = stan_data[n].tolist()

with fdd.open("w") as fo:
    json.dump(stan_data, fo, indent=4)

fi = fout / f"{basename}_inits_TASK{taskid}.json"
for key, val in stan_inits.items():
    stan_inits[key] = val.tolist()
with fi.open("w") as fo:
    json.dump(stan_inits, fo, indent=4)

LOGGER.completed()

