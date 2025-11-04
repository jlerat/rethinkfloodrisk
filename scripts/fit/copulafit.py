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

from hydrodiy.io import csv, iutils

from floodstan import marginals
from floodstan import report
from floodstan.sample import get_logger

from pyrethink import datahub
from pyrethink import sample
from pyrethink import mv_censored_sampling

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fit copula model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ns", "--nsamples", help="Number of MCMC samples",
                    type=int, default=10000)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-p", "--progress", help="Show progress",
                    action="store_true", default=False)
parser.add_argument("-c", "--pcensor", help="Censoring threshold",
                    type=float, default=0.3)
args = parser.parse_args()

debug = args.debug
pcensor = args.pcensor

if debug:
    stan_nwarm = 200
    stan_nchains = 3
    stan_nsamples = 200
    stan_logger = True
else:
    stan_nwarm = 10000
    stan_nchains = 10
    stan_nsamples = args.nsamples
    stan_logger = False

stan_progress = args.progress

stan_seed = 5446

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

basename = source_file.stem
fout = froot / "outputs" / basename
fout.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
flog = froot / "logs" / basename / f"{basename}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
if flog.exists():
    try:
        flog.unlink()
    except:
        pass

LOGGER = get_logger(stan_logger=stan_logger, flog=flog)

if debug:
    fout = flog.parent / "outputs"
    fout.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
stations = datahub.get_stations()

truepeaks = datahub.get_truepeaks().drop("WATERYEAR", axis=1)

if debug:
    truepeaks = truepeaks.iloc[:, :4]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Configure stan sampler")
sv = sample.StanSamplingMultivariate(truepeaks, pcensor=pcensor)
stan_data = sv.to_dict()
stan_inits = sv.initial_parameters

fout_stan = fout / "stan"
fout_stan.mkdir(exist_ok=True)
for f in fout_stan.glob("*.*"):
    f.unlink()

kw = dict(data=stan_data,
          seed=stan_seed,
          iter_sampling=stan_nsamples // stan_nchains,
          output_dir=fout_stan,
          chains=stan_nchains,
          parallel_chains=stan_nchains,
          iter_warmup=stan_nwarm,
          show_progress=stan_progress,
          inits=stan_inits)

LOGGER.info("Start sampling")
smp = mv_censored_sampling(**kw)

LOGGER.info("Process samples and save to disk")
df = smp.draws_pd()
diag = report.process_stan_diagnostic(smp.diagnose())

fd = fout / f"{basename}_samples.csv"
csv.write_csv(df, fd, "STAN samples",
              source_file, compress=True)

fd = fout / f"{basename}_diagnostic.json"
with fd.open("w") as fo:
    json.dump(diag, fo, indent=4)

LOGGER.info("Process completed")

