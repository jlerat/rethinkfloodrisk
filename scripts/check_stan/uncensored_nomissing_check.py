#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-26 Sun 11:37 AM
## Comment : Check indexing data
##
## ------------------------------

import os
import sys
import re
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cmdstanpy import CmdStanModel

from hydrodiy.io import csv, iutils

from pyrethink import sample
from floodstan.sample import get_logger
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "check_stan" / "uncensored_nomissing_check"
fout.mkdir(exist_ok=True, parents=True)

for f in fout.glob("*.*"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
LOGGER = get_logger(stan_logger=False)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
data = datahub.get_truepeaks()
sv = sample.StanSamplingMultivariate(data)
stan_data = sv.to_dict()
stan_inits = sv.initial_parameters
for k, v in stan_inits.items():
    stan_data[k] = v

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "uncensored_nomissing_check.stan"
suffix = ".exe" if os.name == "nt" else ""
exe_file = stan_file.parent / f"{stan_file.stem}{suffix}"
if not exe_file.exists():
    exe_file = None
kwargs = dict()
model = CmdStanModel(stan_file=stan_file,
                     exe_file=exe_file)
LOGGER.info(".. done")

LOGGER.info("Run model")
kwargs["chains"] = 1
kwargs["seed"] = 5446
kwargs["iter_warmup"] = 1
kwargs["iter_sampling"] = 1
kwargs["fixed_param"] = True
kwargs["show_progress"] = False
smp = model.sample(data=stan_data, **kwargs)
df = smp.draws_pd()
z = df.filter(regex="^z", axis=1).values.reshape(y.T.shape).T
LOGGER.info(".. done")

LOGGER.info("Process completed")
