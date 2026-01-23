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
from scipy.stats import t as student

from cmdstanpy import CmdStanModel

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from floodstan.sample import get_logger

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

nu = 1000
ndf = 20

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "check_stan" / "student_check"
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

N = 100
u = np.linspace(1./N, 1 - 1./N, N)

P = 50
df = np.linspace(0.5, 5, P)

stan_data = {
    "N": N,
    "u": u,
    "P": P,
    "df": df
}

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "student_check.stan"
suffix = ".exe" if os.name == "nt" else ""
exe_file = stan_file.parent / f"{stan_file.stem}{suffix}"
if not exe_file.exists():
    exe_file = None
kwargs = dict()
model = CmdStanModel(stan_file=stan_file,
                     exe_file=exe_file)
LOGGER.info(".. done")

LOGGER.info("Run model")

kwargs = {}
kwargs["chains"] = 1
kwargs["seed"] = 5446
kwargs["iter_warmup"] = 1
kwargs["iter_sampling"] = 1
kwargs["fixed_param"] = True
kwargs["show_progress"] = False
smp = model.sample(data=stan_data, **kwargs)
x = smp.draws_pd().filter(regex="^z", axis=1)
x = x.values.reshape((P, N)).T

expected = np.zeros_like(x)
for i in range(N):
    for j in range(P):
        expected[i, j] = student.ppf(u[i], df[j], loc=0, scale=1)

diff = np.arcsinh(x) - np.arcsinh(expected)
assert np.abs(diff).max() < 1e-5

LOGGER.info(".. done")

LOGGER.info("Process completed")
