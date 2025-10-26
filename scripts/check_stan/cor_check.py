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
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyrethink import sample
from floodstan.sample import get_logger
from floodstan import report
from pyrethink import datahub


# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

nwarm = 5000
nsamples = 5000
nchains = 5

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "check_stan" / "cor_check"
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
stan_inits = sv.initial_parameters
L_cor = stan_inits["L_cor"]
cor = L_cor @ L_cor.T
z = np.random.multivariate_normal(mean=np.zeros(len(cor)),
                                  cov=cor,
                                  size=len(data))
stan_data = {
    "N": len(z),
    "P": z.shape[1],
    "z": z,
    "eta_prior": 4.
    }

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "cor_check.stan"
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
kwargs["chains"] = nchains
kwargs["seed"] = 5446
kwargs["iter_warmup"] = nwarm
kwargs["iter_sampling"] = nsamples // nchains
kwargs["fixed_param"] = False
kwargs["show_progress"] = True
smp = model.sample(data=stan_data, **kwargs)
df = smp.draws_pd()
diag = report.process_stan_diagnostic(smp.diagnose())
LOGGER.info(".. done")

plt.close("all")
idx1, idx2 = np.tril_indices(stan_data["P"])
idx1 = idx1[1:]
idx2 = idx2[1:]

nax = len(idx1)
nrows = int(math.sqrt(nax))
ncols = nax // nrows
w = 2
fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                        figsize=((ncols * w, nrows * w * 0.8)),
                        layout="constrained")

for iax, idx in enumerate(zip(idx1, idx2)):
    i1, i2 = idx
    ref = L_cor[i1, i2]
    smp = df.loc[:, f"L_cor[{i1 + 1},{i2 + 1}]"]

    ax = axs.flat[iax]
    ax.hist(smp, density=True)
    putils.line(ax, 0, 1, ref, 0, "k--", lw=0.9)

    x0 = max(-1., ref - 0.1)
    x1 = min(1., ref + 0.1)
    ax.set(xlim=(x0, x1))

plt.show()

LOGGER.info("Process completed")
