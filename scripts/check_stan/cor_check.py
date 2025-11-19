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
parser = argparse.ArgumentParser(description="Run stan check",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-o", "--overwrite", help="Overwrite executable",
                    action="store_true", default=False)
parser.add_argument("-cm", "--covmodel", help="Covariance model",
                    type=str, choices=["LKJ", "IW", "BA"], default="LKJ")
parser.add_argument("-e", "--eta", help="LKJ eta parameter",
                    type=float, default=1.)
parser.add_argument("-ri", "--rho_min", help="Rho min",
                    type=float, default=0.)
parser.add_argument("-rx", "--rho_max", help="Rho max",
                    type=float, default=1.)
parser.add_argument("-p", "--P", help="Dimension",
                    type=int, default=8)
args = parser.parse_args()
overwrite = args.overwrite
covmodel = args.covmodel
eta = args.eta
rho_min = args.rho_min
rho_max = args.rho_max
P = args.P

nwarm = 5000
nsamples = 10000
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
stan_data = {
    "P": P,
    "eta_prior": eta,
    "rho_min": rho_min,
    "rho_max": rho_max
    }

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
#LOGGER.info("check sample from inverse wishart")
#
#nsmp = 10000
#nu = P + 1
#C = np.zeros((nsmp, P, P))
#for i in range(nsmp):
#    x = np.random.randn(P, nu)
#    S = np.cov(x)
#    Si = np.linalg.inv(S)
#    inv_sigs = 1. / np.sqrt(np.diag(Si))[:, None]
#    C[i] = inv_sigs * Si * inv_sigs
#
#plt.close("all")
#bins = np.linspace(-1, 1, 50)
#fig, axs = plt.subplots(ncols=3, layout="constrained")
#i1 = [1, P-1, P-1]
#i2 = [0, 0, P-2]
#for iax, ax in enumerate(axs):
#    ax.hist(C[:, i1[iax], i2[iax]], bins=bins, ec="0.2", fc="0.8")
#plt.show()
#sys.exit()


LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "cor_check.stan"
suffix = ".exe" if os.name == "nt" else ""
exe_file = stan_file.parent / f"{stan_file.stem}{suffix}"
if not exe_file.exists():
    LOGGER.info("File does not exists. Re-compile.")
    exe_file = None
elif overwrite:
    LOGGER.info("Erase exe file and re-compile")
    exe_file.unlink()
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

for cm in ["LKJ", "IW"]:
    for i in range(1, P + 1):
        se = df.loc[:, f"cor_{cm}[{i},{i}]"]
        assert np.allclose(se, 1.)

plt.close("all")
idx1, idx2 = np.tril_indices(stan_data["P"], -1)

nax = len(idx1)
ncols = min(P, 5)
nrows = nax // ncols + int(nax % ncols > 0)
mosaic = [[ncols * ir + ic if ncols * ir + ic < nax else "."
           for ic in range(ncols)] for ir in range(nrows)]

w = 2
fig = plt.figure(figsize=((ncols * w, nrows * w * 0.8)),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic)
bins = np.linspace(rho_min, rho_max, 50)

for iax, ax in axs.items():
    i1 = idx1[iax]
    i2 = idx2[iax]
    smp = df.loc[:, f"cor_{covmodel}[{i1 + 1},{i2 + 1}]"]
    ax.hist(smp, bins=bins, density=True, ec="0.2", fc="0.8")

    putils.line(ax, 0, 1, rho_min, 0, "k--", lw=0.9)
    putils.line(ax, 0, 1, rho_max, 0, "k--", lw=0.9)

    title = f"C[{i1},{i2}]"
    ax.set(title=title, yticks=[])

plt.show()

LOGGER.info("Process completed")
