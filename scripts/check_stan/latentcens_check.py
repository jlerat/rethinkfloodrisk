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
from scipy.stats import norm
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel, write_stan_json

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from floodstan.sample import get_logger
from floodstan import report
from floodstan.marginals import Gumbel

from pyrethink import sample
from pyrethink import datahub

import importlib
importlib.reload(sample)

np.random.seed(5446)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run stan check",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-o", "--overwrite", help="Overwrite specific executable (or all if =0)",
                    type=int, default=-1)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-c", "--clean", help="Clean files",
                    action="store_true", default=False)
args = parser.parse_args()
overwrite = args.overwrite
debug = args.debug
clean = args.clean

versions = [1, 3]

# Data generation
nval = 200
P = 2
pcens = 0.8

if debug:
    nwarm = 100
    nchains = 5
    nsamples = 100
else:
    nwarm = 10000
    nchains = 5
    nsamples = 10000

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "check_stan" / "latentcens_check"
fout.mkdir(exist_ok=True, parents=True)

if clean:
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
gum = Gumbel()

truepeaks = datahub.get_truepeaks()
truepeaks = truepeaks.iloc[:, :P]
truepeaks = truepeaks.loc[truepeaks.iloc[:, 0].notnull()]
sv = sample.StanSamplingMultivariate(truepeaks, pcensor=pcens)

ylocn_true = sv.initial_parameters["ylocn"][0]
ylogscale_true = sv.initial_parameters["ylogscale"][0]
gum.params = [ylocn_true, ylogscale_true, 0]

data = gum.rvs(size=nval * 2).reshape((nval, 2))
sv = sample.StanSamplingMultivariate(data, pcensor=pcens)

stan_data = sv.to_dict()
stan_data["y"] = stan_data["y"][:, 0]

stan_data["censor"] = stan_data["censors"][0]

for dtype in ["obs", "cens"]:
    stan_data[f"idx_{dtype}"] = [idx for idx in stan_data[f"idx_{dtype}"] if idx[1] == 1]
    stan_data[f"N{dtype}"] = len(stan_data[f"idx_{dtype}"])

assert stan_data["Nobs"] + stan_data["Ncens"] == len(stan_data["y"])
assert stan_data["Ncens"] == (stan_data["y"] < stan_data["censor"]).sum()

stan_data.pop("P")
stan_data.pop("censors")
stan_data.pop("Nmiss")
stan_data.pop("idx_miss")
stan_data.pop("eta_prior")

stan_inits = sv.initial_parameters
stan_inits["ylocn"] = stan_inits["ylocn"][0]
stan_inits["ylogscale"] = stan_inits["ylogscale"][0]

ncens = stan_data["Ncens"]
stan_inits["wlat_cens"] = stan_inits["wlat_cens"][:ncens]
stan_inits["ulat_cens"] = np.random.uniform(0, 0.5, size=ncens)
stan_inits.pop("L_cor")
stan_inits.pop("zlat_miss")

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
df = {}
diag = {}
for version in versions:
    LOGGER.info(f"Loading stan model v{version}")
    stan_file = froot / "scripts" / "check_stan" / f"latentcens{version}_check.stan"
    suffix = ".exe" if os.name == "nt" else ""
    exe_file = stan_file.parent / f"{stan_file.stem}{suffix}"
    if not exe_file.exists():
        exe_file = None
    elif overwrite in [0, version]:
        exe_file.unlink()
        exe_file = None

    model = CmdStanModel(stan_file=stan_file,
                         exe_file=exe_file)

    LOGGER.info(".. done")

    LOGGER.info("Run model")
    kwargs = {}
    kwargs["chains"] = nchains
    kwargs["seed"] = 5446
    kwargs["iter_warmup"] = nwarm
    kwargs["iter_sampling"] = nsamples // nchains
    kwargs["show_progress"] = True
    kwargs["output_dir"] = fout

    if version == 3:
        stan_inits["wlat_cens"] = np.random.uniform(0, 1, ncens)
        kwargs["adapt_delta"] = 0.99

    smp = model.sample(data=stan_data,
                       inits=stan_inits,
                       **kwargs)
    LOGGER.info(".. done")

    # Process
    diag = report.process_stan_diagnostic(smp.diagnose())
    #for n in ["treedepth", "rhat", "ebfmi"]:
    #    assert diag[n] == "satisfactory"

    df[version] = smp.draws_pd()

plt.close("all")

# Chains
mosaic = [["ylocn", "ylogscale"],
          ["ucensor", "yrnd"]]
nrows, ncols = len(mosaic), len(mosaic[0])
w = 6
fig = plt.figure(figsize=((ncols * w, nrows * w * 0.8)),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic)
for pname, ax in axs.items():
    ses = []
    for v, ddf in df.items():
        se = ddf.loc[:, pname]
        ses.append(se)

    xa = np.min([se.min() for se in ses])
    xb = np.max([se.max() for se in ses])
    bins = np.linspace(xa, xb, 30)

    labs = ["censored", "latent norm", "latent unif"]
    for ise, se in enumerate(ses):
        ax.hist(se, bins=bins, alpha=0.5, edgecolor="0.4", label=labs[ise])

    if pname == "ylocn":
        putils.line(ax, 0, 1, ylocn_true, 0, color="tab:red", lw=2)
    elif pname == "ylogscale":
        putils.line(ax, 0, 1, ylogscale_true, 0, color="tab:red", lw=2)
    elif pname == "ucensor":
        putils.line(ax, 0, 1, pcens, 0, color="tab:red", lw=2)

    ax.set(title=pname)
    ax.legend()

fp = fout / f"params.png"
fig.savefig(fp)
plt.show()


LOGGER.info("Process completed")
