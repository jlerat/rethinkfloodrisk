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

from cmdstanpy import CmdStanModel

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from floodstan.sample import get_logger
from floodstan import report

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
parser.add_argument("-o", "--overwrite", help="Overwrite executable",
                    action="store_true", default=False)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-n", "--nvars", help="Nb of variables",
                    type=int, default=3)
args = parser.parse_args()
overwrite = args.overwrite
debug = args.debug

# Data generation
nval = 120
P = args.nvars
sig = 0.1
pmiss = 0.
pcens = 0.2

if debug:
    nwarm = 100
    nchains = 5
    nsamples = 100
else:
    nwarm = 5000
    nchains = 5
    nsamples = 10000

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "check_stan" / "censored_check"
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
L_cor = np.zeros((P, P))
for i in range(P):
    v = np.exp(sig * np.random.normal(size=i+1))
    L_cor[i, :i+1] = v / math.sqrt((v*v).sum())

Cref = L_cor @ L_cor.T
assert np.allclose(np.diag(Cref), 1.)

c = Cref[np.triu_indices(P, 1)]
cp = np.percentile(c, [0, 50, 100])
txt = " ".join([f"{v:0.2f}" for v in cp])
LOGGER.info(f"Corr: {txt}")

z = np.zeros(P)
yref = np.random.multivariate_normal(mean=z, cov=Cref, size=nval)
y = yref.copy()

if pmiss > 0:
    n = np.prod(y.shape)
    imiss = np.random.choice(np.arange(n), int(n * pmiss),
                             replace=False)
    y.flat[imiss] = np.nan

    allmiss = np.where(pd.isnull(y).sum(axis=1) == P)[0]
    icol = np.random.choice(np.arange(P), len(allmiss))
    y[allmiss, icol] = yref[allmiss, icol]

sv = sample.StanSamplingMultivariate(y, pcensor=pcens)
stan_data = sv.to_dict()

stan_inits = sv.initial_parameters
stan_inits["wlat_cens"] = np.random.uniform(-1, 1, size=stan_data["Ncens"])

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "censored_check.stan"
suffix = ".exe" if os.name == "nt" else ""
exe_file = stan_file.parent / f"{stan_file.stem}{suffix}"
if not exe_file.exists():
    exe_file = None
elif overwrite:
    exe_file.unlink()
    exe_file = None
kwargs = dict()
model = CmdStanModel(stan_file=stan_file,
                     exe_file=exe_file)
LOGGER.info(".. done")

LOGGER.info("Run model")
kwargs["chains"] = nchains
kwargs["seed"] = 5446
kwargs["iter_warmup"] = nwarm
kwargs["iter_sampling"] = nsamples // nchains
kwargs["show_progress"] = True
kwargs["output_dir"] = fout
smp = model.sample(data=stan_data,
                   inits=stan_inits,
                   **kwargs)
LOGGER.info(".. done")

# Process
diag = report.process_stan_diagnostic(smp.diagnose())
df = smp.draws_pd()
lcor = df.filter(regex="^L_cor", axis=1)

Csmp = []
for _, smp in lcor.iterrows():
    L = smp.values.reshape((P, P)).T
    C = L @ L.T
    Csmp.append(C)

Csmp = np.array(Csmp)

plt.close("all")

# missing values
#for dtype in ["miss", "cens"]:
for dtype in ["cens"]:
    ncols, nrows = 2, 2
    w = 5
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=((ncols * w, nrows * w * 0.8)),
                            layout="constrained")
    y = stan_data["y"].copy()
    isin = np.zeros_like(y, dtype=int)
    for idx in stan_data[f"idx_{dtype}"]:
        isin[idx[0] - 1, idx[1] - 1] = 1

    if dtype == "cens":
        censors = stan_data["censors"][None, :]
        # Remove censored data
        y[y - censors < 0] = np.nan

    oneval = pd.isnull(y).sum(axis=1) == 1
    oneval = np.repeat(oneval[:, None], 3, axis=1) & pd.isnull(y)
    oneval &= isin == 1

    yrt = np.percentile(yref[oneval], 100 - 100 * nrows * ncols / oneval.sum())
    oneval &= yref > yrt
    oneval = np.where(oneval)

    for iax, ax in enumerate(axs.flat):
        i1 = oneval[0][iax] + 1
        i2 = oneval[1][iax] + 1

        yi = y[i1 - 1]
        imiss = np.isnan(yi)
        obs = yi[~imiss]
        true = yref[i1 - 1][imiss].squeeze()

        Cov11i = np.linalg.inv(Cref[~imiss][:, ~imiss])
        Cov22 = Cref[imiss][:, imiss]
        Cov21 = Cref[imiss][:, ~imiss]
        mu = (Cov21 @ Cov11i @ obs).squeeze()
        sig = np.sqrt(Cov22 - Cov21@Cov11i@Cov21.T).squeeze()

        idxs = stan_data[f"idx_{dtype}"]
        idx = next(i + 1 for i, ii in enumerate(idxs)
                   if ii[0] == i1 and ii[1] == i2)
        z = df.loc[:, f"zlat_{dtype}[{idx}]"]

        x0 = min(min(true, mu - 2 * sig), z.quantile(0.1))
        x1 = max(max(true, mu + 2 * sig), z.quantile(0.9))
        x0 = x0 - (x1 - x0) * 0.05
        x1 = x1 + (x1 - x0) * 0.05

        bins = np.linspace(x0, x1, 30)
        ax.hist(z, bins=bins, density=True,
                facecolor="0.8", edgecolor="0.2")

        xx = np.linspace(x0, x1, 500)
        yy = norm.pdf(xx, loc=mu, scale=sig)
        if dtype == "cens":
            cens = stan_data["censors"][i2 - 1]
            yy[xx > cens] = 0.
            yy = yy / (1 - norm.cdf(cens, loc=mu, scale=sig))

        ax.plot(xx, yy, "k--", lw=0.9)

        putils.line(ax, 0, 1, true, color="tab:red", lw=2)

        title = f"true = {true:0.2f}"
        ax.set(title=title, xlim=(x0, x1))

    fp = fout / f"{dtype}_values.png"
    fig.savefig(fp)

    plt.show()
    sys.exit()


# Correlations
idx1, idx2 = np.triu_indices(P, 1)
nax = len(idx1)
nrows = int(math.sqrt(nax))
ncols = nax // nrows
if nrows * ncols < len(idx1):
    ncols += nax - nrows * ncols
w = 3
fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                        figsize=((ncols * w, nrows * w * 0.8)),
                        layout="constrained")
delta_rho = 0.2
for iax, idx in enumerate(zip(idx1, idx2)):
    i1, i2 = idx
    ref = Cref[i1, i2]
    smp = Csmp[:, i1, i2]

    ax = axs.flat[iax]
    ax.hist(smp, density=True)
    putils.line(ax, 0, 1, ref, 0, "k--", lw=0.9)

    x0 = max(-1., ref - delta_rho)
    x1 = min(1., ref + delta_rho)
    ax.set(xlim=(x0, x1))
    title = f"C[{i1 + 1}, {i2 + 1}]"
    ax.set_title(title, x=0.05, y=0.95, va="top", ha="left")

for iax in range(nax, len(axs.flat)):
    axs.flat[iax].axis("off")

plt.show()


LOGGER.info("Process completed")
