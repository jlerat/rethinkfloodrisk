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
nval = 200
P = args.nvars
sig = 0.1
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

fout = froot / "outputs" / "check_stan" / "mv_censored_check"
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
gum = Gumbel()

truepeaks = datahub.get_truepeaks()
truepeaks = truepeaks.iloc[:, :P]

sv = sample.StanSamplingMultivariate(truepeaks, pcensor=pcens)
stan_data = sv.to_dict()
stan_inits = sv.initial_parameters

# Generate data
L_cor = stan_inits["L_cor"]
Cref = L_cor @ L_cor.T
z = np.zeros(len(Cref))

zref = np.random.multivariate_normal(mean=z, cov=Cref, size=nval)
ylocn = stan_inits["ylocn"]
ylogscale = stan_inits["ylogscale"]
yref = np.zeros_like(zref)
yref_withmiss = np.zeros_like(zref)
N = stan_data["N"]
for ivar in range(P):
    gum.params = [ylocn[ivar], ylogscale[ivar], 0.]
    yr = gum.ppf(norm.cdf(zref[:, ivar]))
    yref[:, ivar] = yr.copy()
    pmiss = pd.isnull(truepeaks.iloc[:, ivar]).sum() / N
    imiss = np.random.binomial(1, pmiss, size=nval)
    yr[imiss == 1] = np.nan
    yref_withmiss[:, ivar] = yr

sv = sample.StanSamplingMultivariate(yref_withmiss, pcensor=pcens)
stan_data = sv.to_dict()
stan_inits = sv.initial_parameters

fj = source_file.parent / "stan_data.json"
write_stan_json(fj, stan_data)

fj = source_file.parent / "stan_inits.json"
write_stan_json(fj, stan_inits)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "mv_censored_check.stan"
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
for cn in df.columns.to_series().filter(regex="wlat_miss"):
    cnu = re.sub("^w", "u", cn)
    df.loc[:, cnu] = df.loc[:, cn]

lcor = df.filter(regex="^L_cor", axis=1)

Csmp = []
for _, smp in lcor.iterrows():
    L = smp.values.reshape((P, P)).T
    C = L @ L.T
    Csmp.append(C)

Csmp = np.array(Csmp)
Cmean = Csmp.mean(axis=0)

ylocm = df.filter(regex="yloc", axis=1).mean()
ylogscalem = df.filter(regex="ylogscale", axis=1).mean()

plt.close("all")

# Chains
mosaic = [["ylocn[1]", "ylogscale[1]"],
          ["ulat_miss[1]", "ulat_cens[1]"]]
nrows, ncols = len(mosaic), len(mosaic[0])
w = 6
fig = plt.figure(figsize=((ncols * w, nrows * w * 0.8)),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic)
for pname, ax in axs.items():
    dd = pd.pivot_table(df, index="iter__", columns="chain__",
                       values=pname)
    dd.index.name = "iter"
    dd.columns.name = "chain"
    dd.plot(ax=ax, title=pname)

fp = fout / f"chains.png"
fig.savefig(fp)

# missing values
for dtype in ["miss", "cens"]:
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
    oneval = np.where(oneval)

    for iax, ax in enumerate(axs.flat):
        i1 = oneval[0][iax] + 1
        i2 = oneval[1][iax] + 1

        true = zref[i1 - 1, i2 - 1]

        yi = y[i1 - 1]
        imiss = np.isnan(yi)
        obs = yi[~imiss]
        pobs = np.zeros_like(obs)

        for io, ivar in enumerate(np.where(~imiss)[0]):
            loc = ylocm[ivar]
            sc = math.exp(ylogscalem[ivar])
            gum.params = [loc, sc, 0.]
            pobs[io] = gum.cdf(obs[io])

        zobs = norm.ppf(pobs)

        Cov11i = np.linalg.inv(Cmean[~imiss][:, ~imiss])
        Cov22 = Cmean[imiss][:, imiss]
        Cov21 = Cmean[imiss][:, ~imiss]
        mu = (Cov21 @ Cov11i @ zobs).squeeze()
        sig = np.sqrt(Cov22 - Cov21@Cov11i@Cov21.T).squeeze()

        idxs = stan_data[f"idx_{dtype}"]
        idx = next(i + 1 for i, ii in enumerate(idxs)
                   if ii[0] == i1 and ii[1] == i2)
        z = pd.Series(norm.ppf(df.loc[:, f"ulat_{dtype}[{idx}]"]))

        x0 = min(true, min(mu - 2 * sig, z.quantile(0.1)))
        x1 = max(true, max(mu + 2 * sig, z.quantile(0.9)))
        x0 = x0 - (x1 - x0) * 0.05
        x1 = x1 + (x1 - x0) * 0.05

        bins = np.linspace(x0, x1, 30)
        ax.hist(z, bins=bins, density=True,
                facecolor="0.8", edgecolor="0.2")

        xx = np.linspace(x0, x1, 500)
        yy = norm.pdf(xx, loc=mu, scale=sig)
        if dtype == "cens":
            cens = stan_data["censors"][i2 - 1]
            loc = ylocm[i2 - 1]
            sc = math.exp(ylogscalem[i2 - 1])
            gum.params = [loc, sc, 0.]
            zcens = norm.ppf(gum.cdf(cens))

            yy[xx > zcens] = 0.
            yy = yy / (1 - norm.cdf(zcens, loc=mu, scale=sig))

            zc = norm.ppf(df.loc[:, f"ucensors[{i2}]"].mean())
            putils.line(ax, 0, 1, zc, color="tab:purple", lw=1.2,
                        label="zcens mean")

        ax.plot(xx, yy, "k--", lw=0.9, label="norm trunc")
        putils.line(ax, 0, 1, mu, color="k", ls=":", lw=1.2, label="mu")

        putils.line(ax, 0, 1, true, color="tab:red", lw=1.2, label="true")

        ax.set(xlim=(x0, x1))
        ax.legend()

    fp = fout / f"values_{dtype}.png"
    fig.savefig(fp)

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
    putils.line(ax, 0, 1, ref, 0, "tab:red", lw=0.9)

    x0 = max(-1., ref - delta_rho)
    x1 = min(1., ref + delta_rho)
    ax.set(xlim=(x0, x1))
    title = f"C[{i1 + 1}, {i2 + 1}]"
    ax.set_title(title, x=0.05, y=0.95, va="top", ha="left")

for iax in range(nax, len(axs.flat)):
    axs.flat[iax].axis("off")

plt.show()


LOGGER.info("Process completed")
