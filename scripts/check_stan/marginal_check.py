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
from scipy.stats import gumbel_r, norm
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
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

nwarm = 500
nsamples = 500
nchains = 5

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "check_stan" / "marginal_check"
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
data = data.iloc[:, :3]
data = data.loc[data.notnull().all(axis=1)]

sv = sample.StanSamplingMultivariate(data)
stan_data = sv.to_dict()

stan_inits = sv.initial_parameters
L_cor = stan_inits["L_cor"]
stan_data["L_cor"] = L_cor

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Loading stan model")
stan_file = froot / "scripts" / "check_stan" / "marginal_check.stan"
suffix = ".exe" if os.name == "nt" else ""
exe_file = stan_file.parent / f"{stan_file.stem}{suffix}"
if not exe_file.exists():
    exe_file = None
kwargs = dict()
model = CmdStanModel(stan_file=stan_file,
                     exe_file=exe_file)
LOGGER.info(".. done")

LOGGER.info("Run model")
#kwargs = {}
#kwargs["seed"] = 5446
#out = model.optimize(data=stan_data, **kwargs)
#opt = out.optimized_params_pd
#yloc_stan = opt.filter(regex="yloc", axis=1).squeeze().values
#ylogscale_stan = opt.filter(regex="ylogscale", axis=1).squeeze().values
#
#cor = L_cor @ L_cor.T
#y = stan_data["y"]
#z = np.zeros_like(stan_data["y"])
#rvn = mvn(mean=np.zeros(len(cor)), cov=cor)
#N = stan_data["N"]
#P = stan_data["P"]

#def trans(yv, loc, logscale):
#    rvg = gumbel_r(loc=loc, scale=math.exp(logscale))
#    return norm.ppf(rvg.cdf(yv))
#
#def difftrans(yv, loc, logscale):
#    rvg = gumbel_r(loc=loc, scale=math.exp(logscale))
#    z = norm.ppf(rvg.cdf(yv))
#    return rvg.pdf(yv) / norm.pdf(z)
#
#loc = stan_inits["ylocn"][0]
#logscale = stan_inits["ylogscale"][0]
#
#y0, y1 = y[:, 0].min(), y[:, 0].max()
#yy = np.linspace(y0, y1, 1000)
#plt.close("all")
#fig, axs = plt.subplots(nrows=2, figsize=(7, 8),
#                        layout="constrained")
#yt = trans(yy, loc, logscale)
#ax = axs[0]
#ax.plot(yy, trans(yy, loc, logscale))
#
#ax = axs[1]
#dy = yy[1] - yy[0]
#dyt = (yt[2:] - yt[:-2]) / 2 / dy
#ax.plot(yy[1:-1], dyt)
#ax.plot(yy, difftrans(yy, loc, logscale))
#plt.show()


#def negloglike(thetas):
#    ylocn = thetas[:P]
#    ylogscale = thetas[P:]
#    if np.any(ylogscale < -10) or np.any(ylogscale > 10):
#        return np.inf
#
#    ll = 0.
#    for ivar in range(P):
#        yv = y[:, ivar]
#        rvg = gumbel_r(loc=yloc[ivar], scale=math.exp(ylogscale[ivar]))
#
#        zv = norm.ppf(rvg.cdf(yv))
#        f = rvg.pdf(yv)
#        z[:, ivar] = zv
#        # + jacobian
#        ll += np.log(f / norm.pdf(zv)).sum()
#
#    ll += rvn.logpdf(z).sum()
#    txt = " ".join([f"{t:6.1f}" for t in thetas])
#    print(f"ll = {ll:9.3e} / {txt}")
#    return -ll
#
#yloc = stan_inits["ylocn"]
#ylogscale = stan_inits["ylogscale"]
#thetas0 = np.concatenate([yloc, ylogscale])
#ll0 = negloglike(thetas0)
#
#opt = minimize(negloglike, thetas0)
#yloc_opt = opt.x[:P]
#ylogscale_opt = opt.x[P:]


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

sys.exit()


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
