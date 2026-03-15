#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-03-15 18:27:04.968566
## Comment : Explore xi functions
##
## ------------------------------


import sys
import os
import re
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import laplace
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fimg = froot / "images" / "explore_xi"
fimg.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

aep = np.logspace(-1, math.log10(50), 100)

mean = np.zeros(2)
prob = 1 - aep * 1e-2

plt.close("all")
w, h = 5, 4
ncols, nrows = 3, 2
fig = plt.figure(figsize=(w * ncols, h * nrows),
                 layout="constrained")
rhos = np.linspace(0.1, 0.9, 6)
mosaic = [[r for r in rs] for rs in np.array_split(rhos, 2)]
axs = fig.subplot_mosaic(mosaic)

for rho, ax in axs.items():
    cov = np.array([[1, rho], [rho, 1]])

    # Gaussian
    z = norm.ppf(prob)
    zz = np.repeat(z[:, None], 2, axis=1)
    rv = mvn(mean=mean, cov=cov)
    p0 = rv.cdf(zz)
    xin = 2 - np.log(p0) / np.log(prob)
    p1 = rv.cdf(-zz)
    xibarn = 2 * np.log(1 - prob) / np.log(p1) - 1
    taun = p1 / (1 - prob)

    # Laplace
    ns = 100000
    y = rv.rvs(size=ns)
    w = expon.rvs(size=ns)
    x = np.sqrt(w[:, None]) * y
    taul = np.zeros(len(aep))
    taun2 = np.zeros(len(aep))
    for iq, q in enumerate(prob):
        z = laplace.ppf(q)
        iabove = (x - z >= 0).all(axis=1)
        p1 = iabove.sum() / ns
        taul[iq] = p1 / (1 - q)

        z = norm.ppf(q)
        iabove = (y - z >= 0).all(axis=1)
        p1 = iabove.sum() / ns
        taun2[iq] = p1 / (1 - q)

    # Student df = 3
    xit, xibart, taut = {}, {}, {}
    for df in [3, 5, 10]:
        scale = math.sqrt((df - 2) / df)
        z = student_t.ppf(prob, scale=scale, df=df)
        zz = np.repeat(z[:, None], 2, axis=1)
        rv = mvt(loc=mean, shape=scale**2 * cov, df=df)
        p0 = rv.cdf(zz)
        xit[df] = 2 - np.log(p0) / np.log(prob)
        p1 = rv.cdf(-zz)
        xibart[df] = 2 * np.log(1 - prob) / np.log(p1) - 1
        taut[df] = p1 / (1 - prob)

    # plots
    #ax.plot(aep, xin, "k-", label="xi gaussian")
    #ax.plot(aep, xibarn, "k--", label="xibar gaussian")
    ax.plot(prob, taun, "k-", label="tau gaussian")
    ax.plot(prob, taun2, "k:", label="tau gaussian (sample)")

    ax.plot(prob, taul, "-", color="purple", label="tau Laplace")

    cols = {3: "tab:blue", 5:"tab:red", 10:"tab:green"}
    for df in xit:
        col = cols[df]
        #ax.plot(aep, xit[df], "-", color=col, label=f"xi student {df}")
        #ax.plot(aep, xibart[df], "--", color=col, label=f"xibar student {df}")
        ax.plot(prob, taut[df], "-", color=col, label=f"tau student {df}")

    title = f"rho = {rho:0.2f}"
    x0, x1 = ax.get_xlim()
    x0 = max(prob.min(), x0)
    ylim = (0, 1)
    ax.set(title=title, ylim=ylim, xlabel="CDF [-]")

    ax.legend(loc=2, framealpha=0.)

fp = fimg / "explore_xi.png"
fig.savefig(fp)



LOGGER.completed()

