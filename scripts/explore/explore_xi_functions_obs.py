#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-03-15 18:27:04.968566
## Comment : Explore xi functions
##
## ------------------------------

import re
import sys
from itertools import combinations
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

from pyrethink import datahub
from pyrethink import postpredchecks as ppc


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

def gumbel_cdf(uv, rho):
    theta = 1. / (1. - rho)
    xy = -np.log(uv)
    expsum = np.power(xy, theta).sum(axis=1)
    return np.exp(-np.power(expsum, 1. / theta))

def gumbel_survival(uv, rho):
    return 1 - uv.sum(axis=1) + gumbel_cdf(uv, rho)


u = np.linspace(1e-2, 1 - 1e-2, 10)
uv = np.repeat(u[:, None], 2, axis=1)

aep = np.logspace(-1, math.log10(50), 100)

mean = np.zeros(2)
prob = 1 - aep * 1e-2

ams, _, _, stations = datahub.get_ams_concat()
nsta = ams.shape[1]
pairs = np.array(list(combinations(np.arange(nsta), 2)))

plt.close("all")
w, h = 5, 4
ncols, nrows = 5, len(pairs) // 5
fig = plt.figure(figsize=(w * ncols, h * nrows),
                 layout="constrained")
mosaic = [[re.sub("\\[|\\]", "", str(pairs[ncols * ir + ic]))
           for ic in range(ncols)] for ir in range(nrows)]
axs = fig.subplot_mosaic(mosaic)

prob = np.linspace(0.5, 0.95, 30)

for aname, ax in axs.items():
    i1, i2 = aname.split(" ")
    i1 = int(i1)
    i2 = int(i2)
    sid1 = ams.columns[i1]
    sid2 = ams.columns[i2]

    biv = ppc.bivariate_dependence_statistics(ams.iloc[:, [i1, i2]])
    dep = biv["dependence"]

    # plots
    ax.plot(dep.index * 1e-2, dep.tau, "o-", label=f"Data {sid1}/{sid2}")

    uv = np.repeat(prob[:, None], 2, axis=1)
    zn = norm.ppf(prob)
    zzn = np.repeat(zn[:, None], 2, axis=1)

    df = 3
    scale = math.sqrt((df - 2) / df)
    zt = student_t.ppf(prob, scale=scale, df=df)
    zzt = np.repeat(zt[:, None], 2, axis=1)

    mean = [0, 0]
    for rho in np.linspace(0.1, 0.9, 3):
        # Gumbel
        p1 = gumbel_survival(uv, rho)
        taug = p1 / (1 - prob)
        ax.plot(prob, taug, "-", label=f"tau Gumbel $\\rho$={rho:0.2f}")

        # Gaussian
        theta = math.sin(math.pi * rho / 2)
        cov = np.array([[1, theta], [theta, 1]])
        rv = mvn(mean=mean, cov=cov)
        p0 = rv.cdf(zzn)
        xin = 2 - np.log(p0) / np.log(prob)
        p1 = rv.cdf(-zzn)
        xibarn = 2 * np.log(1 - prob) / np.log(p1) - 1
        taun = p1 / (1 - prob)

        col = ax.get_lines()[-1].get_color()
        ax.plot(prob, taun, "--",
                color=col, label=f"tau Gaussian $\\rho$={rho:0.2f}")

        # Student 3
        df = 3
        rv = mvt(loc=mean, df=df, shape=scale**2 * cov)
        p1 = rv.cdf(-zzt)
        taut = p1 / (1 - prob)

        col = ax.get_lines()[-1].get_color()
        ax.plot(prob, taut, ":",
                color=col, label=f"tau Student 3 $\\rho$={rho:0.2f}")


    title = f"{sid1} / {sid2}"
    x0, x1 = ax.get_xlim()
    x0 = max(prob.min(), x0)
    ylim = (0, 1)
    ax.set(title=title, ylim=ylim, xlabel="CDF [-]")

    ax.legend(loc=3, fontsize="small", framealpha=0.)

fp = fimg / "explore_xi_obs.png"
fig.savefig(fp)



LOGGER.completed()

