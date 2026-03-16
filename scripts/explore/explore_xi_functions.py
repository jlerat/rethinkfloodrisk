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

def clayton_cdf(uv, rho):
    theta = 2. * rho / (1. - rho)
    expsum = np.power(uv, -theta).sum(axis=1) - 1.
    return np.power(expsum, -1. / theta)

def clayton_survival(uv, rho):
    return 1 - uv.sum(axis=1) + clayton_cdf(uv, rho)

def frank_cdf(uv, rho):
    theta = 2. * rho / (1. - rho)
    x = np.exp(-theta * uv[:, 0])
    y = np.exp(-theta * uv[:, 1])
    w = 1 - math.exp(-theta)
    z = w - (1 - x) * (1 - y)
    return -1/theta*np.log(z/w)

def frank_survival(uv, rho):
    return 1 - uv.sum(axis=1) + frank_cdf(uv, rho)


def mtcj_cdf(uv, rho):
    theta = 2. * rho / (1. - rho)
    x = np.power(uv[:, 0], -theta)
    y = np.power(uv[:, 1], -theta)
    return np.power(x + y - 1, -1./theta)

def mtcj_survival(uv, rho):
    return 1 - uv.sum(axis=1) + mtcj_cdf(uv, rho)


u = np.linspace(1e-2, 1 - 1e-2, 10)
uv = np.repeat(u[:, None], 2, axis=1)

aep = np.logspace(-1, math.log10(50), 100)

mean = np.zeros(2)
prob = 1 - aep * 1e-2

ams, _, _, stations = datahub.get_ams_concat()
pair = (0, 4)
sid1 = ams.columns[pair[0]]
sid2 = ams.columns[pair[1]]
ams = ams.iloc[:, list(pair)]
biv = ppc.bivariate_dependence_statistics(ams)
dep = biv["dependence"]

plt.close("all")
w, h = 5, 4
ncols, nrows = 2, 2
fig = plt.figure(figsize=(w * ncols, h * nrows),
                 layout="constrained")
rhos = np.linspace(0.1, 0.9, 4)
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

    # Student
    xit, xibart, taut = {}, {}, {}
    for df in [3]:
        scale = math.sqrt((df - 2) / df)
        z = student_t.ppf(prob, scale=scale, df=df)
        zz = np.repeat(z[:, None], 2, axis=1)
        rv = mvt(loc=mean, shape=scale**2 * cov, df=df)
        p0 = rv.cdf(zz)
        xit[df] = 2 - np.log(p0) / np.log(prob)
        p1 = rv.cdf(-zz)
        xibart[df] = 2 * np.log(1 - prob) / np.log(p1) - 1
        taut[df] = p1 / (1 - prob)

    # Gumbel
    uv = np.repeat(prob[:, None], 2, axis=1)
    p1 = gumbel_survival(uv, rho)
    taug = p1 / (1 - prob)

    # Clayton
    p1 = clayton_cdf(1 - uv, rho)
    tauc = p1 / (1 - prob)

    # Frank
    p1 = frank_survival(uv, rho)
    tauf = p1 / (1 - prob)

    # MTCJ
    p1 = mtcj_cdf(1 - uv, rho)
    taum = p1 / (1 - prob)

    # plots
    ax.plot(dep.index * 1e-2, dep.tau, "o-", label=f"Data {sid1}/{sid2}")

    #ax.plot(aep, xin, "k-", label="xi gaussian")
    #ax.plot(aep, xibarn, "k--", label="xibar gaussian")
    ax.plot(prob, taun, "k-", label="tau gaussian")
    ax.plot(prob, taun2, "k:", label="tau gaussian (sample)")

    ax.plot(prob, taul, ":", color="purple", label="tau Laplace (sample)")
    ax.plot(prob, taug, "-", color="orange", label="tau Gumbel")
    ax.plot(prob, tauc, "-", color="0.5", label="tau Clayton - survival")
    ax.plot(prob, tauf, "-", color="pink", label="tau Frank")
    ax.plot(prob, taum, "-", color="chocolate", label="tau MTCJ - survival")

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

    ax.legend(loc=3, fontsize="small", framealpha=0.)

fp = fimg / "explore_xi.png"
fig.savefig(fp)



LOGGER.completed()

