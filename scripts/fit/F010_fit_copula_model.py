#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from pathlib import Path

from scipy.stats import norm, multivariate_normal as mvn
from scipy.linalg import toeplitz
from scipy.integrate import nquad

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from floodstan import marginals

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
marginal_name = "GEV"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs" / "fit"
fout.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
stations = datahub.get_stations()

truepeaks = datahub.get_truepeaks().filter(regex="^2", axis=1)
nstations = truepeaks.shape[1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

## - DEBUG -
## Check normal integral
#nvars = 4
#n2 = 2
#rho = 0.7
#cor = toeplitz(rho**np.arange(nvars))
#mu = np.zeros(nvars)
#
#i2 = np.random.choice(np.arange(nvars), n2, replace=False)
#i2.sort()
#
#i1 = np.array([i for i in range(nvars) if i not in i2])
#n1 = len(i1)
#
#x2p = np.random.uniform(-0.1, 0.1, size=n2)
#x1 = np.random.uniform(-0.1, 0.1, size=n1)
#
#rv = mvn(mean=mu, cov=cor)
#
## .. brute force
#xx = np.zeros(nvars)
#xx[i1] = x1
#def fun(*args):
#    xx[i2] = args
#    return rv.pdf(xx)
#
#ranges = [(-20, x2p[i]) for i in range(len(i2))]
#cI = nquad(fun, ranges=ranges, full_output=True)[0]
#LOGGER.info(f"Result nquad: {cI:3.3e}")
#
#R11 = cor[i1][:, i1]
#R22 = cor[i2][:, i2]
#R21 = cor[i2][:, i1]
#rv1 = mvn(mean=mu[i1], cov=R11)
#mup = mu[i2] + R21 @ (x1 - mu[i1])
#Rp = R22 - R21@np.linalg.inv(R11)@R21.T
#rvp = mvn(mean=mup, cov=Rp)
#cII = rv1.pdf(x1) * rvp.cdf(x2p)
#LOGGER.info(f"Result analy: {cII:3.3e}")
#
#nsmp = 100000
#xx2 = rvp.rvs(size=nsmp)
#xx = np.zeros((nsmp, nvars))
#xx[:, i1] = x1[None, :]
#xx[:, i2] = xx2
#p = rv.pdf(xx)
#cIII = p.mean()
#LOGGER.info(f"Result sampl: {cIII:3.3e}")

# - end -


marginal_stations = [marginals.factory(marginal_name)
                     for i in range(nstations)]

# total number of params:
# - 3 * nstations => GEVs
# - nstations * (nstations - 1) / 2 => correlation
def vect2raw(thetas, nstations):
    gev_params = thetas[:3 * nstations].reshape((nstations, 3))

    cor = 0.5 * np.eye(nstations)
    cor_elems = thetas[3 * nstations:]
    cor[np.triu_indices(nstations, 1)] = cor_elems
    cor = cor + cor.T

    return gev_params, cor

# Setup data and initial parameters
stnorm = np.zeros_like(truepeaks)
cases = np.zeros_like(truepeaks, dtype=int)
thetas0 = []
censors = np.zeros(nstations)
stcensors = np.zeros(nstations)

for ista in range(nstations):
    values = truepeaks.iloc[:, ista]

    # censoring threshold
    censor = values.quantile(0.3)
    censors[ista] = censor

    # cases
    censored = values < censor
    cases[censored, ista] = 1

    missing = pd.isnull(values)
    cases[missing, ista] = 2

    # Fit LH moment
    marg = marginal_stations[ista]
    marg.fit_lh_moments(values)
    thetas0.extend(marg.params.tolist())

rk = truepeaks.rank()
cor0 = ((rk / rk.max()).corr()).values
eig, P = np.linalg.eig(cor0)
eig = np.maximum(eig, eig.max() * 1e-3)
cor0 = P@np.diag(eig)@P.T
d = (1 / np.sqrt(np.diag(cor0)))[:, None]
cor0 = d * cor0 * d.T
pcor0 = cor0[np.triu_indices(nstations, 1)]
thetas0.extend(pcor0.tolist())
thetas0 = np.array(thetas0)

# unique cases valid/censored/missing
cases_unique = np.unique(cases, axis=0)


# Log-likelihood function
def loglike(thetas, nstations, censors, data, stnorm, stcensors):
    gparams, cor = vect2raw(thetas, nstations)

    # Set GEV params
    for ista in range(nstations):
        marg = marginal_stations[ista]
        marg.params = gparams[ista]
        stcensors[ista] = norm.ppf(marg.cdf(censors[ista]))

    # Compute lpdf
    lpdf = 0
    for case in cases_unique:
        match = (cases == case[None, :]).all(axis=1)
        nmatch = match.sum()

        for ista in range(nstations):
            marg = marginal_stations[ista]

            # Marginal logpdf
            if case[ista] == 0:
                lpdf += marg.logpdf(data[match, ista]).sum()
            elif case[ista] == 1:
                lpdf += marg.logcdf(censors[ista]) * nmatch

            # probabilities
            stnorm[match, ista] = norm.ppf(marg.cdf(data[match, ista]))

        valid = case == 0
        cens = case == 1
        R11 = cor[valid][:, valid]
        if cens.sum() == 0:
            lpdf += mvn(cov=R11).logpdf(stnorm[match][:, valid]).sum()
            continue

        R22 = cor[cens][:, cens]
        import pdb; pdb.set_trace()

    return lpdf

data = truepeaks.values
ll = loglike(thetas0, nstations, censors, data,
             stnorm, stcensors)

LOGGER.completed()

