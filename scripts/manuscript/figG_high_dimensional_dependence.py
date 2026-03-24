#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
import math
from collections import namedtuple
from itertools import combinations as comb
import re
import argparse
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvt
from scipy.optimize import minimize_scalar


import matplotlib.pyplot as plt
from matplotlib import ticker

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import sample

import figA_impact_of_period_on_FFA
import importlib
importlib.reload(figA_impact_of_period_on_FFA)

from figA_impact_of_period_on_FFA import get_script_paths
from figA_impact_of_period_on_FFA import get_logger, get_taskids, get_data
from figA_impact_of_period_on_FFA import get_iter_options, select_data


def process(config, script_paths, logger, data):
    for pcensor, rho_min, has_cluster, copula_shape in get_iter_options(data):
        if has_cluster:
            continue

        _, obs_data, _, _, _ = select_data(data,
                                           pcensor=pcensor,
                                           rho_min=rho_min,
                                           has_cluster=has_cluster,
                                           copula_shape=copula_shape)
        if len(obs_data) == 0:
            continue
        assert len(obs_data) == 1

        rn = next(rn for rn in obs_data if rn.exclude == "NONE")
        obs_data = obs_data[rn]

        # Equivalent to
        # U_i = Phi(X_i)
        # W = Phi^-1(v) -> quantile transform
        # X_i = sqrt(rho) W + sqrt(1 - rho) eps_i
        # v ~ U(0, 1)    eps_i ~ N(0, 1)
        # As a result,
        # X_i | V=v ~ N(sqrt(rho) w, 1 - rho)
        # Hence
        # Pr(X_i < x0 | V=v) = Phi( [x0 - w sqrt(rho)] / sqrt(1 - rho) )
        #
        # Finally, by independence
        # P(X_1 < x_1, ..., X_2 < x_2 | V=v) = prod Phi((x_i - w sqrt(rho)) / sqrt(1-rho))
        #
        # Hence:
        # P(X_1 < x_1, ..., X_2 < x_2) = int[0-1] prod Phi((x_i - w sqrt(rho)) / sqrt(1-rho)) dv
        #

        def const(rho):
            sqr = math.sqrt(rho)
            csqr = math.sqrt(1 - rho)
            return sqr, csqr

        def fun(v, x, rho):
            sqr, csqr = const(rho)
            w = norm.ppf(v)
            return norm.cdf((x - w * sqr) / csqr).prod(axis=1)

        def approx_cdf_basic(x, rho, n):
            cdf = np.zeros(len(x))
            eps = 0.5 / n
            v = np.linspace(eps, 1 - eps, n)
            dv = v[1] - v[0]
            for i, vv in enumerate(v):
                cdf += fun(vv, x, rho) * dv
            return cdf

        def ofun(u, nsta, rho, lmaep):
            xx = norm.ppf(u) * np.ones((1, nsta))
            napp = 10 if config.debug else 50
            c = approx_cdf_basic(xx, rho, napp)[0]
            err = (math.log(c) - lmaep)**2
            return err

        def fit(maep, nsta, rho):
            ua = maep
            ub = maep**(1./nsta)
            lmaep = math.log(maep)
            opt = minimize_scalar(ofun, bracket=(ua, ub),
                                  bounds=(ua, ub),
                                  args=(nsta, rho, lmaep,))
            return opt.x

        plt.close("all")
        rhos = [0.5, 0.9]
        nstas = [2, 10, 30]
        w = 6
        fig, axs = plt.subplots(ncols=len(nstas),
                                figsize=(w * len(nstas), w),
                                layout="constrained",
                                sharex=True, sharey=True)

        naeps = 8 if config.debug else 30
        maeps = np.logspace(-3, -1, naeps)

        for iax, (nsta, ax) in enumerate(zip(nstas, axs)):
            for rho in rhos:
                logger.info(f"Dealing with nsta={nsta} rho={rho:0.2f}")
                uaeps = np.zeros_like(maeps)
                for iaep, maep in enumerate(maeps):
                    if iaep % 5 == 0 :
                        logger.info(f"maep #{iaep + 1:2d}", ntab=1)
                    u = fit(maep, nsta, rho)
                    p = approx_cdf_basic(norm.ppf(u) * np.ones((1, nsta)),
                                         rho, 20)[0]
                    if abs(p - maep) < 1e-5:
                        logger.warning(f"High error : {abs(p - maep):3.3e}",
                                       ntab=1)

                    uaeps[iaep] = u

                ax.plot(maeps * 100, uaeps * 100,
                        "-", lw=3, label=f"ρ={rho:0.2f}")

            ylab = "Univariate Equivalent AEP [%]" if nsta == 2 else ""
            ax.set(title=f"({letters[iax]}) {nsta} stations",
                   xscale="log",
                   yscale="log",
                   xlabel="Multivariate 'AND' AEP [%]",
                   ylabel=ylab)

            ax.plot(maeps * 100, maeps * 100, "k-",
                    label="Co-monotone", lw=0.8)
            ax.plot(maeps * 100, maeps**(1./nsta) * 100, "k--",
                    label="Indedendent", lw=0.8)

            fmt = lambda x, pos: f"1:{100/x:0.0f}"
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            if iax == 0:
                ax.legend(loc=4, fontsize="large")

            ax.grid()

        basename = script_paths.basename
        fp = f"{basename}.png"
        fp = script_paths.fimg / fp
        fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_mins",
                               "awidth", "aheight", "fdpi", "ncols",
                               "excludes", "copula_shapes",
                               "diag", "debug",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "exclude"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ncols = 4
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = True
    load_mvnproc = False
    load_expected_params = False
    load_postpred_checks = False
    exclude = "NONE"

    pcensor = 0.3
    rho_mins = "-1"
    copula_shapes = "0"
    diag = False


    config = CF(args.version, pcensor,
                rho_mins,
                awidth, aheight, fdpi, ncols, excludes,
                copula_shapes,
                diag, args.debug,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks, exclude)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
