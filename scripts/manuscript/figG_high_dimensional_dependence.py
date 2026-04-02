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
from scipy.stats import gamma
from scipy.special import gamma as gamma_fun


import matplotlib.pyplot as plt
from matplotlib import ticker

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pyrethink import marginal_exceedance_score as mes

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

        # Configure
        rhos = [0.8] if config.debug else [0.3, 0.9]
        nstas = [3] if config.debug else [2, 10, 30]
        aep_types = ["KENDALL"] #["AND", "OR", "KENDALL"]

        naeps = 11 if config.debug else 31
        maeps = np.logspace(-3, -1, naeps)

        napp = 50
        use_approx = False

        # Create plot
        plt.close("all")
        mosaic = [[f"{t}_{n}" for n in nstas]
                  for t in aep_types]
        w = 6
        ncols, nrows = len(mosaic[0]), len(mosaic)
        fig = plt.figure(figsize=(w * ncols, w * nrows),
                         layout="constrained")
        axs = fig.subplot_mosaic(mosaic,
                                 sharex=True,
                                 sharey=True)


        for iax, (aname, ax) in enumerate(axs.items()):
            aep_type, nsta = aname.split("_")
            nsta = int(nsta)
            nsmp = 100000 if aep_type != "KENDALL" else 10000

            for rho in rhos:
                logger.info(f"Dealing with {aep_type}/nsta={nsta} rho={rho:0.2f}")

                cop = mes.GaussianFactorCopula(nsta)
                cop.params = rho

                mex1 = mes.MarginalExceedanceScore(aep_type, cop)
                mex1.logger = logger

                if aep_type != "KENDALL":
                    mex2 = mes.MarginalExceedanceScoreEmpirical(aep_type, cop)
                    mex2.logger = logger

                uaeps = np.zeros((len(maeps), 2))
                for iaep, maep in enumerate(maeps):
                    if iaep % 5 == 0 :
                        logger.info(f"maep #{iaep + 1:2d}", ntab=1)

                    uaeps[iaep, 0] = mex1.common_marginal_exceedance_score(maep)
                    if aep_type != "KENDALL":
                        uaeps[iaep, 1] = mex2.common_marginal_exceedance_score(maep)


                ax.plot(maeps * 100, uaeps[:, 0] * 100,
                        "o-", lw=3, label=f"ρ={rho:0.2f}")

                if aep_type != "KENDALL":
                    col = ax.get_lines()[-1].get_color()
                    ax.plot(maeps * 100, uaeps[:, 1] * 100,
                            "o--", lw=1.5, label=f"ρ={rho:0.2f} (sample)",
                            color=col)

            ylab = "Univariate Equivalent AEP [%]" if nsta == 2 else ""
            ax.set(title=f"({letters[iax]}) '{aep_type}' event - {nsta} stations",
                   xscale="log",
                   yscale="log",
                   xlabel=f"Multivariate '{aep_type}' AEP [%]",
                   ylabel=ylab)

            comonot = maeps * 100
            if aep_type == "AND":
                indep = maeps**(1./nsta)
            elif aep_type == "OR":
                indep = (1 - (1 - maeps)**(1./nsta))
            else:
                kfi = mes.KendallFunctionIndependence(nsta)
                indep = kfi.cdf(1 - maeps)

            ax.plot(maeps * 100, comonot, "k-",
                    label="Co-monotone", lw=0.8)
            ax.plot(maeps * 100, indep * 100, "k:",
                    label="Indedendent", lw=0.8)


            fmt = lambda x, pos: f"1:{int(100/x):,d}"
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
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
