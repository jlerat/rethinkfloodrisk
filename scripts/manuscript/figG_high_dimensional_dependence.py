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

        def true_cdf(x, rho, napprox):
            nsta = x.shape[1]
            mean = np.zeros(nsta)
            cov = rho * np.eye(nsta) + (1 - rho) * np.ones((nsta, nsta))
            rv = mvt(mean=mean, cov=cov)
            return [rv.cdf(x)]

        def approx_cdf(x, rho, napprox):
            cdf = np.zeros(len(x))
            eps = 0.5 / napprox
            v = np.linspace(eps, 1 - eps, napprox)
            dv = v[1] - v[0]
            for i, vv in enumerate(v):
                cdf += fun(vv, x, rho) * dv
            return cdf

        plt.close("all")
        fig, axs = plt.subplots(ncols=3)
        z = np.linspace(-3, 3, 50)
        nsta = 10

        for rho, ax in zip([0.2, 0.5, 0.9], axs):
            cc = np.zeros((len(z), 2))
            for ix, x in enumerate(z):
                xx = x * np.ones((1, nsta))
                cc[ix, 0] = true_cdf(xx, rho, 50)[0]
                cc[ix, 1] = approx_cdf(xx, rho, 50)[0]

            ax.plot(z, cc[:, 0])
            ax.plot(z, cc[:, 1])
            ax.set(title=f"rho = {rho}",
                   xlabel="z score")
        plt.show()
        import pdb; pdb.set_trace()



        def ofun(u, nsta, use_approx, rho, u_smp,
                 kendall, lmaep, kind):
            napp = 10 if config.debug else 50
            cdf_fun = approx_cdf if use_approx else true_cdf
            if kind == "AND":
                xx = norm.ppf(u) * np.ones((1, nsta))
                c = cdf_fun(xx, rho, napp)[0]
            elif kind == "OR":
                xx = norm.ppf(u) * np.ones((1, nsta))
                c = 1 - cdf_fun(-xx, rho, napp)[0]
            elif kind == "KENDALL":
                xx = norm.ppf(1 - u) * np.ones((1, nsta))
                c0 = cdf_fun(xx, rho, napp)[0]
                c = 1 - np.interp(c0, kendall.p, kendall.t)
            elif kind == "AND_SMP":
                c = np.all(u_smp < u, axis=1).sum() / nsmp
            elif kind == "OR_SMP":
                c = 1 - np.all(u_smp < 1 - u, axis=1).sum() / nsmp

            err = (math.log(c) - lmaep)**2
            return err

        def fit(maep, nsta, use_approx, rho, u_smp,
                kendall, kind):
            if kind.startswith("AND"):
                ua = maep
                ub = maep**(1./nsta)
            elif kind.startswith("OR"):
                ua = 1 - (1 - maep)**(1./nsta)
                ub = maep
            elif kind.startswith("KENDALL"):
                ua = maep
                ub = 1 - math.exp(-gamma.ppf(gamma_fun(nsta) * maep, a=nsta))

            lmaep = math.log(maep)
            args = (nsta, use_approx, rho, u_smp,
                    kendall, lmaep, kind,)
            opt = minimize_scalar(ofun, bracket=(ua, ub),
                                  bounds=(ua, ub),
                                  args=args)
            return opt

        # Configure
        rhos = [0.8] if config.debug else [0.3, 0.9]
        nstas = [2] if config.debug else [2, 10, 30]
        aep_types = ["AND"] if config.debug else ["AND", "OR", "KENDALL"]

        naeps = 11 if config.debug else 31
        maeps = np.logspace(-3, -1, naeps)

        napp = 50
        use_approx = True

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

        nsmp = 1000000
        v = np.random.normal(size=nsmp)[:, None]

        for iax, (aname, ax) in enumerate(axs.items()):
            aep_type, nsta = aname.split("_")
            nsta = int(nsta)
            eps = np.random.normal(size=(nsmp, nsta))

            for rho in rhos:
                logger.info(f"Dealing with {aep_type}/nsta={nsta} rho={rho:0.2f}")

                # Exact / approx
                uaeps = np.nan * np.zeros((len(maeps), 2))

                sqr, csqr = const(rho)
                x = sqr * v + csqr * eps

                u_smp = norm.cdf(x)
                t = np.linspace(0, 1, napp)
                p = np.array([np.all(u_smp - tt < 0, axis=1).sum() / nsmp
                              for tt in t])
                kendall = pd.DataFrame({"t": t, "p": p})

                for iaep, maep in enumerate(maeps):
                    if iaep % 5 == 0 :
                        logger.info(f"maep #{iaep + 1:2d}", ntab=1)

                    opt = fit(maep, nsta, use_approx, rho,
                              u_smp, kendall, aep_type)
                    uaeps[iaep, 0] = opt.x
                    if aep_type != "KENDALL":
                        opt_smp = fit(maep, nsta, use_approx,
                                      rho, u_smp,
                                      kendall, aep_type + "_SMP")
                        uaeps[iaep, 1] = opt_smp.x

                if aep_type != "KENDALL":
                    ax.plot(maeps * 100, uaeps[:, 0] * 100,
                            "o-", lw=3, label=f"ρ={rho:0.2f}")
                else:
                    ax.plot([], [], "-",
                            lw=3, label=f"ρ={rho:0.2f}")

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
                indep = 1 - np.exp(-gamma.ppf(gamma_fun(nsta) * maeps, a=nsta))

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
