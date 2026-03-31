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
from itertools import product as prod
import re
import argparse
import json
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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import marginal_exceedance_score as mes

from floodstan import marginals
from floodstan import freqplots

from figA_impact_of_period_on_FFA import get_script_paths
from figA_impact_of_period_on_FFA import get_logger, get_taskids, get_data
from figA_impact_of_period_on_FFA import get_iter_options, select_data

import importlib
importlib.reload(mes)


def smooth_basis(x, nodes):
    M = [np.ones_like(x), x]
    for ai in nodes:
        diff = x - ai
        M.append(diff * np.abs(diff))
    return np.column_stack(M)


def smooth(x, x0, y0):
    iok = ~np.isnan(x0) & ~np.isnan(y0)
    iok &= ~np.isinf(x0) & ~np.isinf(y0)
    x0, y0 = x0[iok], y0[iok]
    nodes = x0[1:-1]
    M0 = smooth_basis(x0, nodes)
    theta = np.linalg.inv(M0) @ y0
    return smooth_basis(x, nodes) @ theta


def process(config, script_paths, logger, data):
    for pcensor, rho_min, has_cluster, copula_shape in get_iter_options(data):
        copula_type = 1 if copula_shape > 0 else 0

        _, obs_data, _, expected, _ = select_data(data,
                                                  pcensor=pcensor,
                                                  rho_min=rho_min,
                                                  has_cluster=has_cluster,
                                                  copula_shape=copula_shape)
        if len(expected) == 0:
            continue
        assert len(expected) == 1

        rn = next(iter(expected))
        obs_data = obs_data[rn]
        expected = expected[rn]

        ylocs = pd.Series(expected["ylocs"])
        ylogscales = pd.Series(expected["ylogscales"])
        yshape1 = pd.Series(expected["yshape1"])
        cor = pd.Series(expected["corr_IW"])
        nstations = len(ylocs)
        cor = cor.values.reshape((nstations, nstations)).T

        sids = obs_data.columns.to_list()
        isid1 = sids.index(config.sta1)
        isid2 = sids.index(config.sta2)
        #isid3 = sids.index(config.sta3)
        #isid4 = sids.index(config.sta4)
        rho = cor[isid1, isid2]

        logger.info(f"-- Plotting {rn.text} / rho={rho:0.2f} --", nret=1)

        kinds = ["AND", "OR", "KENDALL"]
        ptypes = ["aep+2d", "aep+nd"]
        mosaic = [[f"{ptype}_{kind}" for kind in kinds]
                  for ptype in ptypes]

        plt.close("all")
        nrows = len(mosaic)
        ncols = len(mosaic[0])
        fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                         layout="tight")
        axs = fig.subplot_mosaic(mosaic)

        for iax, (aname, ax) in enumerate(axs.items()):
            kind = re.sub(".*_", "", aname)
            logger.info(f"Plot {aname}", ntab=1)

            if aname.startswith("aep+2d"):
                if config.use_indep:
                    cop = mes.IndependenceCopula(2)
                else:
                    cop = mes.GaussianCopula(2)
                    cop.params = np.array([[1, rho], [rho, 1]])

                mex = mes.MarginalExceedanceScore(kind, cop)

                gevs = []
                for isid in [isid1, isid2]:
                    g = marginals.GEV()
                    g.locn = ylocs.iloc[isid]
                    g.logscale = ylogscales.iloc[isid]
                    g.shape1 = yshape1.iloc[isid]
                    gevs.append(g)

                p0, p1 = 0.1, 0.999
                x0, x1 = np.maximum(gevs[0].ppf([p0, p1]), 10)
                x0 = round(x0 / 10) * 10
                x1 = (round(x1 / 100) + 1) * 100

                y0, y1 = np.maximum(gevs[1].ppf([p0, p1]), 10)
                y0 = round(y0 / 10) * 10
                y1 = (round(y1 / 100) + 1) * 100

                # plot data
                x = obs_data.loc[:, sta1]
                y = obs_data.loc[:, sta2]
                ax.plot(x, y, "o", mfc="w", mec="k", alpha=0.8,
                        label="Observed streamflow maxima")

                # plot pdf
                ngrid = config.ngrid
                x = np.linspace(x0, x1, ngrid)
                y = np.linspace(y0, y1, ngrid)
                xx, yy = np.meshgrid(x, y)
                uv = np.column_stack([gevs[0].cdf(xx.ravel()),
                                      gevs[1].cdf(yy.ravel())])

                zz = cop.pdf(uv).reshape(xx.shape)
                px = gevs[0].pdf(xx)
                py = gevs[1].pdf(yy)
                zz *= px * py
                cnt = ax.contourf(xx, yy, zz, cmap="Blues",
                                  levels=30, alpha=0.5, norm="log",
                                  label="Survival cumulative density",
                                  antialiased=True)
                for maep in config.maep_target:
                    logger.info(f"AEP 1:{1 / maep:0.0f}", ntab=2)
                    if maep == 1e-1:
                        color = config.col_aep10
                    else:
                        color = config.col_aep100

                    # Plot MAEP solution set
                    npoints = 20 if config.debug else 100
                    df, _ = mex.compute_set(maep, npoints=npoints)
                    x = gevs[0].ppf(df.u)
                    y = gevs[1].ppf(df.v)
                    ax.plot(x, y, "-", lw=2, color=color,
                            label=f"Solution set for AEP 1:{1. / maep:0.0f}")

                    # Plot common MAEP value
                    mex0, _ = mex.compute_score(maep)
                    x = gevs[0].ppf(mex0)
                    y = gevs[1].ppf(mex0)
                    ax.plot(x, y, "o",
                            ms=12, mfc=color, mec="w",
                            label=f"Common MEXS for AEP 1:{1. / maep:0.0f}")
                    ax.plot([x]*2, [y0, y], "--", lw=0.9, color=color)
                    txt = f"1:{1. / (1 - mex0):0.0f}"
                    ax.annotate(txt, xy=(x, y0), xytext=(7, 10),
                                textcoords="offset pixels",
                                color=color, fontweight="bold")

                if kind == "AND":
                    ax.legend(loc=3, framealpha=1)

                #x = np.linspace(x0, x1, 100)
                #y = gevs[1].ppf(gevs[0].cdf(x))
                #ax.plot(x, y, "k--", lw=0.9, alpha=0.6)

                xlabel = f"Streamflow peak {config.sta1} [m3.s-1]"

                ax.set(xscale="log", yscale="log")

                if kind == "AND":
                    ylabel = f"Streamflow peak {config.sta2} [m3.s-1]"
                else:
                    ylabel = ""
                    ax.set_yticklabels([])

                title = f"({letters[iax]}) Bivariate flood peak distribution\n"\
                        + f"'{kind}' exceedance"
                ax.set(xlim=(x0, x1), ylim=(y0, y1),
                       xlabel=xlabel, ylabel=ylabel,
                       title=title)

            else:
                nstas = [2, 5, 10, 20]
                nstas_txt = [f"{n} stations" for n in nstas]
                maeps = config.maep_target
                maeps_txt = [f"1:{1/a:0.0f} AEP" for a in maeps]
                values = pd.DataFrame(np.nan, index=nstas_txt,
                                      columns=maeps_txt)
                for ista, nsta in enumerate(nstas):
                    logger.info(f"Nstations {nsta}", ntab=2)
                    if config.use_indep:
                        cop = mes.IndependenceCopula(nsta)
                    else:
                        cop = mes.GaussianCopula(nsta)
                        cop.params = (1 - rho) * np.eye(nsta) \
                            + rho * np.ones((nsta, nsta))

                    mex = mes.MarginalExceedanceScore(kind, cop)
                    for iaep, maep in enumerate(maeps):
                        mex0, _ = mex.compute_score(maep)
                        cn = values.columns[iaep]
                        idx = values.index[ista]
                        values.loc[idx, cn] = 1 - mex0

                values.plot(kind="bar", ax=ax,
                            rot=0, legend=False)

                aep = [1000, 100, 10, 1]
                tk = [1./a for a in aep]

                if kind == "AND":
                    ax.legend(loc=2)
                    ylabel = f"Common Marginal Exceedance Score [-]"
                    tkl = [f"1:{a}" for a in aep]
                else:
                    ylabel = ""
                    tkl = []

                x0, x1 = 1. / 2000, 1.
                title = f"({letters[iax]}) Common Marginal Exceedance "\
                        + f" Score\n'{kind}' exceedance"
                ax.set(ylim=(x0, x1), ylabel=ylabel,
                       yscale="log",
                       title=title)

                ax.set_yticks(tk, labels=tkl)
                ax.grid(axis="y")

        basename = script_paths.basename
        fp = f"{basename}_{rn.text}_v{config.version}.png"
        fp = script_paths.fimg / fp
        fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood frequency plots",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                        type=float, default=0.3)
    parser.add_argument("-di", "--diag", help="Show stan diagnostics",
                        action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    parser.add_argument("-i", "--use_indep", help="Use independent copula",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_mins", help="Minimum rho value",
                        type=str, default="-1")
    parser.add_argument("-s", "--copula_shapes", help="Copula shapes selected",
                        type=str, default="0")
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_mins",
                               "awidth", "aheight", "fdpi",
                               "excludes", "copula_shapes",
                               "diag", "debug",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "sta1", "sta2", "sta3", "sta4",
                               "ngrid", "maep_target",
                               "use_indep", "col_aep100",
                               "col_aep10"])
    awidth = 5
    aheight = 4
    fdpi = 300
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = True
    load_mvnproc = False
    load_expected_params = True
    load_postpred_checks = False

    col_aep100 = "tab:red"
    col_aep10 = "tab:orange"

    sta1 = "203002"
    sta2 = "203012"
    sta3 = "203005"
    sta4 = "203010"

    ngrid = 50
    maep_target = [1e-2]

    config = CF(args.version, args.pcensor,
                args.rho_mins.split("|"),
                awidth, aheight, fdpi,
                excludes,
                args.copula_shapes.split("|"),
                args.diag, args.debug,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks,
                sta1, sta2, sta3, sta4,
                ngrid, maep_target,
                args.use_indep, col_aep100,
                col_aep10)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
