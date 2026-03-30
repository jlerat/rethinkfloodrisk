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

from figA_impact_of_period_on_FFA import get_script_paths
from figA_impact_of_period_on_FFA import get_logger, get_taskids, get_data
from figA_impact_of_period_on_FFA import get_iter_options, select_data

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
        isid3 = sids.index(config.sta3)
        isid4 = sids.index(config.sta4)

        logger.info(f"-- Plotting {rn.text} --", nret=1)

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
                #cop = mes.IndependenceCopula(2)
                cop = mes.GaussianCopula(2)
                cop.params = cor[[isid1, isid2]][:, [isid1, isid2]]

                mex = mes.MarginalExceedanceScore(kind, cop)
                x0, x1 = 1e-3, 0.5

                for maep in config.maep_target:
                    logger.info(f"AEP 1:{1 / maep:0.0f}", ntab=2)

                    mex0, _ = mex.compute_score(maep)
                    npoints = 50 if config.debug else 200
                    df, _ = mex.compute_set(maep, npoints=npoints)

                    ax.plot(df.u, df.v, "-", lw=2,
                            label=f"AEP solution set for 1:{1. / maep:0.0f}")

                    col = ax.get_lines()[-1].get_color()
                    ax.plot(mex0, mex0, "o", color=col, ms=10)
                    ax.plot([mex0] * 2, [0, mex0], "--", lw=1, color=col, alpha=0.8)

                    txt = f"1:{1. / mex0:0.0f}"
                    ax.annotate(txt, xy=(mex0, x0), xytext=(7, 10),
                                textcoords="offset pixels",
                                color=col, fontweight="bold")

                if kind == "AND":
                    ax.legend(loc=2, fontsize="large", framealpha=1)

                ax.plot([x0, x1], [x0, x1], "k-", lw=0.9, alpha=0.8)

                xlabel = f"{config.sta1} - Marginal Exceedance Score [-]"
                ylabel = f"{config.sta2} - Marginal Exceedance Score [-]"
                title = f"({letters[iax]}) Marginal Exceedance Score '{kind}'"\
                        + " - Bivariate"
                ax.set(xlim=(x0, x1), ylim=(x0, x1),
                       xlabel=xlabel, ylabel=ylabel,
                       xscale="log", yscale="log",
                       title=title)

                aep = [500, 100, 10, 2]
                tk = [1./a for a in aep]
                tkl = [f"1:{a}" for a in aep]
                ax.set_xticks(tk, labels=tkl)
                ax.set_yticks(tk, labels=tkl)

            else:
                nstas = [2, 4, 6]
                nstas_txt = [f"{n} stations" for n in nstas]
                maeps = config.maep_target
                maeps_txt = [f"1:{1/a:0.0f} AEP" for a in maeps]
                values = pd.DataFrame(np.nan, index=nstas_txt,
                                      columns=maeps_txt)
                for ista, nsta in enumerate(nstas):
                    logger.info(f"Nstations {nsta}", ntab=2)
                    if nsta == 2:
                        isids = [isid1, isid2]
                    elif nsta == 4:
                        isids = [isid1, isid2, isid3, isid4]
                    else:
                        isids = np.arange(nstations)

                    #cop = mes.IndependenceCopula(nsta)
                    cop = mes.GaussianCopula(nsta)
                    cop.params = cor[isids][:, isids]

                    mex = mes.MarginalExceedanceScore(kind, cop)
                    for iaep, maep in enumerate(maeps):
                        mex0, _ = mex.compute_score(maep)
                        cn = values.columns[iaep]
                        idx = values.index[ista]
                        values.loc[idx, cn] = mex0

                values.plot(kind="bar", ax=ax,
                            rot=0, legend=False)

                x0, x1 = 1. / 500, 1. / 2
                ylabel = f"Common Marginal Exceedance Score [-]"
                title = f"({letters[iax]}) Marginal Exceedance Score '{kind}'"\
                        + " - Multivariate"
                ax.set(ylim=(x0, x1), ylabel=ylabel,
                       yscale="log",
                       title=title)

                aep = [500, 100, 10, 2]
                tk = [1./a for a in aep]
                tkl = [f"1:{a}" for a in aep]
                ax.set_yticks(tk, labels=tkl)
                ax.grid(axis="y")

                if kind == "AND":
                    ax.legend(loc=2)

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
                               "ngrid", "maep_target"])
    awidth = 6
    aheight = 5
    fdpi = 300
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = True
    load_mvnproc = False
    load_expected_params = True
    load_postpred_checks = False

    sta1 = "203002"
    sta2 = "203005"
    sta3 = "203014"
    sta4 = "203010"

    ngrid = 20 if args.debug else 50
    maep_target = [1e-2, 1e-1]

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
                ngrid, maep_target)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
