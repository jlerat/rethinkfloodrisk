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

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import sample

from floodstan import marginals

import figA_impact_of_period_on_FFA
import importlib
importlib.reload(figA_impact_of_period_on_FFA)

from figA_impact_of_period_on_FFA import get_script_paths
from figA_impact_of_period_on_FFA import get_logger, get_taskids, get_data
from figA_impact_of_period_on_FFA import get_iter_options, select_data


def process(config, script_paths, logger, data):
    for pcensor, rho_min, has_cluster, copula in get_iter_options(data):
        _, obs_data, mvnproc, _, _ = select_data(data,
                                                 pcensor=pcensor,
                                                 rho_min=rho_min,
                                                 has_cluster=has_cluster,
                                                 copula=copula)
        if len(mvnproc) == 0:
            continue
        assert len(mvnproc) == 1
        rn = next(iter(mvnproc))
        logger.info(f"-- Plotting {rn.text} --", nret=1)

        mvnproc = mvnproc[rn]
        obs_data = obs_data[rn]

        groups = mvnproc.columns.str.replace("_.*", "", regex=True).unique()
        groups = [g for g in groups if g not in ["", "mvn", "mv"]]

        ng = len(groups)
        nrows = ng // config.ncols + int(ng % config.ncols > 0)
        mosaic = [[groups[ncols * ir + ic] if ncols * ir + ic < ng else "."
                   for ic in range(ncols)] for ir in range(nrows)]

        for event in config.events:
            logger.info(f"Plotting {event}", ntab=1)
            plt.close("all")
            fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                             layout="constrained")
            axs = fig.subplot_mosaic(mosaic)
            for aname, ax in axs.items():
                grp = aname
                logger.info(f"Group {grp}", ntab=2)

                cn = f"{grp}_obs_log10aep_{event}"
                if cn not in mvnproc.columns:
                    logger.info(f"No data for group {grp}, skip", ntab=3)
                    ax.axis("off")
                    continue

                # value -> %
                aep = 10**(mvnproc.loc[:, cn] + 2)
                prob = aep

                if prob.notnull().sum() == 0:
                    logger.info(f"No data for group {grp}, skip", ntab=3)
                    ax.axis("off")
                    continue

                x0 = round(math.log10(max(1e-9, prob.quantile(0.001))), 2)
                x1 = round(math.log10(prob.max()), 2)
                bins = np.logspace(x0, x1, 50)

                ax.hist(prob, bins=bins, facecolor="0.8", edgecolor="0.2")

                m = prob.mean()
                y0, y1 = ax.get_ylim()
                xy = (m, (y0 + y1) / 2)
                paef = pe.withStroke(linewidth=4,
                                     foreground="w")

                ax.annotate(f"Mean\n{m:0.2f}%", xy, (0, 5),
                            xycoords="data", textcoords="offset points",
                            fontweight="bold", va="bottom", ha="center",
                            fontsize=15,
                            path_effects=[paef])
                ax.plot([m, m], [y0, y1], "k-", lw=2)

                title = f"{grp[1:]}"
                xlab = "AEP [%]"
                ax.set(title=title, xscale="log",
                       xlabel=xlab, ylim=(y0, y1))

            ftitle = f"Event {event} [{rn.text}]\n"
            fig.suptitle(ftitle, fontsize=20, fontweight="bold")

            basename = script_paths.basename
            fp = f"{basename}_{rn.text}_{event}_v{config.version}.png"
            fp = script_paths.fimg / fp
            fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                        type=float, default=0.3)
    parser.add_argument("-di", "--diag", help="Show stan diagnostics",
                        action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                        type=float, default=-1.)
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_min",
                               "awidth", "aheight", "fdpi", "ncols",
                               "excludes", "diag", "debug",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "events", "exclude"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ncols = 3
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = True
    load_mvnproc = True
    load_expected_params = False
    load_postpred_checks = False
    exclude = "NONE"
    events = ["2022-02-27"]

    config = CF(args.version, args.pcensor, args.rho_min,
                awidth, aheight, fdpi, ncols,
                excludes, args.diag, args.debug,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks, events,
                exclude)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
