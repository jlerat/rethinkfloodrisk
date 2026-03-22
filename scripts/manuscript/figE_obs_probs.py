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
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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

# Schematic
RIVERS = {
    "richmond": [
        [0.026, 0.876],
        [0.144, 0.815],
        [0.144, 0.556],
        [0.535, 0.369],
        [0.535, 0.303],
        [0.590, 0.274],
        [0.683, 0.351]
        ],
    "wilsons": [
        [0.867, 0.776],
        [0.540,	0.580],
        [0.535,	0.369]
        ],
    "leycester": [
        [0.272, 0.725],
        [0.540,	0.580]
        ],
    "coopers": [
        [0.687, 0.670],
        [0.791,	0.907]
        ]
    }

STATIONS = {
        "203004": [0.200, 0.532],
        "203005": [0.144, 0.815],
        "203010": [0.336, 0.694],
        "203014": [0.754, 0.707],
        "203012": [0.867, 0.776],
        "203002": [0.773, 0.873]
    }


def process(config, script_paths, logger, data):
    for pcensor, rho_min, has_cluster, copula_shape in get_iter_options(data):
        _, obs_data, mvnproc, _, _ = select_data(data,
                                                 pcensor=pcensor,
                                                 rho_min=rho_min,
                                                 has_cluster=has_cluster,
                                                 copula_shape=copula_shape)
        if len(mvnproc) == 0:
            continue

        assert len(mvnproc) == 2

        rn_isin = next(rn for rn in mvnproc if rn.exclude == "NONE")
        rn_isout = next(rn for rn in mvnproc if rn.exclude != "NONE")

        logger.info(f"-- Plotting {rn_isin.text} --", nret=1)

        mvnproc_in = mvnproc[rn_isin]
        mvnproc_out = mvnproc[rn_isout]

        #potpeaks, _, _ = datahub.get_potpeaks()
        #rk = potpeaks.rank(ascending=False)
        #obs = rk.index[(rk <= 2).any(axis=1)].astype(str).tolist()

        ncols = config.ncols
        nev = len(config.events)
        nrows = nev // ncols + (nev % ncols != 0)
        mosaic = [[config.events[ir * ncols + ic] for ic in range(ncols)]
                  for ir in range(nrows)]
        nrows = len(mosaic)
        plt.close("all")
        fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                         layout="constrained")
        kw = dict(hspace=0.05, wspace=0.05)
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

        for event, ax in axs.items():
            logger.info(f"Plotting {event}", ntab=1)

            # River lines
            for rname, pts in RIVERS.items():
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], "-",
                        color=config.river_color,
                        lw=8, solid_capstyle="round")

            # Stations
            for sid, pts in STATIONS.items():
                # AEP data
                cn = f"G{sid}_obs_{event}_log10aep"
                p_in = 10**mvnproc_in.loc[:, cn]
                pm_in = p_in.mean()
                ps_in = p_in.std()
                p_out = 10**mvnproc_out.loc[:, cn]
                pm_out = p_out.mean()
                ps_out = p_out.std()

                x, y = pts
                col = cm.Reds_r(Normalize(vmin=0, vmax=20)(pm_in * 100))
                ax.plot(x, y, "o",
                        ms=12, mec="k", mfc=col)

                xy = [x, y]
                delta = 18
                txt = f"{sid}\n" if event == config.events[0] else ""
                txt += f"{pm_in * 100:0.1f}% $\\pm$ {ps_in * 100:0.1f}%"
                #txt += f"{pm_out * 100:0.1f}% $\pm$ {ps_out * 100:0.1f}%"

                if sid == "203014":
                    xytext = [delta, -1.5 * delta]
                    va = "top"
                    ha = "left"
                elif sid == "203002":
                    xytext = [-delta, delta]
                    va = "bottom"
                    ha = "right"
                else:
                    xytext = [delta, delta]
                    va = "bottom"
                    ha = "left"

                ax.annotate(txt, xy,
                            ha=ha, va=va,
                            xytext=xytext,
                            fontsize="large",
                            textcoords="offset pixels")

            ax.axis("off")

            cn = f"GALL_obs_{event}_log10aep"
            p_in = 10**mvnproc_in.loc[:, cn]
            pm_in = p_in.mean()
            ps_in = p_in.std()
            p_out = 10**mvnproc_out.loc[:, cn]
            pm_out = p_out.mean()
            ps_out = p_out.std()

            if re.search("MAX", event):
                dt = "Max 2017/2008"
                title = f"{dt}\n"
            else:
                dt = pd.to_datetime(event).strftime("%b %Y")
                title = f"{dt} flood\n"

            title += f"{pm_in * 100:0.2f}% $\\pm$ {ps_in * 100:0.2f}%"
            #title += f"{pm_out * 100:0.1f}% $\pm$ {ps_out * 100:0.1f}%"
            ax.set_title(title, x=0.3, y=0.12,
                         fontsize="x-large",
                         fontweight="bold")

        basename = script_paths.basename
        fp = f"{basename}_{rn_isin.text}_v{config.version}.png"
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
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_mins", help="Minimum rho value",
                        type=str, default="-1")
    parser.add_argument("-s", "--copula_shapes", help="Copula shapes selected",
                        type=str, default="0")
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_mins",
                               "awidth", "aheight", "fdpi", "ncols",
                               "excludes", "copula_shapes",
                               "debug", "diag",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "events", "exclude",
                               "river_color"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ncols = 2
    load_ffa = False
    load_obs_data = False
    load_mvnproc = True
    load_expected_params = False
    load_postpred_checks = False
    excludes = ["NONE", "2021"]
    events = ["2008-01-04", "2017-03-31",
              "MAX-17-08", "2022-02-27"]

    river_color = "0.4"

    config = CF(args.version, args.pcensor,
                args.rho_mins.split("|"),
                awidth, aheight, fdpi, ncols, excludes,
                args.copula_shapes.split("|"),
                args.debug, False,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks, events,
                excludes, river_color)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
