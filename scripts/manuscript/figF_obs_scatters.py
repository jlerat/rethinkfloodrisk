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
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib import ticker

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import sample

import figA_impact_of_period_on_FFA
import importlib
importlib.reload(figA_impact_of_period_on_FFA)

from figA_impact_of_period_on_FFA import get_script_paths, get_data, \
    get_logger


def process(config, script_paths, logger, data):
    ams = data.obs_data.loc[:, config.stationids]
    wy = ams.index

    awra = datahub.get_ams_awra(config.awraid)
    # .. convert mm/day -> m3/s
    area  = datahub.get_awra_cookies().loc[config.awraid, "CATCHMENTAREA_SRTM[km2]"]
    awra *= area / 86.4

    awra.name = "AWRA-L"
    start = ams.index[0]
    ams = pd.concat([ams, awra], axis=1).sort_index().loc[start:]

    nstations = ams.shape[1]
    nplots = nstations * (nstations - 1) // 2
    nc = config.ncols
    nr = nplots // nc + (1 if nplots % nc > 0 else 0)

    putils.set_mpl(font_size=22)
    plt.close("all")
    fig, axs = plt.subplots(ncols=nc, nrows=nr,
                            figsize=(nc * config.awidth, nr * config.awidth),
                            layout="constrained")
    for iplot, (i1, i2) in enumerate(comb(np.arange(nstations), 2)):
        sidx = ams.columns[i1]
        sidy = ams.columns[i2]

        logger.info(f"Plotting X={sidx} Y={sidy}")

        # Get data
        xy = ams.iloc[:, [i1, i2]]
        isok = xy.notnull().all(axis=1)
        xy = xy.loc[isok].values

        # plot
        ax = axs.flat[iplot]

        lxy = np.log10(xy)
        ax.plot(lxy[:, 0], lxy[:, 1], "o",
                mec="w", mfc="0.5",
                ms=8)

        lm0, lm1 = config.lims
        lxy = np.row_stack([lxy, [[lm0, lm0], [lm1, lm1]]])
        xx, yy, zz = putils.kde(lxy)

        ax.contourf(xx, yy, zz, cmap="Blues")

        xlim = config.lims
        xtk = [2, 3]
        ylim = config.lims
        ytk = [2, 3]

        xtkl = [f"$10^{{{k}}}$" for k in xtk]
        xstk = [i * 10**j for j in xtk for i in range(1, 10)]
        xstk = [math.log10(s) for s in xstk if s < xlim[1]]
        xtkl = [] if iplot < nc * (nr - 1) else xtkl

        ytkl = [f"$10^{{{k}}}$" for k in ytk]
        ystk = [i * 10**j for j in ytk for i in range(1, 10)]
        ystk = [math.log10(s) for s in ystk if s < ylim[1]]
        ytkl = [] if iplot % nc != 0 else ytkl

        ax.set(xlabel="", ylabel="",
               xlim=xlim, ylim=ylim)
        ax.set_xticks(xtk, labels=xtkl)
        ax.set_xticks(xstk, minor=True)
        ax.set_yticks(ytk, labels=ytkl)
        ax.set_yticks(ystk, minor=True)

        ax.grid(axis="both", color="0.5", lw=0.5)

        title = f"({iplot + 1}) X={sidx} Y={sidy}"
        ax.set_title(title, x=0.02, y=0.96,
                     fontsize=22,
                     va="top", ha="left")

        iok = np.all(~np.isnan(xy), axis=1)
        rho = spearmanr(xy[iok, 0], xy[iok, 1])[0]
        ax.text(0.98, 0.02, f"ρ={rho:0.2f}",
                transform=ax.transAxes, va="bottom",
                ha="right")


    lab = r"Annual streamflow maximum [$m^3.s^{-1}$]"
    fig.supxlabel(lab, fontsize=25)
    fig.supylabel(lab, fontsize=25)

    basename = script_paths.basename
    fp = f"{basename}_v{config.version}.png"
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
    parser.add_argument("-c", "--clean", help="Clean image folder",
                        action="store_true", default=False)
    parser.add_argument("-s", "--stationids", help="Selected stationids",
                        type=str, default="203010-203024-203014")
    parser.add_argument("-a", "--awraid", help="Selected AWRA-L covariate catchment",
                        type=str, default="WILSONSRIVER")
    args = parser.parse_args()
    stationids = args.stationids.split("-")

    # Config
    CF = namedtuple("Config", ["version", "awidth", "aheight",
                                "fdpi", "ncols", "clean", "lims",
                                "stationids", "awraid", "awra_lims"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ncols = 2
    lims = [math.log10(20), math.log10(4e3)]
    alims = [math.log10(20), math.log10(4e3)]

    config = CF(args.version,
                awidth, aheight, fdpi, ncols,
                args.clean, lims, stationids,
                args.awraid, alims)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
