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
from hydrodiy.stat import sutils
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import sample
from pyrethink import marginal_exceedance_score as mes
from pyrethink.marginal_exceedance_score import MARGINAL_EXCEEDANCE_SCORE_KINDS\
    as MEXS_KINDS
from floodstan import marginals

import figA_impact_of_period_on_FFA
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
        _, obs_data, mvnproc, expected, _ = select_data(data,
                                                        pcensor=pcensor,
                                                        rho_min=rho_min,
                                                        has_cluster=has_cluster,
                                                        copula_shape=copula_shape)
        if len(mvnproc) == 0:
            continue

        assert len(mvnproc) == 1

        rn = next(rn for rn in mvnproc if rn.exclude == "NONE")

        logger.info(f"-- Plotting {rn.text} --", nret=1)

        ams = obs_data[rn].set_index("WATER_YEAR")
        stationids = ams.columns
        nstations = len(stationids)

        mvnproc = mvnproc[rn]
        expected = expected[rn]

        # Dependence
        cop = mes.GaussianCopula(nstations)
        cor = pd.Series(expected["corr_IW"]).values.reshape((nstations, nstations))
        cop.params = cor
        cop.logger = logger

        # Historical grid
        # nstations + multivar
        nval = len(ams)
        marg_cdf = np.nan * np.zeros((nval, nstations))
        aep = np.zeros(nval)
        gev = marginals.GEV()

        for k in range(nstations):
            gev.locn = expected["ylocs"][f"ylocn[{k + 1}]"]
            gev.logscale = expected["ylogscales"][f"ylogscale[{k + 1}]"]
            gev.shape1 = expected["yshape1"][f"yshape1[{k + 1}]"]
            marg_cdf[:, k] = gev.cdf(ams.iloc[:, k])

        aeps = np.nan * np.zeros((nval, nstations + 2))
        # .. marginal aeps
        aeps[:, :nstations] = (1 - marg_cdf) * 1e2
        # .. joint aep
        aeps[:, -1] = cop.aep(marg_cdf, "KENDALL") * 1e2

        # Configure
        ncols_years = config.ncols_years
        nev = len(config.years)
        nrows = nev // ncols_years + (nev % ncols_years != 0)
        mosaic = [["grid"] + [config.years[ir * ncols_years + ic]
                              for ic in range(ncols_years)]
                  for ir in range(nrows)]
        nrows = len(mosaic)
        plt.close("all")
        fig = plt.figure(figsize=((ncols_years + 1) * awidth,
                                  nrows * aheight),
                         layout="constrained")
        wr = [1] + [1.5] * ncols_years
        kw = dict(hspace=0.05, wspace=0.05, width_ratios=wr)
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

        for iax, (aname, ax) in enumerate(axs.items()):
            logger.info(f"Plotting {aname}", ntab=1)

            if aname == "grid":
                im = ax.imshow(aeps, cmap="Reds_r",
                               vmin=0, vmax=config.aep_max)

                # Show AEP values
                #for i in np.arange(nval):
                #    a = aeps[i, -1]
                #    if a > 20:
                #        continue
                #    ax.text(nstations + 1, i, f"{a:0.1f}",
                #            color="w", ha="center",
                #            va="center", fontweight="bold",
                #            fontsize="x-small")

                labs = ams.columns.to_list() + ["", "Multiv. K."]

                ax.set_xticks(np.arange(nstations + 2),
                              labels=labs,
                              rotation=90)
                ax.set_yticks(np.arange(nval), labels=ams.index)

                cbar = fig.colorbar(im,
                                    ticks=np.arange(10, config.aep_max, 10),
                                    shrink=0.5)
                cbar.ax.set(title="AEP\n[%]")
                ax.set(title=f"({letters[iax]}) Expected AEP")
                continue

            year = aname
            # River lines
            for rname, pts in RIVERS.items():
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], "-",
                        color=config.river_color,
                        lw=8, solid_capstyle="round")

            # Stations
            for sid, pts in STATIONS.items():
                # AEP data
                cn = f"G{sid}_ams_UNIV_{year - 1}_log10aep"
                p = 10**mvnproc.loc[:, cn]
                pm = p.mean() * 100
                ps = p.std() * 100

                x, y = pts
                col = cm.Reds_r(Normalize(vmin=0, vmax=config.aep_max)(pm))
                ax.plot(x, y, "o",
                        ms=14, mec="w", mfc=col,
                        markeredgewidth=2)

                xy = [x, y]
                delta = 18
                txt = f"({sid})\n" if year == config.years[0] else ""
                txt += f"{pm:0.1f}% $\\pm$ {ps:0.1f}%"

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
                            textcoords="offset pixels")

            ax.axis("off")

            kind = "KENDALL"
            cn = f"GALL_ams_{kind}_{year - 1}_log10aep"
            p = 10**mvnproc.loc[:, cn]
            pm = p.mean() * 100
            ps = p.std() * 100
            txt = "Multivariate 'KENDALL'\n"\
                  + f"AEP = {pm:4.1f}% $\\pm$ {ps:4.1f}%"
            x0 = 0.65
            y0 = 0.35
            ax.annotate(txt, (x0, y0),
                        xycoords="axes fraction",
                        ha="left", va="top",
                        fontsize="large")

            ax.set_title(f"({letters[iax]}) Details of {year} AEPs",
                         x=0.01, y=1, ha="left", va="top")

        basename = script_paths.basename
        fp = f"{basename}_{rn.text}_v{config.version}.png"
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
                               "awidth", "aheight", "fdpi",
                               "ncols_years",
                               "excludes", "copula_shapes",
                               "debug", "diag",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "years", "exclude",
                               "river_color",
                               "aep_max"])
    awidth = 4
    aheight = 5
    fdpi = 300
    ncols_years = 1
    load_ffa = False
    load_obs_data = True
    load_mvnproc = True
    load_expected_params = True
    load_postpred_checks = False
    excludes = ["NONE"]
    years = [2017, 2022]
    aep_max = 40

    river_color = "0.5"

    config = CF(args.version, args.pcensor,
                args.rho_mins.split("|"),
                awidth, aheight, fdpi, ncols_years, excludes,
                args.copula_shapes.split("|"),
                args.debug, False,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks, years,
                excludes, river_color, aep_max)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
