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
    for pcensor, rho_min, has_cluster, copula_shape in get_iter_options(data):
        _, obs_data, _, _, postpred = select_data(data,
                                                  pcensor=pcensor,
                                                  rho_min=rho_min,
                                                  has_cluster=has_cluster,
                                                  copula_shape=copula_shape)
        if len(postpred) == 0:
            continue

        assert len(postpred) == 1
        rn = next(iter(postpred))
        logger.info(f"-- Plotting {rn.text} --", nret=1)

        postpred = postpred[rn]
        obs_data = obs_data[rn]

        # Generic postpredictive checks
        stationids = [cn for cn in obs_data.columns if not cn.startswith("WATER")]
        ncols = config.ncols
        variables = config.variables
        varnames = [f"{ppt}/{vn}" if vn != "." else "."
                    for ppt in variables for vn in variables[ppt]]
        nv = len(varnames)
        nrows = nv // ncols + int(nv % ncols > 0)
        mosaic = [[varnames[ncols * ir + ic] if ncols * ir + ic < nv else "."
                  for ic in range(ncols)] for ir in range(nrows)]

        arrowprops = dict(arrowstyle="wedge", facecolor="0.6", edgecolor="none")
        paeff = pe.withStroke(linewidth=4, foreground="w")

        plt.close("all")
        fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                         layout="constrained")
        axs = fig.subplot_mosaic(mosaic)

        for aname, ax in axs.items():
            ppt, varname = aname.split("/")
            df = postpred[ppt]

            if ppt in ["univ", "biv"]:
                df = df.loc[df.VARIABLE == varname]
                df = df.filter(regex="pvalue\\[", axis=1)
                if ppt == "univ":
                    df.columns = stationids
                else:
                    cc = ["/".join([stationids[int(i) - 1]
                                    for i in re.sub(".*\\[|\\]", "",
                                                    cn).split(",")])
                          for cn in df.columns]
                    df.columns = cc

                df.squeeze().plot(ax=ax, kind="barh")

            #elif ppt == "biv":
            #    df = df.loc[df.VARIABLE == varname]
            #    obj = putils.ecdfplot(ax, df.T)
            #    obj = obj[df.index[0]]

            #    ax.legend(loc=2, fontsize="small",
            #              framealpha=0.)

            #    idx = obj["index"]
            #    x = obj["values"]
            #    y = obj["position"]
            #    out = {
            #        "low": x < 0.05,
            #        "high": x > 0.95
            #        }

            #    for name, ipb in out.items():
            #        npb = ipb.sum()
            #        if npb == 0:
            #            continue

            #        xpb, ypb, idxp = x[ipb], y[ipb], idx[ipb]
            #        col = "tab:red"
            #        for cnt, (xx, yy, ii) in enumerate(zip(xpb, ypb, idxp)):
            #            ax.plot(xx, yy, "o", color=col)
            #            i1 = int(re.sub(".*\\[|,.*", "", ii)) - 1
            #            sta1 = stationids[i1]
            #            i2 = int(re.sub(".*,|\\].*", "", ii)) - 1
            #            sta2 = stationids[i2]
            #            txt = f"{sta1}\n{sta2}"

            #            xt = 0.2 if name == "low" else 0.8
            #            ha = "left" if name == "low" else "right"
            #            yt = np.linspace(0, 1,  2 * npb)[1 + cnt]
            #            ax.annotate(txt, xy=(xx, yy),
            #                        xytext=(xt, yt),
            #                        textcoords="axes fraction",
            #                        va="bottom", ha=ha,
            #                        arrowprops=arrowprops)
            else:
                idx = df.VARIABLE.str.startswith(varname + "_")
                q = df.VARIABLE.loc[idx].str.replace(varname + "_q", "")
                df = df.loc[idx].filter(regex="pvalue$", axis=1)
                se = df.set_index(q).squeeze()
                se.plot(ax=ax, kind="barh")

            mv = df.mean().mean()
            md = (df - 0.5).abs().mean().mean()
            txt = f"mean pval\n{mv:0.2f}\n\nmean diff\n{md:0.2f}"
            ax.text(0.5, 0.5, txt,
                    transform=ax.transAxes,
                    va="center", ha="center",
                    fontweight="bold", fontsize="x-large",
                    path_effects=[paeff])

            for x in [0.05, 0.95]:
                putils.line(ax, 0, 1, x, 0, "r--")

            putils.line(ax, 0, 1, 0.5, 0, "k-", lw=2)

            title = f"{ppt} post pred checks / {varname}"
            xlab = "check pvalue [-]"
            ax.set(title=title, xlabel=xlab, xlim=(0, 1))

        fig.suptitle(rn.text, fontweight="bold")

        basename = script_paths.basename
        fp = f"{basename}_{rn.text}_v{config.version}.png"
        fp = script_paths.fimg / fp
        fig.savefig(fp, dpi=config.fdpi)

        # Bivariate xi functions
        if not config.xi_plots:
            continue

        obs = data.obs_data[rn]

        df = postpred["biv"]
        pairs = df.filter(regex="obs\\[", axis=1)\
            .columns\
            .to_series()\
            .str.replace("obs\\[|\\]", "", regex=True)\
            .values

        fd = script_paths.fimg / f"xifunctions_{rn.text}"
        fd.mkdir(exist_ok=True)
        for pair in pairs:
            mosaic = [["scatter", "xi"], ["xibar", "tau"]]
            ncols, nrows = len(mosaic[0]), len(mosaic)
            plt.close("all")
            fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                             layout="constrained")
            axs = fig.subplot_mosaic(mosaic)

            ls = dict(xi="--", xibar=":", tau="-")
            cols = dict(xi="tab:blue", xibar="tab:red",
                        tau="tab:green")

            sid1 = stationids[int(pair[0]) - 1]
            sid2 = stationids[int(pair[-1]) - 1]
            logger.info(f"xi fun plots for pair {sid1}/{sid2}", ntab=1)

            for varname, ax in axs.items():
                if varname == "scatter":
                    putils.bivarnplot(ax, obs.loc[:, [sid1, sid2]].values,
                                      namex=sid1, namey=sid2)
                    continue

                idx = df.VARIABLE.str.startswith(varname + "_")
                q = df.VARIABLE.loc[idx].str.replace(varname + "_q", "")
                dd = df.loc[idx]\
                        .filter(regex=pair, axis=1)\
                        .set_index(q)
                dd.columns = [re.sub("\\[.*", "", cn) for cn in dd.columns]

                ax.plot(dd.index, dd.obs, "k", ls=ls[varname],
                        label=f"{varname} obs")

                col = cols[varname]
                ax.plot(dd.index, dd.simmean, "-", color=col,
                        label=f"{varname} sim")
                ax.fill_between(dd.index, dd.simmean - dd.simstd / 2,
                                dd.simmean + dd.simstd / 2,
                                fc=col, alpha=0.5,
                                ec="none")
                ax.legend(loc=2)

                title = f"{varname} functions for {sid1} / {sid2}"
                ylim = (0, 1)
                ax.set(title=title, ylim=ylim)

            fp = f"{basename}_{rn.text}_xi_{sid1}_{sid2}_v{config.version}.png"
            fp = fd / fp
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
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_mins", help="Minimum rho value",
                        type=str, default="-1|0")
    parser.add_argument("-s", "--copula_shapes", help="Copula shapes selected",
                        type=str, default="0|3")
    parser.add_argument("-x", "--xi_plots", help="Draw xi plots",
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
                               "variables", "exclude",
                               "xi_plots"])
    awidth = 6
    aheight = 5
    ncols = 3
    fdpi = 300
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = True
    load_mvnproc = False
    load_expected_params = False
    load_postpred_checks = True
    exclude = "NONE"

    # Post pred checks to plot
    variables = {
        "univ": ["lcoeffvar2", "lskewness2", "lkurtosis2"],
        "biv": ["kendalltau_high", "xibar_q90", "krupskii7"],
        "multi": ["xi", "xibar", "."]
    }

    config = CF(args.version, args.pcensor,
                args.rho_mins.split("|"),
                awidth, aheight, fdpi, ncols,
                excludes,
                args.copula_shapes.split("|"),
                args.diag, args.debug,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks, variables,
                exclude, args.xi_plots)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
