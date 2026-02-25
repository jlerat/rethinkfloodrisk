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
        _, obs_data, _, _, postpred = select_data(data,
                                                  pcensor=pcensor,
                                                  rho_min=rho_min,
                                                  has_cluster=has_cluster,
                                                  copula=copula)
        assert len(postpred) == 1
        rn = next(iter(postpred))
        logger.info(f"-- Plotting {rn.text} --", nret=1)

        postpred = postpred[rn]
        obs_data = obs_data[rn]

        stationids = [cn for cn in obs_data.columns if not cn.startswith("WATER")]
        ncols = 3
        variables = config.variables
        varnames = [f"{ppt}/{vn}" for ppt in variables for vn in variables[ppt]]
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
            if ppt != "multi":
                df = df.loc[df.VARIABLE == varname]
                df = df.filter(regex="pvalue\\[", axis=1)

            if ppt == "univ":
                df.columns = stationids
                df.squeeze().plot(ax=ax, kind="barh")

                m = (df - 0.5).abs().mean().mean()
                ax.text(0.5, 0.5, f"mean diff\n{m:0.2f}",
                        transform=ax.transAxes,
                        va="center", ha="center",
                        fontweight="bold", fontsize="x-large",
                        path_effects=[paeff])
            elif ppt == "biv":
                #bins = np.concatenate([[0], np.linspace(0.05, 0.95, 5), [1]])
                #df.squeeze().plot(ax=ax, kind="hist", bins=bins,
                #                  edgecolor="0.2", facecolor="0.8")
                obj = putils.ecdfplot(ax, df.T)
                obj = obj[df.index[0]]

                idx = obj["index"]
                x = obj["values"]
                y = obj["position"]
                out = {
                    "low": x < 0.05,
                    "high": x > 0.95
                    }

                for name, ipb in out.items():
                    npb = ipb.sum()
                    if npb == 0:
                        continue

                    xpb, ypb, idxp = x[ipb], y[ipb], idx[ipb]
                    col = "tab:red"
                    for cnt, (xx, yy, ii) in enumerate(zip(xpb, ypb, idxp)):
                        ax.plot(xx, yy, "o", color=col)
                        i1 = int(re.sub(".*\\[|,.*", "", ii)) - 1
                        sta1 = stationids[i1]
                        i2 = int(re.sub(".*,|\\].*", "", ii)) - 1
                        sta2 = stationids[i2]
                        txt = f"{sta1}\n{sta2}"

                        xt = 0.2 if name == "low" else 0.8
                        ha = "left" if name == "low" else "right"
                        yt = np.linspace(0, 1,  2 * npb)[1 + cnt]
                        ax.annotate(txt, xy=(xx, yy),
                                    xytext=(xt, yt),
                                    textcoords="axes fraction",
                                    va="bottom", ha=ha,
                                    arrowprops=arrowprops)
            else:
                se = pd.Series(df.pvalue.values, index=df.VARIABLE)
                se.plot(ax=ax, kind="barh")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                        type=float, default=0.3)
    parser.add_argument("-d", "--diag", help="Show stan diagnostics",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                        type=float, default=-1.)
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_min",
                               "awidth", "aheight", "fdpi",
                               "excludes", "diag",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "variables", "exclude"])
    awidth = 6
    aheight = 5
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
        "biv": ["taildep_q75", "taildep_q90"],
        "multi": ["taildep_q75"]
    }

    config = CF(args.version, args.pcensor, args.rho_min,
                awidth, aheight, fdpi,
                excludes, args.diag,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks, variables,
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
