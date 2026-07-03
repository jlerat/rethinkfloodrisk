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
from pyrethink import postpredchecks as ppc

from floodstan import marginals

from figA_impact_of_period_on_FFA import get_script_paths, get_logger, copulafit

def get_data(config, script_paths, logger):
    opm = copulafit.get_options(config.version)
    stations = datahub.get_stations()

    fu = script_paths.fproc / "copulaconcat_postpredcheck_univ.zip"
    pp_univ = pd.read_csv(fu, skiprows=15)

    fb = script_paths.fproc / "copulaconcat_postpredcheck_biv.zip"
    pp_biv = pd.read_csv(fb, skiprows=15)

    fm = script_paths.fproc / "copulaconcat_postpredcheck_multivar.zip"
    pp_mv = pd.read_csv(fb, skiprows=15)

    DT = namedtuple("Data", ["stations", "options", "pp_univ",
                             "pp_biv", "pp_mv"])
    return DT(stations, opm, pp_univ, pp_biv, pp_mv)


def process(config, script_paths, logger, data):

    univ = data.pp_univ
    biv = data.pp_biv
    import pdb; pdb.set_trace()


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
        obs_data = obs_data[rn].set_index("WATER_YEAR")

        stationids = obs_data.columns
        univ = postpred["univ"]
        multi = postpred["multi"]

        stats = pd.DataFrame("", index=stationids,
                             columns=["ams_min", "ams_max",
                                      "lh_skew_obs", "lh_skew_sim",
                                      "tau_q80_obs", "tau_q80_sim"])

        s = stationids[len(stationids) // 2]
        idx = multi.VARIABLE == "tau_q80"
        x = multi.loc[idx, f"obs"].squeeze()
        stats.loc[s, "tau_q80_obs"] = f"{x:0.2f}"
        x = multi.loc[idx, "simmean"].squeeze()
        xs = multi.loc[idx, "simstd"].squeeze()
        stats.loc[s, "tau_q80_sim"] = f"{x:0.2f} ±{xs:0.2f}"

        mname = "lskewness2"
        idx = univ.VARIABLE == mname
        for ista, stationid in enumerate(stationids):
            istap = ista + 1

            x = obs_data.loc[:, stationid].min()
            stats.loc[stationid, "ams_min"] = f"{int(x):,d}"

            x = obs_data.loc[:, stationid].max()
            stats.loc[stationid, "ams_max"] = f"{int(x):,d}"

            x = univ.loc[idx, f"obs[{istap}]"].squeeze()
            stats.loc[stationid, "lh_skew_obs"] = f"{x:0.2f}"

            x = univ.loc[idx, f"simmean[{istap}]"].squeeze()
            xs = univ.loc[idx, f"simstd[{istap}]"].squeeze()
            stats.loc[stationid, "lh_skew_sim"] = f"{x:0.2f} ±{xs:0.2f}"

        basename = script_paths.basename
        fs = f"{basename}_{rn.text}_v{config.version}_{mname}.csv"
        fs = script_paths.fimg / fs
        stats = stats.round(2)
        csv.write_csv(stats, fs,
                      f"Statistics {mname} for posterior",
                      source_file, compress=False,
                      write_index=True,
                      lineterminator="\n")

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
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    parser.add_argument("-x", "--xi_plots", help="Draw xi plots",
                        action="store_true", default=False)
    args = parser.parse_args()

    exclude = "NONE"
    copula_spec = "GaussianFactor_0_1"

    # Config
    CF = namedtuple("Config", ["version", "debug", "xi_plots",
                               "awidth", "aheight", "fdpi", "ncols",
                               "exclude", "copula_spec", "variables"])
    awidth = 6
    aheight = 5
    ncols = 3
    fdpi = 300

    # Post pred checks to plot
    variables = {
        "univ": ["lcoeffvar2", "lskewness2", "lkurtosis2"],
        "biv": ["kendalltau_high", "xibar_q90", "krupskii7"],
        "multi": ["xi", "xibar", "."]
    }

    config = CF(args.version, args.debug, args.xi_plots,
                awidth, aheight, fdpi, ncols,
                exclude, copula_spec, variables)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
