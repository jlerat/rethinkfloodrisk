#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from floodstan import freqplots
from floodstan.report import STAN_DIAGNOSTIC_VARIABLES as SDV
from pyrethink import datahub, processing

import importlib.util

ffit = Path(__file__).resolve().parent.parent / "fit" / "copulafit.py"
spec = importlib.util.spec_from_file_location("copulafit", ffit)
copulafit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copulafit)


def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    fdata = froot / "data"
    fproc = froot / "outputs" / f"copulaprocess_v{config.version}"

    basename = source_file.stem
    fimg = froot / "images" / "manuscript" / basename

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fdata", "fproc", "fimg"])
    script_paths = SP(source_file, basename, froot, fdata,
                      fproc, fimg)

    fimg.mkdir(exist_ok=True)

    for f in fimg.glob("*"):
        for ff in f.glob("*.*"):
            ff.unlink()
        if f.is_dir():
            f.rmdir()
        else:
            f.unlink()

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.source_file.stem
    logger = iutils.get_logger(basename)
    return logger


def get_data(config, script_paths, logger):
    opm = copulafit.get_options(config.version)

    obs_data, _, _, stations = datahub.get_ams_concat()

    fr = script_paths.fproc / "copulaconcat_ffa.csv"
    ffa, _ = csv.read_csv(fr, dtype={"STATIONID": str})

    DT = namedtuple("Data", ["stations", "ffa", "obs_data",
                             "options"])
    return DT(stations, ffa, obs_data, opm)


def process(config, script_paths, logger, data):
    stations = data.stations
    if config.debug:
        stations = stations.iloc[:2]

    options = data.options
    obs_data = data.obs_data
    ffa = data.ffa

    copula_specs = options.options["copula_spec"][1:]
    if config.debug:
        copula_specs = [copula_specs[0]]

    excludes = config.excludes + ["NONE"]
    fptype = config.freq_plot_type

    for copula_spec in copula_specs:
        for stationid in stations.index:
            sinfo = stations.loc[stationid]

            # Rating curve analysis
            rc, _ = datahub.get_rating_curves(stationid, True)
            rc_h = rc.loc[:, "WATERLEVEL[m]"]
            rc_q = rc.loc[:, "STREAMFLOW[m3_s-1]"]
            ipos = (rc_q > 1e-2) & (rc_h > 0)
            rc_h = rc_h.loc[ipos]
            rc_q = rc_q.loc[ipos]

            # Plot
            plt.close("all")
            mosaic = [[f"{ex}_univ-noninf",
                       f"{ex}_univ-inf",
                       f"{ex}_mv-inf"] for ex in excludes]
            nrows = len(mosaic)
            ncols = len(mosaic[0])
            figsize = (ncols * config.awidth, nrows * config.aheight)
            fig = plt.figure(figsize=figsize,
                             layout="constrained")
            axs = fig.subplot_mosaic(mosaic, sharey=True)

            for iax, (aname, ax) in enumerate(axs.items()):
                exclude, axcfg = aname.split("_")

                # Find tasks
                if re.search("univ", axcfg):
                    cs = "Univariate"
                else:
                    cs = copula_spec

                prior = "uninformative" if re.search("noninf", axcfg) \
                    else "informative"

                grp = next(g for g in options.options["group"]
                           if len(g.split("-")) == 8)
                group = stationid if cs == "Univariate" else grp

                taskid = options.find(prior=prior, copula_spec=cs,
                                      exclude=exclude,
                                      group= f"^{group}$")
                taskid = next(t for t in taskid)

                task = options.get_task(taskid)
                pcensor = task.pcensor

                # Get ffa data
                idx = ffa.STATIONID == stationid
                idx &= ffa.TASKID == taskid
                idx &= ffa.VARIABLE.str.contains("DESIGN")
                if idx.sum() == 0:
                    continue

                df = ffa.loc[idx].copy()
                ERI = df.VARIABLE.replace(".*ERI|\.$", "", regex=True)
                df.loc[:, "ERI"] = ERI.astype(float)

                # Plot obs data
                peaks = obs_data.loc[:, str(stationid)]
                x, y = freqplots.plot_data(ax, peaks, fptype, zorder=10)

                same = np.abs(y[:, None] - peaks.values[None, :]) < 1e-10
                _, same = np.where(same)
                time = obs_data.index[same]

                ythresh = y[-3]
                arrowprops = {
                    "edgecolor": "0.4",
                    "arrowstyle": "-"
                    }
                for wy, xx, yy in zip(time, x, y):
                    if yy < ythresh:
                        continue
                    txt = str(wy)
                    ax.annotate(txt, xy=(xx, yy),
                                xycoords="data",
                                xytext=(-40, 40),
                                va="bottom", ha="right",
                                textcoords="offset points",
                                arrowprops=arrowprops,
                                zorder=5)

                # Plot FFA
                aris = df.ERI
                quantiles = df

                iok = aris <= ari_max
                aris = aris[iok]
                quantiles = quantiles.loc[iok]

                inocens = 1 - 1./aris >= pcensor
                quantiles = quantiles.loc[inocens]
                aris = aris[inocens]
                freqplots.plot_marginal_quantiles(ax, aris, quantiles, fptype,
                                                  center_column="POSTERIOR_PREDICTIVE",
                                                  q0_column="5%",
                                                  q1_column="95%",
                                                  alpha=0.3,
                                                  facecolor="tab:blue",
                                                  edgecolor="k")

                retp = [10, 100, 500]
                aeps, xpos = freqplots.add_aep_to_xaxis(ax, fptype, True, retp)

                if exclude == "NONE":
                    exctxt = "All data"
                else:
                    ev = int(re.sub("-.*", "", exclude)) + 1
                    exctxt = f"Without {ev} flood"

                title = f"({letters[iax]}) Flood Frequency Curve\n"\
                        + f"{exctxt} - {prior} prior - {copula_spec} model"
                xlab = "Gumbel reduced variable $-log(-log(P))$ [-]"
                ylab = "Peak flow [m3.s-1]" if iax == 0 else ""
                ax.set(title=title, ylabel=ylab, xlabel=xlab)

                i100 = df.ERI == 100
                q100 = quantiles.loc[i100].squeeze()

                txt = "Uncertainty in the 1:100 event\n\n"
                kw = dict(va="top", ha="left", transform=ax.transAxes)
                ax.text(0.03, 0.97, txt, **kw, fontweight="bold")

                delta = 0.06
                for ist, st in enumerate(["5%", "POSTERIOR_PREDICTIVE", "95%"]):
                    q = q100.loc[st]
                    h = processing.linear_interpolation(q, rc_q, rc_h)
                    stt = "post pred" if st.startswith("POST") else st

                    txt = f"{stt:<12s}"
                    ytxt = 0.97 - delta * (ist + 1)
                    ax.text(0.03, ytxt, txt, **kw)

                    txt = f"{q:>5,.0f} $m^3.s^{{{-1}}}$ ({h:>4.1f}m)"
                    ax.text(0.18, ytxt, txt, **kw)

            ftitle = f"{sinfo.NAME} ({stationid})"
            fig.suptitle(ftitle, fontweight="bold")

            basename = script_paths.basename
            fp = f"{basename}_{stationid}_{copula_spec}_v{config.version}.png"
            fp = script_paths.fimg / fp
            fp.parent.mkdir(exist_ok=True)
            fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Univariate FFA plots",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "debug",
                               "awidth", "aheight", "fdpi",
                               "ptype", "ari_max", "excludes",
                               "freq_plot_type"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    excludes = ["2021"]
    freq_plot_type = "gumbel"

    config = CF(args.version,  args.debug,
                awidth, aheight, fdpi, ptype, ari_max,
                excludes,
                freq_plot_type)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
