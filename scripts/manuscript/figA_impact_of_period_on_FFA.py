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
from pyrethink import datahub


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"
    fout = froot / "outputs" / f"copulafit_v{config.version}"

    basename = source_file.stem
    fimg = froot / "images" / "manuscript" / basename

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fdata", "fout", "fimg"])
    script_paths = SP(source_file, basename, froot, fdata, fout, fimg)

    for pa in script_paths:
        if isinstance(pa, str):
            continue
        if pa.is_file():
            continue
        pa.mkdir(exist_ok=True)

    for f in fimg.glob("*.png"):
        f.unlink()
    for f in fimg.glob("*/*.png"):
        f.unlink()

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.source_file.stem
    logger = iutils.get_logger(basename)
    return logger


def diag2run_config(diag):
    RN = namedtuple("Run",
                    ["exclude", "pcensor", "rho_min", "copula",
                     "has_cluster", "text"])
    ex = diag["task_exclude"]
    pc = diag["task_pcensor"]
    rm = diag["task_rho_min"]
    cop = diag["task_copula"]
    hc = diag["task_has_clusters"]
    txt = f"PC{pc}_RM{rm}_C{cop}_HC{hc}_EX{ex}"
    return RN(ex, pc, rm, cop, hc, txt)


def display_stan_diagnostics(diag, logger):
    for vn in SDV:
        logger.info(f"{vn}: {diag[vn][:50]}", ntab=1)


def get_taskids(config, script_paths):
    fopm = script_paths.fout / "copulafit_options.json"
    opm = hyruns.OptionManager.from_file(fopm)
    kw = {}

    if hasattr(config, "pcensor"):
        kw["pcensor"] = f"{config.pcensor:0.1f}"

    if hasattr(config, "excludes"):
        kw["exclude"] = "|".join(config.excludes)

    if hasattr(config, "rho_min"):
        kw["rho_min"] = f"{config.rho_min:0.1f}"

    if hasattr(config, "copula"):
        kw["copula"] = config.copula

    return opm.find(**kw)


def get_data(config, script_paths, logger):
    _, _, _, stations = datahub.get_ams_concat()
    taskids = get_taskids(config, script_paths)

    ffa = {}
    obs_data = {}

    options = {
        "pcensors": set(),
        "rho_mins": set(),
        "has_clusters": set(),
        "copulas": set()
    }

    for taskid in taskids:
        ftask = script_paths.fout / f"copulafit_TASK{taskid}"
        fd = ftask / f"copulafit_diagnostic_TASK{taskid}.json"
        with fd.open("r") as fo:
            diag = json.load(fo)

        rn = diag2run_config(diag)
        options["pcensors"].add(rn.pcensor)
        options["rho_mins"].add(rn.rho_min)
        options["has_clusters"].add(rn.has_cluster)
        options["copulas"].add(rn.copula)

        mess = f"Load report TASK {taskid} {rn.text}"
        logger.info(mess, nret=1)
        if rn.rho_min != config.rho_min \
                or rn.pcensor != config.pcensor:
            errmsg = "Pb with run config"
            raise ValueError(errmsg)

        if config.diag:
            display_stan_diagnostics(diag, logger)

        fr = ftask / f"postprocess_report_TASK{taskid}.csv"
        df, _ = csv.read_csv(fr, index_col=0)
        ffa[rn] = df

        fd = ftask / f"copulafit_data_TASK{taskid}.json"
        with fd.open("r") as fo:
            d = json.load(fo)
            y = pd.DataFrame(d["y"], columns=d["stationids"])
            cn = "WATER_YEAR"
            y.loc[:, cn] = d["ams_time"]
            y.loc[:, cn] += 1 # adds 1 because starts in Oct
            obs_data[rn] = y

    DT = namedtuple("Data", ["stations", "ffa", "obs_data",
                             "options"])
    return DT(stations, ffa, obs_data, options)


def process(config, script_paths, logger, data):

    pcensors = data.options["pcensors"]
    rho_mins = data.options["rho_mins"]
    has_clusters = data.options["has_clusters"]
    copulas = data.options["copulas"]

    for pcensor, rho_min, has_cluster, copula in \
            prod(pcensors, rho_mins, has_clusters, copulas):
        # Select data
        ffa = {}
        obs_data = {}
        selected = set()
        for rn in data.ffa.keys():
            if rn.pcensor == pcensor and rn.rho_min == rho_min \
                    and rn.has_cluster == has_cluster\
                    and rn.copula == copula:
                selected.add(rn)
                ffa[rn.exclude] = data.ffa[rn]
                obs_data[rn.exclude] = data.obs_data[rn]

        assert len(selected) == 2
        rtxt = re.sub("_EX.*", "", next(iter(selected)).text)
        logger.info(f"-- Plotting {rtxt} --", nret=1)

        for stationid, sinfo in data.stations.iterrows():
            logger.info(f"Plotting {stationid}", ntab=1)

            plt.close("all")
            mosaic = [[per for per in obs_data.keys()]]
            nrows = len(mosaic)
            ncols = len(mosaic[0])
            fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                             layout="constrained")
            axs = fig.subplot_mosaic(mosaic, sharey=True)

            # Rating curve analysis
            rc, _ = datahub.get_rating_curves(stationid, True)
            rc_h = rc.loc[:, "WATERLEVEL[m]"]
            rc_q = rc.loc[:, "STREAMFLOW[m3_s-1]"]
            ipos = (rc_q > 1e-2) & (rc_h > 0)
            rc_h = rc_h.loc[ipos]
            rc_q = rc_q.loc[ipos]

            # Plot ffa
            for iax, (aname, ax) in enumerate(axs.items()):
                exclude = aname

                # Plot data
                peaks = obs_data[exclude].loc[:, str(stationid)]

                x, y = freqplots.plot_data(ax, peaks, ptype, zorder=10)
                same = np.abs(y[:, None] - peaks.values[None, :]) < 1e-10
                _, same = np.where(same)
                time = obs_data[exclude].WATER_YEAR.iloc[same]

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
                df = ffa[exclude]
                istation = obs_data[exclude].columns.tolist().index(str(stationid))
                quantiles = df.filter(regex=f"DESIGN.*\\[{istation + 1}\\]", axis=0)
                aris = quantiles.index.to_series().str\
                        .replace(".*ERI|\\[.*", "", regex=True).astype(float).values

                iok = aris <= ari_max
                aris = aris[iok]
                quantiles = quantiles.loc[iok]

                inocens = 1 - 1./aris >= pcensor
                quantiles = quantiles.loc[inocens]
                aris = aris[inocens]
                freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                                  center_column="POSTERIOR_PREDICTIVE",
                                                  q0_column="5%",
                                                  q1_column="95%",
                                                  alpha=0.3,
                                                  facecolor="tab:blue",
                                                  edgecolor="k")

                retp = [10, 100, 500]
                aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, True, retp)

                if exclude == "NONE":
                    exctxt = "All data"
                else:
                    ev = int(re.sub("-.*", "", exclude)) + 1
                    exctxt = f"Without {ev} flood"

                title = f"({letters[iax]}) {exctxt}"
                xlab = "Gumbel reduced variable $-log(-log(P))$ [-]"
                ylab = "Peak flow [m3.s-1]" if iax == 0 else ""
                ax.set(title=title, ylabel=ylab, xlabel=xlab)

                q100 = quantiles.filter(regex="DESIGN_ERI100\\[", axis=0).squeeze()

                txt = "Uncertainty in 1:100 event\n\n"
                kw = dict(va="top", ha="left", transform=ax.transAxes)
                ax.text(0.03, 0.97, txt, **kw, fontweight="bold")

                delta = 0.06
                for ist, st in enumerate(["5%", "POSTERIOR_PREDICTIVE", "95%"]):
                    q = q100.loc[st]
                    h = datahub.linear_interpolation(q, rc_q, rc_h)
                    stt = "post pred" if st.startswith("POST") else st

                    txt = f"{stt:<12s}"
                    ytxt = 0.97 - delta * (ist + 1)
                    ax.text(0.03, ytxt, txt, **kw)

                    txt = f"{q:>5,.0f} $m^3.s^{{{-1}}}$ ({h:>4.1f}m)"
                    ax.text(0.18, ytxt, txt, **kw)

            ftitle = f"{sinfo.NAME} ({stationid}) [{rtxt}]"
            fig.suptitle(ftitle, fontweight="bold")

            basename = script_paths.basename
            fp = f"{basename}_{stationid}_{rtxt}_v{config.version}.png"
            fp = script_paths.fimg / rtxt / fp
            fp.parent.mkdir(exist_ok=True)
            fig.savefig(fp, dpi=fdpi)


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
                               "ptype", "ari_max", "excludes",
                               "diag"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    excludes = ["NONE", "2021"]
    config = CF(args.version, args.pcensor, args.rho_min,
                awidth, aheight, fdpi, ptype, ari_max,
                excludes, args.diag)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
