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


def get_script_paths(config, source_file):
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
    mvnproc = {}
    expected_params = {}
    postpred_checks = {}

    options = {
        "pcensors": set(),
        "rho_mins": set(),
        "has_clusters": set(),
        "copulas": set()
    }

    run_configs = []
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
        run_configs.append(rn)

        mess = f"Load data from TASK {taskid} {rn.text}"
        logger.info(mess, nret=1)
        if rn.rho_min != config.rho_min \
                or rn.pcensor != config.pcensor:
            errmsg = "Pb with run config"
            raise ValueError(errmsg)

        if config.diag:
            display_stan_diagnostics(diag, logger)

        if config.load_ffa:
            fr = ftask / f"postprocess_report_TASK{taskid}.csv"
            df, _ = csv.read_csv(fr, index_col=0)
            ffa[rn] = df

        if config.load_mvnproc:
            fs = ftask / f"copulafit_mvnprocess_TASK{taskid}.zip"
            mvnproc[rn], _ = csv.read_csv(fs)

        if config.load_postpred_checks:
            pp = {}
            for ppt in ["univ", "biv"]:
                fp = f"postprocess_postpredchecks_{ppt}_TASK{taskid}.csv"
                fp = ftask / fp
                if not fp.exists():
                    continue
                df = pd.read_csv(fp, skiprows=15)
                df.columns = ["VARIABLE"] + df.columns[1:].tolist()
                pp[ppt] = df

            postpred_checks[rn] = pp

        if config.load_expected_params:
            fe = ftask / f"expected_parameters_TASK{taskid}.json"
            if not fe.exists():
                fs = ftask / f"copulafit_samples_TASK{taskid}.zip"
                samples = pd.read_csv(fs, skiprows=15)
                ylocs = samples.filter(regex="ylocn", axis=1).mean()
                ylogscales = samples.filter(regex="ylogsca", axis=1).mean()
                yshape1 = samples.filter(regex="yshape1", axis=1).mean()
                cor = samples.filter(regex="corr_IW", axis=1).mean()
                expected = {
                    "ylocs": ylocs.to_dict(),
                    "ylogscales": ylogscales.to_dict(),
                    "yshape1": yshape1.to_dict(),
                    "corr_IW": cor.to_dict()
                    }
                with fe.open("w") as fo:
                    json.dump(expected, fo, indent=4)

            else:
                with fe.open("r") as fo:
                    expected = json.load(fo)

            expected_params[rn] = expected

        if config.load_obs_data:
            fd = ftask / f"copulafit_data_TASK{taskid}.json"
            with fd.open("r") as fo:
                d = json.load(fo)
                y = pd.DataFrame(d["y"], columns=d["stationids"])
                cn = "WATER_YEAR"
                y.loc[:, cn] = d["ams_time"]
                y.loc[:, cn] += 1 # adds 1 because starts in Oct
                obs_data[rn] = y

    DT = namedtuple("Data", ["stations", "run_configs", "ffa", "obs_data",
                             "mvnproc", "expected_params", "postpred_checks",
                             "options"])
    return DT(stations, run_configs, ffa, obs_data, mvnproc,
              expected_params, postpred_checks, options)

def get_iter_options(data):
    pcensors = data.options["pcensors"]
    rho_mins = data.options["rho_mins"]
    has_clusters = data.options["has_clusters"]
    copulas = data.options["copulas"]
    return prod(pcensors, rho_mins, has_clusters, copulas)


def select_data(data, **kwargs):
    selected = []
    ffa = {}
    obs_data = {}
    mvnproc = {}
    expected_params = {}
    postpred_checks = {}
    for rn in data.run_configs:
        isin = True
        if "pcensor" in kwargs:
            isin &= rn.pcensor == kwargs["pcensor"]

        if "rho_min" in kwargs:
            isin &= rn.rho_min == kwargs["rho_min"]

        if "has_cluster" in kwargs:
            isin &= rn.has_cluster == kwargs["has_cluster"]

        if "copula" in kwargs:
            isin &= rn.copula == kwargs["copula"]

        if isin:
            selected.append(rn)
            if rn in data.ffa:
                ffa[rn] = data.ffa[rn]

            if rn in data.obs_data:
                obs_data[rn] = data.obs_data[rn]

            if rn in data.mvnproc:
                mvnproc[rn] = data.mvnproc[rn]

            if rn in data.expected_params:
                expected_params[rn] = data.expected_params[rn]

            if rn in data.postpred_checks:
                postpred_checks[rn] = data.postpred_checks[rn]

    return ffa, obs_data, mvnproc, expected_params, postpred_checks


def process(config, script_paths, logger, data):

    for pcensor, rho_min, has_cluster, copula in get_iter_options(data):
        ffa, obs_data, _, _, _ = select_data(data,
                                             pcensor=pcensor,
                                             rho_min=rho_min,
                                             has_cluster=has_cluster,
                                             copula=copula)
        assert len(ffa) == 2
        rn_isin = next(rn for rn in ffa if rn.exclude == "NONE")
        rn_isout = next(rn for rn in ffa if rn.exclude != "NONE")

        rtxt = re.sub("_EX.*", "", rn_isin.text)
        logger.info(f"-- Plotting {rtxt} --", nret=1)

        for stationid, sinfo in data.stations.iterrows():
            logger.info(f"Plotting {stationid}", ntab=1)

            plt.close("all")
            mosaic = [["isin", "isout"]]
            nrows = len(mosaic)
            ncols = len(mosaic[0])
            figsize = (ncols * config.awidth, nrows * config.aheight)
            fig = plt.figure(figsize=figsize,
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
                rn = rn_isin if aname == "isin" else rn_isout

                # Plot data
                peaks = obs_data[rn].loc[:, str(stationid)]

                x, y = freqplots.plot_data(ax, peaks, ptype, zorder=10)
                same = np.abs(y[:, None] - peaks.values[None, :]) < 1e-10
                _, same = np.where(same)
                time = obs_data[rn].WATER_YEAR.iloc[same]

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
                df = ffa[rn]
                istation = obs_data[rn].columns.tolist().index(str(stationid))
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

                if aname == "isin":
                    exctxt = "All data"
                else:
                    ev = int(re.sub("-.*", "", rn.exclude)) + 1
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
                               "ptype", "ari_max", "excludes",
                               "diag",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    excludes = ["NONE", "2021"]
    load_ffa = True
    load_obs_data = True
    load_mvnproc = False
    load_expected_params = False
    load_postpred_checks = False
    config = CF(args.version, args.pcensor, args.rho_min,
                awidth, aheight, fdpi, ptype, ari_max,
                excludes, args.diag,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
