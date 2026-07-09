#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-09 Thu 09:43 AM
## Comment : Check posterior predictive distribution
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import namedtuple

#import warnings
#warnings.filterwarnings("error")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils, hyruns

from floodstan import freqplots

from copulafit import get_options, get_stationids

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata0 = froot / "outputs" / f"copulafit_v{config.version}" / f"TASK{config.taskid}"
    fdata1 = froot / "outputs" / f"copulaprocess1_v{config.version}" / f"TASK{config.taskid}"
    fdata2 = froot / "outputs" / f"copulaprocess2_v{config.version}" / f"TASK{config.taskid}"
    fout = froot / "outputs" / f"{source_file.stem}"

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata0", "fdata1", "fdata2",
                              "fout"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata0, fdata1, fdata2, fout)

    fout.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    fd = script_paths.fdata0 / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        stan_data = json.load(fo)
    stationids = stan_data["stationids"]

    fa = script_paths.fdata1 / f"copulaprocess_ffa_TASK{taskid}.csv"
    fz = fa.parent / f"{fa.stem}.zip"
    ffa = pd.read_csv(fz, skiprows=15)
    ffa.columns = ["VARIABLE"] + ffa.columns[1:].to_list()
    ffa.loc[:, "STATIONID"] = ""
    for istation, stationid in enumerate(stationids):
        idx = ffa.VARIABLE.str.contains(f"\\[{istation + 1}\\]$", regex=True)
        ffa.loc[idx, "STATIONID"] = stationid

    fs = script_paths.fdata2 / f"copulaprocess_sum_ffa_TASK{taskid}.csv"
    fz = fs.parent / f"{fs.stem}.zip"
    sum_ffa = pd.read_csv(fz, skiprows=15)

    Data = namedtuple("Data", ["ffa", "sum_ffa", "stationids"])
    data = Data(ffa, sum_ffa, stationids)
    return data


def process(config, script_paths, logger, data):
    logger.info(f"Start processing", nret=1)
    taskid = config.taskid

    ffa = data.ffa
    sum_ffa = data.sum_ffa
    stationids = data.stationids
    nsta = len(stationids)

    plt.close("all")
    ncols = 3
    nrows = nsta // ncols + (0 if nsta % ncols == 0 else 1)
    mosaic = [["." if ir * ncols + ic >= nsta else stationids[ir * ncols + ic]
               for ic in range(ncols)] for ir in range(nrows)]
    aw, ah = 6, 4
    fig = plt.figure(figsize=(aw * ncols, ah * nrows),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)
    ptype = "gumbel"

    for stationid, ax in axs.items():
        ista = stationids.index(stationid)

        # Univariate FFA
        idx = ffa.VARIABLE.str.contains(f"DESIGN_.*\\[{ista + 1}\\]$", regex=True)
        aris = ffa.VARIABLE.loc[idx].str.replace(".*_ERI|\\[.*", "", regex=True)
        aris = aris.values.astype(float)
        cn = "POSTERIOR_PREDICTIVE"
        quantiles1 = ffa.loc[idx, [cn]]
        freqplots.plot_marginal_quantiles(ax, aris, quantiles1, ptype,
                                          center_column=cn,
                                          label="FFA", color="tab:blue")

        # Sampling
        aris = sum_ffa.ERI.str.replace(".*_ERI|\\[.*", "", regex=True)
        aris = aris.values.astype(float)
        cn = f"{stationid}_POSTERIOR_PREDICTIVE"
        quantiles2 = sum_ffa.loc[:, [cn]]
        x = freqplots.plot_marginal_quantiles(ax, aris, quantiles2, ptype,
                                              center_column=cn,
                                              label="Sampling", color="tab:red")

        title = f"{stationid}"
        y0 = quantiles2.loc[aris == 2].iloc[0, 0] * 0.9
        y1 = quantiles2.loc[aris == 200].iloc[0, 0] * 1.1

        ax.set(title=title,
               xlim=(1, 5.5),
               ylim=(y0, y1),
               ylabel="Streamflow [m3.s-1]")

        retp = [2, 5, 10, 100, 200]
        aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

        freqplots.xaxis_label(ax, ptype)
        ax.legend(loc=2)

        tax = ax.twinx()
        tax.plot(x, quantiles2.values - quantiles1.values,
                 "-", color="0.5", lw=0.8)
        y0, y1 = tax.get_ylim()
        y1 = max(y1, abs(y0))
        tax.set(ylim=(-y1, y1), ylabel="difference [m3.s-1]")

    ftitle = f"{config.copula_spec} - {config.prior} prior (task {taskid})"
    fig.suptitle(ftitle, fontweight="bold", fontsize="x-large")

    logger.info("saving image")
    fp = script_paths.fout / f"postpred_compare_TASK{taskid}.png"
    fig.savefig(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit copula to data",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    args = parser.parse_args()

    # Config
    version = args.version
    version_priors = 2

    copula_spec = "GaussianFactor_0_2"
    prior = "uninformative"

    opm = get_options(version, version_priors)

    #group = "203010-203014-203024"
    group = next(g for g in opm.options["group"] if len(re.split("-", g)) == 8)

    # .. options
    taskid = opm.search(copula_spec=f"^{copula_spec}$",
                        exclude="NONE",
                        prior=f"^{prior}$",
                        group=f"^{group}$")
    taskid = next(t for t in taskid)

    Config = namedtuple("Config",
                        ["version", "taskid",
                         "copula_spec", "prior"])
    config = Config(version, taskid, copula_spec, prior)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
