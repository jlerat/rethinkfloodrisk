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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot

from pyrethink import datahub

import importlib.util

ffit = Path(__file__).resolve().parent.parent / "fit" / "copulafit.py"
spec = importlib.util.spec_from_file_location("copulafit", ffit)
copulafit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copulafit)


def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    fsum = froot / "outputs" / f"copulaprocess2_v{config.version}"

    basename = source_file.stem
    fimg = froot / "images" / "manuscript" / basename

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fsum", "fimg"])
    script_paths = SP(source_file, basename, froot,
                      fsum, fimg)

    fimg.mkdir(exist_ok=True)
    if config.clean:
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

    prior = config.prior
    copula_spec = config.copula_spec
    exclude = config.exclude
    awra_covariate = config.awra_covariate
    group = config.group
    taskid = opm.find(prior=prior,
                      copula_spec=copula_spec,
                      exclude=exclude,
                      awra_covariate=awra_covariate,
                      group= f"^{group}$")
    taskid = next(t for t in taskid)

    obs_data, _, _, stations = datahub.get_ams_concat()

    fs = script_paths.fsum / f"TASK{taskid}" / \
        f"copulaprocess_sum_samples_TASK{taskid}.csv"
    sum_samples, _ = csv.read_csv(fs)

    fa = script_paths.fsum / f"TASK{taskid}" / \
        f"copulaprocess_sum_ffa_TASK{taskid}.csv"
    sum_ffa, _ = csv.read_csv(fa)

    DT = namedtuple("Data", ["stations", "obs_data",
                             "sum_samples", "sum_ffa",
                             "options"])
    return DT(stations, obs_data,
              sum_samples, sum_ffa,
              opm)


def process(config, script_paths, logger, data):
    group = config.group
    stationids = group.split("-")
    stations = data.stations.loc[stationids]

    options = data.options
    obs_data = data.obs_data
    sum_samples = data.sum_samples

    # FFA threshold
    sum_ffa = data.sum_ffa
    cn = f"{group}_SUM_POSTERIOR_PREDICTIVE"
    idx = sum_ffa.loc[:, "ERI"] == f"DESIGN_ERI{config.ari}"
    Qsum_thresh = sum_ffa.loc[idx, cn].squeeze()

    # Samples above threshold
    cc = [f"{sid}_SAMPLE" for sid in stationids]
    nval = len(sum_samples)
    cst = 0.3
    ppos = (sum_samples.loc[:, cc].rank() - cst) / (nval + 1 - 2 * cst)

    cn = re.sub("POSTERIOR.*", "SAMPLE", cn)
    eps = Qsum_thresh * 1e-1
    idx = sum_samples.loc[:, cn] >= Qsum_thresh - eps
    idx &= sum_samples.loc[:, cn] < Qsum_thresh + eps

    aeps = (1 - ppos.loc[idx]) * 100
    aeps.columns = [re.sub("_.*", "", cn) for cn in aeps.columns]

    # plot
    plt.close("all")
    ncols, nrows = 2, 1
    figsize = (ncols * config.awidth, nrows * config.aheight)
    fig, axs = plt.subplots(ncols=2, figsize=figsize, layout="constrained")

    def draw(ax, aeps, sid1, sid2):
        x = aeps.loc[:,  sid1]
        y = aeps.loc[:,  sid2]
        ax.plot(x, y, "o", alpha=0.5)

        xy = aeps.loc[:, [sid1, sid2]].values
        xx, yy, zz = putils.kde(xy, logx=True, logy=True)
        ax.contourf(xx, yy, zz, cmap="Blues")

        ax.set(xscale="log", yscale="log")
        tk = [0.1, 1, 10]
        ax.set_xticks(tk)
        ax.set_xticklabels([f"{t}%" for t in tk])
        ax.set_yticks(tk)
        ax.set_yticklabels([f"{t}%" for t in tk])

    ax = axs[0]
    draw(ax, aeps, "203014", "203010")

    ax = axs[1]
    draw(ax, aeps, "203024", "203010")

    # save
    fp = f"{script_paths.basename}_v{config.version}.png"
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
    parser.add_argument("-c", "--clean", help="Clean image folder",
                        action="store_true", default=False)
    parser.add_argument("-a", "--use_awra", help="Use copula including awra covariate",
                        action="store_true", default=False)
    parser.add_argument("-i", "--use_informative", help="Use copula including informative prior",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "debug",
                               "awidth", "aheight", "fdpi",
                               "ptype", "ari_max",
                               "freq_plot_type", "clean",
                               "prior", "copula_spec",
                               "exclude", "awra_covariate",
                               "group", "ari"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    freq_plot_type = "gumbel"

    prior = "uninformative"
    copula_spec = "Gaussian"
    exclude = "NONE"
    awra_covariate = True
    group = "203010-203014-203024"

    ari = 100

    config = CF(args.version,  args.debug,
                awidth, aheight, fdpi, ptype, ari_max,
                freq_plot_type, args.clean,
                prior, copula_spec, exclude,
                awra_covariate, group, ari)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
