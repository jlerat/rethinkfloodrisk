#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-21 13:08:16.079895
## Comment : Compute sensitivity of Q100 to max obs
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyrethink import datahub


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / "sensitivity"
    fimg = froot / "images" / "sensitivity"
    fimg.mkdir(exist_ok=True)
    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fimg"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fimg)
    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    fs = script_paths.fdata / "sensitivity.csv"
    sens, _ = csv.read_csv(fs, index_col="STATIONID")
    data = namedtuple("Data", ["sensitivity"])(sens)
    return data


def process(config, script_paths, logger, data):
    sensitivity = data.sensitivity
    nsensitivity = len(sensitivity)
    logger.info(f"Start processing", nret=1)

    cna = "LENGTH_AMS[yr]"

    #etas = [re.search("(?<=_ETA)\\d", cn) for cn in sensitivity.columns]
    #etas = set([int(e.group()) for e in etas if e is not None])
    etas = [2]

    aris = [re.search("(?<=^Q)\\d+", cn) for cn in sensitivity.columns]
    aris = list(set([int(a.group()) for a in aris if a is not None]))
    aris.sort()

    plt.close("all")
    ncols = config.ncols
    cfg = [f"{a}_{e}_{o}" for a in aris for e in etas for o in [1, 2]]
    ncfg = len(cfg)
    nrows = ncfg // ncols + (1 if ncfg % ncols != 0 else 0)
    mosaic = [[cfg[ir * ncols + ic] if ir * ncols + ic < ncfg else "."
               for ic in range(ncols)] for ir in range(nrows)]

    aw, ah = 6, 6
    fig = plt.figure(figsize=(aw * ncols, ah * nrows),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)

    tforward = lambda x, nu: nu * np.asinh(x / nu)
    tbackward = lambda y, nu: nu * np.sinh(y / nu)
    nu = 1e-3

    x = sensitivity.loc[:, cna]
    X = np.column_stack([np.ones_like(x), tforward(x, nu)])
    iok = ~np.isnan(X[:, 1])

    tforward = lambda x, nu: nu * np.asinh(x / nu)
    for aname, ax in axs.items():
        ari, eta, order = aname.split("_")
        logger.info(f"Plotting ari={ari} eta={eta} order={order}")

        # Scatter plot
        pos = {}
        for method in ["DIFF", "OLS"]:
            cns = f"Q{ari}_SENSITIVITY{order}_ETA{eta}_{method}[%/%dQ]"
            y = sensitivity.loc[:, cns]
            pos[method] = (y > 0).sum() / len(y)
            ax.plot(x, y, "o", label=method, alpha=0.5)
            col = ax.get_lines()[-1].get_color()

            ty = tforward(y, nu)
            theta, _, _, _ = np.linalg.lstsq(X[iok], ty[iok], rcond=1e-6)
            xx = np.linspace(x.min(), x.max(), 100)

            a = theta[0]
            b = theta[1]
            yy = tbackward(a + b * tforward(xx, nu), nu)
            lab = f"{method}: y = t({a:+3.1e} {b:+3.1e}x)"
            ax.plot(xx, yy, "--", label=lab, color=col)

        title = f"Q{ari} sensitivity versus record duration\n" \
                + f"Order={order} η={eta}"
        xlab = "Record duration [yr]"
        ylab = f"Sensitivity Q{ari} [%/%dQ]"
        ax.set(title=title, xlabel=xlab, ylabel=ylab)
        if order == "2":
            ax.set_yscale("asinh", linear_width=1e-3)
            putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)

            txt = "\n".join([f"%pos({m}) = {p * 100:0.1f}%"
                             for m, p in pos.items()])
            ax.text(0.98, 0.98, txt,
                    transform=ax.transAxes, va="top",
                    ha="right", fontweight="bold")
            ax.set(ylim=[-1e-1, 1e-1])

        ax.legend()

    fp = script_paths.fimg / "sensitivity.png"
    fig.savefig(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sensitivity of Q100 to max obs",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    debug = args.debug
    eta = 2
    ncols = 2

    Config = namedtuple("Config", ["debug", "lh_moment_eta", "ncols"])
    config = Config(debug, eta, ncols)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
