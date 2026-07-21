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
    fs = script_paths.fdata / "sensitivity_Q100.csv"
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
    aris = set([int(a.group()) for a in aris if a is not None])

    plt.close("all")
    cfg = [f"{a}_{e}_{o}" for a in aris for e in etas for o in [1, 2]]
    mosaic = [[f"hist_{c}", f"scatter_{c}"] for c in cfg]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    aw, ah = 8, 7
    fig = plt.figure(figsize=(aw * ncols, ah * nrows),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)

    x = sensitivity.loc[:, cna]
    X = np.column_stack([np.ones_like(x), np.log(x)])

    for c in cfg:
        ax_h = axs[f"hist_{c}"]
        ax_s = axs[f"scatter_{c}"]

        ari, eta, order = c.split("_")
        logger.info(f"Plotting ari={ari} eta={eta} order={order}")
        cns = f"Q{ari}_SENSITIVITY{order}_ETA{eta}[%/%dQ]"
        sens = sensitivity.loc[:, cns].copy()
        if order == "1":
            smin, smax = 0, 2
        else:
            smin, smax = -1e1, 1e1
        sens = sens.clip(smin, smax)

        sens.plot(ax=ax_h, kind="hist", bins=20,
                  ec="0.5", fc="lightblue",
                  density=True,
                  orientation="horizontal")
        title = f"Distribution of Q{ari} sensitivity - η={eta}"
        ylab = f"Sensitivity Q{ari} / ΔQmax [%/%]"
        xlab = "Density [-]"
        x0, x1 = ax_h.get_xlim()
        ax_h.set(title=title, xlabel=xlab, ylabel=ylab, xlim=[x1, x0])
        if order == "2":
            ax_h.set_yscale("asinh")

        # Scatter plot
        y = sensitivity.loc[:, cns].clip(0, smax)
        ax_s.plot(x, y, "o")

        iok = sensitivity.loc[:, cns] > smin
        iok &= sensitivity.loc[:, cns] < smax
        theta, _, _, _ = np.linalg.lstsq(X[iok], np.log(y[iok]), rcond=1e-6)
        xx = np.linspace(x.min(), x.max(), 100)

        a = math.exp(theta[0])
        b = theta[1]
        yy = a * xx**b
        lab = f"y = {a:+0.1f} x$^{{{b:+0.2f}}}$"
        ax_s.plot(xx, yy, "k--", label=lab)

        title = f"Q{ari} sensitivity versus record duration - η={eta}"
        xlab = "Record duration [yr]"
        ax_s.set(title=title, xlabel=xlab, ylabel="")
        if order == "2":
            ax_s.set_yscale("asinh")
        ax_s.sharey(ax_h)
        ax_s.legend()

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

    Config = namedtuple("Config", ["debug", "lh_moment_eta"])
    config = Config(debug, eta)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
