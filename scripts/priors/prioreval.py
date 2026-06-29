#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-06-23 18:20:43.944643
## Comment : Evaluation of prior data
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from string import ascii_lowercase as letters

import matplotlib as mpl

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

# Select backend
mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"priorfit_v{config.version}"
    fimg = froot / "images" / source_file.stem

    if config.debug:
        fimg = flogs / source_file.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fimg"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fimg)
    fimg.mkdir(exist_ok=True, parents=True)
    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)

    return logger


def get_data(config, script_paths, logger):
    fp = script_paths.fdata / "priors.csv"
    priors, _ = csv.read_csv(fp)

    data = namedtuple("Data", ["priors"])(priors)

    return data


def process(config, script_paths, logger, data):
    fimg = script_paths.fimg
    priors = data.priors

    logger.info(f"Start plotting", nret=1)

    plt.close("all")

    # Create mosaic with named axes
    mosaic = [priors.PARAMETER.unique().tolist()]
    fnrows = len(mosaic)
    fncols = len(mosaic[0])

    # Create figure
    awidth, aheight = config.awidth, config.aheight
    figsize = (awidth*fncols, aheight*fnrows)
    fig = plt.figure(constrained_layout=True,
                     figsize=figsize)
    axs = fig.subplot_mosaic(mosaic)

    for aname, ax in axs.items():
        param = aname
        df = priors.loc[priors.PARAMETER == param]

        df.loc[:, "ERROR"] = np.abs(df.PREDICTAND - df.PRIOR_MEAN)
        df = pd.pivot_table(df, index="STATIONID",
                            columns="PREDICTORS",
                            values="ERROR")
        df.loc["MEAN", :] = df.mean()

        df.columns = [re.sub("_", " ",
                             re.sub("/", "\n",
                                    re.sub("\[[^\[]+\]|_VALID", "", cn)))
                      for cn in df.columns]
        cc = [cn for cn in df.columns if cn != "INTERCEPT"]
        df = df.loc[:, ["INTERCEPT"] + cc]

        df.plot(ax=ax, kind="bar")
        ax.set(title=param)

    # Save file
    fp = fimg / f"priors.{config.imgext}"
    fig.savefig(fp, dpi=config.fdpi,
                transparent=config.ftransparent)
    #putils.blackwhite(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of prior data",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    parser.add_argument("-t", "--taskid", help="JobID",
                        type=int, default=-1)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-o", "--overwrite", help="Overwrite data",
                        action="store_true", default=False)
    parser.add_argument("-s", "--sitepattern", help="Site selection pattern",
                        type=str, default=".*")
    args = parser.parse_args()

    # Config
    version = args.version
    debug = args.debug
    imgext = "png"
    fdpi = 120
    awidth = 6
    aheight = 6
    ftransparent = False
    create_folders = True
    clean_folders_extension = imgext

    Config = namedtuple("Config",
                        ["version", "debug", "imgext",
                         "fdpi",
                         "awidth", "aheight", "ftransparent",
                         "create_folders",
                         "clean_folders_extension"])
    config = Config(version, debug, imgext,
                    fdpi,
                    awidth, aheight, ftransparent,
                    create_folders,
                    clean_folders_extension)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
