#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-06-20 17:36:03.746063
## Comment : Compute regional priors for parameters
##
## ------------------------------


import sys
import os
import re
import json
import math
import shutil
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import sutils

from floodstan import gls, sample, report
from floodstan import gls_spatial_sampling

from priorfit import get_script_paths

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"

    version = config.version
    fout = froot / "outputs" / f"priorfit_v{config.version}"
    flogs = froot / "logs" / "priorconcat"

    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs)
    flogs.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    taskid = config.taskid
    fl = script_paths.flogs / f"priorconcat_TASK{taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def process(config, script_paths, logger):
    # Concatenate
    lf = list(script_paths.fout.glob("*.csv"))

    logger.info(f"Found {len(lf)} files")
    res = []
    for f in lf:
        if f.stem == "priors":
            logger.info("Skipping priors.csv")
            continue
        df, _ = csv.read_csv(f)
        res.append(df)

    res = pd.concat(res)

    # To disk
    fp = script_paths.fout / f"priors.csv"
    csv.write_csv(res, fp, "Concatenated prior data",
                  script_paths.source_file,
                  compress=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate result files",
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
    args = parser.parse_args()

    # Config
    version = args.version
    taskid = args.taskid
    overwrite = args.overwrite
    debug = args.debug
    Config = namedtuple("Config",
                        ["version", "taskid",
                        "overwrite", "debug"])
    config = Config(version, taskid, overwrite,
                    debug)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Process
    process(config, script_paths, logger)

    logger.completed()
