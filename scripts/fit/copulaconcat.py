#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-06-26 10:26:52.387430
## Comment : Fit copula to data
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
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

from pyrethink import report
from pyrethink import postpredchecks as ppc

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"copulafit_v{config.version}"
    fout = froot / "outputs" / f"copulaprocess_v{config.version}"

    flogs = froot / "logs" / source_file.stem

    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs)

    flogs.mkdir(exist_ok=True, parents=True)
    fout.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}_TASK{config.taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.info("", nret=1)
    return logger


def process(config, script_paths, logger):
    logger.info(f"Start processing", nret=1)

    ftypes = [
        "ffa",
        "postpredcheck_univ",
        "postpredcheck_biv",
        "postpredcheck_multivar"
        ]

    for ftype in ftypes:
        logger.info(f"Concatenating {ftype} files")
        df = []
        for f in script_paths.fout.glob(f"*/*{ftype}*.zip"):
            # Get data
            taskid = re.sub(".*_TASK", "", f.stem)
            fd = script_paths.fdata / f"TASK{taskid}" / f"copulafit_data_TASK{taskid}.json"
            with fd.open("r") as fo:
                data = json.load(fo)

            stationids = data["task_group"].split("-")

            # Get results
            #ddf, _ = csv.read_csv(f)
            ddf = pd.read_csv(f, skiprows=15)
            ddf.columns = ["VARIABLE"] + ddf.columns[1:].to_list()
            ddf.loc[:, "TASKID"] = taskid

            # Set station ID
            ddf.loc[:, "STATIONNB"] = ddf.VARIABLE.str[-2]
            ddf.loc[:, "STATIONID"] = "NA"
            for istation, stationid in enumerate(stationids):
                idx = ddf.VARIABLE.str.contains(f"\[{istation + 1}\]$", regex=True)
                ddf.loc[idx, "STATIONID"] = stationid

            ddf.loc[:, "VARIABLE"] = ddf.VARIABLE.str.replace("\[.*", "",
                                                              regex=True)

            df.append(ddf)

        df = pd.concat(df)
        fr = script_paths.fout / f"copulaconcat_{ftype}.csv"
        csv.write_csv(df, fr, f"Concatenation of {ftype} results",
                      script_paths.source_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit copula to data",
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
                        ["version", "taskid", "overwrite",
                         "debug"])
    config = Config(version, taskid, overwrite,
                    debug)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Process
    process(config, script_paths, logger)

    logger.completed()
