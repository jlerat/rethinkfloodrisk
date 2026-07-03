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

from copulafit import get_options, get_stationids, get_data

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"copulafit_v{config.version}" / f"TASK{config.taskid}"
    fout = froot / "outputs" / f"{source_file.stem}_v{config.version}" / f"TASK{config.taskid}"

    flogs = froot / "logs" / source_file.stem

    fpriors = froot / "outputs" / f"priorfit_v{config.version_priors}"
    if not fpriors.exists():
        errmsg = "Prior data folder does not exist"
        raise ValueError(errmsg)


    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs",
                              "fpriors"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs, fpriors)

    flogs.mkdir(exist_ok=True, parents=True)
    fout.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}_TASK{config.taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.info("", nret=1)
    config.task.log(logger)
    return logger


def process(config, script_paths, logger, data):
    logger.info(f"Start processing", nret=1)
    taskid = config.taskid
    copula_spec = config.task.copula_spec

    fp = script_paths.fdata / f"copulafit_samples_TASK{taskid}.csv"
    fz = fp.parent / f"{fp.stem}.zip"
    samples = pd.read_csv(fz, skiprows=15)

    if config.debug:
        samples = samples.iloc[:200]

    # Observed FFA (assumes GEV marginal)
    if config.task.copula_spec == "Univariate":
        # Rename columns to match ffa_report conventions
        samples.columns = [f"{cn}[1]" if re.search("^y", cn) else cn
              for cn in samples.columns]

    stat, df = report.ffa_report(samples,
                                 design_eris=config.design_eris,
                                 logger=logger)
    fout = script_paths.fout
    source_file = script_paths.source_file
    fs = fout / f"copulaprocess_ffa_TASK{taskid}.csv"
    csv.write_csv(stat, fs, "FFA for task {taskid}",
                  source_file, write_index=True)

    # Post pred checks for copula models
    if config.task.copula_spec == "Univariate":
        return

    fdd = script_paths.fdata / f"copulafit_data_TASK{taskid}.json"
    with fdd.open() as fd:
        cdata = json.load(fd)
    yobs = np.array(cdata["y"])
    ppu, ppb, ppm, pdata = ppc.posterior_predictive_checks(yobs, samples,
                                                          copula_spec,
                                                          logger=logger)
    fp = fout / f"copulaprocess_postpredcheck_univ_TASK{taskid}.csv"
    csv.write_csv(ppu, fp, "Checks for task {taskid}",
                  source_file, write_index=True)

    fp = fout / f"copulaprocess_postpredcheck_biv_TASK{taskid}.csv"
    csv.write_csv(ppb, fp, "Checks for task {taskid}",
                  source_file, write_index=True)

    fp = fout / f"copulaprocess_postpredcheck_multivar_TASK{taskid}.csv"
    csv.write_csv(ppm, fp, "Checks for task {taskid}",
                  source_file, write_index=True)


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
    version_priors = 1
    awraid = "WILSONSRIVER"

    design_eris = [1.1, 1.2, 1.4, 1.6, 1.8,
                   2, 5, 10, 20, 50, 70, 100, 150,
                   200, 300, 500, 700, 1000]

    # .. options
    opm = get_options(version, version_priors)
    if debug:
        ctype = "cop"
        if ctype == "univ":
            taskid = opm.search(copula_spec="Univariate",
                                exclude="NONE",
                                prior="^informative",
                                group="203014")
        else:
            taskid = opm.search(copula_spec="GaussianFactor_0_2$",
                                exclude="NONE",
                                prior="^informative",
                                group=opm.options["group"][1])

        taskid = taskid[0]

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "task", "version_priors",
                         "design_eris", "awraid"])
    config = Config(version, taskid, overwrite,
                    debug, opm.get_task(taskid),
                    version_priors, design_eris, awraid)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)
    logger.info(f"Number of tasks : {opm.ntasks}", nret=1)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
