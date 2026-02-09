#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
import re
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

from hydrodiy.io import csv, iutils, hyruns

from pyrethink import datahub

np.random.seed(5446)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process mvn samples",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--version", help="version",
                    type=int, required=True)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=0)
args = parser.parse_args()
version = args.version
taskid = args.taskid

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / f"copulafit_v{version}"

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / basename / f"{basename}_TASK{taskid}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
LOGGER = iutils.get_logger(basename, flog=flog)
LOGGER.log_dict(vars(args), "Command line arguments")

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
for ftask in fout.glob("*TASK*"):
    lf = list((ftask / "mvnprocess").glob("*.zip"))
    if len(lf) == 0:
        continue

    taskid = int(re.sub("^.*TASK", "", ftask.stem))
    LOGGER.info(f"Dealing with task {taskid}")

    concat = []
    for f in lf:
        try:
            df, comments = csv.read_csv(f)
        except Exception as err:
            LOGGER.warning("Issue with zipfile opening. Switching to os.")
            ftmp = f.parent / "tmp"
            ftmp.mkdir(exist_ok=True)
            fcsv = ftmp / f"{f.stem}.csv"
            if fcsv.exists():
                fcsv.unlink()
            cmd = f"unzip {f} -d {ftmp}"

            subprocess.run(cmd, shell=True, check=True)
            df, comments = csv.read_csv(fcsv)
            fcsv.unlink()
            ftmp.rmdir()

        concat.append(df)

    cc = ["comment", "period", "pcensor", "fit_taskid",
          "stationid_cond", "aep_target", "zcdf"]
    comments = {k: v for k, v in comments.items() if k in cc}

    concat = pd.concat(concat)
    fn = re.sub("_BATCH.*", "", f.stem)
    fc = ftask / f"{fn}.csv"

    # Keep the comments from latest batch file
    csv.write_csv(concat, fc, comments,
                  source_file)

LOGGER.completed()
