#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-02-03 16:09:06.542813
## Comment : Check mcmc samples
##
## ------------------------------


import sys
import os
import re
import json
import math
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fit copula model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--version", help="version",
                    type=str, required=True)
args = parser.parse_args()
version = args.version

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
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
fopm = fout / "copulafit_options.json"
opm = hyruns.OptionManager.from_file(fopm)

diags = []
for f in fout.glob("*/*diagnostic*"):
    with f.open("r") as fo:
        d = json.load(fo)

    taskid = int(re.sub(".*TASK", "", f.stem))
    task = opm.get_task(taskid)
    d["taskid"] = taskid
    d.update({f"opt_{k}": v for k, v in task.to_dict()["options"].items()})

    allgood = all(d[k] == "satisfactory" for k in ["rhat", "effsamplesz"])
    d["allgood"] = allgood
    diags.append(d)

diags = pd.DataFrame(diags)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
iall = diags.opt_exclude == "NONE"
stats = pd.pivot_table(diags.loc[iall], index=["opt_pcensor", "opt_has_clusters"],
                       columns=["opt_copula", "opt_rho_min"],
                       values="allgood", aggfunc="sum")

icop4 = diags.opt_copula == 4

LOGGER.completed()

