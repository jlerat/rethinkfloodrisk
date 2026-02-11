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
diags = []
pp_biv = []
for f in fout.glob("*/*diagnostic*"):
    with f.open("r") as fo:
        d = json.load(fo)
    allgood = all(d[k] == "satisfactory" for k in ["rhat", "effsamplesz"])
    d["stan_allgood"] = allgood

    # Post pred checks

    # .. univariate
    funiv = f.parent / f"postprocess_postpredchecks_univ_TASK{d['taskid']}.csv"
    ppu, _ = csv.read_csv(funiv, index_col=0)
    pvals = ppu.filter(regex="pvalue\\[", axis=1)
    check = ((pvals > 0.05) & (pvals < 0.95)).all().all()
    d["ppu_allgood"] = check

    # .. bivariate
    fbiv = f.parent / f"postprocess_postpredchecks_biv_TASK{d['taskid']}.csv"
    ppb = pd.read_csv(fbiv, skiprows=15, index_col=0)
    pvals = ppb.filter(regex="pvalue\\[", axis=1)\
            .filter(regex="_high|_q90", axis=0)
    eps = 0.05
    check = ((pvals > eps) & (pvals < 1 - eps)).all().all()
    d["ppb_allgood"] = check
    diags.append(d)

    #cn = "taildep_q75"
    cn = "taildep_q90"
    dd = ppb.loc[cn].filter(regex="pvalue\\[").to_dict()
    dd.update({k: v for k, v in d.items() if k.startswith("task_")})
    pp_biv.append(dd)

diags = pd.DataFrame(diags)
pp_biv = pd.DataFrame(pp_biv)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
idx = pp_biv.task_pcensor == 0.3
idx &= pp_biv.task_exclude == "NONE"

ccp = pp_biv.columns.to_series()\
        .filter(regex="pvalue").tolist()
ccv = pp_biv.columns.to_series()\
        .filter(regex="clusters|copula").tolist()
me = pd.melt(pp_biv.loc[idx, ccp + ccv], id_vars=ccv)
me.loc[:, "value"] = np.minimum(me.value, 1 - me.value)
pvv = pd.pivot_table(me.drop("variable", axis=1),
                     index="task_has_clusters",
                     columns="task_copula",
                     aggfunc="mean")


iall = diags.task_exclude == "NONE"
for value in ["stan_allgood", "ppu_allgood", "ppb_allgood",
              "ppb_dev_from_0-1"]:
    if not value.startswith("ppb_dev"):
        stats = pd.pivot_table(diags.loc[iall], index=["task_pcensor", "task_has_clusters"],
                               columns=["task_copula", "task_rho_min"],
                               values=value, aggfunc="sum")
    else:
        stats = pvv.round(2)

    print("")
    print("-" * 80)
    print(f"Stats {value}\n")
    print(stats)
    print("")

LOGGER.completed()

