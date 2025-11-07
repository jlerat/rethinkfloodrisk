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
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA spatial variability",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

awidth = 6
aheight = 5
fdpi = 300

stationid_target = "203002"
eep_target = 1 - 1e-2

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs"

basename = source_file.stem
fimg = froot / "images" / "manuscript" / basename
fimg.mkdir(exist_ok=True, parents=True)
for f in fimg.glob("*.png"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

stations = datahub.get_stations()
if debug:
    stations = stations.iloc[:1]

for ftask in fout.glob("*TASK*"):
    # Setup folders
    taskid = int(re.sub("^.*TASK", "", ftask.stem))
    if taskid <= 0 :
        continue

    # Get data
    fd = ftask / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    if diag["pcensor"] != 0.3 or diag["timeperiod"] != "ALL":
        continue

    period = diag["timeperiod"]

    fd = ftask / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        data = json.load(fo)

    nvar = data["P"]
    stationids = np.array(data["stationids"])

    LOGGER.info(f"Load report TASK {taskid} period={period}")
    fs = ftask / f"copulafit_samples_TASK{taskid}.zip"
    df = pd.read_csv(fs, skiprows=15)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
if debug:
    df = df.iloc[:100]

# Compute probs
i1 = np.where(stationids == stationid_target)[0]
i2 = np.where(stationids != stationid_target)[0]
ztarget = np.atleast_1d(norm.ppf(eep_target))
nsamples = len(df)

cols_mu = [f"{sid}_mu" for sid in stationids[i2]]
cols_sig = [f"{sid}_sig" for sid in stationids[i2]]
cols_smp = [f"{sid}_smp" for sid in stationids[i2]]
eeps = pd.DataFrame(np.nan, index=df.index,
                    columns=cols_mu + cols_sig)

for i, smp in df.iterrows():
    if i % 100 == 0:
        LOGGER.info(f"Processing sample {i + 1} / {nsamples}")

    L_cor = smp.filter(regex="L_cor").values.reshape((nvar, nvar)).T
    cor = L_cor @ L_cor.T

    # Conditional distribution
    S11 = cor[i1][:, i1]
    S11i = np.linalg.inv(S11)
    S22 = cor[i2][:, i2]
    S21 = cor[i2][:, i1]

    muc = S21 @ S11i @ ztarget
    Sc = S22 - S21 @ S11i @ S21.T
    z = np.random.multivariate_normal(mean=muc, cov=Sc)

    eeps.loc[i, cols_mu] = muc
    eeps.loc[i, cols_sig] = np.sqrt(np.diag(Sc))
    eeps.loc[i, cols_smp] = norm.cdf(z)


plt.close("all")
fig, ax = plt.subplots(figsize=(awidth, aheight),
                       layout="constrained")

ee = (1 - eeps.filter(regex="smp", axis=1)) * 100
ee.columns = ee.columns.to_series().str.replace("_.*", "", regex=True)
vm = violinplot.Violin(ee, number_format="0.1f")
vm.draw()

ax.set(xlabel="Station", ylabel="Event Exceedance Probability [%]")

fp = fimg / f"{basename}.png"
fig.savefig(fp)

LOGGER.completed()
