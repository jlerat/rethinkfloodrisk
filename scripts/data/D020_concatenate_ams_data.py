#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-20 16:27:36.472013
## Comment : Collect streamflow data
##
## ------------------------------

import sys
import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
version = datahub.DATA_VERSION

maxwindow = 31

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata = froot / "data"

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ams = []
daily = []
dailymaxs = []
for f in (fdata / "ams").glob(f"*_v{version}.csv"):
    df, _ = csv.read_csv(f, index_col="WATER_YEAR_START")
    wateryear_start = pd.to_datetime(df.index[0]).month

    stationid = re.sub(".*_streamflow_|_v.*", "", f.stem)
    df.drop(f"{stationid}_NVALID", axis=1)
    df.index = df.index.str.replace("-.*", "", regex=True).astype(int)
    df.index.name = "WATERYEAR"
    ams.append(df)

    f = f"dailymax_streamflow_{stationid}_v{version}.csv"
    f = fdata / "dailymax" / f
    df, _ = csv.read_csv(f, index_col="DAY", parse_dates=True)
    df.columns = [f"{stationid}"]
    q = df.squeeze()
    daily.append(q)

    # Rolling window to avoid peak time differences
    qm = q.fillna(-1).rolling(maxwindow, center=True).max()
    qm.name = stationid
    dailymaxs.append(qm)

ams = pd.concat(ams, axis=1).sort_index()
daily = pd.concat(daily, axis=1).sort_index()
dailymaxs = pd.concat(dailymaxs, axis=1).sort_index()

def process(x):
    x = x[x.notnull()]
    x.index = x.index.str.replace("_.*", "", regex=True)
    return x

truepeaks = []
for year, values in ams.iterrows():
    ams_peaks = process(values.filter(regex="_PEAK"))

    start = pd.to_datetime(f"{year}-{wateryear_start}-01")
    end = start + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    qd = daily.loc[start:end, ams_peaks.index]
    diff = np.abs(qd - ams_peaks)
    times = qd.index[(diff < 1e-10).any(axis=1)]

    qmaxs = dailymaxs.loc[times].copy().drop_duplicates()
    qmaxs["WATERYEAR"] = year
    truepeaks.append(qmaxs)

truepeaks = pd.concat(truepeaks)

# Write data
fc = fdata / f"peak_streamflow_concatenated_v{version}.csv"
csv.write_csv(truepeaks, fc, "Peak flow data",
              source_file,
              write_index=True,
              compress=False,
              write_sys_info=False,
              lineterminator="\n")

LOGGER.completed()

