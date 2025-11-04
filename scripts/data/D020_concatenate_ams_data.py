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
from hydrodiy.data import dutils

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
version = datahub.DATA_VERSION

maxwindow = 4

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
    df = df.filter(regex="MAX", axis=1)
    df.columns = [f"{stationid}"]
    q = df.squeeze()
    daily.append(q)

ams = pd.concat(ams, axis=1).sort_index()
daily = pd.concat(daily, axis=1).sort_index()

# Define POT threshold
ams_peaks = ams.filter(regex="_PEAK", axis=1)
ams_peaks.columns = ams_peaks.columns.to_series().str.replace("_PEAK", "")
pot_thresh = ams_peaks.median()

# Fine POT peak time
timepeaks = []
for stationid in ams_peaks.columns:
    qmax = daily.loc[:, stationid]
    qthresh = pot_thresh[stationid]
    above = qmax - qthresh >= 0
    seq = dutils.sequence_true(above)
    tp = [qmax.iloc[i1:i2].idxmax() for i1, i2 in seq]
    timepeaks.extend(tp)


# Build event data base
timepeaks = pd.Series(list(timepeaks)).sort_values()
timepeaks = timepeaks.reset_index(drop=True)
diff = timepeaks.diff().dt.days
new = (diff > maxwindow) | pd.isnull(diff)
peak_index = new.astype(int).cumsum()

potpeaks = []
for peak in peak_index.unique():
    times = timepeaks.loc[peak_index == peak]
    idx = set(times.tolist())
    for shift in range(1, maxwindow):
        offset = pd.DateOffset(days=shift)
        times_min = times - offset
        times_max = times + offset
        idx |= set(times_min.tolist())
        idx |= set(times_max.tolist())

    idx = pd.Series(list(idx)).sort_values().values
    qmaxs = daily.loc[idx].max()
    qmaxs.index = [f"{i}_PEAK" if re.search("[0-9]{6}", i) else i
                   for i in qmaxs.index]
    mid = pd.Interval(times.min(), times.max()).mid
    year = mid.year - 1 if mid.month < 10 else mid.year

    qmaxs["WATERYEAR"] = year
    qmaxs["DAY"] = mid - pd.DateOffset(hours=mid.hour)
    qmaxs["START"] = idx.min()
    qmaxs["END"] = idx.max()

    potpeaks.append(qmaxs)

potpeaks = pd.DataFrame(potpeaks).set_index("DAY")
LOGGER.info(f"{len(potpeaks)} peaks found")

nyears = potpeaks.groupby("WATERYEAR").apply(len, include_groups=False)
nyears.sort_values(inplace=True, ascending=False)
for i in range(6):
    y = nyears.index[i]
    n = nyears.iloc[i]
    LOGGER.info(f"{n} events in year {y:0.0f}")

# Write data
fc = fdata / f"peak_streamflow_concatenated_v{version}.csv"
comments = {"comment": "POT flow data"}
for sid, value in pot_thresh.items():
    comments[f"POT_thresh_{sid}[m3/s]"] = np.round(value, 1)

csv.write_csv(potpeaks, fc, comments,
              source_file,
              write_index=True,
              compress=False,
              write_sys_info=False,
              lineterminator="\n")

LOGGER.completed()

