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
timepeaks = set()
offset = pd.DateOffset(days=maxwindow)

for f in (fdata / "ams").glob(f"*_v{version}.csv"):
    df, _ = csv.read_csv(f, index_col="WATER_YEAR_START")
    wateryear_start = pd.to_datetime(df.index[0]).month

    stationid = re.sub(".*_streamflow_|_v.*", "", f.stem)
    df.drop(f"{stationid}_NVALID", axis=1)
    df.index = df.index.str.replace("-.*", "", regex=True).astype(int)
    df.index.name = "WATERYEAR"
    ams.append(df)

    tp = pd.to_datetime(df.filter(regex="TIMEPEAK").squeeze())
    iok = pd.notnull(tp)
    timepeaks |= set(tp[iok].tolist())

    f = f"dailymax_streamflow_{stationid}_v{version}.csv"
    f = fdata / "dailymax" / f
    df, _ = csv.read_csv(f, index_col="DAY", parse_dates=True)
    df = df.filter(regex="MAX", axis=1)
    df.columns = [f"{stationid}"]
    q = df.squeeze()
    daily.append(q)

ams = pd.concat(ams, axis=1).sort_index()
daily = pd.concat(daily, axis=1).sort_index()

# Build event data base
timepeaks = pd.Series(list(timepeaks)).sort_values()
timepeaks = timepeaks.reset_index(drop=True)
diff = timepeaks.diff().dt.days
new = (diff > maxwindow) | pd.isnull(diff)
peak_index = new.astype(int).cumsum()

truepeaks = []
ams_peaks = ams.filter(regex="_PEAK", axis=1)
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

    truepeaks.append(qmaxs)

truepeaks = pd.DataFrame(truepeaks).set_index("DAY")
LOGGER.info(f"{len(truepeaks)} peaks found")

nyears = truepeaks.groupby("WATERYEAR").apply(len, include_groups=False)
nyears.sort_values(inplace=True, ascending=False)
for i in range(6):
    y = nyears.index[i]
    n = nyears.iloc[i]
    LOGGER.info(f"{n} events in year {y:0.0f}")

# Check all ams are accounted for
tp = truepeaks.filter(regex="_PEAK|WATERYEAR", axis=1)
am = tp.groupby("WATERYEAR").max()
assert (am.notnull().sum() >= 28).all()
diff = np.abs(am - ams_peaks)
assert diff.max().max() == 0

# Write data
fc = fdata / f"peak_streamflow_concatenated_v{version}.csv"
csv.write_csv(truepeaks, fc, "Peak flow data",
              source_file,
              write_index=True,
              compress=False,
              write_sys_info=False,
              lineterminator="\n")

LOGGER.completed()

