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

from netCDF4 import Dataset
from hyncu import nc4io
from hyncu import nc4stationdata as nc4sd

from floodstan import annual_maximum_series

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Collect streamflow data",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="Overwrite data",
                    action="store_true", default=False)
args = parser.parse_args()

version = datahub.DATA_VERSION
overwrite = args.overwrite
debug = args.debug

col_obs = "STREAMFLOW_DAILYMAX_9AM[m3.s-1]"

duration_min = 30

dataname_daily = "catchment_data_daily"

water_year_start = 10
end_timeseries = pd.to_datetime(f"2022-{water_year_start}-01")\
    - pd.DateOffset(days=1)

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fnc_path = froot.parent.parent / "Data" / "characterisation_paper"

fdata = froot / "data"
fdata.mkdir(exist_ok=True, parents=True)

for subfold in ["ams", "dailymax"]:
    (fdata / subfold).mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / basename / f"{basename}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
LOGGER = iutils.get_logger(basename, console=True, contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
fncin = fnc_path / f"combined_daily_v{version}.nc"
with Dataset(fncin, "r") as ncin:
    metavar = nc4sd.StationMetaData(ncin)
    stations, _ = metavar.read_metadata_from_dataset()

# Select valid stations in Northern Rivers + Logan
idx = stations.IS_VALID==1
idx &= stations.index.str.contains("^203[0-9]{3}")
stations = stations.loc[idx]

# Compute duration
start = pd.to_datetime(stations.loc[:, "START[day]"])
end = pd.to_datetime(stations.loc[:, "END[day]"])
dur = (end-start).dt.days/365.25
stations.loc[:, "DURATION[yr]"] = dur

selected = dur > duration_min
stations = stations.loc[selected]

if debug:
    pat = "^2030(10|14)"
    idx = stations.index.str.contains(pat, regex=True)
    stations = stations.loc[idx]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
with Dataset(fncin, "r") as ncin:
    svarin = nc4sd.StationVariable(ncin, dataname_daily)

    nstations = len(stations)
    for istation, (stationid, sinfo) in enumerate(stations.iterrows()):
        fams = fdata / "ams" / f"AMS_streamflow_{stationid}_v{version}.csv"
        if fams.exists() and not overwrite:
            continue

        sinfo = stations.loc[stationid]
        LOGGER.context = f"{stationid} ({istation+1}/{nstations})"
        LOGGER.info("Processing")
        daily, _ = svarin.read_data_from_single_station(stationid)

        qobs = daily.loc[:, col_obs]
        start, end = qobs.index[qobs.notnull()][[0, -1]]
        qobs = qobs[start:end]

        wys = water_year_start
        ams = annual_maximum_series.compute_ams(qobs,
                                                water_year_start=wys)

        drop = ["EVENTID", "NVALYEAR", "WATER_YEAR", "WATER_YEAR_END"]
        ams = ams.loc[ams.NVALYEAR >= 365]\
            .drop(drop, axis=1)\
            .set_index("WATER_YEAR_START")

        ams.columns = stationid + "_" + ams.columns

        comment = f"AMS streamflow data for station {stationid}."\
                  + " Water year starts on the 1/{wys}."
        csv.write_csv(ams, fams, comment,
                      source_file, compress=False,
                      write_sys_info=False,
                      write_index=True,
                      lineterminator="\n")

        qobs = pd.DataFrame(qobs)
        qobs.index.name = "DAY"
        comment = f"Daily streamflow maximum for station {stationid}."
        fq = fdata / "dailymax" / f"dailymax_streamflow_{stationid}_v{version}.csv"
        csv.write_csv(qobs, fq, comment,
                      source_file, compress=False,
                      write_sys_info=False,
                      write_index=True,
                      lineterminator="\n")

        stations.loc[stationid, "DURATION[yr]"] = len(ams)

pat = "NAME|LONGITUDE\\[|LATITUDE\\[|CATCHMENTAREA\\["\
      + "|XOUT|YOUT|^STREAMFLOW_MAX|G_MAX_PROV"
stations = stations.filter(regex=pat, axis=1)
cc = ["NAME"] + [cn for cn in stations.columns.sort_values() if cn != "NAME"]

fs = fdata / f"AMS_stations_v{version}.csv"
stations = stations.loc[:, cc]
stations.index.name = "STATIONID"
csv.write_csv(stations, fs, "Station metadata.",
              source_file, compress=False,
              write_index=True, write_sys_info=False,
              lineterminator="\n")

LOGGER.completed()

