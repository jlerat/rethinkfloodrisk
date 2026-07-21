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
import zipfile

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils
from hydrodiy.data import qualitycontrol

from netCDF4 import Dataset, num2date

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

col_max = "STREAMFLOW_DAILYMAX_9AM[m3.s-1]"

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

fdata = froot / "data" / "ams"
fdata.mkdir(exist_ok=True, parents=True)

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
# Select valid stations in Northern Rivers + Logan
fs = fdata.parent / f"AMS_stations_v{version}.csv"
stations, _ = csv.read_csv(fs, index_col="STATIONID")

if debug:
    pat = "^2030(10|14)"
    idx = stations.index.str.contains(pat, regex=True)
    stations = stations.loc[idx]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

fncin = fnc_path / f"combined_daily_v{version}.nc"
fz = fdata / "ams_all_stations.zip"

with Dataset(fncin, "r") as ncin, \
        zipfile.ZipFile(fz, "w", compression=zipfile.ZIP_DEFLATED) as archive:

    stationids = list(ncin["station_id"][:])
    cols = list(ncin["catchment_data_daily.column_name.numerical"][:])
    icol = cols.index("STREAMFLOW_DAILYMAX_9AM")
    time = ncin["station_data_index"]
    time = pd.to_datetime(num2date(time[:], time.units,
                                   only_use_cftime_datetimes=False))
    data = ncin["catchment_data_daily.numerical"]

    nstations = len(stations)
    for istation, (stationid, sinfo) in enumerate(stations.iterrows()):
        sinfo = stations.loc[stationid]
        LOGGER.context = f"{stationid} ({istation+1}/{nstations})"
        LOGGER.info("Processing")

        ista = stationids.index(stationid)
        qmax = pd.Series(data[ista, :, icol], index=time)

        i0, i1 = qmax.index[qmax.notnull()][[0, -1]]
        qmax = qmax.loc[i0: i1]

        # qaqc
        values = np.ascontiguousarray(qmax.values).astype(np.float64)
        islin = qualitycontrol.islinear(values)
        qmax[islin > 0] = np.nan

        # ams
        wys = water_year_start
        ams = annual_maximum_series.compute_ams(qmax,
                                                water_year_start=wys)

        drop = ["EVENTID", "NVALYEAR", "WATER_YEAR", "WATER_YEAR_END"]
        ams = ams.loc[ams.NVALYEAR >= 365]\
            .drop(drop, axis=1)\
            .set_index("WATER_YEAR_START")

        # Peak day of year
        start = pd.Series([pd.to_datetime(f"{y}-{wys}-01")
                           for y in ams.index], index=ams.index)
        start = start.dt.tz_localize(None)
        dow = ams.TIMEPEAK - start

        ams.loc[:, "DAYOFYEAR"] = -1
        iok = ams.TIMEPEAK.notnull()
        ams.loc[iok, "DAYOFYEAR"] = dow[iok].dt.days

        ams.columns = stationid + "_" + ams.columns

        comment = f"AMS streamflow data for station {stationid}."\
                  + f" Water year starts on the 1/{wys}."
        fams = f"AMS_streamflow_{stationid}_v{version}.csv"
        csv.write_csv(ams, fams, comment,
                      source_file, compress=False,
                      write_sys_info=False,
                      write_index=True,
                      lineterminator="\n",
                      archive=archive)

LOGGER.completed()

