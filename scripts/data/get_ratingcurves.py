#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-20 16:27:36.472013
## Comment : Collect streamflow data
##
## ------------------------------

import sys
import re
import argparse
from pathlib import Path
import requests
from datetime import datetime

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

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

# Configure kiwis download
url_kiwis = "https://realtimedata.waternsw.com.au/cgi/webservice.exe"
datasource = "A"
api_version = "1"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata = froot / "data" / "rating_curves"
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
stations = datahub.get_stations()
potpeaks, _, _ = datahub.get_potpeaks()

if debug:
    stations = stations.iloc[:1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
nstations = len(stations)

# Utility function to generate url corresponding to query
def query2url(query):
    c1, c2 = "'", "\""
    txt = re.sub(c1, c2, str(query))
    return f"{url_kiwis}?{txt}".strip()



for istation, (stationid, sinfo) in enumerate(stations.iterrows()):
    LOGGER.context = f"{stationid} ({istation + 1}/{nstations})"

    se = potpeaks.loc[:, f"{stationid}"]
    se = se.loc[se.notnull()]
    start, end = se.index[[0, -1]]
    times = pd.date_range(start, end, freq="5YS").to_list()
    today = pd.to_datetime(datetime.now()).round("D")
    times.append(today)

    rc = []
    metas = []
    for time in times:
        LOGGER.info(f"Downloading rating curve for time={time}", nret=1)

        fr = fdata / f"{stationid}_rating_table_{time.date()}.csv"
        fz = fr.parent / f"{fr.stem}.zip"
        if fz.exists() and not overwrite:
            LOGGER.info("File exists. Skip.", ntab=1)
            continue

        LOGGER.info("Downloading..", ntab=1)
        query = {\
            "function": "get_effective_rating", \
            "version": api_version, \
            "params": {\
                "site_list": str(stationid), \
                "table_from": "100", \
                "table_to": "141", \
                "interval": "0.1",\
                "datetime": time.strftime("%Y%m%d%H%M%S"), \
                "force_range": "1", \
                "quantised": "1", \
                "shifts": "1"
            }
        }

        url = query2url(query)
        req = requests.get(url)

        if req.status_code == 200:
            js = req.json()
            res = js["return"]["sites"][0]
            if "error_num" in res:
                LOGGER.info("Data problem. Skip.", ntab=1)
                continue

            LOGGER.info("Got data. Processing..", ntab=1)
            df = pd.DataFrame(res["points"])

            df = df.astype({"vt": float, "vf": float})

            # Convert ML/d -> m3/s
            df.loc[:, "vt"] /= 86.4

            # Rename columns
            df = df.rename(columns={
                "q": "QUALITY",
                "vt": "STREAMFLOW[m3.s-1]",
                "vf": "WATERLEVEL[m]"
                })
            df.loc[:, "TIME_VALIDITY"] = time

            meta = {
                "stationid": stationid,
                "short_name": res["site_details"]["short_name"],
                "var_from": res["varfrom_details"]["short_name"].strip(),
                "var_from_precision": res["varfrom_details"]["precision"],
                "var_to": res["varto_details"]["short_name"].strip(),
                "var_to_precision": res["varto_details"]["precision"],
                "time_validity": time
            }
            for q, mess in res["quality_codes"].items():
                meta[f"quality_code_{q}"] = re.sub("\n", " ", mess)
            metas.append(meta)

            rc.append(df)

        else:
            LOGGER.info("Data problem. Skip.", ntab=1)
            continue

    if len(rc) > 0:
        LOGGER.info("Storing..", ntab=1, nret=2)
        rc = pd.concat(rc)
        fr = fdata / f"{stationid}_rating_tables.csv"
        csv.write_csv(rc, fr, f"Rating curves for station {stationid}",
                      source_file)

        metas = pd.DataFrame(metas)
        fm = fdata / f"{stationid}_rating_tables_metadata.csv"
        csv.write_csv(metas, fm, f"Rating curves meta data for station {stationid}",
                      source_file)

LOGGER.completed()

