#!usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : Julien Lerat, CSIRO L&W
## Created : 2022-08-16 Tue 05:57 PM
## Comment : Download rating curves from Water NSW
##           API reference : https://kisters.com.au/doco/hydllp.htm
##
## ------------------------------
import sys, os, re, json, math
from datetime import datetime
from pathlib import Path
from datetime import datetime
import requests
import argparse
import numpy as np
import pandas as pd
import zipfile

from hydrodiy.io import csv, iutils
from hydrodiy.gis import gutils

from nrivdata import config

from tqdm import tqdm

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="Download rating curve from WaterNSW api", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-o", "--overwrite", help="Overwrite data", \
                    action="store_true", default=False)
args = parser.parse_args()

overwrite = args.overwrite

url_kiwis = "https://realtimedata.waternsw.com.au/cgi/webservice.exe"

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata  = config.get_config("DIRECTORIES", "DIR_TS_WNSW") / "WaterNSW_api_downloads"
fdata.mkdir(exist_ok=True)

fratings = fdata / "rating_tables"
fratings.mkdir(exist_ok=True, parents=True)

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

fs = fratings.parent / "WaterNSW_sites.csv"
sites, _ = csv.read_csv(fs, index_col="STATIONID")
sites = sites.loc[sites.loc[:, "STATION_TYPE[undef]"]=="STR", :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Utility function to generate url corresponding to query
def query2url(query):
    c1, c2 = "'", "\""
    txt = re.sub(c1, c2, str(query))
    return f"{url_kiwis}?{txt}".strip()

# Download rating tables
datasource = "A"
api_version = "1"
nsites = len(sites)

for isite, (siteid, info) in enumerate(sites.iterrows()):
    start = pd.to_datetime(info["START[day]"])
    end = pd.to_datetime(info["END[day]"])
    if pd.isnull(end):
        end = pd.to_datetime("2022-07-01")

    times = pd.date_range(start, end, freq="6MS")

    # Download rating table for each time
    tbar = tqdm(times, desc=f"{siteid} {isite+1}/{nsites}", total=len(times))
    fz = fratings / f"{siteid}_rating_tables.zip"
    if fz.exists() and not overwrite:
        continue

    with zipfile.ZipFile(fz, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for time in tbar:
            query = {\
                "function": "get_effective_rating", \
                "version": api_version, \
                "params": {\
                    "site_list": str(siteid), \
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
                    continue

                df = pd.DataFrame(res["points"])
                df = df.rename(columns={"q": "QUALITY[undef]", \
                                    "vt": "STREAMFLOW[ML/day]", \
                                    "vf": "WATERLEVEL[m]"})
                meta = {
                    "siteid": siteid, \
                    "short_name": res["site_details"]["short_name"], \
                    "var_from": res["varfrom_details"]["short_name"].strip(), \
                    "var_from_precision": res["varfrom_details"]["precision"], \
                    "var_to": res["varto_details"]["short_name"].strip(), \
                    "var_to_precision": res["varto_details"]["precision"]
                }
                for q, mess in res["quality_codes"].items():
                    meta[f"quality_code_{q}"] = re.sub("\n", " ", mess)

                fr = f"{siteid}_rating_table_{time.date()}.csv"
                csv.write_csv(df, fr, meta, source_file, \
                            archive=archive)
            else:
                continue

LOGGER.info("Process completed")

