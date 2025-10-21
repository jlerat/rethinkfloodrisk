from pathlib import Path
import re
import json
import numpy as np
from hydrodiy.io import csv

FHERE = Path(__file__).resolve().parent
FROOT = FHERE.parent.parent

ENV = "LOCAL"

with (FHERE / "config.json").open("r") as fo:
    CONFIG = json.load(fo)[ENV]


def replace_root(path):
    root_label = "package_root_folder"
    if path.startswith(root_label):
        return FROOT / re.sub(root_label + "/", "", path)
    else:
        return path


DATA_FOLDER = replace_root(CONFIG["data_folder"])

DATA_VERSION = CONFIG["data_version"]


def get_stations():
    fs = DATA_FOLDER / f"AMS_stations_v{DATA_VERSION}.csv"
    df, _ = csv.read_csv(fs, index_col="STATIONID")
    return df


def get_truepeaks():
    ft = DATA_FOLDER / f"peak_streamflow_concatenated_v{DATA_VERSION}.csv"
    df, _ = csv.read_csv(ft, index_col="DAY", parse_dates=True)

    # Remove -1
    values = df._get_numeric_data()
    values[values < 0] = np.nan

    return df
