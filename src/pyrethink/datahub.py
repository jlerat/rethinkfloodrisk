from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
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
    df, _ = csv.read_csv(fs, index_col="STATIONID",
                         dtype={"STATIONID": str})
    potpeaks, _, _ = get_potpeaks()
    return df.loc[potpeaks.columns, :]


def get_potpeaks():
    ft = DATA_FOLDER / f"peak_streamflow_concatenated_v{DATA_VERSION}.csv"
    df, _ = csv.read_csv(ft, index_col="DAY", parse_dates=True)

    # Remove -1
    values = df._get_numeric_data()
    values[values < 0] = np.nan

    wy = df.WATERYEAR.astype(int)
    df = df.filter(regex="_PEAK", axis=1)
    df.columns = df.columns.str.replace("_PEAK", "")

    # Count number of events per year
    nu = wy.value_counts().mean()

    return df, wy, nu


def get_potpeaks_thresh():
    ft = DATA_FOLDER / f"peak_streamflow_concatenated_v{DATA_VERSION}.csv"
    _, comments = csv.read_csv(ft, index_col="DAY", parse_dates=True)

    qthresh = {re.sub(".*_|\\[.*", "", key): float(val)
               for key, val in comments.items()
               if re.search("^pot", key)}

    qthresh = pd.Series(qthresh)
    qthresh.name = "POT_thresh[m3.s-1]"

    return qthresh


def get_ams(stationid):
    fa = f"AMS_streamflow_{stationid}_v{DATA_VERSION}.csv"
    fa = DATA_FOLDER / "ams" / fa

    if not fa.exists():
        errmsg = f"Cannot find ams data for station {stationid}."
        raise ValueError(errmsg)

    ams, _ = csv.read_csv(fa, parse_dates=True)

    return ams


def get_ams_concat():
    stations = get_stations()
    peaks = pd.DataFrame(np.nan,
                         columns=stations.index,
                         index=np.arange(1957, 2023))
    times = pd.DataFrame(pd.NaT,
                         columns=stations.index,
                         index=np.arange(1957, 2023))


    for stationid in stations.index:
        ams = get_ams(stationid)
        peak = ams.filter(regex="_PEAK$", axis=1).squeeze().values
        time = pd.to_datetime(ams.filter(regex="_TIMEPEAK$", axis=1).squeeze()).values
        wy = ams.WATER_YEAR_START.str[:4].astype(int).values

        peaks.loc[wy, stationid] = peak
        times.loc[wy, stationid] = time

    return peaks, times


def get_censors(pcensor):
    if pcensor < 0 or pcensor > 1:
        errmsg = f"Expected pcensor in [0, 1], got {pcensor}."
        raise ValueError(errmsg)

    ams, _ = get_ams_concat()
    censors = pd.Series(np.nan, index=ams.columns)

    for stationid, qmax in ams.items():
        censors.loc[stationid] = qmax.quantile(pcensor)

    return censors


def get_rating_curves(stationid, only_last=False):
    frc = DATA_FOLDER / "rating_curves" / f"{stationid}_rating_tables.csv"
    fz = frc.parent / f"{frc.stem}.zip"
    if not fz.exists():
        errmsg = f"Cannot find rating data for station {stationid}."
        raise ValueError(errmsg)

    rc, _ = csv.read_csv(frc)

    fm = frc.parent / f"{frc.stem}_metadata.csv"
    meta, _ = csv.read_csv(fm)

    times = rc.TIME_VALIDITY.unique()
    if only_last:
        times = times[[-1]]

    rcs = {}
    metas = {}
    for time in times:
        rcs[time] = rc.loc[rc.TIME_VALIDITY == time]
        metas[time] = meta.loc[meta.time_validity == time]

    if only_last:
        return rcs[time], metas[time]
    else:
        return rcs, metas


def eep2aep(nu, eep):
    return 1 - np.exp(-nu * eep)


def linear_interpolation(xx, x, y):
    """ Linear interpolation """
    # Sort values
    isort = np.argsort(x)
    x = np.array(x)[isort]
    if np.any(np.diff(x) <= 0):
        errmsg = "Cannot process duplicates in x."
        raise ValueError(errmsg)

    if len(y) != len(x):
        errmsg = "Expected x and y of same length."
        raise ValueError(errmsg)

    y = np.array(y)[isort]
    xx = np.atleast_1d(xx)

    # interpolation coefficients
    D = np.abs(x[:, None] - x[None, 1:-1])
    D = np.column_stack([D, np.ones(len(x)), x])
    coefs = np.linalg.solve(D, y)

    # Run interpolation
    D = np.abs(xx[:, None] - x[None, 1:-1])
    D = np.column_stack([D, np.ones(len(xx)), xx])
    return (D @ coefs).squeeze()
