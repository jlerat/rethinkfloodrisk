from pathlib import Path
import re
import numpy as np
import pandas as pd
from hydrodiy.io import csv

FHERE = Path(__file__).resolve().parent
FROOT = FHERE.parent.parent
DATA_FOLDER = FROOT / "data"
DATA_VERSION = "5.0"


def get_stations(no_missing=True):
    fs = DATA_FOLDER / f"AMS_stations_v{DATA_VERSION}.csv"
    df, _ = csv.read_csv(fs, index_col="STATIONID",
                         dtype={"STATIONID": str})
    return df


def get_potpeaks(no_missing=True):
    ft = DATA_FOLDER / f"peak_streamflow_concatenated_v{DATA_VERSION}.csv"
    df, _ = csv.read_csv(ft, index_col="DAY", parse_dates=True)

    # Remove -1
    values = df._get_numeric_data()
    values[values < 0] = np.nan

    wy = df.WATERYEAR.astype(int)
    df = df.filter(regex="_PEAK", axis=1)
    df.columns = df.columns.str.replace("_PEAK", "")

    ams, _, _, _ = get_ams_concat(no_missing)
    df = df.loc[:, ams.columns]
    if no_missing:
        iok = df.notnull().all(axis=1)
        df = df.loc[iok]
        wy = wy.loc[iok]

    # Count number of events per year
    nu = wy.value_counts().mean()

    return df, wy, nu


def get_potpeaks_thresh(no_missing=True):
    ft = DATA_FOLDER / f"peak_streamflow_concatenated_v{DATA_VERSION}.csv"
    _, comments = csv.read_csv(ft, index_col="DAY", parse_dates=True)

    potpeaks, _, _ = get_potpeaks(no_missing)

    qthresh = {re.sub(".*_|\\[.*", "", key): float(val)
               for key, val in comments.items()
               if re.search("^pot", key)}

    qthresh = pd.Series(qthresh)
    qthresh.name = "POT_thresh[m3.s-1]"
    qthresh = qthresh.loc[potpeaks.columns]

    return qthresh


def get_ams(stationid):
    fa = f"AMS_streamflow_{stationid}_v{DATA_VERSION}.csv"
    fa = DATA_FOLDER / "ams" / fa

    if not fa.exists():
        errmsg = f"Cannot find ams data for station {stationid}."
        raise ValueError(errmsg)

    ams, _ = csv.read_csv(fa, parse_dates=True)

    return ams


def get_ams_concat(no_missing=True):
    stations = get_stations()
    peaks = pd.DataFrame(np.nan,
                         columns=stations.index,
                         index=np.arange(1957, 2023))
    times = pd.DataFrame(pd.NaT,
                         columns=stations.index,
                         index=np.arange(1957, 2023))
    dows = pd.DataFrame(-1,
                        columns=stations.index,
                        index=np.arange(1957, 2023))

    for stationid in stations.index:
        ams = get_ams(stationid)
        peak = ams.filter(regex="_PEAK$", axis=1).squeeze().values
        time = ams.filter(regex="_TIMEPEAK$", axis=1).squeeze()
        time = pd.to_datetime(time).values
        dow = ams.filter(regex="_DAYOFYEAR", axis=1).squeeze()
        dow = dow.values
        wy = ams.WATER_YEAR_START.str[:4].astype(int).values

        peaks.loc[wy, stationid] = peak
        times.loc[wy, stationid] = time
        dows.loc[wy, stationid] = dow

    if no_missing:
        miss = peaks.isnull().sum()
        selected = miss < 30
        isok = peaks.loc[:, selected].notnull().all(axis=1)
        peaks = peaks.loc[isok, selected]
        times = times.loc[isok, selected]
        dows = dows.loc[isok, selected]
        stations = stations.loc[selected]

    return peaks, times, dows, stations


def get_censors(pcensor, no_missing=True):
    if pcensor < 0 or pcensor > 1:
        errmsg = f"Expected pcensor in [0, 1], got {pcensor}."
        raise ValueError(errmsg)

    ams, _, _, _ = get_ams_concat(no_missing)
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
