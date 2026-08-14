#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-21 Tue 08:19 AM
## Comment : Table with site characteristics
##
## ------------------------------

import sys
import math
from collections import namedtuple
from itertools import combinations as comb
import re
import argparse
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import matplotlib.pyplot as plt
from matplotlib import ticker

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from floodstan import marginals

from pyrethink import datahub, processing

from figA_impact_of_period_on_FFA import copulafit

def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    basename = source_file.stem
    fout = froot / "images" / "manuscript"
    fdata = froot / "outputs" / f"copulaconcat_v{config.version}"
    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fdata", "fout"])
    script_paths = SP(source_file, basename, froot,
                      fdata, fout)
    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.source_file.stem
    logger = iutils.get_logger(basename)
    return logger


def get_data(config, script_paths, logger):
    stationids = config.stationids
    obs_data, _, _, stations = datahub.get_ams_concat()
    stations = stations.loc[stationids]
    obs_data = obs_data.loc[:, stationids]

    version = config.version
    opm = copulafit.get_options(version)

    grp = "-".join(stationids)
    cspec = config.copula_spec
    acov = "True"
    taskid = opm.find(exclude=config.exclude,
                      awra_covariate=acov,
                      group=grp,
                      copula_spec=cspec)
    taskid = next(t for t in taskid)

    ppred = {}
    for vtype in ["multivar", "univ"]:
        fp = script_paths.fdata / f"copulaconcat_postpredcheck_{vtype}.zip"
        pp = pd.read_csv(fp, skiprows=15)
        idx = pp.TASKID == taskid
        cns = ["obs", "simmean", "simstd"]

        if vtype == "multivar":
            idx &= pp.VARIABLE == config.multivar_metric
            pp = pp.loc[idx, cns].squeeze()
            ppred[vtype] = pp
        else:
            idx &= pp.VARIABLE == config.univar_metric
            pp = pp.loc[idx].filter(regex="^(obs|simmean|simstd)", axis=1).squeeze()
            sids = stationids + ["AWRA-L"]
            df = pd.DataFrame({cn: np.nan for cn in cns}, index=sids)
            for ista, sid in enumerate(sids):
                for cn in cns:
                    df.loc[sid, cn] = pp.loc[f"{cn}[{ista + 1}]"]
            ppred[vtype] = df

    DT = namedtuple("Data", ["stations", "obs_data", "ppred"])
    return DT(stations, obs_data, ppred)


def process(config, script_paths, logger, data):
    stations = data.stations
    obs = data.obs_data

    cc = ["NAME", "CATCHMENTAREA[km2]"]
    charac = stations.loc[:, cc]
    charac.loc[:, "RECORD_LENGTH[yr]"] = obs.notnull().sum()
    dd = {
          "NAME": "Lismore catchment",
          "CATCHMENTAREA[km2]": 1390,
          "RECORD_LENGTH[yr]": 110
          }
    charac.loc["AWRA-L"] = dd

    um = config.univar_metric
    pp = data.ppred["univ"]

    charac.loc[:, f"{um}_obs"] = pp.loc[:, "obs"].apply(lambda x: f"{x:0.2f}")
    fun = lambda x: f"{x.iloc[0]:0.2f} ±{x.iloc[1]:0.2f}"
    charac.loc[:, f"{um}_sim"] = pp.loc[:, ["simmean", "simstd"]].apply(fun, axis=1)

    mm = config.multivar_metric
    pp = data.ppred["multivar"]
    charac.loc[stations.index[0], f"{mm}_obs"] = f"{pp.loc['obs']:0.2f}"
    charac.loc[stations.index[0], f"{mm}_sim"] = fun(pp.loc[["simmean", "simstd"]])

    fd = script_paths.fout / "tabA_site_characteristics.csv"
    charac_s = charac.astype(str)
    for cn, se in charac.items():
        if cn == "NAME" or re.search("obs|sim", cn):
            continue
        else:
            digit = 0
        charac_s.loc[:, cn] = se.apply(lambda x: f"{float(x):03.{digit}f}")

    csv.write_csv(charac_s, fd, "Site characteristics",
                  script_paths.source_file, compress=False,
                  write_index=True,
                  lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table containing site characteristics",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-s", "--stationids", help="Selected stationids",
                        type=str, default="203010-203014-203024")
    parser.add_argument("-e", "--eta", help="LH moment shift",
                        type=int, default=2)
    args = parser.parse_args()
    stationids = args.stationids.split("-")
    aris = [100, 500]

    # Copula config
    exclude = "NONE"
    copula_spec = "Gaussian"

    # Evaluation
    multivar_metric = "tau_q50"
    univar_metric = "lskewness2"

    # Config
    CF = namedtuple("Config", ["version", "stationids", "eta", "aris",
                               "exclude", "copula_spec",
                               "multivar_metric", "univar_metric"])
    config = CF(args.version, stationids, args.eta, aris,
                exclude, copula_spec,
                multivar_metric, univar_metric)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
