#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-21 13:08:16.079895
## Comment : Compute sensitivity of Q100 to max obs
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from itertools import product as prod
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from floodstan import marginals

from pyrethink import datahub


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fout = froot / "outputs" / "sensitivity"
    fout.mkdir(exist_ok=True)
    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fout"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fout)
    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    stations = datahub.get_stations(False)
    if config.debug:
        stations = stations.iloc[:5]
    data = namedtuple("Data", ["stations"])(stations)
    return data


def process(config, script_paths, logger, data):
    stations = data.stations
    nstations = len(stations)
    logger.info(f"Start processing", nret=1)

    eps = 1e-2
    aris = config.aris
    etas = config.lh_moment_etas

    for ari, eta, order in prod(aris, etas, [1, 2]):
        cns = f"Q{ari}_SENSITIVITY{order}_ETA{eta}_DIFF[%/%dQ]"
        stations.loc[:, cns] = np.nan

        cns = f"Q{ari}_SENSITIVITY{order}_ETA{eta}_OLS[%/%dQ]"
        stations.loc[:, cns] = np.nan

    cna = "LENGTH_AMS[yr]"
    cmax = "QMAX[m3.s-1]"
    cmax2 = "QMAX2[m3.s-1]"
    cratio = "RATIO_QMAX_QMAX2[-]"
    for cn in [cna, cmax, cmax2, cratio]:
        stations.loc[:, cn] = np.nan

    gev = marginals.factory("GEV")

    for isite, (stationid, sinfo) in enumerate(stations.iterrows()):
        ctxt = f"{stationid} ({isite+1}/{nstations})"
        logger.info(f"Processing {ctxt}", nret=1)

        ams = datahub.get_ams(stationid)
        y = ams.filter(regex="_PEAK", axis=1).squeeze()
        y = y[y.notnull()].values.copy()
        if len(y) < 20:
            continue

        stations.loc[stationid, cna] = len(y)

        Qmax = np.max(y)
        Q2 = np.max(y[y < Qmax])
        stations.loc[stationid, cmax] = Qmax
        stations.loc[stationid, cmax2] = Q2
        stations.loc[stationid, cratio] = Qmax / Q2

        imax = np.argmax(y)

        for ari, eta in prod(aris, etas):
            Qref = {}
            for i in [-1, 0, 1]:
                y[imax] = Qmax + i * eps
                gev.fit_lh_moments(y, eta=eta)
                Qref[i] = gev.ppf(1 - 1. / ari)

            ratio = Qmax / Qref[0]
            s1 = (Qref[1] - Qref[-1]) / 2 / eps
            cns1 = f"Q{ari}_SENSITIVITY1_ETA{eta}_DIFF[%/%dQ]"
            stations.loc[stationid, cns1] = s1 * ratio

            s2 = (Qref[1] - 2 * Qref[0] + Qref[-1]) / eps**2
            cns2 = f"Q{ari}_SENSITIVITY2_ETA{eta}_DIFF[%/%dQ]"
            stations.loc[stationid, cns2] = s2 * ratio**2

            qqi = Qmax * np.linspace(1, 2, 10)
            qqo = np.zeros_like(qqi)
            for iq, q in enumerate(qqi):
                y[imax] = q
                gev.fit_lh_moments(y, eta=eta)
                qqo[iq] = gev.ppf(1 - 1. / ari)

            dq = qqi - Qmax
            X = np.column_stack([dq, dq**2])
            theta, _, _, _ = np.linalg.lstsq(X, qqo - Qref[0], rcond=1e-6)

            cns1 = f"Q{ari}_SENSITIVITY1_ETA{eta}_OLS[%/%dQ]"
            stations.loc[stationid, cns1] = theta[0] * ratio

            s2 = (Qref[1] - 2 * Qref[0] + Qref[-1]) / eps**2
            cns2 = f"Q{ari}_SENSITIVITY2_ETA{eta}_OLS[%/%dQ]"
            stations.loc[stationid, cns2] = theta[1] * ratio**2

    fs = script_paths.fout / "sensitivity.csv"
    csv.write_csv(stations, fs, "Sensitivity of reference floods to max obs",
                  script_paths.source_file,
                  compress=False, write_index=True,
                  lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sensitivity of Q100 to max obs",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    debug = args.debug
    etas = [2]
    aris = [10, 100, 500]

    Config = namedtuple("Config", ["debug", "lh_moment_etas", "aris"])
    config = Config(debug, etas, aris)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
