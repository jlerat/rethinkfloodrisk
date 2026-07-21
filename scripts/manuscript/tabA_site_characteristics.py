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

def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    basename = source_file.stem
    fout = froot / "images" / "manuscript"
    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fout"])
    script_paths = SP(source_file, basename, froot,
                      fout)
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

    rating_curves = {}
    for stationid in config.stationids:
        rc, _ = datahub.get_rating_curves(stationid, True)
        ch, cq = "WATERLEVEL[m]", "STREAMFLOW[m3_s-1]"
        rc = rc.loc[:, [ch, cq]].rename(columns={ch:"H", cq:"Q"})
        rating_curves[stationid] = rc

    DT = namedtuple("Data", ["stations", "obs_data", "rating_curves"])
    return DT(stations, obs_data, rating_curves)


def process(config, script_paths, logger, data):
    stations = data.stations
    obs = data.obs_data

    cc = ["NAME", "CATCHMENTAREA[km2]"]
    charac = stations.loc[:, cc]
    charac.loc[:, "RECORD_LENGTH[yr]"] = obs.notnull().sum()

    # Configure gev fit sensitivity analysis
    aris = config.aris
    csens1 = [f"H{ari}_SENSITIVITY_QMAX[cm/%dQmax]" for ari in aris]
    csens2 = [f"Q{ari}_SENSITIVITY_QMAX[%dQ/%dQmax]" for ari in aris]
    charac.loc[:, csens1 + csens2] = np.nan

    eta = config.eta
    eps = 1e-6

    for stationid, ams in obs.items():
        y = ams.loc[ams.notnull()].values.copy()
        logger.info(f"Station {stationid} len(ams)={len(y)}", nret=1)
        imax = np.argmax(y)
        Qmax = y.max()

        rc = data.rating_curves[stationid]

        Qref = {a: {} for a in aris}
        Href = {a: {} for a in aris}
        for i in [-1, 0, 1]:
            y[imax] = Qmax + i * eps
            gev = marginals.factory("GEV")
            gev.fit_lh_moments(y, eta=eta)

            for ari in aris:
                prob = 1 - 1. / ari
                Q = gev.ppf(prob)
                H = processing.linear_interpolation(Q, rc.Q, rc.H)

                Qref[ari][i] = Q
                Href[ari][i] = H

        for iari, ari in enumerate(aris):
            Sh = (Href[ari][1] - Href[ari][-1]) / 2 / eps * Qmax
            charac.loc[stationid, csens1[iari]] = Sh
            logger.info(f"\tSensitivity H{ari} = {Sh:0.2f} cm/%dQmax")

            Sq = (Qref[ari][1] - Qref[ari][-1]) / 2 / eps * Qmax / Qref[ari][0]
            charac.loc[stationid, csens2[iari]] = Sq
            logger.info(f"\tSensitivity Q{ari} = {Sq:0.2f} %dQ/%dQmax")

    fd = script_paths.fout / "tabA_site_characteristics.csv"
    charac_s = charac.astype(str)
    for cn, se in charac.items():
        if cn == "NAME":
            continue
        if cn in csens1:
            digit = 1
        elif cn in csens2:
            digit = 2
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

    parser.add_argument("-s", "--stationids", help="Selected stationids",
                        type=str, default="203010-203024-203014")
    parser.add_argument("-e", "--eta", help="LH moment shift",
                        type=int, default=2)
    args = parser.parse_args()
    stationids = args.stationids.split("-")
    aris = [100, 500]

    # Config
    CF = namedtuple("Config", ["stationids", "eta", "aris"])
    config = CF(stationids, args.eta, aris)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
