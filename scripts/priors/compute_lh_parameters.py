#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-06-12 11:23:40.081704
## Comment : Compute LH parameters for a wide range of stations
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from floodstan import marginals
from pyrethink import datahub

MARGINALS = ["GEV"]


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"
    fout = fdata / "priors"
    fout.mkdir(exist_ok=True, parents=True)

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout)
    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    stations = datahub.get_stations(False)
    ams = datahub.get_ams()
    cc = list(set(ams.columns) & set(stations.index))
    stations = stations.loc[cc, :]
    ams = ams.loc[:, cc]

    data = namedtuple("Data", ["stations", "ams"])(stations, ams)
    return data


def process(config, script_paths, logger, data):
    nstations = len(data.stations)
    logger.info(f"Start processing", nret=1)

    res = []

    for isite, (stationid, sinfo) in enumerate(data.stations.iterrows()):
        ctxt = f"{stationid} ({isite+1}/{nstations})"
        logger.info(f"Processing {ctxt}", nret=1)
        ams = data.ams.loc[:, stationid]
        ams = ams.loc[ams.notnull()]

        for m in MARGINALS:
            marg = marginals.factory(m)
            marg.fit_lh_moments(ams.squeeze())

            dd = {
                "stationid": stationid,
                "marginal": m,
                "nval": len(ams),
                "param_locn": marg.locn,
                "param_logscale": marg.logscale,
                "param_shape1": marg.shape1
                }
            res.append(dd)

        if isite >= 10 and config.debug:
            break

    res = pd.DataFrame(res)
    logger.info("\n" + str(res.describe()), nret=1)

    fr = script_paths.fout / f"params_lh_moments.csv"
    csv.write_csv(res, fr, "Fitted LH moment params",
                  script_paths.source_file,
                  write_index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute LH parameters for a wide range of stations",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    debug = args.debug
    Config = namedtuple("Config", ["debug"])
    config = Config(debug)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
