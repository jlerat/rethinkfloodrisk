#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
from collections import namedtuple
from itertools import product as prod
import re
import argparse
import json
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils, violinplot

from pyrethink import datahub

import importlib
import figA_impact_of_period_on_FFA
importlib.reload(figA_impact_of_period_on_FFA)

from figA_impact_of_period_on_FFA import get_script_paths
from figA_impact_of_period_on_FFA import get_logger, get_taskids, get_data
from figA_impact_of_period_on_FFA import get_iter_options, select_data

def process(config, script_paths, logger, data):
    for pcensor, rho_min, has_cluster, copula_shape in get_iter_options(data):
        _, _, mvnproc, _, _ = select_data(data,
                                          pcensor=pcensor,
                                          rho_min=rho_min,
                                          has_cluster=has_cluster,
                                          copula_shape=copula_shape)
        if len(mvnproc) == 0:
            continue

        assert len(mvnproc) == 1

        rn = next(iter(mvnproc))
        logger.info(f"-- Plotting {rn.text} --", nret=1)

        mvnproc = mvnproc[rn]
        aep_targets = [1, 10]

        for aep_target in aep_targets:
            logger.info(f"Plot violin {aep_target}", ntab=1)

            plt.close("all")
            fig, ax = plt.subplots(figsize=(config.awidth, config.aheight),
                                   layout="constrained")

            aep = mvnproc.filter(regex=f".*aep{aep_target:02d}_.*_smp", axis=1) * 100
            cols = aep.columns.to_series().str.replace(f".*_aep{aep_target:02d}_|_smp",
                                                       "", regex=True)
            aep.columns = cols

            vm = violinplot.Violin(aep, number_format="0.1f")
            vm.draw()

            ax.set(xlabel="Station", ylabel="Annual Exceedance Probability [%]")

            basename = script_paths.basename
            fp = f"{basename}_AEP{aep_target}_{rn.text}_v{config.version}.png"
            fp = script_paths.fimg / fp
            fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                        type=float, default=0.3)
    parser.add_argument("-di", "--diag", help="Show stan diagnostics",
                        action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_mins", help="Minimum rho value",
                        type=str, default="-1|0")
    parser.add_argument("-s", "--copula_shapes", help="Copula shapes selected",
                        type=str, default="0|3")
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_mins",
                               "awidth", "aheight", "fdpi",
                               "excludes", "copula_shapes",
                               "diag", "debug",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks"])
    awidth = 6
    aheight = 5
    fdpi = 300
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = False
    load_mvnproc = True
    load_expected_params = False
    load_postpred_checks = False

    config = CF(args.version, args.pcensor,
                args.rho_mins.split("|"),
                awidth, aheight, fdpi,
                excludes, args.copula_shapes.split("|"),
                args.diag, args.debug,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
