#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-06-26 10:26:52.387430
## Comment : Fit copula to data
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

import warnings
warnings.filterwarnings("error")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

from floodstan import marginals
from pyrethink import copulas
from pyrethink import postpredchecks as ppc

from copulafit import get_options, get_stationids, get_data

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"copulafit_v{config.version}" / f"TASK{config.taskid}"
    fout = froot / "outputs" / f"{source_file.stem}_v{config.version}" / f"TASK{config.taskid}"

    flogs = froot / "logs" / source_file.stem

    fpriors = froot / "outputs" / f"priorfit_v{config.version_priors}"
    if not fpriors.exists():
        errmsg = "Prior data folder does not exist"
        raise ValueError(errmsg)

    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs",
                              "fpriors"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs, fpriors)

    flogs.mkdir(exist_ok=True, parents=True)
    fout.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}_TASK{config.taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.info("", nret=1)
    config.task.log(logger)
    return logger


def process(config, script_paths, logger, data):
    logger.info(f"Start processing", nret=1)
    taskid = config.taskid
    copula_spec = config.task.copula_spec

    fp = script_paths.fdata / f"copulafit_samples_TASK{taskid}.csv"
    fz = fp.parent / f"{fp.stem}.zip"
    samples = pd.read_csv(fz, skiprows=15)

    if config.debug:
        samples = samples.iloc[:200]
    nsamples = len(samples)

    # Skip univariate configs
    if config.task.copula_spec == "Univariate":
        return

    # Marginal
    marginal = marginals.factory(config.marginal_name)

    # Run process
    stationids = data.stationids
    nsta = len(stationids)

    task_grp = config.task.group
    grps = [g for g in set(config.groups + config.sum_groups)
            if re.search("-", g)]

    aeps = []
    sum_samples = []

    for ismp, (i, smp) in enumerate(samples.iterrows()):
        if ismp % 50 == 0:
            logger.info(f"Processing sample {ismp + 1} / {nsamples}", nret=1)

        # Get copula
        cop = copulas.factory(config.task.copula_spec, nsta)

        # retrieve correlation matrix
        if re.search("Factor", config.task.copula_spec):
            is_factor = True
            nf = cop.copula_nfactors
            zrhos = smp.filter(regex="zrhos").values.reshape((nsta, nf + 1))
            cop.set_params_via_zrhos(zrhos)
        else:
            is_factor = False
            corr = smp.filter(regex="corr_IW").values.reshape((nsta, nsta)).T
            cop.params = corr

        # Sample
        x = cop.sample(1)[0]

        # Back transform raw space via marginal quantile
        y = np.empty(len(x))
        dd = {
            "version": config.version,
            "taskid": config.taskid,
            "copula_spec": config.task.copula_spec,
            "isample": ismp
            }
        for ista, (sid, xx) in enumerate(zip(stationids, x)):
            thetas = smp.filter(regex=f"^y(log|loc|sh).*\\[{ista + 1}\\]")
            marginal.params = thetas
            y[ista] = max(0, marginal.ppf(xx))
            dd[f"{sid}_sample"] = float(y[ista])

        for grp in config.sum_groups:
            s = 0
            for sid in grp.split("-"):
                s += dd[f"{sid}_sample"]
            dd[f"{grp}_sum_sample"] = s

        sum_samples.append(dd)

        # Operations on station groups
        for grp in grps:
            grp_sids = grp.split("-")
            nsta_grp = len(grp_sids)

            # check the group is covered by the task model
            if any(not re.search(sid, task_grp) for sid in grp_sids):
                continue

            # Parameterise copula for sub-group
            ista = [stationids.index(sid) for sid in grp_sids]
            subparams = cop.params[ista]
            if not is_factor:
                suparams = subparams[:, ista]

            subcop = copulas.factory(config.task.copula_spec, nsta_grp)
            subcop.params = subparams

            # compute aeps
            for ari in config.design_eris:
                cdf_marginals = (1 - 1./ari) * np.ones(nsta_grp)
                surv = subcop.survival(cdf_marginals)
                l10 = math.log10(surv) if surv > 0 else np.nan
                dd = {
                    "version": config.version,
                    "taskid": config.taskid,
                    "copula_spec": config.task.copula_spec,
                    "group": grp,
                    "isample": ismp,
                    "ari": ari,
                    "log10_aep": l10,
                    }
                aeps.append(dd)

    fout = script_paths.fout
    source_file = script_paths.source_file

    fa = fout / f"copulaprocess_multivar_aeps_TASK{taskid}.csv"
    aeps = pd.DataFrame(aeps)
    csv.write_csv(aeps, fa, "Multivariate AEPs computations for task {taskid}",
                  source_file)

    fs = fout / f"copulaprocess_sum_samples_TASK{taskid}.csv"
    sum_samples = pd.DataFrame(sum_samples)
    csv.write_csv(sum_samples, fs, "Summation samples for task {taskid}",
                  source_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit copula to data",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    parser.add_argument("-t", "--taskid", help="JobID",
                        type=int, default=-1)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-o", "--overwrite", help="Overwrite data",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    version = args.version
    taskid = args.taskid
    overwrite = args.overwrite
    debug = args.debug
    version_priors = 2
    awraid = "WILSONSRIVER"

    design_eris = [10, 50, 100, 200]

    sum_groups = ["203010-203014-203024"] \
                 + ["203004-203010-203014-203024"]

    # .. options
    opm = get_options(version, version_priors)
    groups = opm.options["group"]
    marginal_name = opm.context["marginal_name"]

    if debug:
        ctype = "cop"
        if ctype == "univ":
            taskid = opm.search(copula_spec="Univariate",
                                exclude="NONE",
                                prior="^informative",
                                group="203014")
        else:
            taskid = opm.search(copula_spec="GaussianFactor_0_2$",
                                exclude="NONE",
                                prior="^informative",
                                group=opm.options["group"][1])

        taskid = taskid[0]

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "task", "version_priors",
                         "design_eris", "awraid",
                         "groups", "sum_groups",
                         "marginal_name"])
    config = Config(version, taskid, overwrite,
                    debug, opm.get_task(taskid),
                    version_priors, design_eris, awraid,
                    groups, sum_groups, marginal_name)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)
    logger.info(f"Number of tasks : {opm.ntasks}", nret=1)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
