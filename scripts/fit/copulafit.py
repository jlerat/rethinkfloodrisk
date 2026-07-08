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

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

from floodstan import marginals
from floodstan import report
from floodstan import sample as fsample
from floodstan import univariate_censored_sampling

from pyrethink import datahub
from pyrethink import sample as rsample
from pyrethink import mv_censored_sampling
from pyrethink import mv_censored_factors_sampling

def get_options(version, version_priors, awraid="WILSONSRIVER"):
    copula_specs = [
        "Univariate",
        "Gaussian",
        "GaussianFactor_0_1",
        "GaussianFactor_0_2",
        "Student_5"
        ]
    pcensors = [0.3]
    priors = ["uninformative", "informative"]
    excludes = ["NONE",
                "2016",
                "2021"]

    marginal_name = "GEV"

    stations = datahub.get_stations()
    sids = stations.index.to_list()
    groups = ["203010-203014-203024"] \
             + ["-".join(sids)] \
             + sids
    if int(version) >= 22:
        groups += ["203004-203010-203014-203024"]

    awra_covariate = [False, True]

    opm = hyruns.OptionManager(version=version,
                               version_priors=version_priors,
                               marginal_name=marginal_name,
                               awraid=awraid)
    opm.from_cartesian_product(pcensor=pcensors,
                               exclude=excludes,
                               copula_spec=copula_specs,
                               prior=priors,
                               awra_covariate=awra_covariate,
                               group=groups)

    # .. exclude options with a single stations and
    # multivariate copula_spec
    keep = []
    for task in opm.tasks:
        nsta = len(task["group"].split("-"))
        if nsta == 1 and task["copula_spec"] != "Univariate":
            continue
        if nsta > 1 and task["copula_spec"] == "Univariate":
            continue
        if nsta < 8 and task["copula_spec"] == "GaussianFactor2":
            continue
        if nsta != 3 and task["awra_covariate"]:
            continue
        keep.append(task)
    opm.tasks = keep
    return opm


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"
    fout = froot / "outputs" / f"{source_file.stem}_v{config.version}" / f"TASK{config.taskid}"

    flogs = froot / "logs" / source_file.stem

    fpriors = froot / "outputs" / f"priorfit_v{config.version_priors}"
    if not fpriors.exists():
        errmsg = "Prior data folder does not exist"
        raise ValueError(errmsg)

    if config.debug:
        fout = flogs / fout.stem

    fstan = fout / f"stan_TASK{config.taskid}"

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs",
                              "fstan", "fpriors"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs,
                               fstan, fpriors)

    flogs.mkdir(exist_ok=True, parents=True)
    fout.mkdir(exist_ok=True, parents=True)
    fstan.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}_TASK{config.taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.info("", nret=1)
    config.task.log(logger)
    return logger


def get_stationids(config):
    stationids = config.task["group"].split("-")
    if config.task.awra_covariate:
        stationids += [f"AWRA-{config.awraid}"]
    return stationids


def get_data(config, script_paths, logger):
    df, _, _, _ = datahub.get_ams_concat()

    if config.task.awra_covariate:
        awra = datahub.get_ams_awra(config.awraid)
        awra.name = f"AWRA-{config.awraid}"
        df = pd.concat([df, awra], axis=1).sort_index()

    excl = config.task.exclude
    if excl != "NONE":
        df = df.loc[df.index != int(excl)]

    ams_times = df.index
    stationids = get_stationids(config)
    pcensor = config.task["pcensor"]

    marginal_name = config.marginal_name

    copula_spec = config.task.copula_spec

    if copula_spec == "Univariate":
        marginal = marginals.factory(marginal_name)
        y = df.loc[:, stationids].squeeze()
        censor = np.nanpercentile(y, pcensor * 100)
        nchains = config.stan_nchains if hasattr(config, "stan_nchains")\
                else 10
        sv = fsample.StanSamplingVariable(marginal, y, censor,
                                          ninits=nchains)
    else:
        y = df.loc[:, stationids]
        censors = np.nanpercentile(y, pcensor * 100, axis=0)
        sv = rsample.StanSamplingMultivariate(y,
                                              copula_spec,
                                              censors=censors)

    stan_data = sv.to_dict()

    # set priors
    if config.task.prior == "informative":
        fp = script_paths.fpriors / "priors.csv"
        priors, _ = csv.read_csv(fp, dtype={"STATIONID": str})

        for isite, stationid in enumerate(stationids):
            idx0 = priors.STATIONID == stationid
            idx0 &= priors.MARGINAL == marginal_name

            for pn in ["locn", "logscale", "shape1"]:
                idx = idx0 & (priors.PARAMETER == pn)
                pred = "INTERCEPT" if pn == "shape1"\
                        else "LOG10_CATCHMENTAREA_VALID[km2][-]"
                idx &= priors.PREDICTORS == pred

                # Prior for one station is missing...
                if idx.sum() == 0:
                    logger.warning(f"No informative prior for {stationid} / {pn}")
                    continue

                pv = priors.loc[idx].squeeze()
                pm = float(pv.PRIOR_MEAN)
                ps = float(pv.PRIOR_STD)
                pn2 = f"y{pn}_prior"

                if config.task.copula_spec == "Univariate":
                    stan_data[pn2] = [pm, ps]
                else:
                    stan_data[pn2][isite] = [pm, ps]

    stan_inits = sv.initial_parameters
    Data = namedtuple("Data", ["stan_data", "stan_inits",
                               "stationids", "ams_times"])
    data = Data(stan_data, stan_inits, stationids,
                ams_times)
    return data


def process(config, script_paths, logger, data):
    logger.info(f"Start processing", nret=1)

    kw = dict(data=data.stan_data,
              seed=config.seed,
              iter_sampling=config.stan_nsamples // config.stan_nchains,
              output_dir=script_paths.fstan,
              inits=data.stan_inits,
              chains=config.stan_nchains,
              parallel_chains=config.stan_nchains,
              iter_warmup=config.stan_nwarm,
              show_progress=config.debug)

    if config.task.copula_spec == "Univariate":
        sampler = univariate_censored_sampling
    elif re.search("GaussianFactor", config.task.copula_spec):
        sampler = mv_censored_factors_sampling
    else:
        sampler = mv_censored_sampling

    smp = sampler(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    # Clean stan folder
    for f in script_paths.fstan.glob("*.*"):
        f.unlink()
    script_paths.fstan.rmdir()

    # Report stan diagnostic
    for me in report.STAN_DIAGNOSTIC_VARIABLES:
        logger.info(f"Stan diagnostic {me}: {diag[me]}")

    diag["version"] = config.version
    diag["stan_nchains"] = config.stan_nchains
    diag["stan_nwarm"] = config.stan_nwarm
    diag["taskid"] = config.taskid
    task_opt = {f"task_{k}": v for k, v in config.task.to_dict()["options"].items()}
    diag.update(task_opt)

    basename = script_paths.basename
    fout = script_paths.fout
    source_file = script_paths.source_file
    fd = fout / f"{basename}_samples_TASK{taskid}.csv"
    csv.write_csv(df, fd, f"STAN samples for task {taskid}",
                  source_file, compress=True)

    fd = fout / f"{basename}_diagnostic_TASK{taskid}.json"
    with fd.open("w") as fo:
        json.dump(diag, fo, indent=4)

    # Store data with additional info
    stan_data = data.stan_data
    stan_data["pcensor"] = config.task.pcensor
    stan_data["ams_time"] = data.ams_times.tolist()
    stan_data["stationids"] = data.stationids
    stan_data.update(task_opt)

    fdd = fout / f"{basename}_data_TASK{taskid}.json"
    for n in ["y", "idx_cens", "idx_obs", "idx_miss", "censors"]:
        if n not in stan_data:
            continue
        dt = stan_data[n]
        if hasattr(dt, "tolist"):
            dt = dt.tolist()
        stan_data[n] = dt

    with fdd.open("w") as fo:
        json.dump(stan_data, fo, indent=4)

    fi = fout / f"{basename}_inits_TASK{taskid}.json"
    stan_inits = data.stan_inits
    enum = stan_inits.items() if isinstance(stan_inits, dict) \
            else enumerate(stan_inits)
    for key, val in enum:
        if hasattr(val, "tolist"):
            val = val.tolist()
        stan_inits[key] = val

    with fi.open("w") as fo:
        json.dump(stan_inits, fo, indent=4)


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
    seed = 5446
    awraid = "WILSONSRIVER"

    version_priors = 2

    if debug:
        stan_nwarm = 200
        stan_nchains = 3
        stan_nsamples = 200
    else:
        stan_nwarm = 10000
        stan_nchains = 10
        stan_nsamples = 50000

    # .. options
    opm = get_options(version, version_priors, awraid)
    marginal_name = opm.context["marginal_name"]

    if debug:
        ctype = "mv"
        if ctype == "univ":
            taskid = opm.search(copula_spec="Univariate",
                                exclude="2021",
                                prior="^informative",
                                group="203014")[0]
        else:
            taskid = opm.search(copula_spec="GaussianFactor_0_2$",
                                exclude="2021",
                                prior="^informative",
                                awra_covariate="True",
                                group="203010-203014-203024")[0]

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "task", "version_priors",
                         "stan_nwarm", "stan_nchains",
                         "stan_nsamples", "seed", "awraid",
                         "marginal_name"])
    config = Config(version, taskid, overwrite,
                    debug, opm.get_task(taskid),
                    version_priors, stan_nwarm,
                    stan_nchains, stan_nsamples,
                    seed, awraid, marginal_name)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)
    logger.info(f"Number of tasks : {opm.ntasks}", nret=1)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
