#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-06-20 17:36:03.746063
## Comment : Compute regional priors for parameters
##
## ------------------------------


import sys
import os
import re
import json
import math
import shutil
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import sutils

from floodstan import gls, sample, report
from floodstan import gls_spatial_sampling

from pyrethink import datahub


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"

    version = config.version
    fout = froot / "outputs" / f"priorfit_v{config.version}"
    flogs = froot / "logs" / "priorfit"

    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs)
    flogs.mkdir(exist_ok=True, parents=True)
    fout.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    taskid = config.taskid
    fl = script_paths.flogs / f"priorfit_TASK{taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    # Configuration of task
    task = config.task
    parname = task.parname
    npred = task.npred
    data = datahub.get_params_lh_moments()

    # Get predictand
    y = data.loc[:, f"param_{parname.lower()}"]
    if parname == "LOCN":
        y = np.log(y)

    # get predictors
    xfull = np.log(data.filter(regex="PREDICTOR", axis=1))
    xfull.loc[:, "INTERCEPT"] = 1

    if npred>0:
        # OLS regression to reduce predictor list
        theta, fstat, fpvalue, _ = sutils.lstsq(xfull, y)
        preds = theta.tpvalue.sort_values().index.tolist()[:npred]
    else:
        theta = None
        preds = ["INTERCEPT"]

    x = xfull.loc[:, preds]

    preds = [re.sub("PREDICTOR_", "", p) for p in preds]
    logger.info(f"Predictors chosen: {'/'.join(preds)}")

    # Select stations
    stations = datahub.get_stations()
    isin = stations.index.isin(data.STATIONID)
    stationids = stations.index[isin]

    # .. eliminate predictand values for these sites
    ivalid = data.STATIONID.isin(stationids)
    ysample = y.copy()
    ysample.loc[ivalid] = np.nan

    # station coordinates
    w = data.loc[:, ["XCENTROID_EPSG28356[m]", "YCENTROID_EPSG28356[m]"]]
    w *= 1e-3

    # Filter with a radious of 500 km
    d = pdist(w)
    sid_idx = data.STATIONID.isin(stationids)
    stationids = data.STATIONID[sid_idx]
    idx_dm = squareform(d)[sid_idx].max(axis=0) < config.dist_max

    w = w.loc[idx_dm]
    x = x.loc[idx_dm]
    y = y.loc[idx_dm]
    ysample = ysample.loc[idx_dm]
    data = data.loc[idx_dm]
    sid_idx = np.where(data.STATIONID.isin(stationids))[0]
    stationids = data.STATIONID.iloc[sid_idx].tolist()

    # prior on rho based on min dist
    d = pdist(w)
    logrho_prior = [math.log(d.mean())]*2
    logrho_lower = math.log(d.min()/2)
    logrho_upper = math.log(2*d.max())

    # Prepare stan data for sampling
    stan_data, stan_inits = gls.prepare(x, w, ysample,
                                        logrho_prior=logrho_prior,
                                        logrho_lower=logrho_lower,
                                        logrho_upper=logrho_upper,
                                        logalpha_prior=[2, 5],
                                        logsigma_prior=[2, 5])

    Data = namedtuple("Data", ["stan_data",
                               "stan_inits",
                               "stationids",
                               "stationids_idx",
                               "predictors",
                               "predictand", "theta"])
    return Data(stan_data, stan_inits, stationids, sid_idx,
                preds, y, theta)


def process(config, script_paths, logger, data):
    # Configuration of task
    task = config.task
    parname = task.parname
    npred = task.npred
    marginal = task.marginal
    logger.info(f"Sampling {parname}-{npred}")

    fstan = script_paths.fout / f"stan_{parname}_NP{npred}"
    fstan.mkdir(exist_ok=True)
    for f in fstan.glob("*.*"):
        f.unlink()

    smp = gls_spatial_sampling(data=data.stan_data,
                               inits=data.stan_inits,
                               seed=config.seed,
                               iter_warmup=config.nwarm,
                               iter_sampling=config.nsamples//config.nchains,
                               chains=config.nchains,
                               parallel_chains=config.nchains,
                               show_progress=config.debug,
                               output_dir=fstan)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    # zip stan data
    #base_name = str(fstan)
    #shutil.make_archive(base_name, base_dir=base_name, \
    #                    root_dir=base_name, format="zip")
    # clean
    shutil.rmtree(fstan)

    # Prediction for target sites
    logger.info(f"Generating data for parameter {parname} with npred={npred}")
    ypred = gls.generate(data.stan_data, df, True)

    # Storage
    logger.info(f"Storing data for parameter {parname} with npred={npred}")
    priors = []
    pname = parname.lower()
    pname = f"log{pname}" if parname == "LOCN" else pname
    opts = task.to_dict()["options"]
    for i, isite in enumerate(data.stationids_idx):
        if parname == "LOCN":
            # Transform locn to original scale
            values = np.exp(ypred[:, isite])
            pred = np.exp(data.predictand.iloc[isite])
        else:
            values = ypred[:, isite]
            pred = data.predictand.iloc[isite]

        dd = {
            "STATIONID": data.stationids[i],
            "TASKID": config.taskid,
            "MARGINAL": marginal,
            "NPREDICTORS": npred,
            "PARAMETER": pname,
            "PRIOR_MEAN": float(values.mean().round(3)),
            "PRIOR_STD": float(values.std(ddof=1).round(3)),
            "PREDICTAND": pred,
            "PREDICTORS": "/".join(data.predictors)
        }
        for m in config.stan_diag_metrics:
            dd[f"STAN_DIAG_{m}"] = diag[m]

        # .. store GLS params
        bn = [cn for cn in df.columns if re.search("^beta", cn)]
        for pn in ["logrho", "logalpha", "logsigma"]+bn:
            se = df.loc[:, pn]
            pn2 = re.sub("\\[|\\]", "", pn).upper()
            dd[f"GLS_{pn2}_MEAN"] = float(se.mean().round(3))
            dd[f"GLS_{pn2}_STD"] = float(se.std().round(3))

        priors.append(dd)

    priors = pd.DataFrame(priors)

    # To disk
    fp = script_paths.fout / f"priors_TASK{taskid}_{parname}_{marginal}_NP{npred}.csv"
    comment = f"Prior data for variable {parname} and marginal {marginal}."
    csv.write_csv(priors, fp, comment,
                  script_paths.source_file,
                  compress=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute regional priors for parameters",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    parser.add_argument("-t", "--taskid", help="JobID",
                        type=int, default=-1)
    parser.add_argument("-s", "--seed", help="Random seed",
                        type=int, default=5446)
    parser.add_argument("-dm", "--dist_max", help="Maxsimum distance",
                        type=float, default=1000)
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
    seed = args.seed
    dist_max = 500 if debug else args.dist_max

    stan_diag_metrics = ["treedepth", "rhat", "ebfmi", "effsamplesz"]

    if debug:
        nwarm = 1000
        nsamples = 5000
        nchains = 3
        npreds = [2]
    else:
        nwarm = 10000
        nsamples = 10000
        nchains = 10
        npreds = [0, 1, 2, 3]

    marginals = ["GEV"]
    parnames = ["LOCN", "LOGSCALE", "SHAPE1"]
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(marginal=marginals,
                               npred=npreds,
                               parname=parnames)

    Config = namedtuple("Config",
                        ["version", "taskid",
                        "overwrite", "debug", "task",
                        "seed", "nwarm", "nchains",
                        "nsamples", "dist_max",
                        "stan_diag_metrics"])
    config = Config(version, taskid, overwrite,
                    debug, opm.get_task(taskid),
                    seed, nwarm, nchains, nsamples,
                    dist_max, stan_diag_metrics)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)
    logger.info(f"Number of tasks : {opm.ntasks}")
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
