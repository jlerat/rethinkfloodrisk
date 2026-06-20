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
import shutils
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"

    version = config.version
    fout = froot / "outputs" / "regional_priors_v{version}"
    flogs = froot / "logs" / "regional_priors"

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
    fl = script_paths.flogs / f"regional_priors_TASK{taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def process(config, script_paths, logger):
    # Configuration of task
    task = config.task
    parname = task.parname
    npred = task.npred
    logger.info(f"Prior fitting {parname}-{npred}")

    # Get data
    data = datahub.get_params_lh_moments()

    # Prepare stan
    stan_data = gls.prepare(x, w, y,
                            logrho_prior=logrho_prior,
                            logrho_lower=logrho_lower,
                            logrho_upper=logrho_upper,
                            logalpha_prior=[2, 5],
                            logsigma_prior=[2, 5])

    fstan = fout / f"stan_{parname}_NP{npred}"
    fstan.mkdir(exist_ok=True)
    for f in fstan.glob("*.*"):
        f.unlink()

    smp = gls_spatial.sample(
        data=stan_data,
        seed=config.seed,
        iter_warmup=config.nwarm,
        iter_sampling=config.nsamples//config.nchains,
        chains=config.nchains,
        parallel_chains=config.nchains,
        show_progress=config.debug,
        output_dir=fstan)

    df = smp.draws_pd()
    diag = sample.format_stan_diagnostic(smp.diagnose())

    # zip and clean
    base_name = str(fstan)
    shutil.make_archive(base_name, base_dir=base_name, \
                            root_dir=base_name, format="zip")
    shutil.rmtree(base_name)

    # Prediction for target sites
    ys = gls.generate(stan_data, df, True)
    ypred = ys[:, pd.isnull(y)].squeeze()

    # XV target
    ytarget = float(yfull[pd.isnull(y)].squeeze().round(3))

    # Storage
    dd = {
        "PARAMETER": parname.lower(),
        "PRIOR_MEAN": float(ypred.mean().round(3)),
        "PRIOR_STD": float(ypred.std(ddof=1).round(3)),
        "Y_XVTARGET": ytarget,
        "VARIABLE": varname,
        "DIAGNOSTIC": diag,
        "MARGINAL": marginal,
        "PREDICTORS": "/".join(preds),
    }

    # .. store GLS params
    bn = [cn for cn in df.columns if re.search("^beta", cn)]
    for pn in ["logrho", "logalpha", "logsigma"]+bn:
        se = df.loc[:, pn]
        pn2 = re.sub("\[|\]", "", pn).upper()
        dd[f"GLS_{pn2}_MEAN"] = float(se.mean().round(3))
        dd[f"GLS_{pn2}_STD"] = float(se.std().round(3))

    priors.append(dd)

    # To disk
    df = pd.DataFrame(priors)
    fp = fout.parent / f"priors_{varname}_{marginal}_NP{npred}.csv"
    comment = f"Prior data for variable {varname} and marginal {marginal}."
    csv.write_csv(df, fp, comment, \
            source_file, compress=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute regional priors for parameters",
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

    marginals = ["GEV"]
    npreds = [0, 1, 2, 3]
    parnames = ["LOCN", "LOGSCALE", "SHAPE1"]:
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(marginal=marginal,
                               npred=npreds,
                               parname=parnames)

    Config = namedtuple("Config",
                        ["version", "taskid",
                        "overwrite", "debug", "task"])
    config = Config(version, taskid, overwrite,
                    debug, opm.get_task(taskid))

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Process
    process(config, script_paths, logger)

    logger.completed()
