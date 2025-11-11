#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
import re
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

from hydrodiy.io import csv, iutils, hyruns

from pyrethink import datahub

np.random.seed(5446)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process mvn samples",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of batches",
                    type=int, default=800)
args = parser.parse_args()
taskid = args.taskid
nbatch = args.nbatch

debug = taskid < 0

# Configure mvn conditional
stationid_cond = "203002"
eep_target = 1 - 1e-2
zcond = np.atleast_1d(norm.ppf(eep_target))

# Configure mvn cdf
zcdf = norm.ppf(eep_target)

groups_mvn_cdf = {
    "all": datahub.get_stations().index.astype(str).tolist(),
    "around-repentance": ["203002", "203014", "203010"]
    }


# Select fit task with
# pcens = 0.3
# period = 'ALL'
fit_taskid = 2

# Runner
opm = hyruns.OptionManager(fit_taskid=fit_taskid,
                           stationid_cond=stationid_cond,
                           eep_target=eep_target,
                           zcdf=zcdf)
opm.from_cartesian_product(batch=np.arange(nbatch))

# Load task
task = opm.get_task(max(0, taskid))
batch = task.batch

# Frequency of log report
iterlog = 5

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

ftask = froot / "outputs" / f"copulafit_TASK{fit_taskid}"
fwrite = ftask / "mvnprocess"
fwrite.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / basename / f"{basename}_TASK{fit_taskid}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
LOGGER = iutils.get_logger(basename, flog=flog, console=debug,
                           contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")

if debug:
    fwrite = froot / "logs" / basename / "mvnprocess"
    fwrite.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")
fd = ftask / f"copulafit_diagnostic_TASK{fit_taskid}.json"
with fd.open("r") as fo:
    diag = json.load(fo)

period = diag["timeperiod"]
pcensor = diag["pcensor"]

fd = ftask / f"copulafit_data_TASK{fit_taskid}.json"
with fd.open("r") as fo:
    data = json.load(fo)

nvar = data["P"]
stationids = np.array(data["stationids"])
icond_1 = np.where(stationids == stationid_cond)[0]
icond_2 = np.where(stationids != stationid_cond)[0]

LOGGER.info(f"Load report TASK {fit_taskid} period={period}")
fs = ftask / f"copulafit_samples_TASK{fit_taskid}.zip"
samples = pd.read_csv(fs, skiprows=15)

# Get samples from batch
nsamples = len(samples)
idx = hyruns.get_batch(nsamples, nbatch, batch)
samples = samples.iloc[idx]

if debug:
    samples = samples.iloc[:5]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
nsamples = len(samples)
sidc = stationid_cond
cols_mu = [f"mvn_cond{sidc}_{sid}_mu" for sid in stationids[icond_2]]
cols_sig = [f"mvn_cond{sidc}_{sid}_sig" for sid in stationids[icond_2]]
cols_smp = [f"mvn_cond{sidc}_{sid}_smp_cdf" for sid in stationids[icond_2]]

stats = ["log10_pall", "log10_pall_num",
         "log10_pany", "log10_pany_num"]
cols = [f"{g}_{v}" for g in groups_mvn_cdf for v in stats]\
    + cols_mu + cols_sig + cols_smp

res = pd.DataFrame(np.nan, index=samples.index,
                   columns=cols)

for i, smp in samples.iterrows():
    if i % iterlog == 0:
        LOGGER.info(f"Processing sample {i + 1} / {nsamples}")

    L_cor = smp.filter(regex="L_cor").values.reshape((nvar, nvar)).T
    cor_all = L_cor @ L_cor.T
    k = np.arange(nvar)
    cor_all[k, k] = 1.

    # MVN conditional
    S11 = cor_all[icond_1][:, icond_1]
    S11i = np.linalg.inv(S11)
    S22 = cor_all[icond_2][:, icond_2]
    S21 = cor_all[icond_2][:, icond_1]

    muc = S21 @ S11i @ zcond
    Sc = S22 - S21 @ S11i @ S21.T
    z = np.random.multivariate_normal(mean=muc, cov=Sc)

    res.loc[i, cols_mu] = muc
    res.loc[i, cols_sig] = np.sqrt(np.diag(Sc))
    res.loc[i, cols_smp] = norm.cdf(z)

    # Loop on groups
    for gname, stationids in groups_mvn_cdf.items():
        idx = [data["stationids"].index(sid) for sid in stationids]
        nstations = len(stationids)
        mean = np.zeros(nstations)

        # MVN CDF
        cor = cor_all[idx][:, idx]

        #LOGGER.info("Sampling normals", ntab=1)
        z = np.random.multivariate_normal(mean=mean, cov=cor, size=10000000)
        nz = len(z)

        #LOGGER.info("Computing probs", ntab=1)
        rv = mvn(mean=mean, cov=cor)

        # All above threshold
        x = -zcdf * np.ones(nstations)
        pall = rv.cdf(x)
        lpall = math.log10(pall) if pall > 0 else np.nan
        res.loc[i, f"{gname}_log10_pall"] = lpall

        pall_num = np.all(z - zcdf > 0, axis=1).sum() / nz
        lpall = math.log10(pall_num) if pall > 0 else np.nan
        res.loc[i, f"{gname}_log10_pall_num"] = lpall

        # Any above threshold
        x = zcdf * np.ones(nstations)
        pany = 1 - rv.cdf(x)
        lpany = math.log10(pany) if pany > 0 else np.nan
        res.loc[i, f"{gname}_log10_pany"] = lpany

        pany_num = np.any(z - zcdf > 0, axis=1).sum() / nz
        lpany = math.log10(pany_num) if pany_num > 0 else np.nan
        res.loc[i, f"{gname}_log10_pany_num"] = lpany

# Save data to disk
fr = fwrite / f"copulafit_mvnprocess_TASK{fit_taskid}_BATCH{batch}.csv"

comments = {
    "comment": "MVN process results",
    "period": period,
    "pcensor": pcensor,
    "fit_taskid": fit_taskid,
    "stationid_cond": stationid_cond,
    "eep_target": eep_target,
    "zcdf": zcdf
    }
csv.write_csv(res, fr, comments,
              source_file,
              write_index=True)

LOGGER.completed()
