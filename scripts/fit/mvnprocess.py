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

from floodstan.marginals import GEV

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
                    type=int, default=50)
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
    "GALL": datahub.get_stations().index.astype(str).tolist(),
    "G02-14-10": ["203002", "203014", "203010"],
    "G02-14": ["203002", "203014"]
    }

if debug:
    groups_mvn_cdf = {k: v for k, v in groups_mvn_cdf.items()
                      if k == "GALL"}

# Runner
opm = hyruns.OptionManager(stationid_cond=stationid_cond,
                           eep_target=eep_target,
                           zcdf=zcdf)
# Select fit task with
pcensors = [0.3, 0.5]
rho_mins = [-1, 0]
excludes = ["NONE"]

opm.from_cartesian_product(batch=np.arange(nbatch),
                           pcensor=pcensors,
                           rho_min=rho_mins,
                           exclude=excludes)

# Load task
task = opm.get_task(max(0, taskid))
batch = task.batch
pcensor = task.pcensor
exclude = task.exclude
rho_min = task.rho_min

# Frequency of log report
iterlog = 5

if debug:
    batch = 0
    pcensor = 0.5
    exclude = "NONE"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fopm = froot / "outputs" / "copulafit_options.json"
opm_fit = hyruns.OptionManager.from_file(fopm)
fit_taskid = opm_fit.search(pcensor=pcensor,
                            rho_min=rho_min,
                            exclude=exclude)[0]

ftask = froot / "outputs" / f"copulafit_TASK{fit_taskid}"

fwrite = ftask / "mvnprocess"
fwrite.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / basename / f"{basename}_TASK{taskid}.log"
flog.parent.mkdir(exist_ok=True, parents=True)
LOGGER = iutils.get_logger(basename, flog=flog, console=debug,
                           contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")
task.log(LOGGER)

if debug:
    fwrite = froot / "logs" / basename / "mvnprocess"
    fwrite.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

# Obs events
obs = [event for event in opm_fit.options["exclude"] if event != "NONE"]
if debug:
    obs = ["2022-02-27"]

# Peaks
potpeaks, _ = datahub.get_potpeaks()

fd = ftask / f"copulafit_diagnostic_TASK{fit_taskid}.json"
with fd.open("r") as fo:
    diag = json.load(fo)

fd = ftask / f"copulafit_data_TASK{fit_taskid}.json"
with fd.open("r") as fo:
    data = json.load(fo)

nvar = data["P"]
stationids = np.array(data["stationids"])

def get_station_index(sid):
    return np.where(sid == stationids)[0][0]

icond_1 = np.where(stationids == stationid_cond)[0]
icond_2 = np.where(stationids != stationid_cond)[0]

LOGGER.info(f"Load samples TASK {fit_taskid}")
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

gev = GEV()

nsamples = len(samples)
sidc = stationid_cond
cols_mu = [f"mvn_cond{sidc}_{sid}_mu" for sid in stationids[icond_2]]
cols_sig = [f"mvn_cond{sidc}_{sid}_sig" for sid in stationids[icond_2]]
cols_smp = [f"mvn_cond{sidc}_{sid}_smp_cdf" for sid in stationids[icond_2]]

gsta = [f"G{sid}" for sid in stationids]
cols_obs = [f"{g}_obs_log10eep_{event}" for event in obs
            for g in list(groups_mvn_cdf.keys()) + gsta]

stats = ["log10pall_eeptarget", "log10pany_eeptarget"]

cols = [f"{g}_{v}" for g in groups_mvn_cdf for v in stats]\
    + cols_mu + cols_sig + cols_smp + cols_obs

res = pd.DataFrame(np.nan, index=samples.index,
                   columns=cols)

for ismp, (i, smp) in enumerate(samples.iterrows()):
    if i % iterlog == 0:
        LOGGER.info(f"Processing sample {ismp + 1} / {nsamples}")

    cor_all = smp.filter(regex="cor_IW").values.reshape((nvar, nvar)).T

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
    for gname, grp_stationids in groups_mvn_cdf.items():
        idx = [get_station_index(sid) for sid in grp_stationids]
        ngstations = len(grp_stationids)
        mean = np.zeros(ngstations)

        # MVN CDF
        cor = cor_all[idx][:, idx]

        #LOGGER.info("Computing probs", ntab=1)
        rv = mvn(mean=mean, cov=cor)

        # All above threshold
        x = -zcdf * np.ones(ngstations)
        pall = rv.cdf(x)
        lpall = math.log10(pall) if pall > 0 else np.nan
        res.loc[i, f"{gname}_log10pall"] = lpall

        # Any above threshold
        x = zcdf * np.ones(ngstations)
        pany = 1 - rv.cdf(x)
        lpany = math.log10(pany) if pany > 0 else np.nan
        res.loc[i, f"{gname}_log10pany"] = lpany

        # Obs eep
        for event in obs:
            # Get peak flow data
            pp = potpeaks.loc[event].squeeze()
            zstd = np.nan * np.zeros(ngstations)

            # Compute cdf for each station
            # Careful here, don't mix up the station
            # index sid within the stan data list
            # and the number k used to order them within zstd
            for k, sid in enumerate(grp_stationids):
                isid = get_station_index(sid) + 1
                ylocn = smp.loc[f"ylocn[{isid}]"]
                ylogscale = smp.loc[f"ylogscale[{isid}]"]
                yshape1 = smp.loc[f"yshape1[{isid}]"]
                gev.params = [ylocn, ylogscale, yshape1]
                qo = pp[sid]
                if ~np.isnan(qo):
                    cdf = gev.cdf(qo)
                    # Store individual estimate of event
                    if gname == "GALL":
                        lc = math.log10(1 - cdf)
                        res.loc[i, f"G{sid}_obs_log10eep_{event}"] = lc

                    zstd[k] = norm.ppf(cdf)

            lc = math.log10(rv.cdf(-zstd))
            res.loc[i, f"{gname}_obs_log10eep_{event}"] = lc

# Save data to disk
fr = fwrite / f"copulafit_mvnprocess_TASK{fit_taskid}_BATCH{batch}.csv"

comments = {
    "comment": "MVN process results",
    "exclude": exclude,
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
