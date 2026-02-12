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
from itertools import product as prod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt

from hydrodiy.io import csv, iutils, hyruns

from floodstan.marginals import GEV

from pyrethink import datahub
from pyrethink import sample

import importlib
importlib.reload(sample)

np.random.seed(5446)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process mvn samples",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--version", help="version",
                    type=int, required=True)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=0)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-n", "--nbatch", help="Number of batches",
                    type=int, default=20)
args = parser.parse_args()
version = args.version
taskid = args.taskid
nbatch = args.nbatch
debug = args.debug

# Configure mvn conditional
stationid_cond = "203002"
aep_targets = [1 - 1e-1, 1 - 1e-2]

# Configure mvn cdf

groups_mvn_cdf = {
    "GALL": ["203002", "203004", "203005", "203010",
             "203012", "203014"],
    "G02-14-10": ["203002", "203014", "203010"],
    "G14-10": ["203014", "203010"],
    "G02-14": ["203002", "203014"],
    "G02-04": ["203002", "203004"]
    }

if debug:
    groups_mvn_cdf = {k: v for k, v in groups_mvn_cdf.items()
                      if k == "GALL"}

# Runner
opm = hyruns.OptionManager(stationid_cond=stationid_cond,
                           aep_targets=aep_targets)

# Select certain fit tasks
pcensors = [0.3]
rho_mins = [-1]
copulas = [0, 2.01, 2.1, 3., 4.]
excludes = ["NONE"]
has_clusters_all = [True, False]
dirichlet_alphas = [1., 1.5]

opm.from_cartesian_product(batch=np.arange(nbatch),
                           pcensor=pcensors,
                           rho_min=rho_mins,
                           copula=copulas,
                           has_clusters=has_clusters_all,
                           dirichlet_alpha=dirichlet_alphas,
                           exclude=excludes)

# Load task
task = opm.get_task(max(0, taskid))
batch = task.batch
pcensor = task.pcensor
exclude = task.exclude
rho_min = task.rho_min
copula = task.copula
has_clusters = task.has_clusters
dirichlet_alpha = task.dirichlet_alpha

# Frequency of log report
iterlog = 5

if debug:
    batch = 0
    pcensor = 0.3
    exclude = "NONE"
    copula = 3.
    has_clusters = True
    rho_min = -1

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fopm = froot / "outputs" / f"copulafit_v{version}" / "copulafit_options.json"
opm_fit = hyruns.OptionManager.from_file(fopm)
fit_taskid = opm_fit.search(pcensor=f"{pcensor:0.1f}",
                            rho_min=f"{rho_min:0.1f}",
                            copula=copula,
                            has_clusters=has_clusters,
                            exclude=exclude)
fit_taskid = next(tid for tid in fit_taskid
                  if opm_fit.get_task(tid)["copula"] == copula)

ftask = froot / "outputs" / f"copulafit_v{version}" / f"copulafit_TASK{fit_taskid}"

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

LOGGER.info("Fit task:")
opm_fit.get_task(fit_taskid).log(LOGGER)

if debug:
    fwrite = froot / "logs" / basename / f"copulafit_v{version}" \
        / "mvnprocess"
    fwrite.mkdir(exist_ok=True, parents=True)

    #ftask = froot / "logs" / "copulafit" / f"copulafit_v{version}"

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

# Obs events
ams, _, _, stations = datahub.get_ams_concat()
potpeaks, _, _ = datahub.get_potpeaks()

rk = potpeaks.rank(ascending=False)
obs = rk.index[(rk <= 2).any(axis=1)].astype(str).tolist()

if debug:
    obs = ["2022-02-27", "2022-03-30"]

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

icond = np.where(stationids == stationid_cond)[0]
itarget = np.where(stationids != stationid_cond)[0]

# Sampler
ccs = sample.CopulaSampling(copula, nvar)
pids = np.array(data["partitions_id"])
probs = ccs.partitions.compute_probabilities(pids,
                                             dirichlet_alpha)

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

cols_cond = {}
cols_cond_all = []
sidc = stationid_cond
for st, p in prod(["smp_cdf"], aep_targets):
    cc = [f"mv_cond{sidc}_p{p:0.02f}_{sid}_{st}"
          for sid in stationids if sid != sidc]
    cols_cond[(st, p)] = cc
    cols_cond_all.extend(cc)

gsta = [f"G{sid}" for sid in stationids]
cols_obs = [f"{g}_obs_log10aep_{event}" for event in obs
            for g in list(groups_mvn_cdf.keys()) + gsta]

stats = [f"{st}_p{p:0.02f}"
         for st in ["log10pall_aeptarget", "log10pany_aeptarget"]
         for p in aep_targets]

cols = [f"{g}_{v}" for g in groups_mvn_cdf for v in stats]\
    + cols_cond_all

res = pd.DataFrame(np.nan, index=samples.index,
                   columns=cols)

iparts = ccs.partitions.sample(probs, len(samples))

for ismp, (i, smp) in enumerate(samples.iterrows()):
    if i % iterlog == 0:
        LOGGER.info(f"Processing sample {ismp + 1} / {nsamples}")

    # retrieve correlation matrix
    corr = smp.filter(regex="corr_IW").values.reshape((nvar, nvar)).T
    ccs.corr = corr

    # Get random partition
    ipartition = iparts[ismp]

    # Sample conditional
    for aep_target in aep_targets:
        zcond = sample.copula_marginal_ppf(copula, [aep_target])
        z = ccs.conditional_sample_given_ipart(ipartition, icond, zcond,
                                               itarget)
        u = sample.copula_marginal_cdf(copula, z)
        sid = stationid_cond
        cc = [f"mv_cond{sidc}_p{aep_target:0.02f}_{sid}_smp_cdf"
              for sid in stationids if sid != sidc]
        res.loc[i, cc] = u

    # Loop on groups
    for gname, grp_stationids in groups_mvn_cdf.items():
        grp_idx = [get_station_index(sid) for sid in grp_stationids]

        for aep_target in aep_targets:
            zcdf = sample.copula_marginal_ppf(copula, aep_target)
            z = -zcdf * np.ones(ccs.nstations)
            pall = ccs.cdf_given_ipart(ipartition, z, grp_idx)
            lpall = math.log10(pall) if pall > 0 else np.nan
            res.loc[i, f"{gname}_log10pall_aeptarget_p{aep_target:0.02f}"] = lpall

            # Any above threshold
            z = zcdf * np.ones(ccs.nstations)
            pany = 1 - ccs.cdf_given_ipart(ipartition, z, grp_idx)
            lpany = math.log10(pany) if pany > 0 else np.nan
            res.loc[i, f"{gname}_log10pany_aeptarget_p{aep_target:0.02f}"] = lpany

        # Obs aep
        for event in obs:
            # Get peak flow data
            pp = potpeaks.loc[event].squeeze()
            zev = np.nan * np.zeros(ccs.nstations)

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
                        res.loc[i, f"G{sid}_obs_log10aep_{event}"] = lc

                    zev[isid - 1] = sample.copula_marginal_ppf(copula, cdf)

            pevent = ccs.cdf_given_ipart(ipartition, zev, grp_idx)
            lpev = math.log10(pevent) if pevent > 0 else np.nan
            res.loc[i, f"{gname}_obs_log10aep_{event}"] = lpev

# Save data to disk
fr = fwrite / f"copulafit_mvnprocess_TASK{fit_taskid}_BATCH{batch}.csv"

comments = {
    "comment": "MV N/T process results",
    "exclude": exclude,
    "pcensor": pcensor,
    "rho_min": rho_min,
    "fit_taskid": fit_taskid,
    "stationid_cond": stationid_cond,
    "aep_targets": aep_targets

    }
csv.write_csv(res, fr, comments,
              source_file,
              write_index=True)

LOGGER.completed()
