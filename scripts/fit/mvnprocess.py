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
from pyrethink.sample import Partitions

np.random.seed(5446)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process mvn samples",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--version", help="version",
                    type=int, required=True)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of batches",
                    type=int, default=20)
args = parser.parse_args()
version = args.version
taskid = args.taskid
nbatch = args.nbatch

debug = taskid < 0

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
copulas = [0, 1.5, 2, 3, 4]
excludes = ["NONE"]
has_clusters_all = [True, False]

opm.from_cartesian_product(batch=np.arange(nbatch),
                           pcensor=pcensors,
                           rho_min=rho_mins,
                           copula=copulas,
                           has_clusters=has_clusters_all,
                           exclude=excludes)

# Load task
task = opm.get_task(max(0, taskid))
batch = task.batch
pcensor = task.pcensor
exclude = task.exclude
rho_min = task.rho_min
copula = task.copula
has_clusters = task.has_clusters

# Frequency of log report
iterlog = 5

if debug:
    batch = 0
    pcensor = 0.3
    exclude = "2007"
    copula = 1
    has_clusters = True
    rho_min = 0

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
if len(fit_taskid) != 1:
    errmsg = "opm_fit does not return one task"
    raise ValueError(errmsg)

fit_taskid = fit_taskid[0]

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

if debug:
    fit_taskid = -1
    fwrite = froot / "logs" / basename / f"copulafit_v{version}" \
        / "mvnprocess"
    fwrite.mkdir(exist_ok=True, parents=True)

    ftask = froot / "logs" / "copulafit" / f"copulafit_v{version}"

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

icond_1 = np.where(stationids == stationid_cond)[0]
icond_2 = np.where(stationids != stationid_cond)[0]

# Partitions
parts = Partitions(nvar)

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
stationids_conditional = stationids[icond_2]

cols_cond = {}
cols_cond_all = []
for st, p in prod(["mu", "sig", "smp_cdf"], aep_targets):
    cc = [f"mv_cond{sidc}_p{p:0.02f}_{sid}_{st}"
          for sid in stationids_conditional]
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

for ismp, (i, smp) in enumerate(samples.iterrows()):
    if i % iterlog == 0:
        LOGGER.info(f"Processing sample {ismp + 1} / {nsamples}")

    # retrieve correlation matrix
    corr_all = smp.filter(regex="corr_IW").values.reshape((nvar, nvar)).T

    # retrieve partition
    ipartition = int(smp.filter(regex="ipart").squeeze())
    cnt = parts.subsets_counts[ipartition]
    part = parts.subsets[ipartition][:cnt]

    # MV dist conditional
    S11 = corr_all[icond_1][:, icond_1]
    S11i = np.linalg.inv(S11)
    S22 = corr_all[icond_2][:, icond_2]
    S21 = corr_all[icond_2][:, icond_1]

    for aep_target in aep_targets:
        if copula > 0:
            zcond = student_t.ppf([aep_target], df=copula)
        else:
            zcond = norm.ppf([aep_target])

        muc = S21 @ S11i @ zcond
        Sc = S22 - S21 @ S11i @ S21.T

        if copula > 0:
            # See https://en.wikipedia.org/wiki/Multivariate_t-distribution#Conditional_Distribution
            nu = copula
            p1 = len(zcond)
            d1 = zcond.T @ S11i @ zcond
            a = (nu + d1) / (nu + p1)
            df = nu + p1
            z = mvt.rvs(loc=muc, shape=a * Sc, df=df)
        else:
            z = mvn.rvs(mean=muc, cov=Sc)

        for st in ["mu", "sig", "smp_cdf"]:
            cc = [f"mvn_cond{stationid_cond}_p{aep_target:0.02f}_{sid}_{st}"
                  for sid in stationids_conditional]
            if st == "mu":
                values = muc
            elif st == "sig":
                values = np.sqrt(np.diag(Sc))
            else:
                if copula > 0:
                    values = student_t.cdf(z, df=copula)
                else:
                    values = norm.cdf(z)

            res.loc[i, cc] = values

    # Loop on groups
    for gname, grp_stationids in groups_mvn_cdf.items():
        idx = [get_station_index(sid) for sid in grp_stationids]
        ngstations = len(grp_stationids)
        mean = np.zeros(ngstations)

        # MV CDF
        corr = np.ascontiguousarray(corr_all[idx][:, idx])

        #LOGGER.info("Computing probs", ntab=1)
        if copula > 0:
            rv = mvt(loc=mean, shape=corr, df=copula)
        else:
            rv = mvn(mean=mean, cov=corr)

        for aep_target in aep_targets:
            if copula > 0:
                zcdf = student_t.ppf(aep_target, df=copula)
            else:
                zcdf = norm.ppf(aep_target)

            # All above threshold
            x = -zcdf * np.ones(ngstations)
            pall = rv.cdf(x)
            lpall = math.log10(pall) if pall > 0 else np.nan
            res.loc[i, f"{gname}_log10pall_aeptarget_p{aep_target:0.02f}"] = lpall

            # Any above threshold
            x = zcdf * np.ones(ngstations)
            pany = 1 - rv.cdf(x)
            lpany = math.log10(pany) if pany > 0 else np.nan
            res.loc[i, f"{gname}_log10pany_aeptarget_p{aep_target:0.02f}"] = lpany

        # Obs aep
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
                        res.loc[i, f"G{sid}_obs_log10aep_{event}"] = lc

                    if copula > 0:
                        zstd[k] = student_t.ppf(cdf, df=copula)
                    else:
                        zstd[k] = norm.ppf(cdf)

            cdf = rv.cdf(-zstd)
            lc = math.log10(cdf) if cdf > 0 else np.nan
            res.loc[i, f"{gname}_obs_log10aep_{event}"] = lc

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
