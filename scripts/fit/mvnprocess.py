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
from pyrethink import copulas
from pyrethink import marginal_exceedance_score as mes
from pyrethink.marginal_exceedance_score import MARGINAL_EXCEEDANCE_SCORE_KINDS as MEXS_KINDS

import importlib
importlib.reload(copulas)

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
maeps = [1e-2]

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
                           maeps=maeps)

# Select certain fit tasks
pcensors = [0.3]
copula_shapes = [0, 3, 5]
excludes = ["NONE", "2021"]
rho_mins = [-1]
has_clusters_all = [False]
dirichlet_alphas = [1.]

opm.from_cartesian_product(batch=np.arange(nbatch),
                           pcensor=pcensors,
                           rho_min=rho_mins,
                           copula_shape=copula_shapes,
                           has_clusters=has_clusters_all,
                           dirichlet_alpha=dirichlet_alphas,
                           exclude=excludes)

# Load task
task = opm.get_task(max(0, taskid))
batch = task.batch
pcensor = task.pcensor
exclude = task.exclude
rho_min = task.rho_min
copula_shape = task.copula_shape
has_clusters = task.has_clusters
dirichlet_alpha = task.dirichlet_alpha

if debug:
    batch = 0
    pcensor = 0.3
    exclude = "NONE"
    copula_shape = 0
    has_clusters = False
    rho_min = -1
    dirichlet_alpha = 1.

copula_type = 1 if copula_shape > 0 else 0

if has_clusters:
    raise ValueError("Does not deal with clusters")

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fopm = froot / "outputs" / f"copulafit_v{version}" / "copulafit_options.json"
opm_fit = hyruns.OptionManager.from_file(fopm)
fit_taskid = opm_fit.search(pcensor=pcensor,
                            rho_min=rho_min,
                            copula_shape=copula_shape,
                            has_clusters=has_clusters,
                            exclude=exclude)
assert len(fit_taskid) == 1
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
LOGGER = iutils.get_logger(basename, flog=flog, console=debug)
LOGGER.log_dict(vars(args), "Command line arguments")
task.log(LOGGER)

LOGGER.info("Fit task:")
fit_task = opm_fit.get_task(fit_taskid)
fit_task.log(LOGGER)

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

ams_selected = [2021, 2016, 2000, 2007]
ams = ams.loc[ams_selected, :]

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
ccs = copulas.Copula(copula_type, copula_shape, nvar)

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

gsta = [f"G{sid}" for sid in stationids]
cols_ams = [f"G{sid}_ams_UNIV_{year}_log10aep"
            for sid in stationids
            for year in ams_selected]

cols_ams += [f"{g}_ams_{kd}_{year}_log10aep"
             for g in list(groups_mvn_cdf.keys())
             for kd in MEXS_KINDS
             for year in ams_selected]

stats = [f"CMEXS_{kd}_log10aep{1./aep:0.0f}"
         for kd in MEXS_KINDS
         for aep in maeps]
cols = [f"{g}_{v}" for g in groups_mvn_cdf for v in stats]\
       + cols_ams

res = pd.DataFrame(np.nan, index=samples.index,
                   columns=cols)

for ismp, (i, smp) in enumerate(samples.iterrows()):
    LOGGER.info(f"Processing sample {ismp + 1} / {nsamples}", nret=1)

    # retrieve correlation matrix
    corr = smp.filter(regex="corr_IW").values.reshape((nvar, nvar)).T
    ccs.corr = corr

    # Loop on groups
    for gname, grp_stationids in groups_mvn_cdf.items():
        LOGGER.info(f"CMEXS for group {gname}", ntab=1, nret=1)
        grp_idx = [get_station_index(sid) for sid in grp_stationids]
        nsids = len(grp_stationids)
        cop = mes.GaussianCopula(nsids)
        cop.logger = LOGGER

        idx = [get_station_index(sid) for sid in grp_stationids]
        cop.params = corr[idx][:, idx]

        for kind in MEXS_KINDS:
            mex = mes.MarginalExceedanceScore(kind, cop)
            mex.logger = LOGGER
            for maep in maeps:
                sc, _ = mex.common_marginal_exceedance_score(maep)
                lsc = math.log10(sc) if sc > 0 else np.nan
                res.loc[i, f"{gname}_CMEXS_{kind}_log10aep{1./maep:0.0f}"] = lsc

        # Obs aep
        for year in ams_selected:
            pp = ams.loc[year].squeeze()

            # Compute cdf for each station
            # Careful here, don't mix up the station
            # index sid within the stan data list
            # and the number k used to order them within zstd
            marg_cdfs = np.nan * np.zeros(nsids)
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
                        lc = math.log10(1 - cdf) if 1 - cdf > 0 else np.nan
                        res.loc[i, f"G{sid}_ams_UNIV_{year}_log10aep"] = lc

                    marg_cdfs[k] = cdf

            for kind in MEXS_KINDS:
                aep_event = cop.aep(marg_cdfs, kind)
                lsev = math.log10(aep_event) if aep_event > 0 else np.nan
                res.loc[i, f"{gname}_ams_{kind}_{year}_log10aep"] = lsev

            lsev = res.loc[i, f"{gname}_ams_KENDALL_{year}_log10aep"]
            ken = 10**(lsev + 2)
            LOGGER.info(f"AEP kendall for ams {year} = {ken:0.1f}%", ntab=1)

# Save data to disk
fr = fwrite / f"copulafit_mvnprocess_TASK{fit_taskid}_BATCH{batch}.csv"

comments = {
    "comment": "MV N/T process results",
    "exclude": exclude,
    "pcensor": pcensor,
    "rho_min": rho_min,
    "fit_taskid": fit_taskid,
    "stationid_cond": stationid_cond,
    "maeps": maeps

    }
if not debug:
    csv.write_csv(res, fr, comments,
                  source_file,
                  write_index=True,
                  compress=False)

LOGGER.completed()
