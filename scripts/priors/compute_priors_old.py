#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2022-12-12 Mon 11:48 AM
## Comment : Generate priors
##
## ------------------------------


import sys, os, re, json, math
import argparse
import shutil
from pathlib import Path
from itertools import product as prod

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from hydrodiy.io import csv, iutils
from hydrodiy.io.hyruns import OptionManager, SiteBatch
from hydrodiy.stat import sutils

from tqdm import tqdm

from nrivdata import dataset

from nrivfloodfreqstan import sample, marginals, gls_spatial, gls

SEED = 5446

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="Generate priors", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--nbatch", help="Number of batch processes", \
                    type=int, default=1)
parser.add_argument("-v", "--version", help="AMS fit version number", \
                    type=int, required=True)
parser.add_argument("-t", "--taskid", help="Task id", \
                    type=int, required=True)
parser.add_argument("-p", "--progress", help=" Show progress", \
                    action="store_true", default=False)
parser.add_argument("-d", "--debug", help="Debug mode",\
                    action="store_true", default=False)
args = parser.parse_args()

taskid = args.taskid
progress = args.progress
debug = args.debug
nbatch = args.nbatch
version = args.version
data_version = "1.0"

if debug:
    nwarm = 100
    nsamples = 100
    nchains = 2
else:
    nwarm = 10000
    nsamples = 10000
    nchains = 5

# Get option manager
opm = OptionManager("manager", \
            nbatch=nbatch, \
            dataversion=data_version)

npreds = [0, 1, 2, 3]
batches = np.arange(nbatch).tolist()
marginals = ["GEV", "LogPearson3", "Gumbel", "LogNormal", "Normal"]
varnames = ["STREAMFLOW", "PRECIPITATION",
       "PRECIPITATION2days", "PRECIPITATION3days",
       "PRECIPITATION_CATCHMAX", "QTOT", "QTOT2days", "QTOT3days", "SM",
       "SMPFULL", "SM1daybefore", "SMPFULL1daybefore", "SM2daysbefore",
       "SMPFULL2daysbefore"]

opm.from_cartesian_product(\
    marginal = marginals, \
    npred=npreds, \
    varname = varnames, \
    batch=batches)

# Get task
task = opm.get_task(taskid)
varname = task.varname
marginal = task.marginal
batch = task.batch
npred = task.npred

predictors = ["CATCHMENTAREA_ORIGINAL+SRTM[km2]", \
                "RATIO_DISTANCE_VS_SQRTAREA[adim]", \
                "XCENTER_EPSG28356[m]", "YCENTER_EPSG28356[m]", \
                "ALTITUDE[m]"]

if not re.search("PRECIP", varname):
    predictors += ["PRECIPITATION_2YEARDAILY_1911_2021[mm/day]", \
                        "PRECIPITATION_5YEARDAILY_1911_2021[mm/day]", \
                        "PRECIPITATION_IFD_DUR360_AEP2[mm]", \
                        "PRECIPITATION_IFD_DUR360_AEP50[mm]", \
                        "PRECIPITATION_ANNUAL_MEAN_1911_2021[mm]", \
                        "NETPRECIPITATION_ANNUAL_MEAN_1911_2021[mm]"]

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "data" / f"priors_version{version}" / \
                            f"priors_{varname}_{marginal}_NP{npred}"
fout.mkdir(exist_ok=True, parents=True)

flogs = froot / "logs" / "priorfit"
flogs.mkdir(exist_ok=True, parents=True)
flog = flogs / f"priorfit_TASK{taskid}_version{version}.log"

basename = source_file.stem
LOGGER = sample.get_logger(level="INFO", flog=flog, stan_logger=False)
LOGGER.info("Process started")

task.log(LOGGER)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
zcov = dataset.ZDataset(provider="CSIRO", \
                folder=fout.parent.parent, \
                name="covpaperdata", \
                version=data_version)

datapack = "covpaperams"
stations, _ = zcov.read_tabular_data(datapack, "stations")

sb = SiteBatch(stations.index, opm.context["nbatch"])
stationids = sb[batch]

if debug:
    stationids = stationids[:2]

# .. store station ids in option manager for easy retrieval
opm.context["stationids"] = stations.index.tolist()
fopm = fout.parent / "optionmanager.json"
if not fopm.exists():
    with fopm.open("w") as fo:
        json.dump(opm.to_dict(), fo, indent=4)

# LH parameters
fp = fout.parent.parent / "lhmom_params.csv"
params, _ = csv.read_csv(fp)
# .. restrict to options
idx = (params.loc[:, "MARGINAL[undef]"] == marginal)\
        & (params.loc[:, "VARIABLE[undef]"] == varname)
params = params.loc[idx, :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
priors = []

for parname in ["LOCN", "LOGSCALE", "SHAPE1"]:
    # Skip shape for 2 params marginals
    if marginal in ["Normal", "Gumbel", "LogNormal"] and parname == "SHAPE1":
        continue

    yfull = params.filter(regex=f"STATIONID|PARAM_{parname}", axis=1).copy()
    yfull.loc[:, "STATIONID"] = yfull.STATIONID.astype(str)
    yfull = yfull.set_index("STATIONID").squeeze()


    # get predictors
    xfull = np.log(stations.loc[yfull.index, predictors])
    xfull.loc[:, "INTERCEPT"] = 1

    if npred>0:
        # OLS regression to reduce predictor list
        theta, fstat, fpvalue, _ = sutils.lstsq(xfull, yfull)
        preds = theta.tpvalue.sort_values().index.tolist()[:npred]
    else:
        preds = ["INTERCEPT"]

    x = xfull.loc[:, preds]

    # Get centered coordinates
    w = stations.loc[yfull.index, ["XCENTER_EPSG28356[m]", "YCENTER_EPSG28356[m]"]]
    w *= 1e-3

    # .. prior on rho based on min dist
    d = pd.Series(pdist(w))
    logrho_prior = [math.log(d.mean())]*2
    logrho_lower = math.log(d.min()/2)
    logrho_upper = math.log(2*d.max())

    # loop on stations
    for stationid in stationids:
        LOGGER.info(f"Prior fitting {parname}-{stationid}")

        y = yfull.copy()
        y.loc[stationid] = np.nan

        stan_data = gls.prepare(x, w, y, \
                                logrho_prior=logrho_prior, \
                                logrho_lower=logrho_lower, \
                                logrho_upper=logrho_upper, \
                                logalpha_prior=[2, 5], \
                                logsigma_prior=[2, 5])

        fstan = fout / f"stan_{parname}_{stationid}_NP{npred}"
        for f in fstan.glob("*.*"):
            f.unlink()

        smp = gls_spatial.sample(\
            data=stan_data, \
            seed=SEED, \
            iter_warmup=nwarm, \
            iter_sampling=nsamples//nchains, \
            chains=nchains, \
            show_progress=debug, \
            output_dir=fstan)

        df = smp.draws_pd()
        diag = sample.format_stan_diagnostic(smp.diagnose())

        # zip and clean
        base_name = str(fstan)
        shutil.make_archive(base_name, base_dir=base_name, \
                                root_dir=base_name, format="zip")
        shutil.rmtree(base_name)

        # Prediction for missing site
        ys = gls.generate(stan_data, df, True)
        ypred = ys[:, pd.isnull(y)].squeeze()

        # XV target
        ytarget = float(yfull[pd.isnull(y)].squeeze().round(3))

        # Storage
        dd = {
            "STATIONID_XV": stationid, \
            "PARAMETER": parname.lower(), \
            "PRIOR_MEAN": float(ypred.mean().round(3)), \
            "PRIOR_STD": float(ypred.std(ddof=1).round(3)), \
            "Y_XVTARGET": ytarget, \
            "VARIABLE": varname, \
            "DIAGNOSTIC": diag, \
            "MARGINAL": marginal, \
            "PREDICTORS": "/".join(preds), \
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


LOGGER.info("Process completed")

