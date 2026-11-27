#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import math
from collections import namedtuple
from itertools import product as prod
import re
import argparse
import json
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, boxplot

from pyrethink import datahub, processing

import importlib.util

from figA_impact_of_period_on_FFA import copulafit
from figB_ffa_sum import get_logger, get_task

def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    fffa = froot / "outputs" / f"copulaprocess1_v{config.version}"
    fsum = froot / "outputs" / f"copulaprocess2_v{config.version}"

    basename = source_file.stem
    taskid = config.task.taskid
    fimg = froot / "images" / "manuscript" / basename / f"TASK{taskid}"

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fsum", "fffa", "fimg"])
    script_paths = SP(source_file, basename, froot,
                      fsum, fffa, fimg)

    fimg.mkdir(exist_ok=True)
    if config.clean:
        for f in fimg.glob("*"):
            for ff in f.glob("*.*"):
                ff.unlink()
            if f.is_dir():
                f.rmdir()
            else:
                f.unlink()

    return script_paths


def get_data(config, script_paths, logger):
    task = config.task
    taskid = task.taskid
    obs_data, _, _, stations = datahub.get_ams_concat()

    # Config FFA
    ari = config.ari
    prob = 1 - 1. / ari

    # Original FFA
    #fa = script_paths.fffa / f"TASK{taskid}" / \
    #     f"copulaprocess_ffa_TASK{taskid}.csv"
    #ffa, _ = csv.read_csv(fa, index_col=0)
    #pat = f"DESIGN_ERI{ari}\\[[1-{len(stationids)}]\\]"
    #ffa = ffa.filter(regex=pat, axis=0).T
    #ffa.columns = [f"{sid}_ARI{ari}" for sid in stationids]

    # Get sample and compute FFA for sub-sampled data
    fsc = script_paths.fimg / f"convergence_samples_ARI{ari}.csv"
    fzc = fsc.parent / f"{fsc.stem}.zip"
    fa = script_paths.fimg / f"FFA_ARI{ari}.csv"
    if not fzc.exists():
        # Original sample from multivariate model
        fss = script_paths.fsum / f"TASK{taskid}" / \
            f"copulaprocess_sum_samples_TASK{taskid}.csv"
        sum_samples, _ = csv.read_csv(fss)

        stationids = [re.sub("_SAMPLE", "", cn)
                      for cn in sum_samples.columns
                      if re.search("_SAMPLE", cn)
                        and not re.search("AWRA", cn)]

        nsmp = len(sum_samples)

        # Check FFA
        ffa = sum_samples.filter(regex="^\d.*_SAMPLE", axis=1).quantile(prob)
        ffa = pd.DataFrame(ffa).T
        csv.write_csv(ffa, fa, "FFA full sample",
                      script_paths.source_file)

        # Sub-sample and compute FFA
        idx_full = np.arange(nsmp)
        convergence_samples = []

        for nval in config.nvalues:
            logger.info(f"Resampling using {nval} values")
            smp = []
            for rep in np.arange(config.nreplicates):
                if nval == nsmp and rep > 0:
                    break
                idx_resamp = np.random.choice(idx_full,
                                              size=nval,
                                              replace=False)
                df = sum_samples.filter(regex="^\d.*_SAMPLE", axis=1).iloc[idx_resamp]
                qt = df.quantile(prob)
                qt.index = [f"{re.sub("_SAMPLE", f"_ARI{ari}", idx)}" for idx in qt.index]
                qt.name = rep
                smp.append(qt)

            smp = pd.DataFrame(smp)
            smp.loc[:, "NVAL"] = nval
            convergence_samples.append(smp)

        convergence_samples = pd.concat(convergence_samples)
        csv.write_csv(convergence_samples, fsc, "Computing FFA from sub-samples",
                      script_paths.source_file)
    else:
        logger.info(f"Opening resampling data file")
        convergence_samples, _ = csv.read_csv(fsc)

        ffa, _ = csv.read_csv(fa)

        stationids = [re.sub("_SAMPLE", "", cn)
                      for cn in ffa.columns
                      if re.search("_SAMPLE", cn)
                        and not re.search("AWRA", cn)]


    DT = namedtuple("Data", ["stations", "stationids", "obs_data",
                             "ffa", "convergence_samples"])
    return DT(stations, stationids, obs_data,
              ffa, convergence_samples)


def process(config, script_paths, loogger, data):
    stationids = data.stationids
    ffa = data.ffa
    convergence_samples = data.convergence_samples
    ari = config.ari
    use_rc = config.use_rating_curve

    # plot
    nval = convergence_samples.NVAL
    for stationid in stationids:
        if use_rc and re.search("SUM", stationid):
            continue

        logger.info(f"Plotting {stationid}")

        if use_rc:
            # Rating curve analysis
            rc, _ = datahub.get_rating_curves(stationid, True)
            rc_h = rc.loc[:, "WATERLEVEL[m]"]
            rc_q = rc.loc[:, "STREAMFLOW[m3_s-1]"]
            ipos = (rc_q > 1e-2) & (rc_h > 0)
            rc_h = rc_h.loc[ipos]
            rc_q = rc_q.loc[ipos]

        # plot
        plt.close("all")
        fz = (config.awidth, config.aheight)
        fig, ax = plt.subplots(figsize=fz,
                               layout="constrained")

        se = convergence_samples.loc[:, f"{stationid}_ARI{ari}"]
        bx = boxplot.Boxplot(data=se, by=nval,
                             show_text=True)

        if use_rc:
            # Convert Q to H
            st = bx.stats
            q = st.values.flat
            h = processing.linear_interpolation(q, rc_q, rc_h)
            st.loc[:] = h.reshape(st.shape)

        # Draw box plot
        bx.draw(ax=ax)

        # Add original FFA
        q = ffa.loc[:, f"{stationid}_SAMPLE"].squeeze()
        if use_rc:
            to_plot = processing.linear_interpolation(q, rc_q, rc_h)
        else:
            to_plot = q
        lab = "Posterior predictive from full sample"
        putils.line(ax, 1, 0, 0, to_plot, "k--", label=lab)

        if use_rc:
            ylabel = "Stage level 1:100 AEP [m]"
        else:
            ylabel = "Streamflow 1:100 AEP [m3.s-1]"

        ax.set(xlabel="Number of random samples",
               ylabel=ylabel)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.grid(axis="y")
        ax.legend(loc=1, framealpha=0.)

        # save
        cs = task.copula_spec
        fp = f"{script_paths.basename}_v{config.version}_{stationid}_ARI{ari}_{cs}.png"
        fp = script_paths.fimg / fp
        fp.parent.mkdir(exist_ok=True)
        fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Univariate FFA plots",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-a", "--ari", help="ARI selected",
                        type=int, default=100)
    parser.add_argument("-c", "--clean", help="Clean image folder",
                        action="store_true", default=False)
    parser.add_argument("-rc", "--rating_curve",
                        help="Convert flow to h using rating lean image folder",
                        action="store_true", default=False)
    parser.add_argument("-cs", "--copula_spec", help="Copula specification",
                        type=str, default="Gaussian",
                        choices=["Gaussian", "Student_3", "Student_5"])
    args = parser.parse_args()
    use_rating_curve = args.rating_curve
    ari = args.ari
    version = args.version
    copula_spec = args.copula_spec

    # Config
    CF = namedtuple("Config", ["version",
                               "awidth", "aheight", "fdpi",
                               "ptype", "ari_max",
                               "freq_plot_type", "clean",
                               "task", "ari",
                               "nreplicates", "nvalues",
                               "use_rating_curve"])
    awidth = 7
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    freq_plot_type = "gumbel"

    prior = "uninformative"
    exclude = "NONE"
    awra_covariate = True
    group = "203010-203014-203024"


    nreplicates = 1000
    nvalues = [200, 500, 1000, 2000, 5000, 10000]

    task = get_task(version, prior, copula_spec,
                    exclude, awra_covariate,
                    group)

    config = CF(args.version,
                awidth, aheight, fdpi, ptype, ari_max,
                freq_plot_type, args.clean,
                task, ari,
                nreplicates, nvalues,
                use_rating_curve)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
