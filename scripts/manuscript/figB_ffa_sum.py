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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import matplotlib as mpl
mpl.rcParams["axes3d.mouserotationstyle"] = "azel"

from hydrodiy.io import csv, iutils
from hydrodiy.plot import violinplot

from pyrethink import datahub
from floodstan import marginals
from pyrethink import copulas

from floodstan import freqplots

import importlib.util

ffit = Path(__file__).resolve().parent.parent / "fit" / "copulafit.py"
spec = importlib.util.spec_from_file_location("copulafit", ffit)
copulafit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copulafit)


def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    fsum = froot / "outputs" / f"copulaprocess2_v{config.version}"
    fparams = froot / "outputs" / f"copulafit_v{config.version}"

    basename = source_file.stem
    fimg = froot / "images" / "manuscript" / basename

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fsum", "fparams", "fimg"])
    script_paths = SP(source_file, basename, froot,
                      fsum, fparams, fimg)

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


def get_logger(config, script_paths):
    basename = script_paths.source_file.stem
    logger = iutils.get_logger(basename)
    return logger


def get_data(config, script_paths, logger):
    opm = copulafit.get_options(config.version)

    prior = config.prior
    copula_spec = config.copula_spec
    exclude = config.exclude
    awra_covariate = config.awra_covariate
    group = config.group
    stationids = group.split("-")
    taskid = opm.find(prior=prior,
                      copula_spec=copula_spec,
                      exclude=exclude,
                      awra_covariate=awra_covariate,
                      group= f"^{group}$")
    taskid = next(t for t in taskid)
    task = opm.get_task(taskid)

    obs_data, _, _, stations = datahub.get_ams_concat()

    fs = script_paths.fsum / f"TASK{taskid}" / \
        f"copulaprocess_sum_samples_TASK{taskid}.csv"
    sum_samples, _ = csv.read_csv(fs)

    fa = script_paths.fsum / f"TASK{taskid}" / \
        f"copulaprocess_sum_ffa_TASK{taskid}.csv"
    sum_ffa, _ = csv.read_csv(fa)

    # Get parameters and sample
    fs = script_paths.fimg / "conditional_samples.csv"
    fz = fs.parent / f"{fs.stem}.zip"
    if not fz.exists():
        nsta = len(stationids) + (1 if awra_covariate else 0)
        cop = copulas.factory(copula_spec, nsta)

        # Set conditional and target variables
        scond = "203010"
        icond = stationids.index(scond)
        starg = ["203014", "203024"]
        itarg = [stationids.index(s) for s in starg]

        # Compute posterior conditional flow for conditional variable
        aricond = 100
        marginal = marginals.factory(task.context["marginal_name"])
        qcond = sum_ffa.loc[sum_ffa.ERI == f"DESIGN_ERI{aricond}",
                            f"{scond}_POSTERIOR_PREDICTIVE"].squeeze()

        fp = script_paths.fparams / f"TASK{taskid}" / \
            f"copulafit_samples_TASK{taskid}.zip"
        params = pd.read_csv(fp, skiprows=15)
        if config.debug:
            params = params.iloc[:1000]

        samples_cond = pd.DataFrame(columns=starg + [scond], index=params.index)
        samples_cond.loc[:, scond] = qcond

        for ismp, smp in params.iterrows():
            if ismp % 100 == 0:
                logger.info(f"Sampling conditional {ismp} / {len(params)}")

            # retrieve correlation matrix
            if re.search("Factor", copula_spec):
                is_factor = True
                nf = cop.copula_nfactors
                zrhos = smp.filter(regex="zrhos").values.reshape((nsta, nf + 1))
                cop.set_params_via_zrhos(zrhos)
            else:
                is_factor = False
                corr = smp.filter(regex="corr_IW").values.reshape((nsta, nsta)).T
                cop.params = corr

            p = smp.filter(regex=f"y(l|sh).*\\[{icond + 1}\\]")
            marginal.params = p
            zcond = cop.marginal_ppf(marginal.cdf(qcond))
            ztarg = cop.sample_conditional([icond], np.array([zcond]))
            utarg = cop.marginal_cdf(ztarg)

            for j, sid in enumerate(starg):
                itarg = stationids.index(sid)
                p = smp.filter(regex=f"y(l|sh).*\\[{itarg + 1}\\]")
                marginal.params = p
                samples_cond.loc[ismp, sid] = marginal.ppf(utarg[j])

        # Sum of conditional samples
        samples_cond.loc[:, f"{group}_SUM"] = samples_cond.sum(axis=1)

        csv.write_csv(samples_cond, fs, "Conditional samples",
                      script_paths.source_file,
                      compress=True, write_index=False,
                      lineterminator="\n")
    else:
        logger.info("Loading samples conditional")
        samples_cond, _ = csv.read_csv(fs)


    DT = namedtuple("Data", ["stations", "obs_data",
                             "sum_samples", "sum_ffa",
                             "options", "samples_cond"])
    return DT(stations, obs_data,
              sum_samples, sum_ffa,
              opm, samples_cond)


def process(config, script_paths, logger, data):
    group = config.group
    stationids = group.split("-")
    stations = data.stations.loc[stationids]

    options = data.options
    obs_data = data.obs_data
    sum_samples = data.sum_samples

    # plot
    plt.close("all")
    mosaic = [["FFA", "SCATTER"]]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (ncols * config.awidth, nrows * config.aheight)
    fig = plt.figure(figsize=figsize, layout="compressed")
    kw = dict(width_ratios=[1., 1.])
    axs = fig.subplot_mosaic(mosaic,
                             gridspec_kw=kw)

    # FFA plot
    aris = np.logspace(math.log10(3), math.log10(300), 500)
    qt = sum_samples.filter(regex="SAMPLE", axis=1).quantile(1 - 1. / aris, axis=0)
    qt = pd.DataFrame(qt.values, columns=qt.columns, index=aris)
    df = qt.filter(regex="\d_SAMPLE", axis=1)
    qt.loc[:, "SUM_QT"] = df.sum(axis=1)

    ptype = "gumbel"
    cns1 = next(cn for cn in qt.columns if re.search("SUM", cn))
    ax = axs["FFA"]
    lab = "Q(" + ", ".join(stationids) + ")"
    freqplots.plot_marginal_quantiles(ax, aris, qt, ptype,
                                      label=lab, color="tab:blue",
                                      center_column=cns1, lw=2)
    cns2 = "SUM_QT"
    lab = "+ ".join([f"Q({s})" for s in stationids])
    freqplots.plot_marginal_quantiles(ax, aris, qt, ptype,
                                      label=lab, color="tab:red",
                                      center_column=cns2, lw=2)

    retp = [10, 100]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, return_periods=retp)
    xa, xb = ax.get_ylim()
    ya, yb = ax.get_ylim()

    retp, xpos = [retp[-1]], [xpos[-1]]
    for r, x in zip(retp, xpos):
        ia = aris[np.argmin(np.abs(aris - r))]
        v2 = qt.loc[ia, cns2]
        r1 = qt.loc[np.abs(qt.loc[:, cns1] - v2).idxmin()].name
        x1 = freqplots.cdf_to_reduced_variate(1 - 1. / r1, ptype)

        dy = (yb - ya) * 0.05
        ax.plot([x, x1, x1], [v2, v2, ya + dy / 2], color="tab:red", lw=0.9, ls="--")
        ax.plot(x, v2, "o", mec="tab:red", mfc="w", ms=8)
        ax.plot(x1, v2, "o", color="tab:red", ms=8)

        ax.annotate("", xytext=(x1, ya + dy), xy=(x1, ya),
            arrowprops=dict(arrowstyle="-|>", color="tab:red"),
            size=15)

        ax.text(x1, ya + (yb - ya) * 0.04,
                f" 1:{r1:0.0f}",
                va="bottom", ha="left",
                fontweight="bold", fontsize="large",
                color="tab:red")

    freqplots.set_xlabel(ax, ptype)
    ax.set_xlim((1, 5))
    ax.set_xticks(np.arange(1, 7))
    ax.legend(loc=2)

    title = "(a) Quantiles of flow sum and sum of quantiles"
    ax.set_title(title)
    ax.set_ylabel("Streamflow [$m^3.s^{-1}$]")

    # scatter plot
    ax = axs["SCATTER"]

    # .. compute aep of sum corresponding to 1:100
    df = sum_samples.filter(regex="^[\d].*_S", axis=1)
    df.columns = [re.sub("_.*", "", cn) for cn in df.columns]
    cns1 = re.sub("_.*", "", cns1)

    aep = 100
    th = df.loc[:, cns1].quantile(1 - 1. / aep)

    # .. get conditional samples that match sum = th
    samples_cond = data.samples_cond
    cn = next(cn for cn in samples_cond.columns if cn.endswith("SUM"))
    diff =  np.abs(samples_cond.loc[:, cn] - th)
    tol = 10
    idx = (diff < tol)

    logger.info(f"Scatter plot", nret=1)
    logger.info(f"\tQ{aep} = {th:0.1f}")
    logger.info(f"\tmax error = {diff.max():0.2f} m3.s-1")
    cc = [cn for cn in df.columns if cn not in [group, cns1]]
    ddf = samples_cond.iloc[idx, :2]

    x = np.arange(len(ddf))
    norm = Normalize(vmin=0, vmax=6)
    cmap = plt.cm.Reds_r

    ax.scatter(ddf.iloc[:, 0], ddf.iloc[:, 1], alpha=0.3)

    title = "(b) Flows with a sum having a 1% AEP"
    ax.set(xlabel="Random samples",
           ylabel="Streamflow [$m^3.s^{-1}$]",
           title=title)

    # save
    fp = f"{script_paths.basename}_v{config.version}.png"
    fp = script_paths.fimg / fp
    fp.parent.mkdir(exist_ok=True)
    fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Univariate FFA plots",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    parser.add_argument("-c", "--clean", help="Clean image folder",
                        action="store_true", default=False)
    parser.add_argument("-a", "--use_awra", help="Use copula including awra covariate",
                        action="store_true", default=False)
    parser.add_argument("-i", "--use_informative", help="Use copula including informative prior",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "debug",
                               "awidth", "aheight", "fdpi",
                               "ptype", "ari_max",
                               "freq_plot_type", "clean",
                               "prior", "copula_spec",
                               "exclude", "awra_covariate",
                               "group", "ari"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    freq_plot_type = "gumbel"

    prior = "uninformative"
    copula_spec = "Gaussian"
    exclude = "NONE"
    awra_covariate = True
    group = "203010-203014-203024"

    ari = 100

    config = CF(args.version,  args.debug,
                awidth, aheight, fdpi, ptype, ari_max,
                freq_plot_type, args.clean,
                prior, copula_spec, exclude,
                awra_covariate, group, ari)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
