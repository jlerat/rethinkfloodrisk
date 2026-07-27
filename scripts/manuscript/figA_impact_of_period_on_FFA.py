#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from floodstan import freqplots, marginals
from pyrethink import datahub, processing

import importlib.util

ffit = Path(__file__).resolve().parent.parent / "fit" / "copulafit.py"
spec = importlib.util.spec_from_file_location("copulafit", ffit)
copulafit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(copulafit)


def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    fconcat = froot / "outputs" / f"copulaconcat_v{config.version}"

    basename = source_file.stem
    fimg = froot / "images" / "manuscript" / basename

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fconcat", "fimg"])
    script_paths = SP(source_file, basename, froot,
                      fconcat, fimg)

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

    obs_data, _, _, stations = datahub.get_ams_concat()

    fr = script_paths.fconcat / "copulaconcat_ffa.csv"
    ffa, _ = csv.read_csv(fr, dtype={"STATIONID": str})

    DT = namedtuple("Data", ["stations", "ffa", "obs_data",
                             "options"])
    return DT(stations, ffa, obs_data, opm)


def process(config, script_paths, logger, data):
    grp_mv = next(g for g in data.options.options["group"] if len(g.split("-")) == 3)
    stationids = grp_mv.split("-")
    stations = data.stations.loc[stationids]

    options = data.options
    obs_data = data.obs_data
    ffa = data.ffa

    copula_specs = options.options["copula_spec"][1:]
    if config.debug:
        copula_specs = ["Gaussian"]
        stations = stations.iloc[:1]

    excludes = config.excludes + ["NONE"]
    fptype = config.freq_plot_type

    for copula_spec in copula_specs:
        for stationid in stations.index:
            logger.info(f"FFA plots - {copula_spec} - {stationid}")

            sinfo = stations.loc[stationid]

            # Get data
            peaks = obs_data.loc[:, str(stationid)]

            # .. Rating curve analysis
            rc, _ = datahub.get_rating_curves(stationid, True)
            rc_h = rc.loc[:, "WATERLEVEL[m]"]
            rc_q = rc.loc[:, "STREAMFLOW[m3_s-1]"]
            ipos = (rc_q > 1e-2) & (rc_h > 0)
            rc_h = rc_h.loc[ipos]
            rc_q = rc_q.loc[ipos]

            # Plot
            plt.close("all")
            cop_prior = "inf" if config.use_informative else "noninf"
            models = ["univ-noninf", f"mv-{cop_prior}"]
            mosaic = [["."] + [f"col_title/{m}" for m in models]]
            mosaic += [[f"row_title/{ex}"] + [f"data/{ex}/{m}" for m in models]
                       for ex in excludes]

            nrows = len(mosaic)
            ncols = len(mosaic[0])
            figsize = ((ncols - 1) * config.awidth, (nrows - 1) * config.aheight)
            fig = plt.figure(figsize=figsize,
                             layout="constrained")
            kw = dict(width_ratios=[1] + [9] * (ncols - 1),
                      height_ratios=[1] + [9] * (nrows - 1))
            axs = fig.subplot_mosaic(mosaic,
                                     gridspec_kw=kw)
            iplot = 0
            for iax, (aname, ax) in enumerate(axs.items()):
                pcfg = aname.split("/")
                if pcfg[0] == "col_title":
                    mod = pcfg[1]
                    model = "Univariate" if re.search("univ", mod) else "Multivariate"
                    model += " model"
                    ax.text(0.5, 0.5, model,
                            va="center", ha="center",
                            fontsize="x-large",
                            fontweight="bold")
                    ax.axis("off")
                    continue

                elif pcfg[0] == "row_title":
                    exclude = pcfg[1]
                    if exclude == "NONE":
                        exctxt = "Fitting using all data"
                    else:
                        ev = int(re.sub("-.*", "", exclude)) + 1
                        exctxt = f"Fitting without {ev} flood"

                    ax.text(0.5, 0.5, exctxt,
                            va="center", ha="center",
                            fontweight="bold",
                            fontsize="x-large",
                            rotation=90)
                    ax.axis("off")
                    continue

                exclude, mod = pcfg[1:]

                # Find tasks
                if re.search("univ", mod):
                    model = "Univariate"
                    cs = "Univariate"
                else:
                    model = "Multivariate"
                    if config.use_awra:
                        model += " with AWRAL covariate"
                    cs = copula_spec

                prior = "uninformative" if re.search("noninf", mod) \
                    else "informative"

                group = stationid if model == "Univariate" else grp_mv
                awra_covariate = False if model == "Univariate" else config.use_awra

                taskid = options.find(prior=prior,
                                      copula_spec=cs,
                                      exclude=exclude,
                                      awra_covariate=awra_covariate,
                                      group= f"^{group}$")
                taskid = next(t for t in taskid)

                task = options.get_task(taskid)
                pcensor = task.pcensor

                # Get ffa data
                idx = ffa.STATIONID == stationid
                idx &= ffa.TASKID == taskid
                idx &= ffa.VARIABLE.str.contains("DESIGN")
                if idx.sum() == 0:
                    continue

                df = ffa.loc[idx].copy()
                ERI = df.VARIABLE.replace(".*ERI|\\.$", "", regex=True)
                df.loc[:, "ERI"] = ERI.astype(float)

                # Get data
                peaks_excluded = peaks.copy()
                if exclude != "NONE":
                    peaks_excluded.loc[int(exclude)] = np.nan

                # Fit GEV
                gev = marginals.factory("GEV")
                gev.fit_lh_moments(peaks_excluded, eta=2)

                # Plot obs data
                x, y = freqplots.plot_data(ax, peaks_excluded, fptype, zorder=10)

                same = np.abs(y[:, None] - peaks.values[None, :]) < 1e-10
                _, same = np.where(same)
                time = obs_data.index[same]

                ythresh = y[-3]
                arrowprops = {
                    "edgecolor": "0.4",
                    "arrowstyle": "-"
                    }
                for wy, xx, yy in zip(time, x, y):
                    if yy < ythresh:
                        continue
                    txt = str(wy)
                    ax.annotate(txt, xy=(xx, yy),
                                xycoords="data",
                                xytext=(40, -40),
                                va="bottom", ha="right",
                                textcoords="offset points",
                                arrowprops=arrowprops,
                                zorder=5)

                # Plot FFA
                aris = df.ERI
                quantiles = df

                iok = aris <= ari_max
                aris = aris[iok]
                quantiles = quantiles.loc[iok]

                inocens = 1 - 1./aris >= pcensor
                quantiles = quantiles.loc[inocens]
                aris = aris[inocens]
                freqplots.plot_marginal_quantiles(ax, aris, quantiles, fptype,
                                                  center_column="POSTERIOR_PREDICTIVE",
                                                  q0_column="5%",
                                                  q1_column="95%",
                                                  alpha=0.3,
                                                  facecolor="tab:blue",
                                                  edgecolor="k")

                # plot LH moment fit
                #if re.search("univ", mod):
                #    freqplots.plot_marginal_cdf(ax, gev, fptype,
                #                                label="LH moment fit",
                #                                linestyle="--")

                if iplot >= ncols - 1:
                    xlab = "Gumbel reduced variable $-log(-log(P))$ [-]"
                    xticks = None
                else:
                    xlab = ""
                    xticks = []

                if iplot % (ncols - 1) == 0:
                    ylab = "Peak flow [m3.s-1]"
                    yticks = None
                else:
                    ylab = ""
                    yticks = []

                ylim = (0, peaks.max() * 1.3)
                ax.set(xlabel=xlab,
                       ylabel=ylab,
                       ylim=ylim)
                if xticks is not None:
                    ax.set_xticks(xticks)
                if yticks is not None:
                    ax.set_yticks(yticks)

                retp = [10, 100, 500]
                aeps, xpos = freqplots.add_aep_to_xaxis(ax, fptype, True, retp)

                i100 = df.ERI == 100
                q100 = quantiles.loc[i100].squeeze()

                design = {}
                cn_pp = "POSTERIOR_PREDICTIVE"
                for ist, st in enumerate(["5%", cn_pp, "95%"]):
                    q = q100.loc[st]
                    h = processing.linear_interpolation(q, rc_q, rc_h)
                    design[st] = {"q": q, "h": h}

                design = pd.DataFrame(design).T
                design.loc["CI90", :] = design.loc["95%"] - design.loc["5%"]

                txt = f"({letters[iplot]})"
                ytxt = 0.96
                kw = dict(va="top", ha="left", transform=ax.transAxes,
                          fontsize="large",
                          fontdict = {"family": "monospace"})
                ax.text(0.03, ytxt, txt, **kw)

                txt = f"Flow [1% AEP] = {design.loc[cn_pp, 'q']:0.0f}"
                txt += f" $\pm$ {design.loc['CI90', 'q']/2:0.0f} $m^3.s^{{{-1}}}$"
                dy = 0.06
                ytxt -= dy
                ax.text(0.03, ytxt, txt, **kw)

                txt = f"Stage[1% AEP] = {design.loc[cn_pp, 'h']:0.1f}"
                txt += f" $\pm$ {design.loc['CI90', 'h']/2:0.1f} $m$"
                ytxt -= dy
                ax.text(0.03, ytxt, txt, fontweight="bold", **kw)

                iplot += 1

            basename = script_paths.basename
            use_a = config.use_awra
            use_i = config.use_informative
            fp = f"{basename}_{stationid}_{copula_spec}_A{use_a}_I{use_i}_v{config.version}.png"
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
                               "ptype", "ari_max", "excludes",
                               "freq_plot_type", "use_awra",
                               "use_informative", "clean"])
    awidth = 6
    aheight = 5
    fdpi = 300
    ptype = "gumbel"
    ari_max = 500
    excludes = ["2021"]
    freq_plot_type = "gumbel"

    config = CF(args.version,  args.debug,
                awidth, aheight, fdpi, ptype, ari_max,
                excludes,
                freq_plot_type,
                args.use_awra, args.use_informative,
                args.clean)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
