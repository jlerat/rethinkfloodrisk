#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-21 13:01:43.360895
## Comment : Fit mvt copula model via max likelihood
##
## ------------------------------

import sys
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
from scipy.stats import percentileofscore

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import sutils
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import sample
from floodstan import marginals

from figA_impact_of_period_on_FFA import get_logger
from figA_impact_of_period_on_FFA import copulafit

X_FLOW_SUM = 153.2720137
Y_FLOW_SUM = -28.8171849

def get_script_paths(config, source_file):
    froot = source_file.parent.parent.parent
    fdata = froot / "data"
    fsum = froot / "outputs" / f"copulaprocess2_v{config.version}"

    basename = source_file.stem
    fimg = froot / "images" / "manuscript" / basename

    SP = namedtuple("ScriptPaths",
                    ["source_file", "basename",
                     "froot", "fsum", "fdata", "fimg"])
    script_paths = SP(source_file, basename, froot,
                      fsum, fdata, fimg)

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
    opm = copulafit.get_options(config.version)
    prior = "uninformative"
    copula_spec = config.copula_spec
    exclude = config.exclude
    awra_covariate = True
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
    obs_data = obs_data.loc[:, stationids]
    obs_data.loc[:, "YEAR"] = obs_data.index + 1
    obs_data = obs_data.set_index("YEAR")

    stations = stations.loc[stationids]

    fs = script_paths.fdata / "schematic_rivers.geojson"
    with fs.open("r") as fo:
        rivers = json.load(fo)["features"]

    fs = script_paths.fsum / f"TASK{taskid}" / \
        f"copulaprocess_sum_samples_TASK{taskid}.csv"
    sum_samples, _ = csv.read_csv(fs)

    DT = namedtuple("Data", ["stations", "obs_data",
                             "sum_samples", "rivers"])
    return DT(stations, obs_data,
              sum_samples, rivers)


def process(config, script_paths, logger, data):
    ams = data.obs_data
    ams = ams.loc["1999":]

    stations = data.stations
    stationids = list(stations.index)
    sum_samples = data.sum_samples
    rivers = data.rivers

    nstations = len(stations)
    nval = len(ams)
    aeps = pd.DataFrame(np.nan,
                        index=ams.index,
                        columns=stationids + ["", "Gauged\nInflows"])
    for stationid in aeps.columns:
        if stationid in stationids:
            s = sum_samples.loc[:, f"{stationid}_SAMPLE"]
            a = ams.loc[:, stationid]
        elif stationid.startswith("Gauged"):
            s = sum_samples.loc[:, f"{config.group}_SUM_SAMPLE"]
            a = ams.loc[:, stationids].sum(axis=1)
        else:
            continue
        p = percentileofscore(s, a)
        aeps.loc[:, stationid] = 100 - p

    # Configure
    ncols_years = config.ncols_years
    nev = len(config.years)
    nrows = nev // ncols_years + (nev % ncols_years != 0)
    mosaic = [["grid"] + [config.years[ir * ncols_years + ic]
                          for ic in range(ncols_years)]
              for ir in range(nrows)]
    nrows = len(mosaic)
    plt.close("all")
    fig = plt.figure(figsize=((ncols_years + 1) * awidth,
                              nrows * aheight),
                     layout="constrained")
    wr = [1] + [1.5] * ncols_years
    kw = dict(hspace=0.15, wspace=0.05, width_ratios=wr)
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    for iax, (aname, ax) in enumerate(axs.items()):
        logger.info(f"Plotting {aname}", ntab=1)

        if aname == "grid":
            im = ax.imshow(aeps, cmap="Reds_r",
                           vmin=0, vmax=config.aep_max)

            # Show AEP values
            #for i in np.arange(nval):
            #    a = aeps[i, -1]
            #    if a > 20:
            #        continue
            #    ax.text(nstations + 1, i, f"{a:0.1f}",
            #            color="w", ha="center",
            #            va="center", fontweight="bold",
            #            fontsize="x-small")

            labs = aeps.columns.to_list()
            ax.set_xticks(np.arange(len(labs)),
                          labels=labs,
                          rotation=90)
            ax.set_yticks(np.arange(nval), labels=ams.index)

            xtk = list(range(nstations)) + [nstations + 1]
            ax.set_xticks(xtk)

            cbar = fig.colorbar(im,
                                ticks=np.arange(10, config.aep_max, 10),
                                shrink=0.5)
            cbar.ax.set(title="AEP\n[%]")
            ax.set(title=f"({letters[iax]}) AEP from posterior\n"
                         + "predictive distribution")
            continue

        year = aname

        # River lines
        for river in data.rivers:
            pts = np.array(river["geometry"]["coordinates"])[0]
            ax.plot(pts[:, 0], pts[:, 1], "-",
                    color=config.river_color,
                    lw=8, solid_capstyle="round")

        # Stations
        for sid, sinfo in data.stations.iterrows():
            # AEP data
            pm = aeps.loc[year, sid]

            # Station data
            x = sinfo.loc["LONGITUDE[arc_degree]"]
            y = sinfo.loc["LATITUDE[arc_degree]"]
            col = cm.Reds_r(Normalize(vmin=0, vmax=config.aep_max)(pm))
            ax.plot(x, y, "o",
                    ms=14, mec="w", mfc=col,
                    markeredgewidth=2)

            xy = [x, y]
            delta = 18
            txt = f"{sid}\n" if year == config.years[0] else ""
            txt += f"{pm:0.1f}%"

            if sid == "203014":
                xytext = [delta, -1.5 * delta]
                va = "top"
                ha = "left"
            elif sid == "203024":
                xytext = [-delta, delta]
                va = "bottom"
                ha = "right"
            else:
                xytext = [delta, delta]
                va = "bottom"
                ha = "left"

            ax.annotate(txt, xy,
                        ha=ha, va=va,
                        xytext=xytext,
                        fontsize="large",
                        textcoords="offset pixels")

        ax.axis("off")

        pm = aeps.loc[year, "Gauged\nInflows"]
        txt = f"Lismore gauged inflows\n" if year == config.years[0] else ""
        txt += f"{pm:4.1f}%"
        x0 = X_FLOW_SUM
        y0 = Y_FLOW_SUM
        ax.plot(x0, y0, "o",
                ms=14, mec="w", mfc=col,
                markeredgewidth=2)
        xytext = (delta, -delta)
        ax.annotate(txt, (x0, y0),
                    ha="left", va="top",
                    xytext=xytext,
                    fontsize="large",
                    textcoords="offset pixels")

        ax.set_title(f"({letters[iax]}) Details of {year} AEPs",
                     x=0.01, y=1, ha="left", va="top")

    basename = script_paths.basename
    fp = f"{basename}_v{config.version}.png"
    fp = script_paths.fimg / fp
    fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-c", "--clean", help="Clean image folder",
                        action="store_true", default=False)
    parser.add_argument("-s", "--copula_spec", help="Copula specification selected",
                        type=str, default="Gaussian")
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "group",
                               "awidth", "aheight", "fdpi",
                               "ncols_years",
                               "excludes", "copula_spec",
                               "debug", "diag",
                               "years", "exclude",
                               "river_color",
                               "aep_max", "clean"])
    awidth = 3.5
    aheight = 5
    fdpi = 300
    ncols_years = 1
    group = "203010-203014-203024"
    excludes = "NONE"
    years = [2017, 2022]
    aep_max = 40

    river_color = "0.5"

    config = CF(args.version, group,
                awidth, aheight, fdpi, ncols_years, excludes,
                args.copula_spec,
                args.debug, False,
                years,
                excludes, river_color, aep_max,
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
