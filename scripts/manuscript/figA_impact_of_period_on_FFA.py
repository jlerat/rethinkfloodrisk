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
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils
from floodstan import freqplots
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA curves",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug


awidth = 6
aheight = 5
fdpi = 300

ptype = "gumbel"

pcensor = 0.5
excludes = ["NONE", "2022-02-27"]

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "data"

fout = froot / "outputs"

basename = source_file.stem
fimg = froot / "images" / "manuscript" / basename
fimg.mkdir(exist_ok=True, parents=True)
for f in fimg.glob("*.png"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
LOGGER.info("Load data")

stations = datahub.get_stations()
if debug:
    stations = stations.iloc[:1]


fopm = fout / "copulafit_options.json"
opm = hyruns.OptionManager.from_file(fopm)
taskids = opm.search(pcensor=pcensor)

ffa = {}
data = {}
for taskid in taskids:
    ftask = fout / f"copulafit_TASK{taskid}"
    fd = ftask / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    exclude = diag["exclude"]
    if excludes is not None:
        if not exclude in excludes:
            continue

    LOGGER.info(f"Load report TASK {taskid} exclude={exclude}")
    fr = ftask / f"postprocess_report_TASK{taskid}.csv"
    df, _ = csv.read_csv(fr, index_col=0)
    ffa[exclude] = df

    fd = ftask / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        d = json.load(fo)
        y = pd.DataFrame(d["y"], columns=d["stationids"])
        t = pd.DataFrame(d["potpeaks_time"]).reset_index(drop=True)
        y = pd.concat([y, t], axis=1)
        data[exclude] = y

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
for stationid, sinfo in stations.iterrows():
    LOGGER.info(f"Plotting {stationid}")
    plt.close("all")
    mosaic = [[per for per in data.keys()]]
    nrows = len(mosaic)
    ncols = len(mosaic[0])
    fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic, sharey=True)

    # Rating curve analysis
    rc, _ = datahub.get_rating_curves(stationid, True)
    rc_h = rc.loc[:, "WATERLEVEL[m]"]
    rc_q = rc.loc[:, "STREAMFLOW[m3_s-1]"]
    ipos = (rc_q > 1e-2) & (rc_h > 0)
    rc_h = rc_h.loc[ipos]
    rc_q = rc_q.loc[ipos]

    # Plot ffa
    for iax, (aname, ax) in enumerate(axs.items()):
        exclude = aname

        # Plot data
        peaks = data[exclude].loc[:, str(stationid)]
        time = data[exclude].loc[:, "DAY"]

        x, y = freqplots.plot_data(ax, peaks, ptype, zorder=10)
        same = np.abs(y[:, None] - peaks.values[None, :]) < 1e-10
        _, same = np.where(same)
        time = time.iloc[same]

        ythresh = y[-3]
        arrowprops = {
            "edgecolor": "0.4",
            "arrowstyle": "-"
            }
        for t, xx, yy in zip(time, x, y):
            if yy < ythresh:
                continue
            d = pd.to_datetime(t).strftime("%b\n%y")
            ax.annotate(d, xy=(xx, yy),
                        xycoords="data",
                        xytext=(-40, 40),
                        va="bottom", ha="right",
                        textcoords="offset points",
                        arrowprops=arrowprops,
                        zorder=5)

        # Plot FFA
        df = ffa[exclude]
        istation = data[exclude].columns.tolist().index(str(stationid))
        quantiles = df.filter(regex=f"DESIGN.*\\[{istation + 1}\\]", axis=0)
        aris = quantiles.index.to_series().str\
                .replace(".*ERI|\\[.*", "", regex=True).astype(float).values

        inocens = 1 - 1./aris >= pcensor
        quantiles = quantiles.loc[inocens]
        aris = aris[inocens]
        freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                          center_column="POSTERIOR_PREDICTIVE",
                                          q0_column="5%",
                                          q1_column="95%",
                                          alpha=0.3,
                                          facecolor="tab:blue",
                                          edgecolor="k")

        retp = [100]
        aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, True, retp)

        if exclude == "NONE":
            exctxt = "All data"
        else:
            ev = re.sub("-.*", "", exclude)
            exctxt = f"Without {ev} flood"

        title = f"({letters[iax]}) {exctxt}"
        xlab = "Gumbel reduced variable $-log(-log(P))$ [-]"
        ylab = "Peak flow [m3.s-1]" if iax == 0 else ""
        ax.set(title=title, ylabel=ylab, xlabel=xlab)

        q100 = quantiles.filter(regex="DESIGN_ERI100\\[", axis=0).squeeze()
        txt = "1:100 uncertainty:\n"
        for st in ["5%", "POSTERIOR_PREDICTIVE", "95%"]:
            q = q100.loc[st]
            h = datahub.linear_interpolation(q, rc_q, rc_h)
            stt = "ppred" if st.startswith("POST") else st
            txt += f"{stt:>5s} {q:6.0f} $m^3.s^{{{-1}}}$ ({h:4.1f}m)\n"

        ax.text(0.03, 0.97, txt, va="top", ha="left",
                transform=ax.transAxes)

    ftitle = f"{sinfo.NAME} ({stationid})"
    fig.suptitle(ftitle, fontweight="bold")

    fp = fimg / f"{basename}_station{istation + 1}.png"
    fig.savefig(fp, dpi=fdpi)

LOGGER.completed()
