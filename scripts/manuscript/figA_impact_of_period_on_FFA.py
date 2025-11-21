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
from floodstan.report import STAN_DIAGNOSTIC_VARIABLES as SDV
from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA curves",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--clear", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                    type=float, default=0.3)
parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                    type=float, default=-1.)
args = parser.parse_args()

clear = args.clear
pcensor = args.pcensor
rho_min = args.rho_min

awidth = 6
aheight = 5
fdpi = 300

ptype = "gumbel"
ari_max = 500

excludes = ["NONE", "2021"]

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
if clear:
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

fopm = fout / "copulafit_options.json"
opm = hyruns.OptionManager.from_file(fopm)
taskids = opm.find(pcensor=f"{pcensor:0.1f}",
                   exclude="|".join(excludes),
                   rho_min=f"{rho_min:0.1f}")
assert len(taskids) == len(excludes)

ffa = {}
data = {}
for taskid in taskids:
    ftask = fout / f"copulafit_TASK{taskid}"
    fd = ftask / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    ex = diag["exclude"]
    pc = diag["pcensor"]
    rm = diag["rho_min"]
    mess = f"Load report TASK {taskid} exclude={ex} pcensor={pc} rho_min={rm}"
    LOGGER.info(mess)
    if rm != rho_min or pc != pcensor:
        errmsg = "Expected pcensor={pcensor} rho_min={rho_min}"
        raise ValueError(errmsg)

    for vn in SDV:
        LOGGER.info(f"{vn}: {diag[vn][:50]}", ntab=1)

    fr = ftask / f"postprocess_report_TASK{taskid}.csv"
    df, _ = csv.read_csv(fr, index_col=0)
    ffa[ex] = df

    fd = ftask / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        d = json.load(fo)
        y = pd.DataFrame(d["y"], columns=d["stationids"])
        cn = "WATER_YEAR"
        y.loc[:, cn] = d["ams_time"]
        y.loc[:, cn] += 1 # adds 1 because starts in Oct
        data[ex] = y

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

        x, y = freqplots.plot_data(ax, peaks, ptype, zorder=10)
        same = np.abs(y[:, None] - peaks.values[None, :]) < 1e-10
        _, same = np.where(same)
        time = data[exclude].WATER_YEAR.iloc[same]

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

        iok = aris <= ari_max
        aris = aris[iok]
        quantiles = quantiles.loc[iok]

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

        retp = [10, 100, 500]
        aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, True, retp)

        if exclude == "NONE":
            exctxt = "All data"
        else:
            ev = int(re.sub("-.*", "", exclude)) + 1
            exctxt = f"Without {ev} flood"

        title = f"({letters[iax]}) {exctxt}"
        xlab = "Gumbel reduced variable $-log(-log(P))$ [-]"
        ylab = "Peak flow [m3.s-1]" if iax == 0 else ""
        ax.set(title=title, ylabel=ylab, xlabel=xlab)

        q100 = quantiles.filter(regex="DESIGN_ERI100\\[", axis=0).squeeze()

        txt = "Uncertainty in 1:100 event\n\n"
        kw = dict(va="top", ha="left", transform=ax.transAxes)
        ax.text(0.03, 0.97, txt, **kw, fontweight="bold")

        delta = 0.06
        for ist, st in enumerate(["5%", "POSTERIOR_PREDICTIVE", "95%"]):
            q = q100.loc[st]
            h = datahub.linear_interpolation(q, rc_q, rc_h)
            stt = "post pred" if st.startswith("POST") else st

            txt = f"{stt:<12s}"
            ytxt = 0.97 - delta * (ist + 1)
            ax.text(0.03, ytxt, txt, **kw)

            txt = f"{q:>5,.0f} $m^3.s^{{{-1}}}$ ({h:>4.1f}m)"
            ax.text(0.18, ytxt, txt, **kw)

    ftitle = f"{sinfo.NAME} ({stationid})"
    fig.suptitle(ftitle, fontweight="bold")

    fp = f"{basename}_{stationid}"\
         + f"_pcensor{pcensor}_rhomin{rho_min}.png"
    fp = fimg / fp
    fig.savefig(fp, dpi=fdpi)

LOGGER.completed()
