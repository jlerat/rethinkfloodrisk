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
import math
import argparse
import json
import time
from string import ascii_letters as letters

import warnings
warnings.simplefilter("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pyrethink import datahub

from floodstan import marginals
from floodstan.report import STAN_DIAGNOSTIC_VARIABLES as SDV

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA 100 ARI",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--clear", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                    type=float, default=0.5)
parser.add_argument("-r", "--rho_min", help="Minimum rho value",
                    type=float, default=-1.)
args = parser.parse_args()

clear = args.clear
pcensor = args.pcensor
rho_min = args.rho_min

awidth = 6
aheight = 5
fdpi = 100 # 300
ngrid = 40

aep_target = 0.99
# aep used for the right-hand side plot (0.99 is too extreme, can't see it)
aep_target_plot = 0.95

sta1 = "203002"
sta2 = "203014"

#sta1 = "203014"
#sta2 = "203010"

exclude = "NONE"

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
# @Utils
# ----------------------------------------------------------------------
# Copied from
# https://stackoverflow.com/questions/10374930/annotating-a-3d-scatter-plot
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
stations = datahub.get_stations()

fopm = fout / "copulafit_options.json"
opm = hyruns.OptionManager.from_file(fopm)
taskids = opm.search(pcensor=f"{pcensor:0.1f}",
                     rho_min=f"{rho_min:0.1f}",
                     exclude=exclude)

data = {}
for taskid in taskids:
    fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_diagnostic_TASK{taskid}.json"
    with fd.open("r") as fo:
        diag = json.load(fo)

    pc = diag["pcensor"]
    ex = diag["exclude"]
    rm = diag["rho_min"]
    rho_min = rm
    mess = f"Load data TASK {taskid} exclude={ex} pcensor={pc} rho_min={rm}"
    LOGGER.info(mess, nret=1)

    for vn in SDV:
        LOGGER.info(f"{vn}: {diag[vn][:50]}", ntab=1)

    fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_data_TASK{taskid}.json"
    with fd.open("r") as fo:
        stan_data = json.load(fo)

    stationids = stan_data["stationids"]
    nstations = len(stationids)
    ista1 = stationids.index(sta1)
    ista2 = stationids.index(sta2)

    fe = fimg / f"expected_parameters_TASK{taskid}.json"
    if not fe.exists():
        LOGGER.info(f"load samples", ntab=1)
        fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_samples_TASK{taskid}.zip"
        samples = pd.read_csv(fs, skiprows=15)
        ylocs = samples.filter(regex="ylocn", axis=1).mean()
        ylogscales = samples.filter(regex="ylogsca", axis=1).mean()
        yshape1 = samples.filter(regex="yshape1", axis=1).mean()
        cor = samples.filter(regex="cor_IW", axis=1).mean()
        expected = {
            "ylocs": ylocs.to_dict(),
            "ylogscales": ylogscales.to_dict(),
            "yshape1": yshape1.to_dict(),
            "cor_IW": cor.to_dict()
            }
        with fe.open("w") as fo:
            json.dump(expected, fo, indent=4)

    else:
        LOGGER.info(f"read expected params from img folder", ntab=1)
        with fe.open("r") as fo:
            expected = json.load(fo)
        ylocs = pd.Series(expected["ylocs"])
        ylogscales = pd.Series(expected["ylogscales"])
        yshape1 = pd.Series(expected["yshape1"])
        cor = pd.Series(expected["cor_IW"])

    cor = cor.values.reshape((nstations, nstations)).T

    LOGGER.info(f"read mvnprocess results", ntab=1)
    fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
    mvnproc, comment = csv.read_csv(fs)

    groups = mvnproc.columns.str.replace("_.*", "", regex=True).unique()
    pat = f"{sta1[-2:]}-{sta2[-2:]}|{sta2[-2:]}-{sta1[-2:]}"
    groups = [g for g in groups if re.search(f"G(ALL|{pat})$", g)]
    LOGGER.info(f"Groups = {groups}", ntab=1)

    data[rho_min] = {
        "ylocs": ylocs,
        "ylogscales": ylogscales,
        "yshape1": yshape1,
        "cor": cor,
        "groups": groups,
        "mvnproc": mvnproc
        }

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
gev = marginals.GEV()

for rho_min, dd in data.items():
    LOGGER.info(f"Plotting rho_min = {rho_min}", nret=1)

    # Get data
    ylocs = dd["ylocs"]
    ylogscales = dd["ylogscales"]
    yshape1 = dd["yshape1"]
    cor = dd["cor"]
    groups = dd["groups"]
    mvnproc = dd["mvnproc"]

    plt.close("all")
    mosaic = [["diagram", stat] for stat in ["pall", "pany"]]
    nrows = len(mosaic)
    ncols = len(mosaic[0])
    fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                     layout="tight")
    axs = {
        "diagram_pall": fig.add_subplot(2, 2, 1, projection="3d"),
        "diagram_pany": fig.add_subplot(2, 2, 3, projection="3d"),
        "pall": fig.add_subplot(2, 2, 2),
        "pany": fig.add_subplot(2, 2, 4),
        }

    cols = ["tab:blue", "tab:orange", "tab:green"]

    paef = pe.withStroke(linewidth=4,
                         foreground="w")

    for iax, (aname, ax) in enumerate(axs.items()):
        LOGGER.info(f"Plot {aname}")
        evtype = "or" if re.search("any", aname) else "and"

        if aname.startswith("diagram"):
            pa, pb = 0.0, 0.995

            xx, zz, marg = {}, {}, {}
            xthresh, xlims = {}, {}
            for ista in [ista1, ista2]:
                gev.params = [ylocs.iloc[ista], ylogscales.iloc[ista],
                              yshape1.iloc[ista]]
                xa, xb = gev.ppf([pa, pb])
                xa = max(xa, 0.)
                xlims[ista] = (xa, xb)
                xx[ista] = np.linspace(xa, xb, ngrid)
                zz[ista] = norm.ppf(gev.cdf(xx[ista]))
                marg[ista] = gev.pdf(xx[ista])
                xthresh[ista] = gev.ppf(aep_target_plot)

            XX1, XX2 = np.meshgrid(xx[ista1], xx[ista2])
            ZZ1, ZZ2 = np.meshgrid(zz[ista1], zz[ista2])
            ZZ = np.dstack((ZZ1, ZZ2))

            ii = [ista1, ista2]
            rv = mvn(cov=cor[ii][:, ii])
            PP = rv.pdf(ZZ)
            ppmax = np.nanmax(PP)

            # Bivariate pdf
            kwargs = dict(cmap="viridis",
                          linewidth=0.0,
                          antialiased=False,
                          alpha=0.4)
            surf = ax.plot_surface(XX1, XX2, PP, **kwargs)

            # integral
            xt1 = xthresh[ista1]
            xt2 = xthresh[ista2]
            if evtype == "and":
                ii = (XX1 >= xt1) & (XX2 >= xt2)
            else:
                ii = (XX1 >= xt1) | (XX2 >= xt2)
            PP[~ii] = np.nan
            kwargs["alpha"] = 0.8
            surf = ax.plot_surface(XX1, XX2, PP, **kwargs)

            elev = 45
            azim = -110
            roll = 0.
            ax.view_init(elev, azim, roll)
            ax.set_proj_type("ortho")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.zaxis.set_major_locator(ticker.MaxNLocator(3))

            txt = r"$Pr(X_1>x_1^* \cap X_2>x_2^*)$" if evtype == "and"\
                else r"$Pr(X_1>x_1^* \cup X_2>x_2^*)$"
            xa = XX1[ii].mean()
            ya = XX2[ii].mean()
            diff = np.abs(XX1 - xa) + np.abs(XX2 - ya)
            iclose = np.where(diff == diff.min())
            za = PP[iclose][0]
            z0, z1 = ax.get_zlim()
            wref = 0.5 if evtype == "and" else 0.85
            zt0, zt1 = [z1 * w + z0 * (1 - w) for w in [wref - 0.08, wref]]
            ax.text(xa, ya, zt1, txt, ha="right",
                    fontsize=12, fontweight="bold")
            ax.plot([xa, xa], [ya, ya], [zt0, za], "k-", lw=1)
            ax.plot(xa, ya, za, "o", mfc="k", mec="k")

            xlab = f"Peak flow {sta1} [m3.s-1]"
            ylab = f"Peak flow {sta2} [m3.s-1]"
            zlab = "Pr(X,Y) [-]"
            ax.set(xlabel=xlab, ylabel=ylab, zlabel=zlab)

            title = f"({letters[iax]}) Bivariate distribution and"\
                    + f"\n'{evtype}' event"
        else:
            stat = aname

            # Adjust bounds
            if stat == "pall":
                x0, x1 = (-5.2, -2.) if rho_min == -1 else (-5., -2.)
            else:
                x0, x1 = -2.0, -1.3

            # value -> %
            x0 += 2
            x1 += 2

            bins = np.logspace(x0, x1, 50)
            ax.set_xlim((10**x0, 10**x1))
            ax.set_xscale("log")

            plot_means = {}
            for ig, gname in enumerate(groups):
                etxt = re.sub("\\.", "_", f"{aep_target:0.02f}")
                cn = f"{gname}_log10{stat}_aeptarget_p{etxt}"
                sel = mvnproc.loc[:, cn]
                # value -> %
                se = 10**(sel + 2)
                plot_means[gname] = se.mean()
                gg = "/".join([f"2030{s}" for s in gname[1:].split("-")])
                lab = "All sites" if gname == "GALL" else f"Bivariate\n{gg}"
                ax.hist(se, bins=bins, edgecolor="0.5",
                        facecolor=cols[ig],
                        alpha=0.6,
                        label=lab)

            # Got to do it outside of previous loop to maintain y0/y1
            y0, y1 = ax.get_ylim()
            for ig, gname in enumerate(groups):
                m = plot_means[gname]
                ml = math.log10(m)
                if (ml - x0) * (x1 - ml) > 0:
                    ax.plot([m, m], [y0, y1], "-", lw=3, color=cols[ig])
                    ax.set_ylim((y0, y1))
                    w = 0.4
                    xy = (m, y1 * w + y0 * (1-w))
                    xytext = (0, 5)
                    txt = f"Mean\n{m:0.2f}%"
                    ax.annotate(txt, xy, xytext,
                                xycoords="data",
                                va="bottom", ha="center",
                                fontweight="bold",
                                textcoords="offset points",
                                path_effects=[paef])

            xlab = "Annual Exceedance Probability [%]"
            ylab = "MCMC Sample count [-]"
            ax.set(xlabel=xlab, ylabel=ylab)

            loc = 8 if stat == "pany" else 6
            ax.legend(framealpha=0, loc=loc)

            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

            title = r"$\bigcap_i\ X_i > x_i^*$" if evtype == "and"\
                else r"$\bigcup_i\ X_i > x_i^*$"
            title = f"({letters[iax]}) Probability of '{evtype}' event {title}"

        ax.set_title(title, x=0.02, y=0.98, va="top", ha="left",
                     transform=ax.transAxes, fontweight="bold",
                     path_effects=[paef])

    LOGGER.info("Saving to disk")
    fp = fimg / f"{basename}_pcensor{pcensor}_rhomin{rho_min:0.02f}.png"
    fig.savefig(fp)

LOGGER.completed()
