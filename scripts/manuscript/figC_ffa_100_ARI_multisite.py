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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot
from pyrethink import datahub
from floodstan import marginals

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot FFA 100 ARI",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

awidth = 6
aheight = 5
fdpi = 100 # 300
ngrid = 40

eep_target_plot = 0.95

sta1 = "203002"
sta2 = "203014"

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
LOGGER.info("Load data")

# Select fit task with
for fold in fout.glob("copulafit_TASK*"):
    lf = [f for f in fold.glob("*.zip")
          if re.search("mvnprocess", f.stem)]
    if len(lf) > 0:
        taskid = int(re.sub(".*TASK", "", fold.stem))
        break

stations = datahub.get_stations()

LOGGER.info(f"Load data TASK {taskid}")
fd = fout / f"copulafit_TASK{taskid}" / f"copulafit_data_TASK{taskid}.json"
with fd.open("r") as fo:
    data = json.load(fo)
stationids = data["stationids"]
nstations = len(stationids)
ista1 = stationids.index(sta1)
ista2 = stationids.index(sta2)

fe = fimg / "expected_parameters.json"
if not fe.exists():
    LOGGER.info(f"Load samples TASK {taskid}")
    fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_samples_TASK{taskid}.zip"
    samples = pd.read_csv(fs, skiprows=15)
    ylocs = samples.filter(regex="ylocn", axis=1).mean()
    ylogscales = samples.filter(regex="ylogsca", axis=1).mean()
    yshape1 = samples.filter(regex="yshape1", axis=1).mean()
    L_cor = samples.filter(regex="L_cor", axis=1).mean()
    expected = {
        "ylocs": ylocs.to_dict(),
        "ylogscales": ylogscales.to_dict(),
        "yshape1": yshape1.to_dict(),
        "L_cor": L_cor.to_dict()
        }
    with fe.open("w") as fo:
        json.dump(expected, fo, indent=4)

else:
    with fe.open("r") as fo:
        expected = json.load(fo)
    ylocs = pd.Series(expected["ylocs"])
    ylogscales = pd.Series(expected["ylogscales"])
    yshape1 = pd.Series(expected["yshape1"])
    L_cor = pd.Series(expected["L_cor"])

L_cor = L_cor.values.reshape((nstations, nstations)).T
cor = L_cor @ L_cor.T
k = np.arange(nstations)
cor[k, k] = 1.

LOGGER.info(f"Load mvnprocess TASK {taskid}")
fs = fout / f"copulafit_TASK{taskid}" / f"copulafit_mvnprocess_TASK{taskid}.zip"
df, comment = csv.read_csv(fs)

groups = df.columns.str.replace("_.*", "", regex=True).unique()
groups = [g for g in groups if re.search("ALL|02-14$", g)]

#eep_target = float(comment["eep_target"])

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
gev = marginals.GEV()

plt.close("all")
mosaic = [["diagram", stat] for stat in ["pall", "pany"]]
nrows = len(mosaic)
ncols = len(mosaic[0])
fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                 layout="tight")
axs = {
    "diagram_pall": fig.add_subplot(2, 2, 1, projection="3d"),
    "diagram_pany": fig.add_subplot(2, 2, 3, projection="3d"),
    "pall": fig.add_subplot(2, 2, 4),
    "pany": fig.add_subplot(2, 2, 2),
    }

cols = ["tab:blue", "tab:orange", "tab:green"]

for iax, (aname, ax) in enumerate(axs.items()):
    LOGGER.info(f"Plot {aname}")
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
            xthresh[ista] = gev.ppf(eep_target_plot)

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
        if re.search("any", aname):
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

        txt = r"$Pr(X>x_0 \vee Y>y_0)$" if re.search("all", aname)\
            else r"$Pr(X>x_0 \wedge Y>y_0)$"
        xa = XX1[ii].mean()
        ya = XX2[ii].mean()
        diff = np.abs(XX1 - xa) + np.abs(XX2 - ya)
        iclose = np.where(diff == diff.min())
        za = PP[iclose][0]
        z0, z1 = ax.get_zlim()
        zt0, zt1 = [z1 * w + z0 * (1 - w) for w in [0.72, 0.8]]
        ax.text(xa, ya, zt1, txt, ha="right",
                fontsize=12, fontweight="bold")
        ax.plot([xa, xa], [ya, ya], [zt0, za], "k-", lw=1)
        ax.plot(xa, ya, za, "o", mfc="k", mec="k")

        xlab = f"Peak flow {sta1} [m3.s-1]"
        ylab = f"Peak flow {sta2} [m3.s-1]"
        zlab = "Pr(X,Y) [-]"
        ax.set(xlabel=xlab, ylabel=ylab, zlabel=zlab)

        title = f"({letters[iax]})"
    else:
        stat = aname

        x0, x1 = (-8.5, -2.2) if stat == "pall" else (-1.9, -1.2)
        bins = np.logspace(x0, x1, 50)
        ax.set_xlim((10**x0, 10**x1))
        ax.set_xscale("log")

        for ig, gname in enumerate(groups):
            sel = df.loc[:, f"{gname}_log10_{stat}"]
            se = 10**sel
            ax.hist(se, bins=bins, edgecolor="0.5",
                    facecolor=cols[ig],
                    alpha=0.6)

            m = se.mean()
            ml = math.log10(m)
            if (ml - x0) * (x1 - ml) > 0:
                putils.line(ax, 0, 1, m, 0, color=cols[ig], lw=3)

        xlab = "Event Exceedance Probability [-]"
        ylab = "Sample count [-]"
        ax.set(xlabel=xlab, ylabel=ylab)

        ax.legend(fontsize="small", loc=1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

        title = f"({letters[iax]}) Statistic {stat}"

    ax.set_title(title, x=0.02, y=0.98, va="top", ha="left",
                 transform=ax.transAxes, fontweight="bold")

LOGGER.info("Saving to disk")
fp = fimg / f"{basename}.png"
fig.savefig(fp)

LOGGER.completed()
