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

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.plot import putils

from pyrethink import datahub
from pyrethink import copulas

from floodstan import marginals

from figA_impact_of_period_on_FFA import get_script_paths
from figA_impact_of_period_on_FFA import get_logger, get_taskids, get_data
from figA_impact_of_period_on_FFA import get_iter_options, select_data

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


def process(config, script_paths, logger, data):
    for pcensor, rho_min, has_cluster, copula_shape in get_iter_options(data):
        copula_type = 1 if copula_shape > 0 else 0

        _, obs_data, mvnproc, expected, _ = select_data(data,
                                                        pcensor=pcensor,
                                                        rho_min=rho_min,
                                                        has_cluster=has_cluster,
                                                        copula_shape=copula_shape)
        if len(mvnproc) == 0:
            continue
        assert len(mvnproc) == 1

        rn = next(iter(mvnproc))
        obs_data = obs_data[rn]
        mvnproc = mvnproc[rn]
        expected = expected[rn]

        ylocs = pd.Series(expected["ylocs"])
        ylogscales = pd.Series(expected["ylogscales"])
        yshape1 = pd.Series(expected["yshape1"])
        cor = pd.Series(expected["corr_IW"])
        nstations = len(ylocs)
        cor = cor.values.reshape((nstations, nstations)).T

        groups = mvnproc.columns.str.replace("_.*", "", regex=True).unique()
        pat = f"{sta1[-2:]}-{sta2[-2:]}|{sta2[-2:]}-{sta1[-2:]}"
        groups = [g for g in groups if re.search(f"G(ALL|{pat})$", g)]

        stationids = obs_data.columns.tolist()
        nstations = len(stationids)
        ista1 = stationids.index(config.sta1)
        ista2 = stationids.index(config.sta2)

        logger.info(f"-- Plotting {rn.text} --", nret=1)

        plt.close("all")
        mosaic = [["diagram", stat] for stat in ["pall", "pany"]]
        nrows = len(mosaic)
        ncols = len(mosaic[0])
        fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                         layout="tight")
        axs = {
            "diagram_all": fig.add_subplot(2, 2, 1, projection="3d"),
            "diagram_any": fig.add_subplot(2, 2, 3, projection="3d"),
            "all": fig.add_subplot(2, 2, 2),
            "any": fig.add_subplot(2, 2, 4),
            }

        cols = ["tab:blue", "tab:orange", "tab:green"]

        paef = pe.withStroke(linewidth=4,
                             foreground="w")

        gev = marginals.GEV()

        for iax, (aname, ax) in enumerate(axs.items()):
            logger.info(f"Plot {aname}", ntab=1)
            evtype = "OR" if re.search("any", aname) else "AND"

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
                    xx[ista] = np.linspace(xa, xb, config.ngrid)
                    zz[ista] =copulas.copula_marginal_ppf(copula_type,
                                                          copula_shape,
                                                          gev.cdf(xx[ista]))
                    marg[ista] = gev.pdf(xx[ista])
                    xthresh[ista] = gev.ppf(config.aep_target_plot)

                XX1, XX2 = np.meshgrid(xx[ista1], xx[ista2])
                ZZ1, ZZ2 = np.meshgrid(zz[ista1], zz[ista2])
                ZZ = np.dstack((ZZ1, ZZ2))

                ii = [ista1, ista2]

                ccs = copulas.Copula(copula_type, copula_shape, 2)
                ccs.corr = cor[ii][:, ii]
                PP = np.zeros((ZZ.shape[0], ZZ.shape[1]))
                for i1 in range(ZZ.shape[0]):
                    for i2 in range(ZZ.shape[1]):
                        zz = ZZ[i1, i2]
                        PP[i1, i2] = ccs.pdf_given_partition(0, zz)

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
                if evtype == "AND":
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

                txt = r"$Pr(X_1>x_1^* \cap X_2>x_2^*)$" if evtype == "AND"\
                    else r"$Pr(X_1>x_1^* \cup X_2>x_2^*)$"
                xa = XX1[ii].mean()
                ya = XX2[ii].mean()
                diff = np.abs(XX1 - xa) + np.abs(XX2 - ya)
                iclose = np.where(diff == diff.min())
                za = PP[iclose][0]
                z0, z1 = ax.get_zlim()
                wref = 0.5 if evtype == "AND" else 0.85
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
                if stat == "all":
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
                    cn = f"{gname}_{stat}_log10aep{aep_target:02d}"
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
                        w = 0.3
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

                title = r"$\bigcap_i\ X_i > x_i^*$" if evtype == "AND"\
                    else r"$\bigcup_i\ X_i > x_i^*$"
                title = f"({letters[iax]}) Probability of '{evtype}' event {title}"

            ax.set_title(title, x=0.02, y=0.98, va="top", ha="left",
                         transform=ax.transAxes, fontweight="bold",
                         path_effects=[paef])

        basename = script_paths.basename
        fp = f"{basename}_{rn.text}_v{config.version}.png"
        fp = script_paths.fimg / fp
        fig.savefig(fp, dpi=config.fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood frequency plots",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="version",
                        type=int, required=True)
    parser.add_argument("-p", "--pcensor", help="Censoring threshold value",
                        type=float, default=0.3)
    parser.add_argument("-di", "--diag", help="Show stan diagnostics",
                        action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="Debug",
                        action="store_true", default=False)
    parser.add_argument("-r", "--rho_mins", help="Minimum rho value",
                        type=str, default="-1|0")
    parser.add_argument("-s", "--copula_shapes", help="Copula shapes selected",
                        type=str, default="0|3")
    args = parser.parse_args()

    # Config
    CF = namedtuple("Config", ["version", "pcensor", "rho_mins",
                               "awidth", "aheight", "fdpi",
                               "excludes", "copula_shapes",
                               "diag", "debug",
                               "load_obs_data",
                               "load_ffa",
                               "load_mvnproc",
                               "load_expected_params",
                               "load_postpred_checks",
                               "sta1", "sta2",
                               "ngrid", "aep_target",
                               "aep_target_plot"])
    awidth = 6
    aheight = 5
    fdpi = 300
    excludes = ["NONE"]
    load_ffa = False
    load_obs_data = True
    load_mvnproc = True
    load_expected_params = True
    load_postpred_checks = False

    sta1 = "203002"
    sta2 = "203014"

    ngrid = 40
    aep_target = 1
    # aep used for the right-hand side plot (0.99 is too extreme, can't see it)
    aep_target_plot = 0.95


    config = CF(args.version, args.pcensor,
                args.rho_mins.split("|"),
                awidth, aheight, fdpi,
                excludes,
                args.copula_shapes.split("|"),
                args.diag, args.debug,
                load_obs_data, load_ffa,
                load_mvnproc, load_expected_params,
                load_postpred_checks,
                sta1, sta2, ngrid,
                aep_target, aep_target_plot)

    # Baseline
    source_file = Path(__file__).resolve()
    script_paths = get_script_paths(config, source_file)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
