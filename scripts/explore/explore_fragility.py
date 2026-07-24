#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-19 Sun 06:02 PM
## Comment : Explore fragility concept
##
## ------------------------------


import sys
import os
import re
import json
from itertools import combinations
import math
import argparse
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from scipy.special import gamma, digamma
import matplotlib.pyplot as plt


from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from floodstan import marginals

from pyrethink import datahub

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
ams, _, _, _ = datahub.get_ams_concat()
nsta = ams.shape[1]
gev = marginals.GEV()

aep = 1. / 500
eta = 2

plt.close("all")
fig, axs = plt.subplots(ncols=3, nrows=3)

for iax, (stationid, se) in enumerate(ams.items()):
    data = se.loc[se.notnull()].values.copy()
    imax = np.argmax(data)



    ax = axs.flat[iax]

    #ax.plot(xmax, sens0.x, "ok", ms=7)
    ax.plot(xmax, Q0, "ok", ms=7)

    ax.plot(xx, sens, label="True")

    #x0, x1 = xx.min(), xx.max()
    #ax.plot([x0, x1], [sens0.x + sens0.dx * (u - xmax) for u in [x0, x1]], "k--")

    dx = (np.max(sens) - np.min(sens)) / (xx[-1] - xx[0])
    ax.set(title=f"{stationid} : n={len(data)} dx={dx:0.2f}")

plt.show()



LOGGER.completed()

