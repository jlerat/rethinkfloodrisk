#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-11-14 22:00:13.975234
## Comment : Explore correlation matrix prior
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import invwishart
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


from hydrodiy.io import csv, iutils

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
nv = 50

# Cholesky of a circulant matrix
rho = 0.9
x = rho * np.ones(nv)
x[0] = 1
C0 = (1 - rho) * np.eye(nv) + rho * np.ones((nv, nv))

def getL(rho, nv):
    L = np.zeros((nv, nv))
    for i in range(nv):
        for k in range(i):
            L[i, k] = (rho - (L[i, :k] * L[k, :k]).sum()) / L[k, k]
        L[i, i] = math.sqrt(1 - (L[i, :i]**2).sum())
    return L

L0 = getL(rho, nv)
assert np.allclose(C0, L0@L0.T)

rv = invwishart(scale=np.eye(nv), df=int(1.1 * nv))

def corrnorm(M):
    si = 1./ np.sqrt(np.diag(M))[:, None]
    return si * M * si.T

# Check corr ok
C = corrnorm(rv.rvs())
L = np.linalg.cholesky(C)

nsmp = 10000
M = np.zeros((nsmp, 4, nv, nv))

rho_min = 0.2
rho_max = 1.0
w1 = (rho_max - rho_min) / 2
w0 = (rho_max + rho_min) / 2

for i in range(nsmp):
    C = rv.rvs()
    c[i, 0] = C
    L = np.linalg.cholesky(C)
    c[i, 1] = L

    c[i, 2] = C * w1 + C0 * w0

bins = np.linspace(-1, 1, 50)
plt.close("all")
i1 = 0
i2 = nv - 1
plt.hist(c[:, i1, i2], bins=bins, fc="0.8", ec="0.2", alpha=0.2)
plt.hist(c2[:, i1, i2], bins=bins, fc="0.8", ec="0.2", alpha=0.9)
plt.show()

LOGGER.completed()

