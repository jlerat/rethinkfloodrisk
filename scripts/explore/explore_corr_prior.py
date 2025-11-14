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
from scipy import fft, linalg
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
nv = 4

# Cholesky of a circulant matrix
rho = 0.9
x = rho * np.ones(nv)
x[0] = 1
M = (1 - rho) * np.eye(nv) + rho * np.ones((nv, nv))

# FFT matrix
w = np.exp(-2*math.pi*1j/nv)
kk = np.mgrid[:nv, :nv]
F = w**(kk[0]*kk[1])
Fc = F.conj()

#lams = fft.fft(x)
lams = (nv - (1 - rho)*(nv - 1)) * np.ones(nv)
lams[1:] = 1 - rho

FF = Fc / math.sqrt(nv)
FFi = Fc.conj() / math.sqrt(nv)
assert np.allclose(M, FF@np.diag(lams)@FFi)

D = np.diag(np.sqrt(lams))
C = Fc @ D / math.sqrt(nv)
assert np.allclose(M, C @ C.conj())

sys.exit()


rv = invwishart(scale=np.eye(nv), df=int(nv * 1.2))

nsmp = 10000
m = rv.rvs(size=nsmp)
c = np.zeros_like(m)
c2 = np.zeros_like(c)

rho = 1 - 0.2
B = toeplitz(rho ** np.arange(nv))

for i in range(nsmp):
    M = m[i]
    si = 1./ np.sqrt(np.diag(M))[:, None]
    C = si * M * si.T
    c[i] = C

    C2 = (B + C) / 2
    c2[i] = C2

bins = np.linspace(-1, 1, 50)
plt.close("all")
i1 = 0
i2 = nv - 1
plt.hist(c[:, i1, i2], bins=bins, fc="0.8", ec="0.2", alpha=0.2)
plt.hist(c2[:, i1, i2], bins=bins, fc="0.8", ec="0.2", alpha=0.9)
plt.show()

LOGGER.completed()

