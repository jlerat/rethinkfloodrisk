#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-10-24 15:22:39.682506
## Comment : Test inverse phi approx
##
## ------------------------------


import re
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm
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
def phi_approx(x):
    return  1. / (1. + np.exp(-0.07056 * x**3 - 1.5976 * x))

def inv_phi_approx(p):
    # See Bowling et al. (2009), Equation 10
    # Phi(x) ~ 1 / (1 + exp(-0.07056 x**3 - 1.5976 x))

    a = 1.5976 / 0.07056
    b = np.log(1./p - 1) / 0.07056

    # We now need to solve z**3 + az + b = 0
    sqdelta = np.sqrt(a**3 / 27 + b**2 / 4)
    u1 = -b / 2 - sqdelta
    u2 = -b / 2 + sqdelta

    return np.cbrt(u1) + np.cbrt(u2)

eps = 1e-4
p = np.linspace(eps, 1 - eps, 1000)
x1 = norm.ppf(p)
x2 = inv_phi_approx(p)

#x = np.linspace(-5, 5, 1000)
#p1 = norm.cdf(x)
#p2 = phi_approx(x)

plt.close("all")
fig, ax = plt.subplots()
ax.plot(p, x1)
ax.plot(p, x2)
plt.show()

LOGGER.completed()

