#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-01-27 11:17:46.104880
## Comment : Generate random partitions
##
## ------------------------------


import sys
import os
import re
import random
import json
import math
from itertools import combinations
from itertools import product as prod
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson
from scipy.special import lambertw, factorial

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

class Partitions():
    def __init__(self, nelements):
        self.nelements = nelements
        self.data = list(range(nelements))

        # Initialise
        self.subsets = []
        self.same_cluster = []
        self.counts = []
        self.nsubsets = 0

        # Populate partitions
        self.add_subsets(0, [])

    def add_subsets(self, index, ans):
        data = self.data
        nel = self.nelements
        ncombs = (nel * (nel - 1)) // 2

        if index == len(data):
            combs = []
            nmax = 0
            same = [0] * ncombs
            for ipart, parts in enumerate(ans):
                comb = [0] * nel
                for d in parts:
                    comb[d] = 1
                combs.append(comb)
                nmax = max(nmax, len(parts))
                same = [comb[a]*comb[b] or same[ic]
                        for ic, (a, b) in enumerate(combinations(range(nel), 2))]

            self.counts.append(len(combs))
            self.subsets.append(combs)
            self.same_cluster.append(same)
            self.nsubsets += 1

            return

        elem = data[index]

        for i in range(len(ans)):
            ans[i].append(elem)
            self.add_subsets(index + 1, ans)
            ans[i].pop()

        ans.append([elem])
        self.add_subsets(index + 1, ans)
        ans.pop()

    def select(self, nevents):
        selected = []
        for i in range(self.nsubsets):
            nev = self.counts[i]
            if nev == nevents:
                selected.append(self.subsets[i])

        return selected

    def random(self, nsamples, tokenize=False):
        nel = self.nelements
        x = lambertw(nel).real

        samples = []
        ex = np.arange(1, nel + 1)
        f = factorial(ex)
        mus = x**ex / f

        for i in range(nsamples):
            nz = 0
            go_on = True
            while go_on:
                s = 0
                z = []
                nz = 0
                for j in range(nel):
                    z.append(poisson.rvs(mu=mus[j]))
                    nz += 1
                    s += (1 + j) * z[j]
                    if s == nel:
                        go_on = False
                        break

            rdata = np.random.choice(self.data, nel, replace=False)
            #rdata = self.data
            combs = []
            nstart = 0
            for j in range(nz):
                nb = z[j]
                nelc = j + 1
                for k in range(nb):
                    comb = [0] * nel
                    for e in rdata[nstart + nelc * k: nstart + nelc * (k + 1)]:
                        comb[e] = 1

                    combs.append(comb)

                nstart += nelc * nb

            if tokenize:
                combs = "-".join("".join(str(e) for e in comb) for comb in combs)

            samples.append(combs)

        return samples


for i in range(4, 5):
    parts = Partitions(i)
    ntot = parts.nsubsets
    LOGGER.info(f"{i} elements = {ntot:5,d} subsets", nret=2)

    smp = parts.random(1000, True)
    nuni = len(set(smp))
    LOGGER.info(f"unique random = {nuni:5,d} subsets", ntab=1)

    ncheck1 = 0
    counts = pd.DataFrame(0, index=np.arange(1, i+1),
                          columns=np.arange(1, i+1))
    counts.index.name = "nevents"
    counts.columns.name = "nstationmax"

    for nev in range(1, i + 1):
        subs = parts.select(nev)

        ntot1 = len(subs)
        ncheck1 += ntot1
        LOGGER.info(f"{nev} elements in partition = {ntot1:5,d} subsets",
                    ntab=1, nret=1)
    assert ncheck1 == ntot

LOGGER.completed()

