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
from pathlib import Path

import numpy as np
import pandas as pd

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
        self.subsets = []
        self.counts = []
        self.nsubsets = 0
        self.add_subsets(0, [])

    def add_subsets(self, index, ans):
        data = self.data
        nel = self.nelements

        if index == len(data):
            combs = []
            nmax = 0
            for ipart, parts in enumerate(ans):
                comb = [0] * nel
                for d in parts:
                    comb[d] = 1
                combs.append(comb)
                nmax = max(nmax, len(parts))

            self.counts.append([len(combs), nmax])
            self.subsets.append(combs)
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

    def select(self, nevents, nelem=None):
        selected = []
        for i in range(self.nsubsets):
            nev, nel = self.counts[i]
            cel = True if nelem is None else nel == nelem
            if nev == nevents and cel:
                selected.append(self.subsets[i])

        return selected

    def random(self, nsamples, tokenize=False):
        nel = self.nelements
        samples = []
        nsubs = self.nsubsets
        k = np.random.choice(np.arange(nsubs), nsamples)

        for i in range(nsamples):
            subs = self.subsets[k[i]]
            if tokenize:
                combs = "-".join("".join(str(c) for c in s) for s in subs)
            else:
                combs = subs

            samples.append(combs)

        return samples


for i in range(2, 6):
    parts = Partitions(i)
    ntot = parts.nsubsets
    LOGGER.info(f"{i} elements = {ntot:5,d} subsets", nret=2)

    smp = parts.random(50000, True)
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

        ncheck2 = 0
        for nel in range(1, i + 1):
            subs = parts.select(nev, nel)
            ntot2 = len(subs)
            counts.loc[nev, nel] = ntot2
            ncheck2 += ntot2
            LOGGER.info(f"{nel} max elements = {ntot2:8,d} subsets", ntab=2)

        assert ncheck2 == ntot1

    assert ncheck1 == ntot
LOGGER.completed()

