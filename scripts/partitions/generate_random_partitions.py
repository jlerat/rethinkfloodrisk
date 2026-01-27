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
import json
import math
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
        self.data = list(range(nelements))
        self.subsets = []
        self.add_subsets(0, [])

    def add_subsets(self, index, ans):
        data = self.data
        if index == len(data):
            self.subsets.append([[el for el in s] for s in ans])
            return

        elem = data[index]

        for i in range(len(ans)):
            ans[i].append(elem)
            self.add_subsets(index + 1, ans)
            ans[i].pop()

        ans.append([elem])
        self.add_subsets(index + 1, ans)
        ans.pop()

    def select(self, cmin, cmax):
        selected = []
        for i, subs in enumerate(self.subsets):
            csub = max(len(s) for s in subs)
            if csub >= cmin and csub <= cmax:
                selected.append(subs)

        return selected

for i in range(2, 9):
    parts = Partitions(i)
    n = len(parts.subsets)
    LOGGER.info(f"Partitions of {i} elements = {n:8,d} subsets", nret=1)

    for cmin in range(1, i + 1):
        subs = parts.select(cmin, cmin)
        LOGGER.info(f"Max card {cmin}: {len(subs):8,d} subsets", ntab=1)


LOGGER.completed()

