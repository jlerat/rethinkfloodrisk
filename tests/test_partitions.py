import re
import json
from pathlib import Path
from itertools import combinations
import math
import warnings

import pytest

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

from scipy.stats import t as student_t
from scipy.stats import multivariate_t as mvt

from scipy.stats import invwishart
from scipy.stats import ttest_ind, ks_2samp
from scipy.stats import kstest
import matplotlib.pyplot as plt

from floodstan import report
from floodstan import sample as fsample
from floodstan import marginals
from floodstan import bivariate_censored_sampling

from pyrethink import partitions
from pyrethink import datahub
from pyrethink import mv_censored_no_missing_sampling


@pytest.mark.parametrize("nelems", [1, 2, 3, 4, 5, 9])
def test_partitions_size(nelems, allclose):
    parts = partitions.Partitions(nelems)
    print(f"N subsets = {parts.nsubsets:6,d}")

    if nelems == 1:
        assert parts.nsubsets == 1
    elif nelems == 2:
        assert parts.nsubsets == 2
    elif nelems == 3:
        assert parts.nsubsets == 5
    elif nelems == 4:
        assert parts.nsubsets == 15
    elif nelems == 5:
        assert parts.nsubsets == 52
    elif nelems == 9:
        assert parts.nsubsets == 21147
    elif nelems == 10:
        assert parts.nsubsets == 115975

    if nelems == 1:
        return

    for itest in range(5):
        k = np.random.randint(0, parts.nsubsets)
        pair_in_same = parts.pair_in_same_cluster[k]
        subs = parts.find_subset(pair_in_same)
        assert len(subs) == 1


def test_partitions_sets(allclose):
    nelems = 6
    parts = partitions.Partitions(nelems)
    for ipart in range(parts.nsubsets):
        sets = parts.ipart2sets(ipart)

        part = parts.subsets[ipart]
        part = part[part.sum(axis=1) > 0]
        expected = [np.where(p == 1)[0][0] for p in part.T]
        assert allclose(sets, expected)


@pytest.mark.parametrize("nelems", [4, 5])
def test_partitions_probabilities(nelems, allclose):
    parts = partitions.Partitions(nelems)
    ns = parts.nsubsets

    # Sample random partition ids
    pp = np.maximum(np.random.uniform(0, 1, size=ns) - 0.8, 0)
    pp /= pp.sum()
    pids = np.random.choice(np.arange(ns), p=pp, size=50)

    dalpha = 2.
    probs = parts.compute_probabilities(pids, dalpha)
    assert len(probs) == ns

    cnt = pd.Series(pids).value_counts()
    total = (cnt + dalpha - 1).sum()
    for ipart in np.unique(pids):
        n = (pids == ipart).sum()
        assert probs[ipart] == (n + dalpha - 1) / total


@pytest.mark.parametrize("nelems", [3, 4, 5])
def test_partitions_sample(nelems, allclose):
    parts = partitions.Partitions(nelems)
    ns = parts.nsubsets

    pp = np.maximum(np.random.uniform(0, 1, size=ns) - 0.2, 0)
    pp /= pp.sum()
    pids = np.random.choice(np.arange(ns), p=pp, size=50)
    dalpha = 2.
    probs = parts.compute_probabilities(pids, dalpha)

    iparts = parts.sample(probs, 1000000)
    pp = pd.Series(iparts).value_counts().sort_index() / len(iparts)
    expected = [probs[i] for i in pp.index]
    assert allclose(pp, expected, atol=1e-2)


