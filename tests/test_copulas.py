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

from pyrethink import datahub
from pyrethink import copulas

FTESTS = Path(__file__).resolve().parent

def get_type(copula_shape):
    return 0 if abs(copula_shape) < 1e-10 else 1

@pytest.mark.parametrize("copula_shape", [0., 2.5, 5., 10.])
def test_copula_marginals(copula_shape, allclose):
    copula_type = get_type(copula_shape)

    n = 10000
    u1 = np.linspace(1./n, 1 - 1./n, n)
    z1 = copulas.copula_marginal_ppf(copula_type, copula_shape, u1)

    u2 = copulas.copula_marginal_cdf(copula_type, copula_shape, z1)
    assert allclose(u1, u2)

    z2 = copulas.copula_marginal_ppf(copula_type, copula_shape, u2)
    assert allclose(z1, z2)

    n = 1000000
    u = np.random.uniform(size=n)
    z = copulas.copula_marginal_ppf(copula_type, copula_shape, u)
    assert allclose(z.mean(), 0, atol=5e-3)
    assert allclose(z.std(), 1, atol=5e-2)


@pytest.mark.parametrize("repeat", range(10))
def test_cov2corr(repeat, allclose):
    nsta = 6
    cov = invwishart.rvs(df=nsta+1, scale=np.eye(nsta))
    corr = copulas.cov2corr(cov)
    d = np.diag(1./np.sqrt(np.diag(cov)))
    assert allclose(corr, d @ cov @ d)


@pytest.mark.parametrize("repeat", range(10))
def test_random_corr(repeat, allclose):
    nsta = 6
    z1 = []

    z2 = []
    rho0, rho1 = 0.2, 0.8
    c0 = copulas.corr_ref(nsta, rho0)
    c1 = copulas.corr_ref(nsta, rho1)

    nrepeat = 100
    for i in range(nrepeat):
        idx = np.triu_indices(nsta, 1)
        x = copulas.random_corr(nsta)[idx]
        z1.append(x)

        x = copulas.random_corr(nsta, c0, c1)[idx]
        z2.append(x)

    z1 = (np.array(z1) + 1) / 2
    pv1 = np.array([kstest(zi, "uniform").pvalue for zi in z1.T])
    assert np.percentile(pv1, 10) > 1e-3

    z2 = (np.array(z2) - rho0) / (rho1 - rho0)
    pv2 = np.array([kstest(zi, "uniform").pvalue for zi in z2.T])
    assert np.percentile(pv2, 10) > 1e-3


@pytest.mark.parametrize("copula_shape", [0., 3., 4.])
def test_conditional_sample(copula_shape, allclose):
    copula_type = get_type(copula_shape)
    nsta = 4
    ccs = copulas.Copula(copula_type, copula_shape, nsta)

    rho0, rho1 = 0.8, 0.99
    c0 = copulas.corr_ref(nsta, rho0)
    c1 = copulas.corr_ref(nsta, rho1)
    ccs.corr = copulas.random_corr(nsta, c0, c1)

    icond = np.array([0])
    itarget = np.array([1, 2, 3])

    ucond = np.array([0.6])

    zcond = copulas.copula_marginal_ppf(copula_type, copula_shape, ucond)
    nrepeat = 5000

    atol_mean = 5./math.sqrt(nrepeat)
    atol_cov = 10./math.sqrt(nrepeat)

    # Check conditional on full cluster
    z = np.array([ccs.conditional_sample_given_partition(0, icond, zcond, itarget)
                  for i in range(nrepeat)])

    zz_cond = np.empty(z.shape)
    zz_target = np.empty(z.shape[0])
    iok = np.zeros(nrepeat)
    nok, niter = 0, 0

    while nok < nrepeat:
        to_sample = np.where(iok == 0)[0]
        n_to_sample = len(to_sample)
        if copula_type == 0:
            zz = mvn.rvs(mean=np.zeros(nsta), cov=ccs.corr_rescaled,
                         size=10 * n_to_sample)
        else:
            zz = mvt.rvs(loc=np.zeros(nsta), shape=ccs.corr_rescaled,
                         df=copula_shape,
                         size=10 * n_to_sample)
        iclose = np.abs(zz[:, icond[0]] - zcond[0]) < 1e-3
        tmp = zz[iclose]
        tmp = tmp[:min(len(tmp), n_to_sample)]

        ok = to_sample[:len(tmp)]
        zz_cond[ok] = tmp[:, itarget]
        zz_target[ok] = tmp[:, icond[0]]
        iok[ok] = 1
        nok = (iok == 1).sum()
        niter += 1

    cov = np.cov(z.T)
    expected = np.cov(zz_cond.T)
    assert np.allclose(cov, expected, atol=3e-2)

    # Check independence of clusters
    for ipart in range(ccs.partitions.nsubsets):
        z = [ccs.conditional_sample_given_partition(ipart, icond, zcond, itarget)
             for i in range(nrepeat)]
        z = np.array(z)

        sets = ccs.partitions.ipart2sets(ipart)
        sets_cond = sets[icond]
        sets_target = sets[itarget]

        # If samples are in different sets, the mean should be 0
        # and covariance should equal to the corresponding elements in corr
        # (i.e. indepent from zcond)
        idiff = sets_target != sets_cond
        if idiff.sum() > 0:
            zm = z[:, idiff].mean()
            assert allclose(zm, 0., atol=atol_mean)

            # WATCH OUT ! THIS IS WEIRD
            #zc = np.cov(z[:, idiff].T)
            zc = np.corrcoef(z[:, idiff].T)

            ii = itarget[idiff]
            expected = ccs.corr[ii][:, ii]
            assert allclose(zc, expected, atol=atol_cov)


@pytest.mark.parametrize("copula_shape", [0., 4.])
def test_copula_sample(copula_shape, allclose):
    copula_type = get_type(copula_shape)
    nsta = 4
    ccs = copulas.Copula(copula_type, copula_shape, nsta)
    ccs.corr = copulas.random_corr(nsta)
    nsamples = 500000

    # Sample given ipart
    for ipart in range(ccs.partitions.nsubsets):
        z = ccs.sample_z_given_partition(ipart, nsamples)

        zm = z.mean(axis=0)
        assert allclose(zm, 0, atol=1e-2)
        zs = z.std(axis=0)
        assert allclose(zs, 1, atol=1e-1)

        sets = ccs.partitions.ipart2sets(ipart)
        for iset in np.unique(sets):
            idx = iset == sets
            cov = np.cov(z[:, idx].T)
            expected = ccs.corr[idx][:, idx]
            assert allclose(cov, expected, atol=5e-2)

    # Random probs
    ns = ccs.partitions.nsubsets
    pp = np.maximum(np.random.uniform(0, 1, size=ns) - 0.2, 0)
    pp /= pp.sum()
    pids = np.random.choice(np.arange(ns), p=pp, size=50)
    probs = ccs.partitions.compute_probabilities(pids, 1)

    # Sample integrating ipart out
    ns = ccs.partitions.nsubsets
    ccs.partitions.probabilities = np.random.uniform(0, 1, ns)
    u, iparts = ccs.sample_u(probs, nsamples)

    pv = np.array([kstest(ui, "uniform").pvalue for ui in u.T])
    assert np.median(pv) > 1e-2

    pp = pd.Series(iparts).value_counts() / nsamples
    expected = pd.Series(probs)[pp.index]
    assert allclose(pp, expected, atol=5e-3)


@pytest.mark.parametrize("copula_shape", [0., 5., 10.])
@pytest.mark.parametrize("repeat", range(10))
def test_copula_cdf(repeat, copula_shape, allclose):
    copula_type = get_type(copula_shape)
    nsta = 4
    ccs = copulas.Copula(copula_type, copula_shape, nsta)
    rho0, rho1 = 0.5, 0.99
    c0 = copulas.corr_ref(nsta, rho0)
    c1 = copulas.corr_ref(nsta, rho1)
    ccs.corr = copulas.random_corr(nsta, c0, c1)
    nsamples = 500000

    z0 = np.random.uniform(-0.5, 0.5, size=nsta)
    z0 = np.array([0.49, -0.5, 0.39, -0.1])
    print()
    print()
    print(f"z0 = {z0.round(2)}")
    print()

    atol = 2e-2

    for ipart in range(ccs.partitions.nsubsets):
        z = ccs.sample_z_given_partition(ipart, nsamples)

        ce = (z - z0[None, :] < 0).all(axis=1).sum() / nsamples
        c0 = ccs.cdf_given_partition(ipart, z0)
        assert allclose(ce, c0, atol=atol)

        se = (z - z0[None, :] >= 0).all(axis=1).sum() / nsamples
        s0 = ccs.survival_given_partition(ipart, z0)
        passed = abs(se - s0) < atol

        print(f"ipart={ipart} / s0={s0:0.3f} se={se:0.3f} passed={passed}")
        #assert allclose(se, s0, atol=atol)

