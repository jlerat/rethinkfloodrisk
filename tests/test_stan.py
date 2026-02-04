import re
from pathlib import Path
import math
import warnings

import pytest

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import norm
from scipy.stats import t as student_t
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_t as mvt
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt

from floodstan import marginals

from pyrethink import datahub, sample
from pyrethink import stan_test_indexing
from pyrethink import stan_test_functions
from pyrethink import stan_test_cor
from pyrethink import stan_test_copula
from pyrethink import stan_test_clusters

FTESTS = Path(__file__).resolve().parent

SEED = 5446

def test_stan_indexing():
    data, _, dows, _ = datahub.get_ams_concat()
    censors = datahub.get_censors(pcensor=0.2)
    sv = sample.StanSamplingMultivariate(data, dows, copula=0., censors=censors)
    stan_data = sv.to_dict()
    df = stan_test_indexing(data=stan_data)

    y = stan_data["y"]
    z = df.filter(regex="^z").values.reshape(y.T.shape).T
    for ivar, zv in enumerate(z.T):
        # Check missing
        ismiss = np.isnan(y[:, ivar])
        assert all(z[ismiss, ivar] == 3)
        assert all(z[~ismiss, ivar] != 3)

        # Check censored
        iscens = y[:, ivar] < stan_data["censors"][ivar]
        assert all(z[iscens, ivar] == 2)
        assert all(z[~iscens, ivar] != 2)


@pytest.mark.parametrize("kappa", [-1., -0.5, 0., 0.5, 1.])
def test_stan_functions(kappa, allclose):
    tau = 100.
    alpha = 50.

    mname = "GEV" if abs(kappa) > 0 else "Gumbel"
    marginal = marginals.factory(mname)
    marginal.params = [tau, math.log(alpha), kappa]

    stan_data = {
        "Q": 10000,
        "tau": tau,
        "alpha": alpha,
        "kappa": kappa
    }
    df = stan_test_functions(data=stan_data)

    u = df.filter(regex="^u\\[").squeeze().values
    q = df.filter(regex="^qq").squeeze().values
    expected = marginal.ppf(u)
    atol = 1e-4
    rtol = 5e-4
    assert allclose(q, expected, atol=atol, rtol=rtol)

    uu = df.filter(regex="^uu").squeeze().values
    expected = marginal.cdf(q)
    assert allclose(uu, expected, atol=atol, rtol=rtol)

    lp = df.filter(regex="^lp\\[").squeeze().values
    expected = marginal.logpdf(q)
    assert allclose(lp, expected, atol=atol, rtol=rtol)


def test_stan_cor(allclose):
    P = 5
    Q = 20000

    rho = 0.9
    cor = toeplitz(rho ** np.arange(P))
    L_cor = np.linalg.cholesky(cor)

    stan_data = {
        "P": P,
        "Q": Q,
        "L_cor": L_cor
        }

    df = stan_test_cor(data=stan_data)

    z = df.filter(regex="^zrnd").values.reshape((P, Q)).T
    zcor = np.corrcoef(z.T)
    assert allclose(zcor, cor, atol=1e-2)


def test_stan_copula(allclose):
    N = 10000
    P = 5
    stan_data = {
        "N": N,
        "P": P
        }

    x = stan_test_copula(data=stan_data)

    p0 = x.filter(regex="^p0\\[").values

    # Normal copula
    zn = x.filter(regex="^zn\\[").values
    assert allclose(zn, norm.ppf(p0), atol=1e-4)
    assert allclose(zn.mean(), 0., atol=5e-3)
    assert allclose(zn.std(), 1., atol=1e-2)

    pn = x.filter(regex="^pn\\[").values
    assert allclose(pn, p0)

    znr = x.filter(regex="^znr\\[").values.reshape((3, N)).T
    assert allclose(znr.mean(axis=0), 0, atol=5e-2)
    assert allclose(znr.std(axis=0), 1, atol=5e-2)
    cor = np.cov(znr.T)
    cor0 = 0.1 * np.eye(3) + 0.9 * np.ones((3, 3))
    assert allclose(cor, cor0, atol=3e-2)

    lpdfn = x.filter(regex="^lpdfn\\[").values
    expected = mvn.logpdf(znr, mean=np.zeros(3), cov=cor0)
    assert allclose(lpdfn, expected)

    # Student copula
    zs = x.filter(regex="^zs\\[").values.reshape((P, N)).T

    df = np.linspace(2.1, 5, P)
    assert allclose(zs.mean(axis=0)[df > 3], 0., atol=5e-3)
    assert allclose(zs.std(axis=0)[df > 3], 1., atol=5e-2)

    ps = x.filter(regex="^ps\\[").values.reshape((P, N)).T
    assert allclose(ps, p0[:, None])

    expected = np.zeros((N, P))
    for j in range(P):
        adjust = math.sqrt((df[j] - 2) / df[j])
        expected[:, j] = student_t.ppf(p0, df[j], loc=0, scale=adjust)

    diff = np.arcsinh(zs) - np.arcsinh(expected)
    assert np.abs(diff).max() < 1e-5

    zsr = x.filter(regex="^zsr\\[").values.reshape((3, N)).T
    assert allclose(zsr.mean(axis=0), 0, atol=5e-2)
    # We should use np.cov here, not corrcoef...
    cor = np.corrcoef(zsr.T)
    assert allclose(cor, cor0, atol=3e-2)

    lpdfs = x.filter(regex="^lpdfs\\[").values
    dfr = x.filter(regex="dfr")[0]
    adjust = (dfr - 2) / dfr
    expected = mvt.logpdf(zsr, df=dfr, loc=np.zeros(3),
                          shape=adjust * cor0)
    assert allclose(lpdfs, expected)



def test_stan_clusters(allclose):
    data, times, dows, _ = datahub.get_ams_concat()
    censors = datahub.get_censors(pcensor=0.3)
    sv = sample.StanSamplingMultivariate(data, dows,
                                         copula=0,
                                         censors=censors)
    stan_data = sv.to_dict()
    N = stan_data["N"]
    P = stan_data["P"]

    x = stan_test_clusters(data=stan_data)

    assert N == x["Ncheck"]
    assert P == x["Pcheck"]

    clust = stan_data["clusters"]
    idx = x.filter(regex="indexes").values.reshape((P + 1, P, N)).T
    cnt = idx[:, :, 0]

    for i in range(N):
        cl = clust[i]
        assert allclose(cl.sum(axis=1), cnt[i])

        idxc = idx[i, :, 1:]
        assert allclose(cnt[i], (idxc > 0).sum(axis=1))

        for j in range(P):
            ii = np.where(cl[j] == 1)[0] + 1
            n = int(cnt[i, j])
            assert allclose(ii, idxc[j, :n])
