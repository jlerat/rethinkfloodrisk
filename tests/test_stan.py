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
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt

from floodstan import marginals

from pyrethink import datahub, sample
from pyrethink import stan_test_indexing
from pyrethink import stan_test_functions
from pyrethink import stan_test_cor
from pyrethink import stan_test_student

FTESTS = Path(__file__).resolve().parent

SEED = 5446

def test_stan_indexing():
    data, _ = datahub.get_ams_concat()
    censors = datahub.get_censors(pcensor=0.2)
    sv = sample.StanSamplingMultivariate(data, copula=0., censors=censors)
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


def test_stan_student(allclose):
    N = 1000
    P = 20
    stan_data = {
        "N": N,
        "P": P
        }

    x = stan_test_student(data=stan_data)
    x = x.filter(regex="^z").values.reshape((P, N)).T

    expected = np.zeros_like(x)
    u = np.linspace(1./N, 1 - 1./N, N)
    df = np.linspace(0.5, 5, P)
    for j in range(P):
        expected[:, j] = student_t.ppf(u, df[j], loc=0, scale=1)

    diff = np.arcsinh(x) - np.arcsinh(expected)
    assert np.abs(diff).max() < 1e-5

