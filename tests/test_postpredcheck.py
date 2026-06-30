from pathlib import Path
from itertools import combinations
import re
import pytest
import math
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import t as student_t

from pyrethink import datahub
from pyrethink import postpredchecks as ppc

from floodstan.marginals import GEV

from test_copulas import COPULA_SPECS

FTESTS = Path(__file__).resolve().parent

_, _, _, sta = datahub.get_ams_concat()
NSTATIONS = len(sta)

DATA = pd.read_csv(FTESTS / "censored_missing_data.zip",
                   index_col=0, parse_dates=True)

SAMPLES = pd.read_csv(FTESTS / "censored_missing_samples.zip")
SAMPLES_FACTORS = pd.read_csv(FTESTS / "censored_missing_factors_samples.zip")

MARGINAL = GEV()


@pytest.mark.parametrize("nsta", [3, 6])
@pytest.mark.parametrize("nsamples", [500, 5000])
def test_joint_exceedance_probabilities_independent(nsta, nsamples, allclose):
    pmax = 95 if nsta == 3 else 75
    perc = np.linspace(50, pmax, 10)
    x = np.random.uniform(0, 1, size=(nsamples, nsta))

    p0, p1 = ppc.joint_exceedance_probabilities(x, perc)

    u = perc * 1e-2
    expected = u ** nsta
    err = np.abs(np.arcsinh(p0) - np.arcsinh(expected))

    atol = 2e-2 if nsamples == 5000 else 7e-2
    assert err.max() < atol

    expected = (1 - u) ** nsta
    err = np.abs(np.arcsinh(p1) - np.arcsinh(expected))
    assert err.max() < atol


@pytest.mark.parametrize("nsta", [3, 6])
@pytest.mark.parametrize("nsamples", [500, 5000])
def test_joint_exceedance_probabilities_correlated(nsta, nsamples, allclose):
    pmax = 95 if nsta == 3 else 75
    perc = np.linspace(50, pmax, 10)

    rho = 0.9
    cov = (1 - rho) * np.eye(nsta) + rho * np.ones((nsta, nsta))
    mean = [0.] * nsta
    x = np.random.multivariate_normal(mean=mean, cov=cov, size=nsamples)

    p0, p1 = ppc.joint_exceedance_probabilities(x, perc)

    u = perc * 1e-2
    rv = mvn(mean=mean, cov=cov)
    z = norm.ppf(u)
    zz = np.repeat(z[:, None], nsta, axis=1)
    expected = rv.cdf(zz)
    err = np.abs(np.arcsinh(p0) - np.arcsinh(expected))

    atol = 2e-2 if nsamples == 5000 else 5e-2
    assert err.max() < atol

    expected = rv.cdf(-zz)
    err = np.abs(np.arcsinh(p1) - np.arcsinh(expected))
    assert err.max() < atol


@pytest.mark.parametrize("station", np.arange(NSTATIONS).tolist())
def test_univariate_statistics(station):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station]

    un = ppc.univariate_statistics(data)
    assert un.notnull().all()
    assert len(un) == 9


@pytest.mark.parametrize("rho", [0.8, 0.95])
@pytest.mark.parametrize("nsta", [3, 6])
def test_multivariate_statistics(rho, nsta, allclose):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks
    mv = ppc.multivariate_dependence_statistics(data)
    assert mv.shape == (9, 3)
    assert mv.notnull().all().all()

    # Test routine for multivariate normal
    mean = np.zeros(nsta)
    cov = (1 - rho) * np.eye(nsta) + rho * np.ones((nsta, nsta))
    rv = mvn(mean=mean, cov=cov)
    nsamples = 5000
    data = rv.rvs(size=nsamples)
    qt = np.linspace(50, 90, 11)
    mv = ppc.multivariate_dependence_statistics(data, qt)

    x = np.repeat(mv.index.values[:, None] / 100, nsta, axis=1)
    z = norm.ppf(x)
    expected = 2 - rv.logcdf(z) / np.log(x[:, 0])
    err = np.abs(np.arcsinh(mv.xi) - np.arcsinh(expected))
    assert err.max() < 7e-2


def tests_krupskii():
    nsamples = 100
    u = np.random.uniform(0, 1, nsamples)
    v = np.random.uniform(0, 1, nsamples)
    k = ppc.krupskii(u, v)


@pytest.mark.parametrize("pair", combinations(np.arange(6), 2))
def test_bivariate_statistics_obs(pair, allclose):
    ams, _, _, stations = datahub.get_ams_concat()
    ams = ams.iloc[:, list(pair)]
    biv = ppc.bivariate_dependence_statistics(ams)
    dep = biv["dependence"]

    u = dep.index * 1e-2
    xi0, xi1 = ppc.xi_bounds(u)

    assert np.all(dep.xi >= xi0)
    assert np.all(dep.xi <= xi1)
    assert np.all(dep.xibar >= -1)
    assert np.all(dep.xibar <= 1)


@pytest.mark.parametrize("repeat", range(10))
@pytest.mark.parametrize("rho", [0.5, 0.9, 0.95])
def test_bivariate_statistics(repeat, rho, allclose):
    mean = np.zeros(2)
    cov = [[1, rho], [rho, 1]]
    rv = mvn(mean=mean, cov=cov)
    nsamples = 5000
    data = rv.rvs(size=nsamples)
    biv = ppc.bivariate_dependence_statistics(data)

    expected = 2/math.pi * math.asin(rho)
    assert allclose(biv["kendalltau"], expected, atol=5e-2)

    mv = biv["dependence"]
    nsta = 2
    x = np.repeat(mv.index.values[:, None] / 100, nsta, axis=1)
    z = norm.ppf(x)
    expected = 2 - rv.logcdf(z) / np.log(x[:, 0])
    assert allclose(mv.xi, expected, atol=1e-1, rtol=2e-2)


@pytest.mark.parametrize("copula_spec", ["Gaussian", "GaussianFactor_0_1"])
def test_posterior_predictive_checks(copula_spec):
    if re.search("Factor", copula_spec):
        samples = SAMPLES_FACTORS.iloc[:200]
    else:
        samples = SAMPLES.iloc[:200]
    ppu, ppb, ppm, data = ppc.posterior_predictive_checks(DATA, samples,
                                                          copula_spec)
    assert ppu.shape == (9, 21)
    assert ppb.shape == (32, 21)
    assert ppm.shape == (27, 7)
    assert ppu.filter(regex="pvalue\\[", axis=1).shape == (9, 3)
    assert ppb.filter(regex="pvalue\\[", axis=1).shape == (32, 3)
    assert ppm.filter(regex="pvalue$", axis=1).shape == (27, 1)

