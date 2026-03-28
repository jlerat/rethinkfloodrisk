from pathlib import Path
import math
import numpy as np

from scipy.stats import norm
from scipy.stats import multivariate_normal as mvt
from scipy.stats import gamma
from scipy.special import gamma as gamma_fun

import pytest

from hydrodiy.io import iutils

from pyrethink import marginal_exceedance_score as mes


LOGGER = iutils.get_logger("test")


@pytest.mark.parametrize("name", mes.COPULAS)
def test_copulas(name, allclose):
    if name == "Gumbel":
        pytest.skip("Gumbel not ready yet.")

    nstations = 10
    if name == "Independence":
        args = ()
    elif name == "Gaussian":
        args = (np.eye(10),)
    elif name == "GaussianOneFactor":
        args = (0.8,)
    elif name == "Gumbel":
        args = (0.8,)

    cop = mes.copula_factory(name, nstations, *args)

    smp = cop.sample(100)
    assert len(smp) == 100

    cdf = cop.cdf(smp)
    assert len(cdf) == 100

@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("napprox", [0, 100, 500])
def test_gaussian_cdf(nstations, rho, napprox, allclose):
    mean = np.zeros(nstations)
    cov = (1 - rho) * np.eye(nstations) + rho * np.ones((nstations, nstations))

    if napprox == 0:
        cop = mes.GaussianCopula(nstations, cov)
    else:
        cop = mes.GaussianOneFactorCopula(nstations, rho, napprox)

    rv = mvt(mean=mean, cov=cov)
    u = np.linspace(1e-5, 1-1e-5, 10)
    maxerr = 0
    zmaxerr = None
    for iu, uu in enumerate(u):
        c1 = rv.cdf(norm.ppf(uu) * np.ones((1, nstations)))
        c2 = cop.cdf_main_diagonal(uu)
        atol = 2e-3 if napprox == 100 else 2e-4
        assert allclose(c1, c2, atol=atol)


@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
def test_gaussian_one_factor_sampling(nstations, rho, allclose):
    mean = np.zeros(nstations)
    cov = (1 - rho) * np.eye(nstations) + rho * np.ones((nstations, nstations))
    rv = mvt(mean=mean, cov=cov)
    nsamples = 1000000
    x1 = rv.rvs(size=nsamples)
    m1 = x1.mean(axis=0)
    cov1 = np.cov(x1.T)

    cop = mes.GaussianOneFactorCopula(nstations, rho, nsamples)
    x2 = norm.ppf(cop.random_samples)
    m2 = x2.mean(axis=0)
    cov2 = np.cov(x2.T)

    assert allclose(m1, m2, atol=1e-2)
    assert allclose(cov1, cov2, atol=1e-2)


@pytest.mark.parametrize("nstations", [2, 3, 5, 10, 20])
def test_kendall_function_independence(nstations, allclose):
    cop = mes.IndependenceCopula(nstations)
    t = np.linspace(0, 1, 1000)
    p = cop.kendall_function(t)
    t2 = cop.inverse_kendall_function(p)
    if nstations < 10:
        assert allclose(t2, t, atol=2e-3)

    expected = np.zeros_like(t)
    nlt = np.log(1./t)
    for i in range(nstations):
        expected += nlt**i / math.factorial(i)
    expected *= t
    iok = ~np.isnan(expected)
    assert allclose(p[iok], expected[iok])

    if nstations == 2:
        # See Joe (2014), page 420
        expected = t - t * np.log(t)
        iok = ~np.isnan(expected)
        assert allclose(p[iok], expected[iok])


@pytest.mark.parametrize("nstations", [2, 5, 7])
@pytest.mark.parametrize("repeat", np.arange(1, 6))
def test_compute_empirical_kendall(nstations, repeat, allclose):
    rho = 1e-3
    nsamples = 20000
    cop = mes.GaussianOneFactorCopula(nstations, rho)

    kind = "AND"

    mex = mes.MarginalExceedanceScore(kind, cdf, nsamples=nsamples,
                                      logger=LOGGER)
    pk = mex.compute_empirical_kendall()
    t = pk.t

    # Expected independent
    kfi = mes.KendallFunctionIndependence(nstations)
    expected = kfi.cdf(t)

    atol = 1e-2 if nstations <= 5 else 4e-2
    assert allclose(expected, pk.Kc, atol=atol)


@pytest.mark.parametrize("kind", ["AND", "OR"])
@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_compute_marginal_score(kind, nstations, rho, allclose):
    cdf = mes.GaussianFactorCopulaCDF(nstations, rho, napprox=500)

    nsamples = 1000000
    mex1 = mes.MarginalExceedanceScore(kind, cdf,
                                       empirical=False,
                                       nsamples=nsamples,
                                       logger=LOGGER)
    mex2 = mes.MarginalExceedanceScore(kind, cdf,
                                       empirical=True,
                                       nsamples=nsamples,
                                       logger=LOGGER)

    maeps = np.logspace(math.log10(1e-2/5), -1, 10)
    scs = np.zeros((len(maeps), 2))
    for iaep, maep in enumerate(maeps):
        scs[iaep, 0] = mex1.compute_score(maep)
        scs[iaep, 1] = mex2.compute_score(maep)

    err = np.abs(np.diff(np.log(scs), axis=1)).squeeze()
    assert err.max() < 1e-1


