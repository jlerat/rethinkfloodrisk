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

@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("napprox", [0, 100, 500])
def test_gaussian_cdf(nstations, rho, napprox, allclose):
    cdf = mes.GaussianFactorCopulaCDF(nstations, rho)

    mean = np.zeros(nstations)
    cov = (1 - rho) * np.eye(nstations) + rho * np.ones((nstations, nstations))
    rv = mvt(mean=mean, cov=cov)
    u = np.linspace(1e-5, 1-1e-5, 10)
    maxerr = 0
    zmaxerr = None
    for ix, x in enumerate(norm.ppf(u)):
        c1 = rv.cdf(x * np.ones((1, nstations)))

        c2 = cdf.cdf_main_diagonal(x)
        atol = 2e-3 if napprox == 100 else 2e-4
        assert allclose(c1, c2, atol=atol)


@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
def test_gaussian_sampling(nstations, rho, allclose):
    cdf = mes.GaussianFactorCopulaCDF(nstations, rho)
    mean = np.zeros(nstations)
    cov = (1 - rho) * np.eye(nstations) + rho * np.ones((nstations, nstations))
    rv = mvt(mean=mean, cov=cov)

    nsamples = 1000000
    x1 = rv.rvs(size=nsamples)
    m1 = x1.mean(axis=0)
    cov1 = np.cov(x1.T)

    x2 = norm.ppf(cdf.sample(nsamples))
    m2 = x2.mean(axis=0)
    cov2 = np.cov(x2.T)

    assert allclose(m1, m2, atol=1e-2)
    assert allclose(cov1, cov2, atol=1e-2)


@pytest.mark.parametrize("nstations", [2, 3, 5, 10, 20])
def test_kendall_function_independence(nstations, allclose):
    t = np.linspace(0, 1, 1000)
    kfi = mes.KendallFunctionIndependence(nstations)
    p = kfi.cdf(t)
    t2 = kfi.ppf(p)
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


@pytest.mark.parametrize("nstations", [2, 5, 8])
def test_compute_kendall(nstations, allclose):
    rho = 1e-3
    cdf = mes.GaussianFactorCopulaCDF(nstations, rho)

    kind = "AND"
    nsamples = 20000
    mex = mes.MarginalExceedanceScore(kind, cdf, nsamples=nsamples,
                                      logger=LOGGER)
    pk = mex.compute_empirical_kendall()
    t = pk.t

    # Expected
    kfi = mes.KendallFunctionIndependence(nstations)
    expected = kfi.cdf(t)

    assert allclose(expected, pk.Kc, atol=2e-2)

