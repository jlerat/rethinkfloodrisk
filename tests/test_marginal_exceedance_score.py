from pathlib import Path
import math
import numpy as np

from scipy.stats import norm
from scipy.stats import multivariate_normal as mvt
from scipy.stats import gamma
from scipy.special import gamma as gamma_fun
from scipy.special import expit

import pytest

from hydrodiy.io import iutils

from pyrethink import marginal_exceedance_score as mes


LOGGER = iutils.get_logger("test")


@pytest.mark.parametrize("name", mes.COPULAS)
def test_copulas(name, allclose):
    if name == "Gumbel":
        pytest.skip("Gumbel not ready yet.")

    nstations = 10
    cop = mes.copula_factory(name, nstations)

    rho = 0.8
    if name == "Gaussian":
        cop.params = (1 - rho) * np.eye(nstations) \
                     + rho * np.ones((nstations, nstations))
    else:
        cop.params = rho

    smp = cop.sample(100)
    assert len(smp) == 100

    pdf = cop.pdf(smp)
    assert len(pdf) == 100

    cdf = cop.cdf(smp)
    assert len(cdf) == 100

    surv = cop.survival(smp)
    assert len(surv) == 100


@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("napprox", [0, 100, 500])
def test_gaussian_cdf_and_pdf(nstations, rho, napprox, allclose):
    mean = np.zeros(nstations)
    cov = (1 - rho) * np.eye(nstations) + rho * np.ones((nstations, nstations))

    if napprox == 0:
        cop = mes.GaussianCopula(nstations)
        cop.params = cov
    else:
        cop = mes.GaussianOneFactorCopula(nstations)
        cop.params = rho
        cop.set_approx(napprox)


    rv = mvt(mean=mean, cov=cov)
    u = expit(np.linspace(-10, 10, 50))
    for iu, uu in enumerate(u):
        c1 = rv.cdf(norm.ppf(uu) * np.ones((1, nstations)))
        c2 = cop.cdf_main_diagonal(uu)
        atol = 2e-3 if napprox == 100 else 2e-4
        assert allclose(c1, c2, atol=atol)

        s1 = rv.cdf(-norm.ppf(uu) * np.ones((1, nstations)))
        s2 = cop.survival_main_diagonal(uu)
        assert allclose(s1, s2, atol=atol)

        u_vect = uu * np.ones((1, nstations))
        x = norm.ppf(u_vect)
        p1 = rv.pdf(x) / np.prod(norm.pdf(x), axis=1)
        p2 = cop.pdf(u_vect)
        assert allclose(p1, p2, atol=atol)


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

    cop = mes.GaussianOneFactorCopula(nstations)
    cop.params = rho
    x2 = norm.ppf(cop.sample(nsamples))
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
    cop = mes.GaussianOneFactorCopula(nstations)
    cop.params = 1e-3
    pk = cop.compute_kendall_function_data()

    # Expected independent
    copi = mes.IndependenceCopula(nstations)
    expected = copi.kendall_function(pk.t)

    err = np.abs(np.arcsinh(expected) - np.arcsinh(pk.Kc))
    atol = 2e-2 if nstations <= 5 else 5e-2
    assert err.max() < atol


@pytest.mark.parametrize("kind", ["AND", "OR"])
@pytest.mark.parametrize("nstations", [2, 5, 10])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_compute_marginal_score(kind, nstations, rho, allclose):
    cop = mes.GaussianOneFactorCopula(nstations)
    cop.params = rho

    mex1 = mes.MarginalExceedanceScore(kind, cop)
    mex1.logger = LOGGER

    nsamples = 1000000
    mex2 = mes.MarginalExceedanceScoreEmpirical(kind, cop,
                                                nsamples=nsamples)
    mex2.logger = LOGGER

    maeps = np.logspace(math.log10(1e-2/5), -1, 10)
    scs = np.zeros((len(maeps), 2))
    for iaep, maep in enumerate(maeps):
        scs[iaep, 0], _ = mex1.compute_score(maep)
        scs[iaep, 1], _ = mex2.compute_score(maep)

    err = np.abs(np.diff(np.log(scs), axis=1)).squeeze()
    assert err.max() < 1e-1


@pytest.mark.parametrize("kind", ["KENDALL", "AND", "OR"])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("maep", [0.1, 0.01])
def test_compute_marginal_score_set(kind, rho, maep, allclose):
    nstations = 2
    cop = mes.GaussianOneFactorCopula(nstations)
    cop.params = rho

    mex = mes.MarginalExceedanceScore(kind, cop)
    df, _ = mex.compute_set(maep)

    if kind == "AND":
        check = cop.survival(df.iloc[:, :2])
    elif kind == "OR":
        check = 1 - cop.cdf(df.iloc[:, :2])
    elif kind == "KENDALL":
        t = cop.cdf(df.iloc[:, :2])
        check = 1 - cop.kendall_function(t)

    err = np.abs(np.log(check) - math.log(maep))
    assert (err < 5e-2).all()


@pytest.mark.parametrize("kind", ["KENDALL", "AND", "OR"])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("maep", [0.1, 0.01])
def test_compare_scores(kind, rho, maep, allclose):
    nstations = 2
    cop = mes.GaussianOneFactorCopula(nstations)
    cop.params = rho

    mex = mes.MarginalExceedanceScore(kind, cop)
    mex.logger = LOGGER
    df, _ = mex.compute_set(maep)

    # Check mex0 is in df
    mex0, _ = mex.compute_score(maep)
    expected = np.interp(mex0, df.u, df.v)
    assert allclose(mex0, expected, 1e-3)

