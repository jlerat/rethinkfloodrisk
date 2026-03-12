from pathlib import Path
import re
import pytest
import math
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import t as student_t

from pyrethink import datahub
from pyrethink.partitions import Partitions
from pyrethink import postpredchecks as ppc

from floodstan.marginals import GEV

FTESTS = Path(__file__).resolve().parent

_, _, _, sta = datahub.get_ams_concat()
NSTATIONS = len(sta)

DATA = pd.read_csv(FTESTS / "censored_missing_data.zip",
                   index_col=0, parse_dates=True)
DATA = DATA[pd.notnull(DATA).any(axis=1)]

SAMPLES = pd.read_csv(FTESTS / "censored_missing_samples.zip")
# .. fix all sample format
SAMPLES.columns = [re.sub("^cor", "corr", cn) for cn in SAMPLES.columns]

MARGINAL = GEV()


def test_joint_exceedance_probabilities(allclose):
    potpeaks, _, _ = datahub.get_potpeaks()
    qt = ppc.PERC_TAILS_DEFAULT
    plj, phj = ppc.joint_exceedance_probabilities(potpeaks, qt)
    nq = len(qt)

    assert len(plj) == nq
    assert len(phj) == nq

    q = 70
    for q in [60, 70, 80, 90]:
        iq = np.argmin(np.abs(qt - q))
        x = np.nanpercentile(potpeaks, [q], axis=0)

        ilow = np.all(potpeaks.values - x <= 0, axis=1)
        N = len(potpeaks)
        assert allclose(plj[iq], ilow.sum() / N)

        ihigh = np.all(potpeaks.values - x >= 0, axis=1)
        N = len(potpeaks)
        assert allclose(phj[iq], ihigh.sum() / N)


@pytest.mark.parametrize("station", np.arange(NSTATIONS).tolist())
def test_univariate_statistics(station):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station]

    un = ppc.univariate_statistics(data)
    assert un.notnull().all()
    assert len(un) == 9


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9, 0.95])
@pytest.mark.parametrize("nsta", [3, 6])
def test_multivariate_statistics(rho, nsta, allclose):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks
    mv = ppc.multivariate_dependence_statistics(data)
    assert mv.shape == (25, 3)
    assert mv.notnull().all().all()

    # Test routine for multivariate normal
    mean = np.zeros(nsta)
    cov = (1 - rho) * np.eye(nsta) + rho * np.ones((nsta, nsta))
    rv = mvn(mean=mean, cov=cov)
    data = rv.rvs(size=100000)
    qt = np.arange(50, 96)
    mv = ppc.multivariate_dependence_statistics(data, qt)

    x = np.repeat(mv.index.values[:, None] / 100, nsta, axis=1)
    z = norm.ppf(x)
    expected = 2 - rv.logcdf(z) / np.log(x[:, 0])
    assert allclose(mv.xi, expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("station", np.arange(NSTATIONS - 1).tolist())
def test_bivariate_statistics(station, allclose):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station:station+2]
    biv = ppc.bivariate_dependence_statistics(data)
    if station != 3:
        assert biv.notnull().all()

    qm = data.median()
    iboth = (data - qm.values[None, :] <= 0).all(axis=1)
    ifirst = data.iloc[:, 0] - qm.values[0] <= 0
    N = len(data)
    expected = 2 - math.log(iboth.sum() / N) / math.log(ifirst.sum() / N)
    assert allclose(biv["xi_q50"], expected)

    iboth = (data - qm.values[None, :] >= 0).all(axis=1)
    assert all(data.loc[iboth].min() - qm >= 0)
    ifirst = data.iloc[:, 0] - qm.values[0] >= 0
    expected = 2 * math.log(ifirst.sum() / N) / math.log(iboth.sum() / N) - 1
    assert allclose(biv["xibar_q50"], expected)


@pytest.mark.parametrize("copula", [0, 2.5, 5])
def test_posterior_predictive_checks(copula):
    parts = Partitions(DATA.shape[1])
    parts_id = np.random.randint(0, parts.nsubsets,
                                 len(DATA))
    dalpha = 1.

    with pytest.raises(ValueError, match="Expected copula in"):
        ppc.posterior_predictive_checks(DATA, SAMPLES.iloc[:200],
                                        1.5, parts_id, dalpha)

    ppu, ppb, ppm, data = ppc.posterior_predictive_checks(DATA, SAMPLES.iloc[:200],
                                                          copula, parts_id, dalpha)

    assert ppu.shape == (28, 21)
    assert ppb.shape == (77, 21)
    assert ppm.shape == (25, 7)
    assert ppu.filter(regex="pvalue\\[", axis=1).shape == (28, 3)
    assert ppb.filter(regex="pvalue\\[", axis=1).shape == (77, 3)
    assert ppm.filter(regex="pvalue$", axis=1).shape == (25, 1)


