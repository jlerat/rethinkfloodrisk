from pathlib import Path
import re
import pytest
import math
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm
from scipy.stats import t as student_t

from pyrethink import datahub
from pyrethink.sample import Partitions
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

@pytest.mark.parametrize("station", np.arange(NSTATIONS).tolist())
def test_univariate_statistics(station):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station]

    un = ppc.univariate_statistics(data)
    assert un.notnull().all()


def test_multivariate_statistics():
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks
    mv = ppc.multivariate_dependence_statistics(data)
    assert mv.notnull().all()


@pytest.mark.parametrize("station", np.arange(NSTATIONS - 1).tolist())
def test_bivariate_statistics(station, allclose):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station:station+2]
    biv = ppc.bivariate_dependence_statistics(data)
    assert biv.notnull().all()

    qm = data.median()
    iabove = (data - qm.values[None, :] >= 0).all(axis=1)
    assert all(data.loc[iabove].min() - qm >= 0)
    expected = iabove.sum() / len(data)
    assert allclose(biv["taildep_q50"], expected)


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

    assert ppu.shape == (7, 21)
    assert ppb.shape == (6, 21)
    assert ppm.shape == (4, 7)
    assert ppu.filter(regex="pvalue\\[", axis=1).shape == (7, 3)
    assert ppb.filter(regex="pvalue\\[", axis=1).shape == (6, 3)
    assert ppm.filter(regex="pvalue$", axis=1).shape == (4, 1)


