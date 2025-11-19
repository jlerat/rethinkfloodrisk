from pathlib import Path
import pytest
import math
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from pyrethink import datahub
from pyrethink import postpredchecks as ppc

from floodstan.marginals import GEV

FTESTS = Path(__file__).resolve().parent

NSTATIONS = len(datahub.get_stations())

DATA = pd.read_csv(FTESTS / "censored_missing_data.zip",
                   index_col=0, parse_dates=True)
DATA = DATA[pd.notnull(DATA).any(axis=1)]

SAMPLES = pd.read_csv(FTESTS / "censored_missing_samples.zip")
MARGINAL = GEV()

@pytest.mark.parametrize("station", np.arange(NSTATIONS).tolist())
def test_univariate_statistics(station):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station]

    un = ppc.univariate_statistics(data)
    assert un.notnull().all()


@pytest.mark.parametrize("station", np.arange(NSTATIONS - 1).tolist())
def test_bivariate_statistics(station):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station:station+2]
    biv = ppc.bivariate_dependence_statistics(data)
    assert biv.notnull().all()


def test_generate_samples(allclose):
    params = SAMPLES.iloc[0]
    x = ppc.generate_samples(params, 50000)

    ylocn = params.filter(regex="ylocn")
    ylogscale = params.filter(regex="ylogscale")
    yshape1 = params.filter(regex="yshape1")
    P = len(ylocn)
    z = np.zeros_like(x)
    for ivar in range(P):
        xx = x[:, ivar]
        MARGINAL.params = [ylocn[ivar], ylogscale[ivar],
                           yshape1[ivar]]
        uu = MARGINAL.cdf(xx)
        res = kstest(uu, "uniform")
        assert res.pvalue > 1e-2

        z[:, ivar] = norm.ppf(uu)

    cor = np.corrcoef(z.T)
    expected = params.filter(regex="cor_IW").values.reshape((P, P)).T
    assert allclose(cor, expected, atol=2e-2)


def test_posterior_predictive_checks():
    ppu, ppb, data = ppc.posterior_predictive_checks(DATA, SAMPLES.iloc[:200])

    assert ppu.shape == (7, 21)
    assert ppb.shape == (2, 21)
    assert ppu.filter(regex="pvalue\\[", axis=1).shape == (7, 3)
    assert ppb.filter(regex="pvalue\\[", axis=1).shape == (2, 3)


