from pathlib import Path
import re
import pytest
import math
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm
from scipy.stats import t as student_t

from pyrethink import datahub
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


@pytest.mark.parametrize("station", np.arange(NSTATIONS - 1).tolist())
def test_bivariate_statistics(station):
    potpeaks, _, _ = datahub.get_potpeaks()
    data = potpeaks.iloc[:, station:station+2]
    biv = ppc.bivariate_dependence_statistics(data)
    assert biv.notnull().all()


@pytest.mark.parametrize("copula", [0, 4, 5])
def test_generate_samples(copula, allclose):
    params = SAMPLES.iloc[0]
    nsamples = 50000
    x = ppc.generate_samples(params, copula, nsamples)

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

        if copula > 0:
            z[:, ivar] = student_t.ppf(uu, df=copula)
        else:
            z[:, ivar] = norm.ppf(uu)

    cor = np.corrcoef(z.T)
    expected = params.filter(regex="corr_IW").values.reshape((P, P)).T
    assert allclose(cor, expected, atol=2e-2)

@pytest.mark.parametrize("copula", [0, 1, 5])
def test_posterior_predictive_checks(copula):
    ppu, ppb, data = ppc.posterior_predictive_checks(DATA, SAMPLES.iloc[:200],
                                                     copula)

    assert ppu.shape == (7, 21)
    assert ppb.shape == (6, 21)
    assert ppu.filter(regex="pvalue\\[", axis=1).shape == (7, 3)
    assert ppb.filter(regex="pvalue\\[", axis=1).shape == (6, 3)


