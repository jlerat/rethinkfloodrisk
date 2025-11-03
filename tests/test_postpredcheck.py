import pytest

import numpy as np

from pyrethink import datahub
from pyrethink import postpredchecks as ppc

NSTATIONS = len(datahub.get_stations())

@pytest.mark.parametrize("station", np.arange(NSTATIONS).tolist())
def test_univariate_statistics(station):
    truepeaks = datahub.get_truepeaks()
    data = truepeaks.iloc[:, station]

    un = ppc.univariate_statistics(data)
    assert un.notnull().all()


@pytest.mark.parametrize("station", np.arange(NSTATIONS - 1).tolist())
def test_bivariate_statistics(station):
    truepeaks = datahub.get_truepeaks()
    data = truepeaks.iloc[:, station:station+2]
    biv = ppc.bivariate_dependence_statistics(data)
    assert biv.notnull().all()


def test_generate_samples():
    pytest.skip("WIP")



