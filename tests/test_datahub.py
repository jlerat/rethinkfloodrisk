from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytest

from pyrethink import datahub

FTESTS = Path(__file__).resolve().parent
FIMG = FTESTS / "images" / "datahub"
FIMG.mkdir(exist_ok=True, parents=True)

def test_data_folder():
    fd = datahub.DATA_FOLDER
    assert fd.exists()

def test_get_stations():
    df1 = datahub.get_stations()
    assert df1.shape[0] == 8

    df2 = datahub.get_stations(False)
    assert df2.shape[0] == 900

def test_get_awra_cookies():
    cookies = datahub.get_awra_cookies()
    assert cookies.shape == (12, 11)

def test_potpeaks():
    df, wy, nu = datahub.get_potpeaks()
    assert df.shape == (86, 8)
    assert abs(nu - 1.8297) < 1e-2
    assert (df.min() > 0).all()

def test_potpeaks_thresh():
    thresh = datahub.get_potpeaks_thresh()
    assert len(thresh) == 8

def test_get_ams_concat():
    ams, times, dows, stations = datahub.get_ams_concat()
    assert ams.shape[1] == 8
    assert stations.shape[0] == 8
    assert ams.shape == times.shape
    assert ams.shape == dows.shape
    assert all([dt == np.int64 for dt in dows.dtypes])

@pytest.mark.parametrize("stationid",
                         datahub.get_awra_cookies().index.tolist())
def test_get_ams_awra(stationid):
    se = datahub.get_ams_awra(stationid)
    assert len(se) == 111
    assert np.all(se.index.values == np.arange(1911, 2022))


@pytest.mark.parametrize("stationid",
                         datahub.get_stations().index.tolist())
def test_rating_curves(stationid):
    with pytest.raises(ValueError, match="Cannot find rating data"):
        rcs, metas = datahub.get_rating_curves("bidule")

    rcs, metas = datahub.get_rating_curves(stationid)
    assert isinstance(rcs, dict)
    assert isinstance(metas, dict)

    rc, meta = datahub.get_rating_curves(stationid, True)
    assert isinstance(rc, pd.DataFrame)
    assert isinstance(meta, pd.DataFrame)


@pytest.mark.parametrize("stationid",
                         ["419005"] + datahub.get_stations().index.tolist())
def test_ams(stationid):
    with pytest.raises(ValueError, match="Cannot find ams data"):
        ams = datahub.get_ams("bidule")

    ams = datahub.get_ams(stationid)
    assert f"{stationid}_PEAK" in ams.columns


@pytest.mark.parametrize("pcensor", [0, 0.5, 1])
def test_censors(pcensor):
    with pytest.raises(ValueError, match="Expected pcensor in"):
        _ = datahub.get_censors(2)

    censors = datahub.get_censors(pcensor)


def test_params_lh_moments():
    params = datahub.get_params_lh_moments()
    assert "STATIONID" in  params.columns



