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
    df = datahub.get_stations()

@pytest.mark.parametrize("no_missing", [False, True])
def test_potpeaks(no_missing):
    df, wy, nu = datahub.get_potpeaks(no_missing)

    if no_missing:
        assert df.shape == (58, 6)
        assert abs(nu - 1.8125) < 1e-2
    else:
        assert df.shape == (86, 8)
        assert abs(nu - 1.82978) < 1e-2

    assert (df.min() > 0).all()

@pytest.mark.parametrize("no_missing", [False, True])
def test_potpeaks_thresh(no_missing):
    thresh = datahub.get_potpeaks_thresh(no_missing)
    if no_missing:
        assert len(thresh) == 6
    else:
        assert len(thresh) == 8

@pytest.mark.parametrize("no_missing", [False, True])
def test_get_ams_concat(no_missing):
    ams, times, dows, stations = datahub.get_ams_concat(no_missing)
    if no_missing:
        assert ams.shape[1] == 6
        assert stations.shape[0] == 6
    else:
        assert ams.shape[1] == 8
        assert stations.shape[0] == 8

    assert ams.shape == times.shape
    assert ams.shape == dows.shape
    assert all([dt == np.int64 for dt in dows.dtypes])


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
                         datahub.get_stations().index.tolist())
def test_ams(stationid):
    with pytest.raises(ValueError, match="Cannot find ams data"):
        ams = datahub.get_ams("bidule")

    ams = datahub.get_ams(stationid)


@pytest.mark.parametrize("pcensor", [0, 0.5, 1])
def test_censors(pcensor):
    with pytest.raises(ValueError, match="Expected pcensor in"):
        _ = datahub.get_censors(2)

    censors = datahub.get_censors(pcensor)

