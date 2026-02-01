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


def test_linear_interpolation(allclose):
    x = np.array([0, 1, 1, 2])
    y = np.array([1, 2, 3, 4])
    xx = 10
    with pytest.raises(ValueError, match="Cannot process duplicates"):
        yy = datahub.linear_interpolation(xx, x, y)

    x = np.array([0, 1, 2, 3])
    y = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Expected x and y"):
        yy = datahub.linear_interpolation(xx, x, y)

    y = np.array([1, 2, 3, 4])
    yy = datahub.linear_interpolation(xx, x, y)
    expected = y[-2] + (y[-1] - y[-2])/(x[-1] - x[-2]) * (xx - x[-2])
    assert allclose(yy, expected)

    for itest in range(100):
        x = np.random.uniform(-1, 1, 10)
        theta = np.random.uniform(0, 2, 3)
        X = np.column_stack([x**d for d in range(len(theta))])
        y = X @ theta

        isort = np.argsort(x)
        xs = x[isort]
        ys = y[isort]
        xxlims = np.concatenate([[xs[0] - 1], xs, [xs[-1] + 1]])
        yylims = np.concatenate([[ys[0] - 1], ys, [ys[-1] + 1]])
        yylims[0] = ys[0] + (ys[1] - ys[0])/(xs[1] - xs[0]) * (xxlims[0] - xs[0])
        yylims[-1] = ys[-1] + (ys[-1] - ys[-2])/(xs[-1] - xs[-2]) * (xxlims[-1] - xs[-1])
        xx = (xxlims[1:] + xxlims[:-1])/2

        yy = datahub.linear_interpolation(xx, x, y)
        for i in  range(len(xx)):
            x0 = xxlims[i]
            x1 = xxlims[i + 1]
            y0 = yylims[i]
            y1 = yylims[i + 1]
            expected = y0 + (y1 - y0)/(x1 - x0) * (xx[i] - x0)
            assert allclose(yy[i], expected)


def test_linear_interpolation_plot(allclose):
    x = np.linspace(-5, 5, 500)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, "-")

    for npts in [5, 10]:
        xx = np.linspace(-4, 4, npts)
        yy = np.sin(xx)
        yi = datahub.linear_interpolation(x, xx, yy)
        ax.plot(xx, yy, "+")
        col = ax.get_lines()[-1].get_color()
        ax.plot(x, yi, "-", color=col)

    fp = FIMG / "linear_interpolation.png"
    fig.savefig(fp)


def test_eep2aep(allclose):
    nu = 2
    x = np.logspace(-4, -1, 200)
    y = datahub.eep2aep(nu, x)
    fig, ax = plt.subplots()
    ax.plot(x, y, "-")
    x0, x1 = x[0], x[-1]
    ax.plot([x0, x1], [x0, x1], "k--", lw=0.9)
    ax.set(xscale="log", yscale="log")
    fp = FIMG / "eep2aep.png"
    fig.savefig(fp)


