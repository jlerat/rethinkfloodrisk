from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytest

from pyrethink import datahub, processing

FTESTS = Path(__file__).resolve().parent
FIMG = FTESTS / "images" / "processing"
FIMG.mkdir(exist_ok=True, parents=True)

def test_linear_interpolation(allclose):
    x = np.array([0, 1, 1, 2])
    y = np.array([1, 2, 3, 4])
    xx = 10
    with pytest.raises(ValueError, match="Cannot process duplicates"):
        yy = processing.linear_interpolation(xx, x, y)

    x = np.array([0, 1, 2, 3])
    y = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Expected x and y"):
        yy = processing.linear_interpolation(xx, x, y)

    y = np.array([1, 2, 3, 4])
    yy = processing.linear_interpolation(xx, x, y)
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

        yy = processing.linear_interpolation(xx, x, y)
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
        yi = processing.linear_interpolation(x, xx, yy)
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


