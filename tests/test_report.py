from pathlib import Path
import pytest
import math
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from pyrethink import datahub
from pyrethink import report

from floodstan.marginals import GEV

FTESTS = Path(__file__).resolve().parent

DATA = pd.read_csv(FTESTS / "censored_missing_data.zip")
SAMPLES = pd.read_csv(FTESTS / "censored_missing_samples.zip")

def test_report(allclose):
    stat, df = report.ffa_report(SAMPLES.iloc[:200])
    assert df.shape == (200, 36)
    assert stat.shape == (36, 15)

    for cn in ["MEAN", "POSTERIOR_PREDICTIVE", "EXPECTED_PARAMETERS"]:
        assert cn in stat.columns

