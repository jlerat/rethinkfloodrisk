import pytest

from pyrethink import datahub

def test_data_folder():
    fd = datahub.DATA_FOLDER
    assert fd.exists()

def test_get_stations():
    df = datahub.get_stations()

def test_potpeaks():
    df = datahub.get_potpeaks()
    df = df.filter(regex="_PEAK", axis=1)
    assert (df.min() > 0).all()

def test_potpeaks_thresh():
    thresh = datahub.get_potpeaks_thresh()
    assert len(thresh) == 8

