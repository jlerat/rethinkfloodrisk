import pytest

from pyrethink import datahub

def test_data_folder():
    fd = datahub.DATA_FOLDER
    assert fd.exists()

def test_get_stations():
    df = datahub.get_stations()

def test_truepeaks():
    df = datahub.get_truepeaks()
    assert (df.min() > 0).all()
