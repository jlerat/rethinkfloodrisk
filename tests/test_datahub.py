import pytest

from mypackage import datahub

def test_data_folder():
    fd = datahub.DATA_FOLDER
    assert fd.exists()
