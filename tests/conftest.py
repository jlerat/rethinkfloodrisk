import pytest
from pytest_allclose import report_rmses


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)


def pytest_addoption(parser):
    parser.addoption("--debug_mode", action="store_true", default=False,
                     help="Activate debug mode")


@pytest.fixture
def debug_mode(request):
    return request.config.getoption("--debug_mode")
