"""Pytest configuration for ASAP7 tests."""
import pytest

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Add test report attribute to node for result tracking."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)


def pytest_sessionfinish(session, exitstatus):
    """Print test summary at end of session."""
    try:
        from test_asap7 import print_test_summary
        print_test_summary()
    except ImportError:
        pass  # Skip if test_asap7 not available
