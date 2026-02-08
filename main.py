#!/usr/bin/env python3
"""
PyCMG Main Entry Point

Simplified CLI for running PyCMG verification tests and data collection.

Usage:
    python main.py test api           # Quick smoke tests
    python main.py test integration   # NGSPICE comparison tests
    python main.py test asap7         # ASAP7 PVT verification
    python main.py test all           # Run all tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent


def _run_pytest(args: List[str]) -> int:
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args
    return subprocess.call(cmd)


def _test_suite(suite: str, pytest_args: List[str]) -> int:
    """Run a specific test suite."""
    test_map = {
        "api": ["tests/test_api.py", "-v"],
        "integration": ["tests/test_integration.py", "-v"],
        "asap7": ["tests/test_asap7.py", "-v"],
        "all": ["tests/", "-v"],
    }

    if suite not in test_map:
        print(f"Error: Unknown test suite '{suite}'")
        print(f"Available suites: {', '.join(test_map.keys())}")
        return 1

    args = test_map[suite]
    return _run_pytest(args + pytest_args)


def _print_usage() -> None:
    """Print usage information."""
    print(__doc__)
    print("\nAvailable test suites:")
    print("  api           Quick API validation (~5s, no NGSPICE required)")
    print("  integration   NGSPICE comparison tests (~30s)")
    print("  asap7         ASAP7 PVT verification (~5min)")
    print("  all           Run all tests (~5.5min)")
    print()
    print("Examples:")
    print("  python main.py test api")
    print("  python main.py test integration")
    print("  python main.py test asap7 -v")
    print("  python main.py test all --tb=short")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyCMG entrypoint for verification tests.",
        usage="python main.py test SUITE [pytest_options]",
    )
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["api", "integration", "asap7", "all"],
        help="Test suite to run",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    if not args.suite:
        _print_usage()
        return 0

    return _test_suite(args.suite, list(args.pytest_args))


if __name__ == "__main__":
    sys.exit(main())
