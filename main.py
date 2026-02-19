#!/usr/bin/env python3
"""
PyCMG Main Entry Point

Simplified CLI for running PyCMG verification tests and data collection.

Usage:
    python main.py test api           # Quick smoke tests
    python main.py test jacobian      # DC Jacobian verification vs NGSPICE
    python main.py test regions       # DC operating region tests vs NGSPICE
    python main.py test transient     # Transient waveform verification vs NGSPICE
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
        "jacobian": ["tests/test_dc_jacobian.py", "-v"],
        "regions": ["tests/test_dc_regions.py", "-v"],
        "transient": ["tests/test_transient.py", "-v"],
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
    print("  jacobian      DC Jacobian verification vs NGSPICE (~30s)")
    print("  regions       DC operating region tests vs NGSPICE (~30s)")
    print("  transient     Transient waveform verification vs NGSPICE (~30s)")
    print("  all           Run all tests (~2min)")
    print()
    print("Examples:")
    print("  python main.py test api")
    print("  python main.py test jacobian")
    print("  python main.py test regions -v")
    print("  python main.py test transient")
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
        choices=["api", "jacobian", "regions", "transient", "all"],
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
