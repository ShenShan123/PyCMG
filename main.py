#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tests import verify_utils

ROOT = Path(__file__).resolve().parent


def _run_pytest(args: List[str]) -> int:
    cmd = [sys.executable, "-m", "pytest"] + args
    return subprocess.call(cmd)


def _test_suite(suite: str, pytest_args: List[str]) -> int:
    if suite == "comprehensive":
        args = ["tests/test_comprehensive.py", "-v"]
    elif suite == "repro":
        args = ["tests/test_reproduce_asap7.py", "-v"]
    elif suite == "asap7-full":
        args = ["tests/test_asap7_full_verify.py", "-v"]
    elif suite == "asap7-pvt":
        args = ["tests/test_asap7_pvt_verify.py", "-v"]
    elif suite == "all":
        args = ["tests/", "-v"]
    else:
        raise ValueError(f"unknown test suite: {suite}")
    return _run_pytest(args + pytest_args)


def _collect_deep_verify(args: argparse.Namespace) -> int:
    deep_args = verify_utils.DeepVerifyArgs(
        out=str(args.out),
        vg_start=args.vg_start,
        vg_stop=args.vg_stop,
        vg_step=args.vg_step,
        vd_start=args.vd_start,
        vd_stop=args.vd_stop,
        vd_step=args.vd_step,
        backend=args.backend,
        temps=args.temps,
        stress=args.stress,
        stress_only=args.stress_only,
        stress_samples=args.stress_samples,
        stress_seed=args.stress_seed,
        tran=args.tran,
        tran_step=args.tran_step,
        tran_stop=args.tran_stop,
    )
    ok = verify_utils.run_deep_verify(deep_args)
    return 0 if ok else 1


def _collect_asap7(args: argparse.Namespace) -> int:
    if args.modelcard:
        os.environ["ASAP7_MODELCARD"] = str(args.modelcard)

    deep_args = verify_utils.DeepVerifyArgs(
        out=str(args.out),
        vg_start=args.vg_start,
        vg_stop=args.vg_stop,
        vg_step=args.vg_step,
        vd_start=args.vd_start,
        vd_stop=args.vd_stop,
        vd_step=args.vd_step,
        backend=args.backend,
        temps=str(args.temp_c),
        tran=args.tran,
    )
    ok = verify_utils.run_asap7_full_verify(deep_args, temp_c=args.temp_c)
    return 0 if ok else 1


def _add_deep_verify_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=verify_utils.CIRCUIT_DIR)
    parser.add_argument("--vg-start", type=float, default=0.0)
    parser.add_argument("--vg-stop", type=float, default=1.2)
    parser.add_argument("--vg-step", type=float, default=0.1)
    parser.add_argument("--vd-start", type=float, default=0.0)
    parser.add_argument("--vd-stop", type=float, default=1.2)
    parser.add_argument("--vd-step", type=float, default=0.1)
    parser.add_argument("--temps", type=str, default="27")
    parser.add_argument(
        "--backend",
        type=str,
        default=verify_utils.BACKEND_PYCMG,
        choices=[verify_utils.BACKEND_PYCMG, verify_utils.BACKEND_OSDI],
    )
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-only", action="store_true")
    parser.add_argument("--stress-samples", type=int, default=20)
    parser.add_argument("--stress-seed", type=int)
    parser.add_argument("--tran", action="store_true")
    parser.add_argument("--tran-step", type=float, default=1e-11)
    parser.add_argument("--tran-stop", type=float, default=1e-8)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyCMG entrypoint for verification tests and data collection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Run pytest suites.")
    test_parser.add_argument(
        "suite",
        choices=["comprehensive", "repro", "asap7-full", "asap7-pvt", "all"],
    )
    test_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)

    collect_parser = subparsers.add_parser("collect", help="Run data collection helpers.")
    collect_sub = collect_parser.add_subparsers(dest="mode", required=True)

    deep_parser = collect_sub.add_parser("deep-verify", help="Run deep-verify sweeps.")
    _add_deep_verify_args(deep_parser)

    asap7_parser = collect_sub.add_parser("asap7", help="Run ASAP7 verification sweeps.")
    _add_deep_verify_args(asap7_parser)
    asap7_parser.add_argument("--modelcard", type=Path)
    asap7_parser.add_argument("--temp-c", type=float, default=27.0)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "test":
        return _test_suite(args.suite, list(args.pytest_args))
    if args.command == "collect" and args.mode == "deep-verify":
        return _collect_deep_verify(args)
    if args.command == "collect" and args.mode == "asap7":
        return _collect_asap7(args)

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
