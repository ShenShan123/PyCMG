from __future__ import annotations

from tests import verify_utils


def test_verify_utils_exposes_ngspice_runner() -> None:
    assert hasattr(verify_utils, "run_ngspice_op_point")
