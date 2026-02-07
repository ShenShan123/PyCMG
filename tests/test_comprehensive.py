from __future__ import annotations

import os
from pathlib import Path

from tests import verify_utils

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-deep-verify"
RESULTS_DIR = BUILD / "comprehensive_test_results"


def _run_comprehensive_suite() -> bool:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_pass = True

    args_nmos = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        tran=True,
        vg_start=0.0,
        vg_stop=1.2,
        vg_step=0.1,
        vd_start=0.0,
        vd_stop=1.2,
        vd_step=0.1,
        temps="27",
    )
    all_pass = all_pass and verify_utils.run_deep_verify(
        args_nmos,
        model_src_nmos=verify_utils.MODEL_SRC_NMOS,
        model_src_pmos=verify_utils.MODEL_SRC_PMOS,
    )

    args_pmos = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        tran=True,
        vg_start=0.0,
        vg_stop=1.2,
        vg_step=0.1,
        vd_start=0.0,
        vd_stop=1.2,
        vd_step=0.1,
        temps="27",
    )
    all_pass = all_pass and verify_utils.run_deep_verify(
        args_pmos,
        model_src_nmos=verify_utils.MODEL_SRC_NMOS,
        model_src_pmos=verify_utils.MODEL_SRC_PMOS,
    )

    args_temp = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        tran=False,
        vg_start=0.0,
        vg_stop=0.7,
        vg_step=0.1,
        vd_start=0.0,
        vd_stop=0.7,
        vd_step=0.1,
        temps="-40,0,27,85,125",
    )
    all_pass = all_pass and verify_utils.run_deep_verify(
        args_temp,
        model_src_nmos=verify_utils.MODEL_SRC_NMOS,
        model_src_pmos=verify_utils.MODEL_SRC_PMOS,
    )

    args_volt = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        tran=False,
        vg_start=0.0,
        vg_stop=1.0,
        vg_step=0.1,
        vd_start=0.0,
        vd_stop=1.0,
        vd_step=0.1,
        temps="27",
    )
    all_pass = all_pass and verify_utils.run_deep_verify(
        args_volt,
        model_src_nmos=verify_utils.MODEL_SRC_NMOS,
        model_src_pmos=verify_utils.MODEL_SRC_PMOS,
    )

    os.environ["ASAP7_MODELCARD"] = str(
        ROOT / "tech_model_cards" / "asap7_pdk_r1p7" / "models" / "hspice" / "7nm_TT.pm"
    )
    args_asap7 = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        tran=True,
        vg_start=0.0,
        vg_stop=0.7,
        vg_step=0.1,
        vd_start=0.0,
        vd_stop=0.7,
        vd_step=0.1,
        temps="27",
    )
    all_pass = all_pass and verify_utils.run_asap7_full_verify(args_asap7, temp_c=27.0)

    args_stress = verify_utils.DeepVerifyArgs(
        backend=verify_utils.BACKEND_PYCMG,
        stress=True,
        stress_samples=50,
        temps="27",
    )
    all_pass = all_pass and verify_utils.run_deep_verify(
        args_stress,
        model_src_nmos=verify_utils.MODEL_SRC_NMOS,
        model_src_pmos=verify_utils.MODEL_SRC_PMOS,
    )

    return all_pass


def test_comprehensive_suite() -> None:
    assert _run_comprehensive_suite()
