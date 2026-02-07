from __future__ import annotations

from pathlib import Path

from pycmg import ctypes_host
from tests import verify_utils


def _assert_close(label: str, py_val: float, ng_val: float) -> None:
    tol = max(verify_utils.ABS_TOL_I, abs(ng_val) * verify_utils.REL_TOL)
    assert abs(py_val - ng_val) <= tol, f"{label} mismatch: py={py_val} ng={ng_val} tol={tol}"


def test_reproduce_asap7_matches_ngspice() -> None:
    modelcard = verify_utils.ASAP7_DIR / "7nm_TT_160803.pm"
    model_name = "nmos_lvt"
    parsed = ctypes_host.parse_modelcard(str(modelcard), target_model_name=model_name)
    assert parsed.name == model_name
    assert "l" in parsed.params
    assert "tfin" in parsed.params

    inst_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}
    ng_modelcard = verify_utils.BUILD / "ngspice_eval" / "asap7_nmos_lvt.osdi"
    verify_utils.make_ngspice_modelcard(modelcard, ng_modelcard, model_name, inst_params)

    ng = verify_utils.run_ngspice_op_point(
        ng_modelcard,
        model_name,
        inst_params,
        vd=0.7,
        vg=0.7,
        vs=0.0,
        ve=0.0,
        out_dir=verify_utils.BUILD / "ngspice_eval" / "asap7_op",
        temp_c=27.0,
    )
    eval_fn = verify_utils.make_pycmg_eval(modelcard, model_name, inst_params, temp_c=27.0)
    py = eval_fn(0.7, 0.7, 0.0, 0.0)

    _assert_close("id", py[0], ng["id"])
    _assert_close("ig", py[1], ng["ig"])
    _assert_close("is", py[2], ng["is"])
    _assert_close("ib", py[3], ng["ib"])
