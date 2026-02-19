"""
DC Jacobian Verification Tests

Compares PyCMG's condensed 4×4 analytical Jacobian against NGSPICE's
numerical Jacobian computed via central finite-difference perturbation.

Central differencing: J[:,j] = (I(V+δ_j) - I(V-δ_j)) / (2δ)
- O(δ²) accuracy (vs O(δ) for forward differencing)
- 9 NGSPICE calls per operating point (1 base + 4×2 perturbations)

Run: pytest tests/test_dc_jacobian.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import (
    OSDI_PATH, run_ngspice_op, assert_close,
    ABS_TOL_G, REL_TOL_JAC,
)
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


def get_nmos_jacobian_op_points(vdd: float) -> list[dict]:
    """Generate NMOS operating points for Jacobian testing."""
    return [
        {"name": "saturation", "d": vdd, "g": 0.8 * vdd, "s": 0.0, "e": 0.0},
        {"name": "linear", "d": 0.3 * vdd, "g": vdd, "s": 0.0, "e": 0.0},
        {"name": "off", "d": vdd, "g": 0.0, "s": 0.0, "e": 0.0},
    ]


def get_pmos_jacobian_op_points(vdd: float) -> list[dict]:
    """Generate PMOS operating points for Jacobian testing.

    PMOS: Vs=Vdd, Vg/Vd referenced to Vdd (mirroring test_dc_regions.py).
    """
    return [
        {"name": "saturation", "d": 0.0, "g": 0.2 * vdd, "s": vdd, "e": 0.0},
        {"name": "linear", "d": 0.7 * vdd, "g": 0.0, "s": vdd, "e": 0.0},
        {"name": "off", "d": 0.0, "g": vdd, "s": vdd, "e": 0.0},
    ]


def compute_numerical_jacobian_central(
    modelcard: Path, model_name: str, inst_params: dict,
    op: dict, delta: float = 1e-6, temp_c: float = 27.0,
    tag_prefix: str = "jac",
) -> np.ndarray:
    """Compute 4×4 Jacobian via central finite-difference perturbation.

    Uses central differencing for O(δ²) accuracy:
        J[:,j] = (I(V+δ_j) - I(V-δ_j)) / (2δ)

    Requires 9 NGSPICE simulations: 1 base + 4×2 perturbations.
    The base run is for diagnostics only; central diff doesn't need it.
    """
    op_keys = ["d", "g", "s", "e"]
    current_keys = ["id", "ig", "is", "ie"]
    n = 4
    J = np.zeros((n, n))

    for j, op_key in enumerate(op_keys):
        # Forward perturbation: V + δ
        fwd_op = dict(op)
        fwd_op[op_key] = op[op_key] + delta
        fwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            fwd_op["d"], fwd_op["g"], fwd_op["s"], fwd_op["e"],
            temp_c, tag=f"{tag_prefix}_fwd_{op_key}",
        )
        fwd_I = np.array([fwd[k] for k in current_keys])

        # Backward perturbation: V - δ
        bwd_op = dict(op)
        bwd_op[op_key] = op[op_key] - delta
        bwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            bwd_op["d"], bwd_op["g"], bwd_op["s"], bwd_op["e"],
            temp_c, tag=f"{tag_prefix}_bwd_{op_key}",
        )
        bwd_I = np.array([bwd[k] for k in current_keys])

        # Central difference
        J[:, j] = (fwd_I - bwd_I) / (2.0 * delta)

    return J


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("op_idx", [0, 1, 2], ids=["saturation", "linear", "off"])
def test_nmos_dc_jacobian_full_matrix(tech_name: str, op_idx: int):
    """Compare NMOS condensed 4×4 Jacobian matrix against NGSPICE numerical Jacobian."""
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")

    op_points = get_nmos_jacobian_op_points(tech["vdd"])
    op = op_points[op_idx]
    op_name = op.pop("name")

    # NGSPICE: numerical Jacobian via central differencing
    ng_J = compute_numerical_jacobian_central(
        modelcard, model_name, inst_params, op,
        tag_prefix=f"jac_{tech_name}_nmos_{op_name}",
    )

    # PyCMG: analytical condensed Jacobian
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_J = inst.get_jacobian_matrix(
        {"d": op["d"], "g": op["g"], "s": op["s"], "e": op["e"]}
    )

    # Compare each entry
    terminals = ["d", "g", "s", "e"]
    for i, term_i in enumerate(terminals):
        for j, term_j in enumerate(terminals):
            label = f"{tech_name}/nmos/{op_name}/d(I{term_i})/d(V{term_j})"
            assert_close(
                label, py_J[i, j], ng_J[i, j],
                abs_tol=ABS_TOL_G, rel_tol=REL_TOL_JAC,
            )


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("op_idx", [0, 1, 2], ids=["saturation", "linear", "off"])
def test_pmos_dc_jacobian_full_matrix(tech_name: str, op_idx: int):
    """Compare PMOS condensed 4×4 Jacobian matrix against NGSPICE numerical Jacobian."""
    tech = TECHNOLOGIES[tech_name]

    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "pmos")
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")

    op_points = get_pmos_jacobian_op_points(tech["vdd"])
    op = op_points[op_idx]
    op_name = op.pop("name")

    # NGSPICE: numerical Jacobian via central differencing
    ng_J = compute_numerical_jacobian_central(
        modelcard, model_name, inst_params, op,
        tag_prefix=f"jac_{tech_name}_pmos_{op_name}",
    )

    # PyCMG: analytical condensed Jacobian
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_J = inst.get_jacobian_matrix(
        {"d": op["d"], "g": op["g"], "s": op["s"], "e": op["e"]}
    )

    # Compare each entry
    terminals = ["d", "g", "s", "e"]
    for i, term_i in enumerate(terminals):
        for j, term_j in enumerate(terminals):
            label = f"{tech_name}/pmos/{op_name}/d(I{term_i})/d(V{term_j})"
            assert_close(
                label, py_J[i, j], ng_J[i, j],
                abs_tol=ABS_TOL_G, rel_tol=REL_TOL_JAC,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
