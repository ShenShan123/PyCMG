"""
AC Capacitance Verification Tests

Compares PyCMG's condensed capacitance matrix (cgg, cgd, cgs, cdg, cdd) from
eval_dc() against NGSPICE's @n1[cXX] operating-point variables.

PyCMG extracts capacitances via _condense_caps() which builds the full reactive
Jacobian (dQ/dV) and applies Schur complement condensation to eliminate internal
nodes. NGSPICE performs an equivalent condensation internally for OSDI models.

This test verifies that:
1. PyCMG's _condense_caps() produces correct capacitance values
2. Sign conventions match between PyCMG and NGSPICE
3. Capacitance condensation (internal node elimination) is consistent

Run: pytest tests/test_ac_caps.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import (
    OSDI_PATH, run_ngspice_ac, assert_close,
    ABS_TOL_C, REL_TOL_CAP,
)
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
@pytest.mark.parametrize("tech_name", TECH_NAMES, ids=TECH_NAMES)
def test_nmos_ac_caps(tech_name: str) -> None:
    """Compare PyCMG NMOS capacitances against NGSPICE at saturation bias.

    Operating point: Vd = Vdd/2, Vg = Vdd/2, Vs = 0, Ve = 0
    This places the device in moderate-to-strong inversion with significant
    capacitance contributions from all terminals.
    """
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")
    vdd = tech["vdd"]

    vd = vdd / 2.0
    vg = vdd / 2.0
    vs = 0.0
    ve = 0.0

    # NGSPICE reference: extract @n1[cgg], @n1[cgd], @n1[cgs], @n1[cdg], @n1[cdd]
    ng = run_ngspice_ac(
        modelcard, model_name, inst_params,
        vd, vg, vs, ve,
        tag=f"ac_{tech_name}_nmos",
    )

    # PyCMG: eval_dc returns condensed capacitances
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": ve})

    # Compare each capacitance element
    prefix = f"{tech_name}/nmos"
    cap_keys = ["cgg", "cgd", "cgs", "cdg", "cdd"]
    for cap in cap_keys:
        assert_close(
            f"{prefix}/{cap}",
            py[cap], ng[cap],
            abs_tol=ABS_TOL_C,
            rel_tol=REL_TOL_CAP,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
