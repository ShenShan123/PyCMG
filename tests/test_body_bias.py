"""
Body Bias Verification Tests

Verifies model accuracy with non-zero body bias (bulk terminal voltage)
for both NMOS and PMOS devices across all 5 technologies.

Existing tests all use ve=0.0 (grounded bulk). This test exercises:
  - Reverse body bias (RBB): increases threshold voltage, reduces leakage
  - Forward body bias (FBB): decreases threshold voltage, increases drive

Key parameter: gmb (bulk transconductance) is the primary target — it is
only meaningfully exercised when ve != 0.

Test matrix: 5 techs x 2 devices x 2 bias conditions = 20 tests

Run: pytest tests/test_body_bias.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import OSDI_PATH, run_ngspice_op, assert_close
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard

BIAS_TYPES = ["reverse", "forward"]


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("bias_type", BIAS_TYPES)
def test_nmos_body_bias(tech_name: str, bias_type: str) -> None:
    """Test NMOS DC outputs match NGSPICE with non-zero body bias.

    Bias conditions (saturation region): vd=vdd/2, vg=vdd/2, vs=0.
      - Reverse body bias: ve = -0.1 (body more negative than source)
      - Forward body bias: ve = +0.1 (body more positive than source)
    """
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")
    vdd = tech["vdd"]

    vd = vdd / 2
    vg = vdd / 2
    vs = 0.0
    ve = -0.1 if bias_type == "reverse" else 0.1

    # PyCMG evaluation
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": ve})

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        vd, vg, vs, ve,
        tag=f"body_bias_{tech_name}_nmos_{bias_type}",
    )

    # Compare currents and derivatives
    prefix = f"{tech_name}/nmos/body_{bias_type}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])
    assert_close(f"{prefix}/gmb", py["gmb"], ng["gmb"])


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("bias_type", BIAS_TYPES)
def test_pmos_body_bias(tech_name: str, bias_type: str) -> None:
    """Test PMOS DC outputs match NGSPICE with non-zero body bias.

    Bias conditions (saturation region): vd=vdd*0.3, vg=vdd*0.3, vs=vdd.
      - Reverse body bias: ve = vdd + 0.1 (body more positive than source for PMOS)
      - Forward body bias: ve = vdd - 0.1 (body less positive than source for PMOS)
    """
    tech = TECHNOLOGIES[tech_name]

    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "pmos")
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")

    vdd = tech["vdd"]

    vd = vdd * 0.3
    vg = vdd * 0.3
    vs = vdd
    ve = vdd + 0.1 if bias_type == "reverse" else vdd - 0.1

    # PyCMG evaluation
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": ve})

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        vd, vg, vs, ve,
        tag=f"body_bias_{tech_name}_pmos_{bias_type}",
    )

    # Compare currents and derivatives
    prefix = f"{tech_name}/pmos/body_{bias_type}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])
    assert_close(f"{prefix}/gmb", py["gmb"], ng["gmb"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
