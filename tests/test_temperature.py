"""
Temperature Verification Tests

Verifies PyCMG temperature handling against NGSPICE ground truth.
Tests at non-default temperatures (-40C, 85C, 125C) for both NMOS and PMOS
devices using ASAP7 technology.

Temperature bugs are model-level (not tech-specific), so ASAP7 alone provides
sufficient coverage. The default temperature (27C) is already verified by
test_dc_regions.py and test_dc_jacobian.py across all 5 technologies.

PyCMG Instance takes temperature in KELVIN: temp_K = temp_C + 273.15
NGSPICE uses .temp in Celsius via the temp_c parameter of run_ngspice_op.

Run: pytest tests/test_temperature.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import OSDI_PATH, run_ngspice_op, assert_close
from tests.conftest import TECHNOLOGIES, get_tech_modelcard

TECH = "ASAP7"
TEMPS_C = [-40.0, 85.0, 125.0]


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("temp_c", TEMPS_C)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
def test_temperature(temp_c: float, device: str) -> None:
    """Test DC currents and derivatives match NGSPICE at non-default temperatures."""
    vdd = TECHNOLOGIES[TECH]["vdd"]
    mc_path, model_name, inst_params = get_tech_modelcard(TECH, device)

    # Saturation operating point
    if device == "nmos":
        vd, vg, vs, ve = vdd / 2, vdd / 2, 0.0, 0.0
    else:
        # PMOS: source at Vdd, gate and drain pulled low
        vd, vg, vs, ve = vdd * 0.3, vdd * 0.3, vdd, vdd

    # PyCMG -- temperature in KELVIN
    temp_k = temp_c + 273.15
    model = Model(str(OSDI_PATH), str(mc_path), model_name)
    inst = Instance(model, params=inst_params, temperature=temp_k)
    py = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": ve})

    # NGSPICE -- temperature in Celsius
    ng = run_ngspice_op(
        mc_path, model_name, inst_params,
        vd, vg, vs, ve,
        temp_c=temp_c,
        tag=f"temp_{device}_{temp_c}",
    )

    prefix = f"ASAP7/{device}/T{temp_c}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
