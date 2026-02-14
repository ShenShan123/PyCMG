"""
DC Operating Region Tests

Verifies model accuracy across voltage-ratio-defined operating regions
for both NMOS and PMOS devices across all 5 technologies.

NMOS: 5 regions with positive voltages, grounded source
PMOS: 5 regions with inverted sense (Vs=Vdd)

Run: pytest tests/test_dc_regions.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import OSDI_PATH, run_ngspice_op, assert_close, REL_TOL
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


def get_nmos_region_ops(vdd: float) -> dict:
    """NMOS operating regions: positive voltages, grounded source."""
    return {
        "off_state":           {"d": vdd,       "g": 0.0,       "s": 0.0, "e": 0.0},
        "weak_inversion":      {"d": vdd,       "g": 0.3 * vdd, "s": 0.0, "e": 0.0},
        "moderate_inversion":  {"d": 0.5 * vdd, "g": 0.6 * vdd, "s": 0.0, "e": 0.0},
        "strong_linear":       {"d": 0.3 * vdd, "g": vdd,       "s": 0.0, "e": 0.0},
        "strong_saturation":   {"d": vdd,       "g": 0.8 * vdd, "s": 0.0, "e": 0.0},
    }


def get_pmos_region_ops(vdd: float) -> dict:
    """PMOS operating regions: Vs=Vdd, Vg/Vd referenced to Vdd."""
    return {
        "off_state":           {"d": 0.0,       "g": vdd,       "s": vdd, "e": 0.0},
        "weak_inversion":      {"d": 0.0,       "g": 0.7 * vdd, "s": vdd, "e": 0.0},
        "moderate_inversion":  {"d": 0.5 * vdd, "g": 0.4 * vdd, "s": vdd, "e": 0.0},
        "strong_linear":       {"d": 0.7 * vdd, "g": 0.0,       "s": vdd, "e": 0.0},
        "strong_saturation":   {"d": 0.0,       "g": 0.2 * vdd, "s": vdd, "e": 0.0},
    }


NMOS_REGIONS = list(get_nmos_region_ops(1.0).keys())
PMOS_REGIONS = list(get_pmos_region_ops(1.0).keys())


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("region", NMOS_REGIONS)
def test_nmos_dc_region(tech_name: str, region: str):
    """Test NMOS DC currents and derivatives match NGSPICE in operating region."""
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")

    vdd = tech["vdd"]
    op = get_nmos_region_ops(vdd)[region]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        op["d"], op["g"], op["s"], op["e"],
        tag=f"region_{tech_name}_nmos_{region}",
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc(op)

    # Compare currents
    prefix = f"{tech_name}/nmos/{region}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/ig", py["ig"], ng["ig"])
    assert_close(f"{prefix}/is", py["is"], ng["is"])

    # Compare derivatives
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])

    # Compare charges
    assert_close(f"{prefix}/qg", py["qg"], ng["qg"])
    assert_close(f"{prefix}/qd", py["qd"], ng["qd"])


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("region", PMOS_REGIONS)
def test_pmos_dc_region(tech_name: str, region: str):
    """Test PMOS DC currents and derivatives match NGSPICE in operating region."""
    tech = TECHNOLOGIES[tech_name]

    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "pmos")
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")

    vdd = tech["vdd"]
    op = get_pmos_region_ops(vdd)[region]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        op["d"], op["g"], op["s"], op["e"],
        tag=f"region_{tech_name}_pmos_{region}",
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc(op)

    prefix = f"{tech_name}/pmos/{region}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/ig", py["ig"], ng["ig"])
    assert_close(f"{prefix}/is", py["is"], ng["is"])
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
