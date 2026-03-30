"""
Core-Voltage Vt Variant Verification Tests

Verifies PyCMG vs NGSPICE agreement for all core-voltage threshold voltage
variants across technologies. Tests DC operating point (currents, derivatives,
charges) in saturation, linear, and subthreshold regions.

This file tests CORE_VT_VARIANTS — the extended Vt flavors (lvt, slvt, sram,
ulvt, elvt, hvt, lnvt) that share the same Vdd and geometry as the base
technology but have different threshold voltage tuning.

Run: pytest tests/test_vt_variants.py -v
"""

from __future__ import annotations

import pytest

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import OSDI_PATH, run_ngspice_op, assert_close
from tests.conftest import (
    ALL_TECHNOLOGIES, CORE_VT_NAMES, get_tech_modelcard,
)


# ---------------------------------------------------------------------------
# Operating point generators (same as test_dc_regions.py)
# ---------------------------------------------------------------------------

def _nmos_ops(vdd: float) -> dict[str, dict[str, float]]:
    """NMOS operating regions: positive voltages, grounded source."""
    return {
        "saturation":  {"d": vdd,       "g": 0.8 * vdd, "s": 0.0, "e": 0.0},
        "linear":      {"d": 0.3 * vdd, "g": vdd,       "s": 0.0, "e": 0.0},
        "subthreshold": {"d": vdd,      "g": 0.0,       "s": 0.0, "e": 0.0},
    }


def _pmos_ops(vdd: float) -> dict[str, dict[str, float]]:
    """PMOS operating regions: Vs=Vdd, Vg/Vd referenced to Vdd."""
    return {
        "saturation":  {"d": 0.0,       "g": 0.2 * vdd, "s": vdd, "e": 0.0},
        "linear":      {"d": 0.7 * vdd, "g": 0.0,       "s": vdd, "e": 0.0},
        "subthreshold": {"d": 0.0,      "g": vdd,       "s": vdd, "e": 0.0},
    }


REGION_NAMES = ["saturation", "linear", "subthreshold"]


# ---------------------------------------------------------------------------
# Helper: run one PyCMG vs NGSPICE comparison
# ---------------------------------------------------------------------------

def _compare_dc(
    tech_name: str,
    device_type: str,
    region: str,
) -> None:
    """Compare PyCMG eval_dc against NGSPICE operating point."""
    tech = ALL_TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, device_type)
    vdd = tech["vdd"]

    ops = _nmos_ops(vdd) if device_type == "nmos" else _pmos_ops(vdd)
    op = ops[region]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        op["d"], op["g"], op["s"], op["e"],
        tag=f"vt_{tech_name}_{device_type}_{region}",
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc(op)

    prefix = f"{tech_name}/{device_type}/{region}"

    # Currents
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/ig", py["ig"], ng["ig"])
    assert_close(f"{prefix}/is", py["is"], ng["is"])

    # Derivatives
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])
    assert_close(f"{prefix}/gmb", py["gmb"], ng["gmb"])

    # Charges
    assert_close(f"{prefix}/qg", py["qg"], ng["qg"])
    assert_close(f"{prefix}/qd", py["qd"], ng["qd"])
    assert_close(f"{prefix}/qs", py["qs"], ng["qs"])
    assert_close(f"{prefix}/qb", py["qb"], ng["qb"])


# ---------------------------------------------------------------------------
# Tests: NMOS Vt variants
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", CORE_VT_NAMES)
@pytest.mark.parametrize("region", REGION_NAMES)
def test_nmos_vt_variant(tech_name: str, region: str) -> None:
    """NMOS DC verification for core-voltage Vt variants."""
    _compare_dc(tech_name, "nmos", region)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", CORE_VT_NAMES)
@pytest.mark.parametrize("region", REGION_NAMES)
def test_pmos_vt_variant(tech_name: str, region: str) -> None:
    """PMOS DC verification for core-voltage Vt variants."""
    try:
        _compare_dc(tech_name, "pmos", region)
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
