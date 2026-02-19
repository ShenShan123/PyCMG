"""
NFIN (Number of Fins) Scaling Sanity Tests

PyCMG-only tests verifying that NFIN scaling produces expected proportional
changes in drain current and gate charge. No NGSPICE comparison needed.

NFIN handling in ctypes_host.py:
  - parse_modelcard() forces NFIN=1.0 in the modelcard params (lines 429-430).
  - Instance.__init__() applies modelcard params first, then instance params.
  - apply_param() has NO NFIN override, so instance-level NFIN values should
    take effect and override the modelcard's forced 1.0.

Therefore, passing NFIN in instance params should correctly control fin count.

Run: pytest tests/test_nfin_scaling.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from pycmg import Model, Instance
from tests.conftest import get_tech_modelcard, TECHNOLOGIES

OSDI_PATH = Path(__file__).resolve().parents[1] / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
TECH = "ASAP7"


def _make_instance(model: Model, inst_params: Dict[str, float], nfin: float) -> Instance:
    """Create an Instance with a specific NFIN value, overriding the default."""
    params = {**inst_params, "NFIN": nfin}
    return Instance(model, params=params)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
@pytest.mark.parametrize("device", ["nmos", "pmos"])
def test_nfin_doubles_current(device: str) -> None:
    """Doubling NFIN should approximately double drain current (id) and charges.

    BSIM-CMG scales most extensive quantities (currents, charges) linearly with
    NFIN. With NFIN=2 vs NFIN=1, we expect:
      - |id(NFIN=2)| / |id(NFIN=1)| ~ 2.0
      - |qg(NFIN=2)| / |qg(NFIN=1)| ~ 2.0

    Tolerance: 1.8x to 2.2x (10% margin for any numerical effects).
    """
    vdd = TECHNOLOGIES[TECH]["vdd"]
    mc_path, model_name, inst_params = get_tech_modelcard(TECH, device)

    # Bias the device in saturation for meaningful current
    if device == "nmos":
        nodes = {"d": vdd / 2, "g": vdd / 2, "s": 0.0, "e": 0.0}
    else:
        nodes = {"d": vdd * 0.3, "g": vdd * 0.3, "s": vdd, "e": vdd}

    model = Model(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(mc_path),
        model_name=model_name,
    )

    inst1 = _make_instance(model, inst_params, nfin=1.0)
    r1 = inst1.eval_dc(nodes)

    inst2 = _make_instance(model, inst_params, nfin=2.0)
    r2 = inst2.eval_dc(nodes)

    # Guard: both evaluations must produce non-trivial current
    assert abs(r1["id"]) > 1e-15, f"NFIN=1 id is effectively zero: {r1['id']}"
    assert abs(r2["id"]) > 1e-15, f"NFIN=2 id is effectively zero: {r2['id']}"

    ratio_id = abs(r2["id"]) / abs(r1["id"])
    assert 1.8 <= ratio_id <= 2.2, (
        f"NFIN scaling failed for id ({device}): "
        f"ratio={ratio_id:.4f}, id(1)={r1['id']:.4e}, id(2)={r2['id']:.4e}"
    )

    # Check charge scaling as well (qg should also scale with NFIN)
    assert abs(r1["qg"]) > 1e-25, f"NFIN=1 qg is effectively zero: {r1['qg']}"
    assert abs(r2["qg"]) > 1e-25, f"NFIN=2 qg is effectively zero: {r2['qg']}"

    ratio_qg = abs(r2["qg"]) / abs(r1["qg"])
    assert 1.8 <= ratio_qg <= 2.2, (
        f"NFIN scaling failed for qg ({device}): "
        f"ratio={ratio_qg:.4f}, qg(1)={r1['qg']:.4e}, qg(2)={r2['qg']:.4e}"
    )


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
@pytest.mark.parametrize("device", ["nmos", "pmos"])
def test_nfin_triples_current(device: str) -> None:
    """Tripling NFIN should approximately triple drain current and charges.

    Complementary to test_nfin_doubles_current: uses NFIN=3 vs NFIN=1
    to verify linearity holds beyond a single doubling.

    Tolerance: 2.7x to 3.3x (10% margin).
    """
    vdd = TECHNOLOGIES[TECH]["vdd"]
    mc_path, model_name, inst_params = get_tech_modelcard(TECH, device)

    if device == "nmos":
        nodes = {"d": vdd / 2, "g": vdd / 2, "s": 0.0, "e": 0.0}
    else:
        nodes = {"d": vdd * 0.3, "g": vdd * 0.3, "s": vdd, "e": vdd}

    model = Model(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(mc_path),
        model_name=model_name,
    )

    inst1 = _make_instance(model, inst_params, nfin=1.0)
    r1 = inst1.eval_dc(nodes)

    inst3 = _make_instance(model, inst_params, nfin=3.0)
    r3 = inst3.eval_dc(nodes)

    # Guard: non-trivial current
    assert abs(r1["id"]) > 1e-15, f"NFIN=1 id is effectively zero: {r1['id']}"
    assert abs(r3["id"]) > 1e-15, f"NFIN=3 id is effectively zero: {r3['id']}"

    ratio_id = abs(r3["id"]) / abs(r1["id"])
    assert 2.7 <= ratio_id <= 3.3, (
        f"NFIN scaling failed for id ({device}): "
        f"ratio={ratio_id:.4f}, id(1)={r1['id']:.4e}, id(3)={r3['id']:.4e}"
    )

    # Charge scaling
    assert abs(r1["qg"]) > 1e-25, f"NFIN=1 qg is effectively zero: {r1['qg']}"
    assert abs(r3["qg"]) > 1e-25, f"NFIN=3 qg is effectively zero: {r3['qg']}"

    ratio_qg = abs(r3["qg"]) / abs(r1["qg"])
    assert 2.7 <= ratio_qg <= 3.3, (
        f"NFIN scaling failed for qg ({device}): "
        f"ratio={ratio_qg:.4f}, qg(1)={r1['qg']:.4e}, qg(3)={r3['qg']:.4e}"
    )
