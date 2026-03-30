"""Tests for technology registry and model_overrides support."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
ASAP7_MODELCARD = ROOT / "modelcards" / "ASAP7" / "7nm_TT_160803.pm"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
def test_instance_model_overrides():
    """model_overrides should shift device behavior (e.g., changing EOT shifts Id)."""
    from pycmg import Model, Instance

    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    nodes = {"d": 0.45, "g": 0.45, "s": 0.0, "e": 0.0}
    params = {"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0}

    # Baseline: no overrides
    inst_base = Instance(model, params=params)
    result_base = inst_base.eval_dc(nodes)

    # Override EOT (thicker oxide -> less current)
    inst_thick = Instance(model, params=params, model_overrides={"eot": 1.5e-9})
    result_thick = inst_thick.eval_dc(nodes)

    # Thicker oxide should reduce drain current
    assert abs(result_thick["id"]) < abs(result_base["id"]), \
        f"Thicker EOT should reduce Id: base={result_base['id']:.3e}, thick={result_thick['id']:.3e}"
