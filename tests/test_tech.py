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


# ---------------------------------------------------------------------------
# Technology registry tests (pycmg.tech)
# ---------------------------------------------------------------------------


def test_device_config_asap7():
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["ASAP7"]
    dev = tech.get_device("nmos_rvt")
    assert dev.modelcard is not None
    assert dev.pdk_device is None
    assert "TFIN" in dev.inst_params
    assert "DEVTYPE" in dev.inst_params
    assert "L" not in dev.inst_params
    assert "NFIN" not in dev.inst_params


def test_device_config_tsmc():
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["TSMC7"]
    dev = tech.get_device("nmos_svt")
    assert dev.modelcard is None
    assert dev.pdk_device == "nch_svt_mac"
    assert "TFIN" in dev.inst_params
    assert "DEVTYPE" in dev.inst_params


def test_tech_config_list_devices():
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["ASAP7"]
    devices = tech.list_devices()
    assert "nmos_rvt" in devices
    assert "pmos_rvt" in devices
    assert len(devices) >= 8


def test_tech_registry_all_techs():
    from pycmg.tech import TECH_REGISTRY, list_techs
    assert set(list_techs()) >= {"ASAP7", "TSMC5", "TSMC7", "TSMC12", "TSMC16"}


def test_tech_config_pdk_path():
    from pycmg.tech import TECH_REGISTRY
    assert TECH_REGISTRY["ASAP7"].pdk_path is None
    assert TECH_REGISTRY["TSMC7"].pdk_path is not None
    assert "cln7" in TECH_REGISTRY["TSMC7"].pdk_path


# ---------------------------------------------------------------------------
# Per-device min_l auto-detection (Task 3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ASAP7_MODELCARD.exists(), reason="ASAP7 modelcard missing")
def test_min_l_asap7():
    """ASAP7 min_l should be auto-detected from modelcard L parameter (21nm)."""
    from pycmg.tech import TECH_REGISTRY
    dev = TECH_REGISTRY["ASAP7"].get_device("nmos_rvt")
    dev._min_l = None  # Reset cache
    min_l = dev.get_min_l()
    assert abs(min_l - 21e-9) < 1e-12, f"Expected 21nm, got {min_l*1e9:.1f}nm"


@pytest.mark.skipif(
    not (ROOT / "modelcards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l").exists(),
    reason="TSMC7 PDK missing"
)
def test_min_l_tsmc7_core():
    """TSMC7 core device min_l should be 8nm (from PDK lmin scanning)."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["TSMC7"]
    dev = tech.get_device("nmos_svt")
    dev._min_l = None  # Reset cache
    min_l = dev.get_min_l(tech.pdk_path)
    assert abs(min_l - 8e-9) < 1e-12, f"Expected 8nm, got {min_l*1e9:.1f}nm"


@pytest.mark.skipif(
    not (ROOT / "modelcards" / "TSMC5" / "cln5_1d2_sp_v1d2_2p2.l").exists(),
    reason="TSMC5 PDK missing"
)
def test_min_l_tsmc5_core():
    """TSMC5 core device min_l should be 6nm."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["TSMC5"]
    dev = tech.get_device("nmos_svt")
    dev._min_l = None  # Reset cache
    min_l = dev.get_min_l(tech.pdk_path)
    assert abs(min_l - 6e-9) < 1e-12, f"Expected 6nm, got {min_l*1e9:.1f}nm"


def test_min_l_cached():
    """get_min_l() should cache result after first call."""
    from pycmg.tech import TECH_REGISTRY
    dev = TECH_REGISTRY["ASAP7"].get_device("nmos_rvt")
    dev._min_l = None  # Reset cache
    l1 = dev.get_min_l()
    l2 = dev.get_min_l()
    assert l1 == l2
    assert dev._min_l is not None
