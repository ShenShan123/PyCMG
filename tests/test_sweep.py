"""Tests for the sweep module."""
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
ASAP7_MODELCARD = ROOT / "modelcards" / "ASAP7" / "7nm_TT_160803.pm"


def test_build_nodes_nmos():
    from pycmg.sweep import build_nodes
    nodes = build_nodes(0.4, 0.5, 0.0, 0.9, "nmos")
    assert nodes == {"g": 0.4, "d": 0.5, "s": 0.0, "e": 0.0}


def test_build_nodes_pmos():
    from pycmg.sweep import build_nodes
    nodes = build_nodes(0.4, 0.5, 0.0, 0.9, "pmos")
    assert nodes["s"] == 0.9
    assert abs(nodes["g"] - 0.5) < 1e-12
    assert abs(nodes["d"] - 0.4) < 1e-12
    assert abs(nodes["e"] - 0.9) < 1e-12


def test_build_voltage_grid_bounds():
    from pycmg.sweep import build_voltage_grid
    vg, vd = build_voltage_grid(0.9, 0.35, vg_points=50, vd_points=50)
    assert vg[0] >= 0.0
    assert vg[-1] <= 0.9
    assert vd[0] >= 0.0
    assert vd[-1] <= 0.9


def test_build_voltage_grid_density():
    from pycmg.sweep import build_voltage_grid
    vg, _ = build_voltage_grid(0.9, 0.35, vg_points=50, vd_points=10, dense_ratio=0.6)
    dense_count = np.sum((vg >= 0.215) & (vg <= 0.485))
    sparse_count = len(vg) - dense_count
    assert dense_count > sparse_count


def test_build_voltage_grid_no_duplicates():
    from pycmg.sweep import build_voltage_grid
    vg, vd = build_voltage_grid(0.75, 0.3, vg_points=50, vd_points=50)
    assert len(vg) == len(np.unique(vg))
    assert len(vd) == len(np.unique(vd))


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_find_threshold_nmos():
    from pycmg import Model, Instance
    from pycmg.sweep import find_threshold
    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    inst = Instance(model, params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0})
    vth = find_threshold(inst, vdd=0.9, device_type="nmos")
    assert 0.1 < vth < 0.7


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_find_threshold_pmos():
    from pycmg import Model, Instance
    from pycmg.sweep import find_threshold
    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "pmos_rvt")
    inst = Instance(model, params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0})
    vth = find_threshold(inst, vdd=0.9, device_type="pmos")
    assert 0.1 < vth < 0.7


def test_resolve_devices_all():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    devices = resolve_devices(None, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert "nmos_rvt" in devices
    assert len(devices) >= 8


def test_resolve_devices_filter():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"ASAP7": ["nmos_rvt", "pmos_rvt"]}
    devices = resolve_devices(filt, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert devices == ["nmos_rvt", "pmos_rvt"]


def test_resolve_devices_glob():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"ASAP7": ["nmos_*"]}
    devices = resolve_devices(filt, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert all(d.startswith("nmos_") for d in devices)
    assert len(devices) >= 4


def test_resolve_devices_missing_skipped():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"TSMC7": ["nmos_rvt"]}
    devices = resolve_devices(filt, "TSMC7", TECH_REGISTRY["TSMC7"])
    assert devices == []


def test_sweep_config_defaults():
    from pycmg.sweep import SweepConfig
    config = SweepConfig(techs=["ASAP7"])
    assert config.l_multipliers == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert config.nfins == [1.0, 2.0, 3.0]
    assert len(config.temperatures) == 5
    assert config.process_vars is None


def test_build_all_columns_no_process():
    from pycmg.sweep import build_all_columns
    cols = build_all_columns([])
    assert len(cols) == 28
    assert cols[0] == "tech"
    assert cols[-1] == "cdd"


def test_build_all_columns_with_process():
    from pycmg.sweep import build_all_columns
    cols = build_all_columns(["eot", "toxp"])
    assert len(cols) == 30
    assert "eot" in cols
    assert "toxp" in cols
    # Process vars should be between geometry and voltage
    eot_idx = cols.index("eot")
    vg_idx = cols.index("Vg")
    temp_idx = cols.index("temp_K")
    assert temp_idx < eot_idx < vg_idx
