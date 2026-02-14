"""
Pytest configuration and technology registry for PyCMG verification tests.

The registry provides deterministic modelcard selection:
- ASAP7: Explicit TT corner + rvt variant (no glob ambiguity)
- TSMC: Explicit file names + per-device instance params (PMOS L=20nm)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"

# Technology registry — single source of truth for all test parametrization.
#
# Each entry specifies:
#   dir:          subdirectory under tech_model_cards/
#   vdd:          core supply voltage (V)
#   nmos_file:    exact modelcard filename for NMOS
#   pmos_file:    exact modelcard filename for PMOS
#   nmos_model:   .model name inside the NMOS modelcard
#   pmos_model:   .model name inside the PMOS modelcard
#   nmos_params:  instance params for NMOS (baked into modelcard for NGSPICE)
#   pmos_params:  instance params for PMOS
#
TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "ASAP7": {
        "dir": "ASAP7",
        "vdd": 0.9,
        "corner": "TT",
        "nmos_file": "7nm_TT_160803.pm",
        "pmos_file": "7nm_TT_160803.pm",
        "nmos_model": "nmos_rvt",
        "pmos_model": "pmos_rvt",
        "nmos_params": {"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 1.0, "DEVTYPE": 1},
        "pmos_params": {"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 1.0, "DEVTYPE": 0},
    },
    "TSMC5": {
        "dir": "TSMC5/naive",
        "vdd": 0.65,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 1},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 0},
    },
    "TSMC7": {
        "dir": "TSMC7/naive",
        "vdd": 0.75,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 1},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 0},
    },
    "TSMC12": {
        "dir": "TSMC12/naive",
        "vdd": 0.80,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 1},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 0},
    },
    "TSMC16": {
        "dir": "TSMC16/naive",
        "vdd": 0.80,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 1},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 0},
    },
}

TECH_NAMES = list(TECHNOLOGIES.keys())


def get_tech_modelcard(tech_name: str, device_type: str = "nmos") -> Tuple[Path, str, Dict[str, float]]:
    """Get modelcard path, model name, and instance params for a technology.

    Args:
        tech_name: Key from TECHNOLOGIES registry
        device_type: "nmos" or "pmos"

    Returns:
        Tuple of (modelcard_path, model_name, inst_params)
    """
    tech = TECHNOLOGIES[tech_name]
    tech_dir = ROOT / "tech_model_cards" / tech["dir"]

    file_key = f"{device_type}_file"
    model_key = f"{device_type}_model"
    params_key = f"{device_type}_params"

    modelcard = tech_dir / tech[file_key]
    if not modelcard.exists():
        raise FileNotFoundError(f"Modelcard not found: {modelcard}")

    return modelcard, tech[model_key], tech[params_key]


# -- pytest hooks (keep existing) --

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Add test report attribute to node for result tracking."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)
