"""
Pytest configuration and technology registry for PyCMG verification tests.

The registry provides deterministic modelcard selection:
- ASAP7: Explicit TT corner + rvt variant (no glob ambiguity)
- TSMC: Explicit file names + per-device instance params (PMOS L=20nm)

Tiered registry:
- TECHNOLOGIES / TECH_NAMES: Original 5 entries (backward-compatible)
- CORE_VT_VARIANTS / CORE_VT_NAMES: Additional core-voltage Vt flavors
- ALL_TECHNOLOGIES / ALL_TECH_NAMES: Union of all
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"

# ---------------------------------------------------------------------------
# Helper: TSMC instance param templates (same geometry within a tech node)
# ---------------------------------------------------------------------------
_ASAP7_NMOS_PARAMS = {"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 1.0, "DEVTYPE": 1}
_ASAP7_PMOS_PARAMS = {"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 1.0, "DEVTYPE": 0}

_TSMC_NMOS_PARAMS = {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 1}
_TSMC_PMOS_PARAMS = {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0, "DEVTYPE": 0}


def _asap7_entry(nmos_model: str, pmos_model: str) -> Dict[str, Any]:
    """Build an ASAP7 registry entry (all variants share the same TT file)."""
    return {
        "dir": "ASAP7", "vdd": 0.9, "corner": "TT",
        "nmos_file": "7nm_TT_160803.pm", "pmos_file": "7nm_TT_160803.pm",
        "nmos_model": nmos_model, "pmos_model": pmos_model,
        "nmos_params": _ASAP7_NMOS_PARAMS.copy(), "pmos_params": _ASAP7_PMOS_PARAMS.copy(),
    }


def _tsmc_entry(
    tech_dir: str, vdd: float, vt: str,
) -> Dict[str, Any]:
    """Build a TSMC registry entry for a core-voltage Vt variant.

    Follows the naming convention: nch_{vt}_mac / pch_{vt}_mac
    NMOS uses L=16nm file, PMOS uses L=20nm file (avoids L=16nm convergence).
    """
    return {
        "dir": tech_dir, "vdd": vdd,
        "nmos_file": f"nch_{vt}_l16nm.l",
        "pmos_file": f"pch_{vt}_l20nm.l",
        "nmos_model": f"nch_{vt}",
        "pmos_model": f"pch_{vt}",
        "nmos_params": _TSMC_NMOS_PARAMS.copy(), "pmos_params": _TSMC_PMOS_PARAMS.copy(),
    }


# ---------------------------------------------------------------------------
# Tier 1: Original technologies (backward-compatible, used by existing tests)
# ---------------------------------------------------------------------------
#
# Each entry specifies:
#   dir:          subdirectory under modelcards/
#   vdd:          core supply voltage (V)
#   nmos_file:    exact modelcard filename for NMOS
#   pmos_file:    exact modelcard filename for PMOS
#   nmos_model:   .model name inside the NMOS modelcard
#   pmos_model:   .model name inside the PMOS modelcard
#   nmos_params:  instance params for NMOS (baked into modelcard for NGSPICE)
#   pmos_params:  instance params for PMOS
#
TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "ASAP7":  _asap7_entry("nmos_rvt", "pmos_rvt"),
    # NOTE: Original TSMC entries use NMOS=svt + PMOS=lvt (historical choice).
    # New Vt variant entries in CORE_VT_VARIANTS use matched Vt for both.
    "TSMC5": {
        "dir": "TSMC5/naive", "vdd": 0.65,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _TSMC_NMOS_PARAMS.copy(), "pmos_params": _TSMC_PMOS_PARAMS.copy(),
    },
    "TSMC7": {
        "dir": "TSMC7/naive", "vdd": 0.75,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _TSMC_NMOS_PARAMS.copy(), "pmos_params": _TSMC_PMOS_PARAMS.copy(),
    },
    "TSMC12": {
        "dir": "TSMC12/naive", "vdd": 0.80,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _TSMC_NMOS_PARAMS.copy(), "pmos_params": _TSMC_PMOS_PARAMS.copy(),
    },
    "TSMC16": {
        "dir": "TSMC16/naive", "vdd": 0.80,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _TSMC_NMOS_PARAMS.copy(), "pmos_params": _TSMC_PMOS_PARAMS.copy(),
    },
}

TECH_NAMES = list(TECHNOLOGIES.keys())

# ---------------------------------------------------------------------------
# Tier 2: Core-voltage Vt variants (same Vdd & geometry, different threshold)
# ---------------------------------------------------------------------------
CORE_VT_VARIANTS: Dict[str, Dict[str, Any]] = {
    # ASAP7 — lvt, slvt, sram (rvt is already in TECHNOLOGIES)
    "ASAP7_lvt":  _asap7_entry("nmos_lvt", "pmos_lvt"),
    "ASAP7_slvt": _asap7_entry("nmos_slvt", "pmos_slvt"),
    "ASAP7_sram": _asap7_entry("nmos_sram", "pmos_sram"),

    # TSMC5 — lvt already tested via TECHNOLOGIES; add ulvt, elvt
    "TSMC5_lvt":  _tsmc_entry("TSMC5/naive", 0.65, "lvt_mac"),
    "TSMC5_ulvt": _tsmc_entry("TSMC5/naive", 0.65, "ulvt_mac"),
    "TSMC5_elvt": _tsmc_entry("TSMC5/naive", 0.65, "elvt_mac"),

    # TSMC7 — lvt already tested; add ulvt
    "TSMC7_lvt":  _tsmc_entry("TSMC7/naive", 0.75, "lvt_mac"),
    "TSMC7_ulvt": _tsmc_entry("TSMC7/naive", 0.75, "ulvt_mac"),

    # TSMC12 — lvt already tested; add hvt, ulvt, lnvt
    "TSMC12_lvt":  _tsmc_entry("TSMC12/naive", 0.80, "lvt_mac"),
    "TSMC12_hvt":  _tsmc_entry("TSMC12/naive", 0.80, "hvt_mac"),
    "TSMC12_ulvt": _tsmc_entry("TSMC12/naive", 0.80, "ulvt_mac"),
    "TSMC12_lnvt": _tsmc_entry("TSMC12/naive", 0.80, "lnvt_mac"),

    # TSMC16 — lvt already tested; add hvt, ulvt, lnvt
    "TSMC16_lvt":  _tsmc_entry("TSMC16/naive", 0.80, "lvt_mac"),
    "TSMC16_hvt":  _tsmc_entry("TSMC16/naive", 0.80, "hvt_mac"),
    "TSMC16_ulvt": _tsmc_entry("TSMC16/naive", 0.80, "ulvt_mac"),
    "TSMC16_lnvt": _tsmc_entry("TSMC16/naive", 0.80, "lnvt_mac"),
}

CORE_VT_NAMES = list(CORE_VT_VARIANTS.keys())

# ---------------------------------------------------------------------------
# Union: all technologies (Tier 1 + Tier 2)
# ---------------------------------------------------------------------------
ALL_TECHNOLOGIES: Dict[str, Dict[str, Any]] = {**TECHNOLOGIES, **CORE_VT_VARIANTS}
ALL_TECH_NAMES = list(ALL_TECHNOLOGIES.keys())


def get_tech_modelcard(tech_name: str, device_type: str = "nmos") -> Tuple[Path, str, Dict[str, float]]:
    """Get modelcard path, model name, and instance params for a technology.

    Searches ALL_TECHNOLOGIES (Tier 1 + Tier 2).

    Args:
        tech_name: Key from ALL_TECHNOLOGIES registry
        device_type: "nmos" or "pmos"

    Returns:
        Tuple of (modelcard_path, model_name, inst_params)
    """
    tech = ALL_TECHNOLOGIES[tech_name]
    tech_dir = ROOT / "modelcards" / tech["dir"]

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
