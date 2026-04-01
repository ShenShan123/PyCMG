"""
Pytest configuration and technology registry for PyCMG verification tests.

The registry provides deterministic modelcard selection:
- ASAP7: Explicit TT corner + rvt variant (no glob ambiguity)
- TSMC: Explicit file names + per-device instance params (PMOS L=20nm)

Tiered registry:
- TECHNOLOGIES / TECH_NAMES: Original 5 entries (backward-compatible)
- CORE_VT_VARIANTS / CORE_VT_NAMES: Additional core-voltage Vt flavors
- ALL_TECHNOLOGIES / ALL_TECH_NAMES: Union of all

This module imports from ``pycmg.tech.TECH_REGISTRY`` as the source of truth
for technology metadata (vdd, tfin), then augments with test-specific fields
(L, NFIN, naive modelcard dirs/filenames) that are only needed for NGSPICE
verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from pycmg.tech import TECH_REGISTRY, TechConfig
from tests.helpers import ROOT, OSDI_PATH

# ---------------------------------------------------------------------------
# Test-specific constants: L and NFIN values used by existing tests.
# These are NOT stored in pycmg.tech (which treats L and NFIN as swept params).
# ---------------------------------------------------------------------------
_ASAP7_TEST_L = 7e-9
_ASAP7_TEST_NFIN = 1.0

_TSMC_NMOS_TEST_L = 16e-9
_TSMC_PMOS_TEST_L = 20e-9   # PMOS uses L=20nm to avoid L=16nm convergence
_TSMC_TEST_NFIN = 2.0

# ---------------------------------------------------------------------------
# Helper: build old-format inst_params dicts from tech registry + test values
# ---------------------------------------------------------------------------

def _make_asap7_params(is_pmos: bool) -> Dict[str, float]:
    """Build ASAP7 inst_params in the old test format (includes L, NFIN)."""
    tfin = TECH_REGISTRY["ASAP7"].tfin
    return {
        "L": _ASAP7_TEST_L,
        "TFIN": tfin,
        "NFIN": _ASAP7_TEST_NFIN,
        "DEVTYPE": 0 if is_pmos else 1,
    }


def _make_tsmc_params(is_pmos: bool, tech_name: str = "TSMC7") -> Dict[str, float]:
    """Build TSMC inst_params in the old test format (includes L, NFIN)."""
    # All TSMC nodes share TFIN=6e-9 — use the registry value for consistency
    tfin = TECH_REGISTRY[tech_name].tfin
    return {
        "L": _TSMC_PMOS_TEST_L if is_pmos else _TSMC_NMOS_TEST_L,
        "TFIN": tfin,
        "NFIN": _TSMC_TEST_NFIN,
        "DEVTYPE": 0 if is_pmos else 1,
    }


# ---------------------------------------------------------------------------
# Helper: ASAP7 / TSMC entry builders (old dict format for test consumption)
# ---------------------------------------------------------------------------

def _asap7_entry(nmos_model: str, pmos_model: str) -> Dict[str, Any]:
    """Build an ASAP7 registry entry (all variants share the same TT file)."""
    return {
        "dir": "ASAP7", "vdd": TECH_REGISTRY["ASAP7"].vdd, "corner": "TT",
        "nmos_file": "7nm_TT_160803.pm", "pmos_file": "7nm_TT_160803.pm",
        "nmos_model": nmos_model, "pmos_model": pmos_model,
        "nmos_params": _make_asap7_params(is_pmos=False),
        "pmos_params": _make_asap7_params(is_pmos=True),
    }


def _tsmc_entry(
    tech_name: str, tech_dir: str, vt: str,
) -> Dict[str, Any]:
    """Build a TSMC registry entry for a core-voltage Vt variant.

    Follows the naming convention: nch_{vt}_mac / pch_{vt}_mac
    NMOS uses L=16nm file, PMOS uses L=20nm file (avoids L=16nm convergence).
    """
    return {
        "dir": tech_dir, "vdd": TECH_REGISTRY[tech_name].vdd,
        "nmos_file": f"nch_{vt}_l16nm.l",
        "pmos_file": f"pch_{vt}_l20nm.l",
        "nmos_model": f"nch_{vt}",
        "pmos_model": f"pch_{vt}",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name=tech_name),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name=tech_name),
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
        "dir": "TSMC5/naive", "vdd": TECH_REGISTRY["TSMC5"].vdd,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC5"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC5"),
    },
    "TSMC7": {
        "dir": "TSMC7/naive", "vdd": TECH_REGISTRY["TSMC7"].vdd,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC7"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC7"),
    },
    "TSMC12": {
        "dir": "TSMC12/naive", "vdd": TECH_REGISTRY["TSMC12"].vdd,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC12"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC12"),
    },
    "TSMC16": {
        "dir": "TSMC16/naive", "vdd": TECH_REGISTRY["TSMC16"].vdd,
        "nmos_file": "nch_svt_mac_l16nm.l", "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC16"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC16"),
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
    "TSMC5_lvt":  _tsmc_entry("TSMC5", "TSMC5/naive", "lvt_mac"),
    "TSMC5_ulvt": _tsmc_entry("TSMC5", "TSMC5/naive", "ulvt_mac"),
    "TSMC5_elvt": _tsmc_entry("TSMC5", "TSMC5/naive", "elvt_mac"),

    # TSMC7 — lvt already tested; add ulvt
    "TSMC7_lvt":  _tsmc_entry("TSMC7", "TSMC7/naive", "lvt_mac"),
    "TSMC7_ulvt": _tsmc_entry("TSMC7", "TSMC7/naive", "ulvt_mac"),

    # TSMC12 — lvt already tested; add hvt, ulvt, lnvt
    "TSMC12_lvt":  _tsmc_entry("TSMC12", "TSMC12/naive", "lvt_mac"),
    "TSMC12_hvt":  _tsmc_entry("TSMC12", "TSMC12/naive", "hvt_mac"),
    "TSMC12_ulvt": _tsmc_entry("TSMC12", "TSMC12/naive", "ulvt_mac"),
    "TSMC12_lnvt": _tsmc_entry("TSMC12", "TSMC12/naive", "lnvt_mac"),

    # TSMC16 — lvt already tested; add hvt, ulvt, lnvt
    "TSMC16_lvt":  _tsmc_entry("TSMC16", "TSMC16/naive", "lvt_mac"),
    "TSMC16_hvt":  _tsmc_entry("TSMC16", "TSMC16/naive", "hvt_mac"),
    "TSMC16_ulvt": _tsmc_entry("TSMC16", "TSMC16/naive", "ulvt_mac"),
    "TSMC16_lnvt": _tsmc_entry("TSMC16", "TSMC16/naive", "lnvt_mac"),
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
