#!/usr/bin/env python3
"""
TSMC7 NF and NFIN Parameter Verification Tests

Tests that NF (number of fins) and NFIN (fin count) parameters
are correctly recognized and applied in PyCMG for TSMC7 modelcards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pycmg
from pycmg.ctypes_host import Model, Instance, parse_modelcard

# Paths
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
TSMC7_DIR = ROOT / "tech_model_cards" / "TSMC7"
TSMC7_MODELCARD = TSMC7_DIR / "tsmc7_simple.l"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_modelcard_has_nf_nfin() -> None:
    """Verify TSMC7 modelcard includes NF and NFIN parameters."""
    if not TSMC7_MODELCARD.exists():
        pytest.skip(f"TSMC7 modelcard not found: {TSMC7_MODELCARD}")

    # Parse modelcard
    parsed = parse_modelcard(str(TSMC7_MODELCARD), "nch_svt_mac")

    # Check NF and NFIN are present
    assert "nf" in parsed.params, "NF parameter not found in TSMC7 modelcard"
    assert "nfin" in parsed.params, "NFIN parameter not found in TSMC7 modelcard"

    # Verify default values (should be 1.0)
    assert parsed.params["nf"] == 1.0, f"NF should be 1.0, got {parsed.params['nf']}"
    assert parsed.params["nfin"] == 1.0, f"NFIN should be 1.0, got {parsed.params['nfin']}"

    print(f"✓ TSMC7 modelcard has NF={parsed.params['nf']} and NFIN={parsed.params['nfin']}")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_instance_with_nf_nfin() -> None:
    """Verify PyCMG can create instances with NF and NFIN parameters."""
    if not TSMC7_MODELCARD.exists():
        pytest.skip(f"TSMC7 modelcard not found: {TSMC7_MODELCARD}")

    # Create model
    model = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), "nch_svt_mac")

    # Create instance with NF and NFIN parameters
    inst_params = {
        "L": 16e-9,
        "TFIN": 6e-9,
        "NF": 2.0,   # Number of fins
        "NFIN": 3.0,  # Fin count multiplier
    }

    inst = Instance(model, params=inst_params)

    # Run DC analysis at nominal bias
    result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

    # Verify we get valid results
    assert result["id"] != 0, "Id should not be zero at Vg=0.75V, Vd=0.75V"
    assert abs(result["ig"]) < 1e-6, "Ig should be very small (leakage only)"
    # For NMOS with Id negative (into drain) and Is positive (out of source),
    # Ids = Id - Is will be negative
    assert result["ids"] < 0, "Ids should be negative for NMOS (current into drain)"

    print(f"✓ Instance created with NF=2.0, NFIN=3.0")
    print(f"  Id = {result['id']:.6e} A")
    print(f"  Ids = {result['ids']:.6e} A")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_nfin_scaling() -> None:
    """Verify that NFIN parameter scales drain current correctly.

    For FinFET devices, Id should scale proportionally with NFIN.
    This test verifies NFIN=2 produces ~2× current vs NFIN=1.
    """
    if not TSMC7_MODELCARD.exists():
        pytest.skip(f"TSMC7 modelcard not found: {TSMC7_MODELCARD}")

    model = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), "nch_svt_mac")

    # Test NFIN=1
    inst1 = Instance(model, params={"L": 16e-9, "TFIN": 6e-9, "NFIN": 1.0})
    result1 = inst1.eval_dc({"d": 0.6, "g": 0.75, "s": 0.0, "e": 0.0})
    ids_nfin1 = result1["ids"]

    # Test NFIN=2
    inst2 = Instance(model, params={"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0})
    result2 = inst2.eval_dc({"d": 0.6, "g": 0.75, "s": 0.0, "e": 0.0})
    ids_nfin2 = result2["ids"]

    # Verify NFIN=2 produces approximately 2× current
    ratio = ids_nfin2 / ids_nfin1
    assert 1.8 < ratio < 2.2, f"NFIN scaling ratio={ratio:.2f}, expected ~2.0"

    print(f"✓ NFIN scaling verified:")
    print(f"  NFIN=1: Ids = {ids_nfin1:.6e} A")
    print(f"  NFIN=2: Ids = {ids_nfin2:.6e} A")
    print(f"  Ratio = {ratio:.2f}× (expected ~2.0×)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
