"""
Full TSMC7 PDK Verification Tests

Tests parse_tsmc7_pdk() function with full TSMC7 PDK modelcard.
"""

import pytest
from pathlib import Path

import pycmg
from pycmg.ctypes_host import parse_tsmc7_pdk

ROOT = Path(__file__).resolve().parents[1]
TSMC7_PDK_PATH = ROOT / "tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l"
OSDI_PATH = ROOT / "build-deep-verify/osdi/bsimcmg.osdi"


def test_tsmc7_pdk_nmos_svtt_extraction():
    """Test parameter extraction from full PDK."""
    L = 16e-9

    # Extract using parse_tsmc7_pdk
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), "nch", "svt_mac", L)

    # Verify extraction
    assert parsed.name == "nch_svt_mac"
    assert "level" in parsed.params
    assert parsed.params["level"] == 72
    assert "version" in parsed.params

    # Check some key parameters exist
    assert "eot" in parsed.params
    assert "tfin" in parsed.params
    assert "hfin" in parsed.params


def test_tsmc7_pdk_nmos_svtt_eval_dc():
    """Test DC evaluation with full TSMC7 PDK."""
    from pycmg.ctypes_host import Model, Instance

    L = 16e-9
    TFIN = 6e-9
    NFIN = 2.0

    # Extract using parse_tsmc7_pdk
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), "nch", "svt_mac", L)

    # Create model and instance
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), "nch_svt_mac", parsed.params)
    inst = Instance(model, L=L, TFIN=TFIN, NFIN=NFIN)

    # Run DC analysis
    result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

    # Verify we get reasonable results
    assert result.id > 0  # Drain current should flow
    assert abs(result.ig) < 1e-6  # Gate current should be small
    assert abs(result.ids) < 1e-6  # Source current should be small
    assert abs(result.ie) < 1e-6  # Bulk current should be small
