"""
Variant Selection Tests (Fixed)

Tests that _find_length_variant correctly selects variants for different L values.
"""

import pytest
from pathlib import Path
from pycmg.ctypes_host import _find_length_variant

ROOT = Path(__file__).resolve().parents[1]
TSMC7_PDK_PATH = ROOT / "tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l"


@pytest.mark.parametrize("L,expected_variant", [
    (120e-9, 1),   # Variant 1: lmin=1.2e-07, lmax=2.4001e-07 (120-240nm)
    (72e-9, 3),    # Variant 3: lmin=3.6e-08, lmax=7.2e-08 (36-72nm)
    (36e-9, 3),    # Variant 3: lmin=3.6e-08, lmax=7.2e-08 (36-72nm)
    (20e-9, 4),    # Variant 4: lmin=2e-08, lmax=3.6e-08 (20-36nm)
    (16e-9, 4),    # Variant 4: lmin=2e-08, lmax=3.6e-08 (20-36nm)
    (12e-9, 4),    # Variant 4: lmin=2e-08, lmax=3.6e-08 (20-36nm)
    (8e-9, 6),     # Variant 6: lmin=8e-09, lmax=1.1e-08 (8-11nm)
])
def test_tsmc7_pdk_variant_selection(L, expected_variant):
    """Test correct variant selection for different L values."""
    # Test with nch_svt_mac (NMOS SVT)
    variant_num = _find_length_variant(str(TSMC7_PDK_PATH), "nch_svt_mac", L)

    assert variant_num == expected_variant, (
        f"L={L*1e9:.1f}nm should select variant {expected_variant}, got {variant_num}"
    )
