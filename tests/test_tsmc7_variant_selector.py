"""
Unit tests for TSMC7 variant selector function.

Tests the select_tsmc7_variant() function which selects the appropriate
model variant based on device length L.
"""

import pytest

from pycmg.ctypes_host import select_tsmc7_variant


class TestTSMC7VariantSelector:
    """Test suite for TSMC7 variant selection."""

    @pytest.fixture
    def tsmc7_modelcard_path(self):
        """Path to the TSMC7 modelcard file."""
        return "/home/shenshan/pycmg-wrapper/.worktrees/more-techs/tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l"

    def test_select_variant_short_device(self, tsmc7_modelcard_path):
        """Test variant selection for very short device (L = 10nm)."""
        L = 10e-9  # 10 nm
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # Should select variant with range covering 10nm
        # Based on the modelcard: nch_svt_mac.6 has lmin=8e-09, lmax=1.1e-08
        assert result == "nch_svt_mac.6"

    def test_select_variant_nominal_device(self, tsmc7_modelcard_path):
        """Test variant selection for nominal device (L = 50nm)."""
        L = 50e-9  # 50 nm
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # Should select variant with range covering 50nm
        # Based on the modelcard: nch_svt_mac.3 has lmin=3.6e-08, lmax=7.2e-08
        assert result == "nch_svt_mac.3"

    def test_select_variant_long_device(self, tsmc7_modelcard_path):
        """Test variant selection for long device (L = 200nm)."""
        L = 200e-9  # 200 nm
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # Should select variant with range covering 200nm
        # Based on the modelcard: nch_svt_mac.1 has lmin=1.2e-07, lmax=2.4e-07
        assert result == "nch_svt_mac.1"

    def test_select_variant_at_lower_boundary(self, tsmc7_modelcard_path):
        """Test variant selection at exact lower boundary (L = 72nm)."""
        L = 72e-9  # Exactly at lower boundary of variant.2 AND upper boundary of variant.3
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # When L is at a boundary, both variants match. The algorithm prefers
        # the variant with the smallest range for precision.
        # variant.3 has lmin=3.6e-08, lmax=7.2e-08 (range=36nm)
        # variant.2 has lmin=7.2e-08, lmax=1.2e-07 (range=48nm)
        # So variant.3 is selected (smaller range)
        assert result == "nch_svt_mac.3"

    def test_select_variant_at_upper_boundary(self, tsmc7_modelcard_path):
        """Test variant selection at exact upper boundary (L = 120nm)."""
        L = 120e-9  # Exactly at upper boundary of variant.2
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # Should select nch_svt_mac.2 which has lmax=1.2e-07
        # (edge case: L == lmax should select this variant, not the next)
        assert result == "nch_svt_mac.2"

    def test_select_variant_very_short(self, tsmc7_modelcard_path):
        """Test variant selection for extremely short device (L = 8.5nm)."""
        L = 8.5e-9  # 8.5 nm
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # Should select nch_svt_mac.6 (lmin=8e-09, lmax=1.1e-08)
        assert result == "nch_svt_mac.6"

    def test_select_variant_mid_range(self, tsmc7_modelcard_path):
        """Test variant selection in middle of range (L = 28nm)."""
        L = 28e-9  # 28 nm
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        # Should select nch_svt_mac.4 (lmin=2e-08, lmax=3.6e-08)
        assert result == "nch_svt_mac.4"

    def test_error_out_of_range_below(self, tsmc7_modelcard_path):
        """Test error when L is below all available ranges."""
        L = 1e-9  # 1 nm - below minimum (8nm)
        with pytest.raises(ValueError) as exc_info:
            select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        error_msg = str(exc_info.value)
        assert "No TSMC7 variant found" in error_msg
        assert f"L={L:.3e}" in error_msg
        assert "Available ranges" in error_msg

    def test_error_out_of_range_above(self, tsmc7_modelcard_path):
        """Test error when L is above all available ranges."""
        L = 1e-6  # 1 um - above maximum (240nm)
        with pytest.raises(ValueError) as exc_info:
            select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        error_msg = str(exc_info.value)
        assert "No TSMC7 variant found" in error_msg
        assert f"L={L:.3e}" in error_msg
        assert "Available ranges" in error_msg

    def test_error_nonexistent_model(self, tsmc7_modelcard_path):
        """Test error when model type doesn't exist."""
        L = 50e-9
        with pytest.raises(ValueError) as exc_info:
            select_tsmc7_variant(tsmc7_modelcard_path, "nonexistent_model", L)
        error_msg = str(exc_info.value)
        assert "No TSMC7 variants found" in error_msg
        assert "nonexistent_model" in error_msg

    def test_error_invalid_file(self):
        """Test error when modelcard file doesn't exist."""
        L = 50e-9
        with pytest.raises(FileNotFoundError):
            select_tsmc7_variant("/nonexistent/file.l", "nch_svt_mac", L)

    def test_pch_model_selection(self, tsmc7_modelcard_path):
        """Test variant selection for pch model."""
        # First check if pch_svt_mac exists in the modelcard
        import re
        with open(tsmc7_modelcard_path, "r") as f:
            content = f.read()
            has_pch = bool(re.search(r'\.model pch_svt_mac\.\d+ pmos', content))

        if has_pch:
            L = 50e-9
            result = select_tsmc7_variant(tsmc7_modelcard_path, "pch_svt_mac", L)
            # Should return a variant like "pch_svt_mac.X"
            assert result.startswith("pch_svt_mac.")
            assert "." in result
        else:
            pytest.skip("pch_svt_mac not found in test modelcard")

    def test_boundary_case_minimum_l(self, tsmc7_modelcard_path):
        """Test at the absolute minimum L in the modelcard."""
        # The minimum L in nch_svt_mac is 8e-09 (variant.6)
        L = 8e-9
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        assert result == "nch_svt_mac.6"

    def test_boundary_case_maximum_l(self, tsmc7_modelcard_path):
        """Test at the absolute maximum L in the modelcard."""
        # The maximum L in nch_svt_mac is 2.4e-07 (variant.1)
        L = 2.4e-7
        result = select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)
        assert result == "nch_svt_mac.1"

    def test_error_message_includes_all_variants(self, tsmc7_modelcard_path):
        """Test that error message lists all available variant ranges."""
        L = 1e-9  # Out of range
        with pytest.raises(ValueError) as exc_info:
            select_tsmc7_variant(tsmc7_modelcard_path, "nch_svt_mac", L)

        error_msg = str(exc_info.value)
        # Check that error contains multiple variant information
        assert "nch_svt_mac.1" in error_msg
        assert "nch_svt_mac.2" in error_msg
        # Check that ranges are shown in scientific notation
        assert "e-" in error_msg or "e+" in error_msg

    def test_case_sensitivity(self, tsmc7_modelcard_path):
        """Test that model_type is case-sensitive."""
        L = 50e-9
        # Try with wrong case - should fail
        with pytest.raises(ValueError):
            select_tsmc7_variant(tsmc7_modelcard_path, "NCH_SVT_MAC", L)
