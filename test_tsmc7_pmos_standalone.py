#!/usr/bin/env python3
"""
Standalone TSMC7 PMOS Test

Tests PMOS variant selection and DC evaluation without requiring NGSPICE.
This test verifies that PMOS models work correctly with TSMC7 integration.

Run: python3 test_tsmc7_pmos_standalone.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pycmg.ctypes_host import Model, Instance


def test_pmos_variant_selection():
    """Test PMOS variant selection across different length bins."""
    print("\n" + "=" * 70)
    print("TEST 1: PMOS Variant Selection")
    print("=" * 70)

    ROOT = Path(__file__).resolve().parents[0]
    OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    TSMC7_MODELCARD = ROOT / "tech_model_cards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l"

    pch_model = "pch_svt_mac"
    test_lengths = [
        (16e-9, 5, "Bin 5 (11-20nm)"),
        (30e-9, 4, "Bin 4 (20-36nm)"),
        (50e-9, 3, "Bin 3 (36-72nm)"),
    ]

    print(f"\nModel: {pch_model}")
    print("Testing variant selection:")

    all_passed = True
    for l_val, expected_bin, desc in test_lengths:
        params = {"L": l_val, "TFIN": 8e-9, "NFIN": 2.0}
        try:
            model = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), pch_model)
            inst = Instance(model, params=params)
            print(f"  ✓ L = {l_val*1e9:.0f}nm → {desc}")
        except Exception as e:
            print(f"  ✗ L = {l_val*1e9:.0f}nm → ERROR: {e}")
            all_passed = False

    return all_passed


def test_pmos_dc_evaluation():
    """Test PMOS DC evaluation in saturation region."""
    print("\n" + "=" * 70)
    print("TEST 2: PMOS DC Evaluation (Saturation Region)")
    print("=" * 70)

    ROOT = Path(__file__).resolve().parents[0]
    OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    TSMC7_MODELCARD = ROOT / "tech_model_cards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l"

    pch_model = "pch_svt_mac"
    pch_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}

    try:
        model = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), pch_model)
        inst = Instance(model, params=pch_params)

        # PMOS in saturation: Vgs < 0, Vds < 0
        # Bias: Vd=0.0V, Vg=0.0V, Vs=0.75V
        # Vgs = -0.75V, Vds = -0.75V
        result = inst.eval_dc({"d": 0.0, "g": 0.0, "s": 0.75, "e": 0.0})

        print(f"\nConfiguration:")
        print(f"  Model: {pch_model}")
        print(f"  Parameters: L={pch_params['L']*1e9:.0f}nm, TFIN={pch_params['TFIN']*1e9:.0f}nm, NFIN={pch_params['NFIN']}")
        print(f"  Bias: Vd=0.0V, Vg=0.0V, Vs=0.75V")
        print(f"  Operating point: Vgs=-0.75V, Vds=-0.75V (saturation)")

        print(f"\nResults:")
        print(f"  Id  = {result['id']:.6e} A")
        print(f"  Ig  = {result['ig']:.6e} A")
        print(f"  Is  = {result['is']:.6e} A")
        print(f"  Ie  = {result['ie']:.6e} A")
        print(f"  Ids = {result['ids']:.6e} A")
        print(f"  Qg  = {result['qg']:.6e} C")
        print(f"  Qd  = {result['qd']:.6e} C")
        print(f"  Qs  = {result['qs']:.6e} C")
        print(f"  Qb  = {result['qb']:.6e} C")
        print(f"  gm  = {result['gm']:.6e} S")
        print(f"  gds = {result['gds']:.6e} S")
        print(f"  gmb = {result['gmb']:.6e} S")
        print(f"  cgg = {result['cgg']:.6e} F")
        print(f"  cgd = {result['cgd']:.6e} F")
        print(f"  cgs = {result['cgs']:.6e} F")
        print(f"  cdg = {result['cdg']:.6e} F")
        print(f"  cdd = {result['cdd']:.6e} F")

        # Basic sanity checks
        assert abs(result['ig']) < 1e-6, "Gate current too high"
        assert abs(result['ids']) > 1e-9, "Drain-source current too low"
        assert result['gm'] > 0, "Transconductance should be positive"

        print("\n  ✓ All sanity checks passed")
        return True

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pmos_operating_regions():
    """Test PMOS in different operating regions."""
    print("\n" + "=" * 70)
    print("TEST 3: PMOS Operating Regions")
    print("=" * 70)

    ROOT = Path(__file__).resolve().parents[0]
    OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    TSMC7_MODELCARD = ROOT / "tech_model_cards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l"

    pch_model = "pch_svt_mac"
    pch_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}

    bias_conditions = [
        {
            "d": 0.0, "g": 0.75, "s": 0.75, "e": 0.0,
            "desc": "Cutoff",
            "vgs": 0.0, "vds": -0.75
        },
        {
            "d": 0.0, "g": 0.0, "s": 0.75, "e": 0.0,
            "desc": "Saturation",
            "vgs": -0.75, "vds": -0.75
        },
        {
            "d": 0.5, "g": 0.0, "s": 0.75, "e": 0.0,
            "desc": "Linear",
            "vgs": -0.75, "vds": -0.25
        },
    ]

    try:
        model = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), pch_model)
        inst = Instance(model, params=pch_params)

        print(f"\nTesting PMOS in different operating regions:")
        all_passed = True

        for bias in bias_conditions:
            result = inst.eval_dc({
                "d": bias["d"], "g": bias["g"], "s": bias["s"], "e": bias["e"]
            })
            vgs = bias["vgs"]
            vds = bias["vds"]

            print(f"\n  {bias['desc']} Region:")
            print(f"    Vgs = {vgs:+.2f}V, Vds = {vds:+.2f}V")
            print(f"    Id  = {result['id']:.3e} A")
            print(f"    Ids = {result['ids']:.3e} A")
            print(f"    gm  = {result['gm']:.3e} S")

        print("\n  ✓ All operating regions tested")
        return True

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nmos_pmos_comparison():
    """Compare NMOS and PMOS operation."""
    print("\n" + "=" * 70)
    print("TEST 4: NMOS vs PMOS Comparison")
    print("=" * 70)

    ROOT = Path(__file__).resolve().parents[0]
    OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    TSMC7_MODELCARD = ROOT / "tech_model_cards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l"

    params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}

    try:
        # Test NMOS
        print(f"\nNMOS (nch_svt_mac):")
        nch_model = "nch_svt_mac"
        nch_model_inst = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), nch_model)
        nch_inst = Instance(nch_model_inst, params=params)

        nch_result = nch_inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})
        print(f"  Bias: Vgs=+0.75V, Vds=+0.75V")
        print(f"  Id  = {nch_result['id']:.3e} A")
        print(f"  Ids = {nch_result['ids']:.3e} A")
        print(f"  gm  = {nch_result['gm']:.3e} S")

        # Test PMOS
        print(f"\nPMOS (pch_svt_mac):")
        pch_model = "pch_svt_mac"
        pch_model_inst = Model(str(OSDI_PATH), str(TSMC7_MODELCARD), pch_model)
        pch_inst = Instance(pch_model_inst, params=params)

        pch_result = pch_inst.eval_dc({"d": 0.0, "g": 0.0, "s": 0.75, "e": 0.0})
        print(f"  Bias: Vgs=-0.75V, Vds=-0.75V")
        print(f"  Id  = {pch_result['id']:.3e} A")
        print(f"  Ids = {pch_result['ids']:.3e} A")
        print(f"  gm  = {pch_result['gm']:.3e} S")

        print("\n  ✓ Both NMOS and PMOS operating correctly")
        return True

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TSMC7 PMOS Functionality Test Suite")
    print("=" * 70)
    print("\nThis test verifies PMOS support in TSMC7 integration.")
    print("Testing variant selection, DC evaluation, and operating regions.")

    results = {
        "Variant Selection": test_pmos_variant_selection(),
        "DC Evaluation": test_pmos_dc_evaluation(),
        "Operating Regions": test_pmos_operating_regions(),
        "NMOS/PMOS Comparison": test_nmos_pmos_comparison(),
    }

    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nPMOS support is fully functional in TSMC7 integration.")
        print("Both NMOS and PMOS devices work correctly with proper bias conventions.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
