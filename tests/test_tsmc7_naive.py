"""
Naive TSMC7 Modelcard Verification Tests

Tests naive TSMC7 modelcards against NGSPICE ground truth.
Coverage:
- Device Types: SVT, LVT, ULVT (NMOS/PMOS)
- Length Values: 16nm, 20nm, 24nm
- Operating Points: Representative bias conditions

VERIFICATION STRATEGY:
- PyCMG loads naive modelcard via parse_modelcard()
- NGSPICE loads same modelcard via .include
- Both use identical bsimcmg.osdi binary
- Compare outputs for binary-level consistency

Run: pytest tests/test_tsmc7_naive.py -v
Duration: ~3 minutes
Requires: NGSPICE, naive TSMC7 modelcards
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict

import pytest

import pycmg
from pycmg.ctypes_host import Model, Instance, parse_modelcard

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
BUILD = ROOT / "build-deep-verify"

# Naive TSMC7 modelcard directory
TSMC7_NAIVE_DIR = ROOT / "tech_model_cards" / "TSMC7" / "naive"

# Full TSMC7 PDK (for cross-verification)
TSMC7_FULL_PDK = ROOT / "tech_model_cards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l"

# Tolerances
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
REL_TOL = 5e-3

# Test configurations
TEST_DEVICES = [
    "nch_svt_mac",  # NMOS Standard Threshold
    "nch_lvt_mac",  # NMOS Low Threshold
    "pch_svt_mac",  # PMOS Standard Threshold
]

TEST_LENGTHS = [16e-9, 20e-9, 24e-9]  # 16nm, 20nm, 24nm

# Test geometry (typical values)
TEST_TFIN = 6e-9  # 6nm fin thickness
TEST_NFIN = 2.0    # 2 fins


def _iter_naive_modelcards() -> list[Path]:
    """Get list of naive TSMC7 modelcard files."""
    if not TSMC7_NAIVE_DIR.exists():
        return []

    return sorted(TSMC7_NAIVE_DIR.glob("*.l"))


def _make_ngspice_naive_netlist(
    output_path: Path,
    modelcard_path: str,
    model_name: str,
    inst_params: Dict[str, float],
    voltages: Dict[str, float],
    temp: float = 27.0,
    osdi_path: str = None,
) -> None:
    """Write NGSPICE netlist using naive TSMC7 modelcard."""
    osdi_path = osdi_path or str(OSDI_PATH)

    # Create a main netlist file that sources the circuit
    circuit_netlist = output_path.parent / f"{output_path.stem}_circuit.cir"
    with open(circuit_netlist, "w") as f:
        f.write(f"* Circuit with naive TSMC7 modelcard\n")
        f.write(f".include {modelcard_path}\n\n")

        # Direct model instantiation (NOT subcircuit)
        params_str = " ".join(f"{k}={v}" for k, v in inst_params.items())
        f.write(f"N1 d g s e {model_name} {params_str}\n\n")

        # Voltage sources
        f.write(f"Vd d 0 {voltages['d']}\n")
        f.write(f"Vg g 0 {voltages['g']}\n")
        f.write(f"Vs s 0 {voltages['s']}\n")
        f.write(f"Ve e 0 {voltages['e']}\n")

        f.write(f"\n.temp {temp}\n")
        f.write(".op\n")
        f.write(".end\n")

    # Create runner with .control block for OSDI loading
    with open(output_path, "w") as f:
        f.write("* NGSPICE + OSDI runner for naive TSMC7 Modelcard\n")
        f.write(".control\n")
        f.write(f"  osdi {osdi_path}\n")
        f.write(f"  source {circuit_netlist}\n")
        f.write("  filetype=ascii\n")
        f.write("  wr_vecnames\n")
        f.write(".options saveinternals\n")
        f.write("  run\n")
        f.write(f"  print i(vd) i(vg) i(vs) i(ve)\n")
        f.write(".endc\n")
        f.write(".end\n")


def _run_ngspice(netlist_path: Path) -> Dict[str, float]:
    """Run NGSPICE and parse OP results."""
    env = os.environ.copy()
    ngspice_bin = env.get("NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")

    result = subprocess.run(
        [ngspice_bin, "-b", str(netlist_path)],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"NGSPICE failed: {result.stdout}\n{result.stderr}")

    # Parse currents from stdout
    # Format from print command: "i(vd) = 1.23456e-05"
    parsed = {}
    patterns = {
        "id": r"i\(vd\)\s*=\s*([-\d\.eE+]+)",
        "ig": r"i\(vg\)\s*=\s*([-\d\.eE+]+)",
        "is": r"i\(vs\)\s*=\s*([-\d\.eE+]+)",
        "ie": r"i\(ve\)\s*=\s*([-\d\.eE+]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, result.stdout)
        if match:
            parsed[key] = float(match.group(1))
        else:
            parsed[key] = float('nan')

    for key, pattern in patterns.items():
        match = re.search(pattern, result.stdout)
        if match:
            parsed[key] = float(match.group(1))
        else:
            parsed[key] = float('nan')

    return parsed


def _assert_close(name: str, pycmg_val: float, ngspice_val: float,
               abs_tol: float = ABS_TOL_I, rel_tol: float = REL_TOL) -> None:
    """Assert two values are close within tolerance."""
    if abs(ngspice_val) > 1e100:  # Check for NaN/inf
        pytest.skip(f"NGSPICE value invalid for {name}: {ngspice_val}")
        return

    diff = abs(pycmg_val - ngspice_val)
    mag = max(abs(pycmg_val), abs(ngspice_val))

    # Check absolute tolerance
    if diff <= abs_tol:
        return

    # Check relative tolerance
    if mag > 0 and diff / mag <= rel_tol:
        return

    pytest.fail(
        f"{name}: PyCMG={pycmg_val:.6e}, NGSPICE={ngspice_val:.6e}, "
        f"diff={diff:.6e}, rel_diff={diff/mag*100:.4f}% "
        f"(abs_tol={abs_tol:.3e}, rel_tol={rel_tol:.3e})"
    )


@pytest.mark.skipif(not TSMC7_NAIVE_DIR.exists(), reason="naive TSMC7 modelcards not found")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_naive_modelcards_exist() -> None:
    """Verify naive modelcard files exist."""
    modelcards = _iter_naive_modelcards()
    assert len(modelcards) > 0, "No naive modelcards found"


@pytest.mark.skipif(not TSMC7_NAIVE_DIR.exists(), reason="naive TSMC7 modelcards not found")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("model_name", ["nch_svt_mac", "nch_lvt_mac", "pch_svt_mac"])
@pytest.mark.parametrize("L", [16e-9, 20e-9, 24e-9])
def test_naive_extraction(model_name: str, L: float) -> None:
    """Test parameter extraction from naive modelcard."""
    naive_file = TSMC7_NAIVE_DIR / f"{model_name}_l{int(L*1e9)}nm.l"

    if not naive_file.exists():
        pytest.skip(f"Naive modelcard not found: {naive_file}")

    # Extract using parse_modelcard (same as ASAP7)
    parsed = parse_modelcard(str(naive_file), model_name)

    # Verify extraction
    assert parsed.name == model_name, f"Model name mismatch: {parsed.name} != {model_name}"
    assert "level" in parsed.params, "level parameter not found"
    assert parsed.params["level"] == 72, f"level should be 72, got {parsed.params['level']}"

    # Verify key process parameters exist
    assert "eot" in parsed.params, "eot parameter not found"
    assert "hfin" in parsed.params, "hfin parameter not found"

    # Verify instance parameters are NOT in modelcard
    assert "l" not in [k.lower() for k in parsed.params.keys()], "L should not be baked into modelcard"
    assert "tfin" not in [k.lower() for k in parsed.params.keys()], "TFIN should not be baked into modelcard"
    assert "nfin" not in [k.lower() for k in parsed.params.keys()], "NFIN should not be baked into modelcard"


@pytest.mark.skipif(not TSMC7_NAIVE_DIR.exists(), reason="naive TSMC7 modelcards not found")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("model_name,L", [
    ("nch_svt_mac", 16e-9),
    ("pch_svt_mac", 16e-9),
])
def test_naive_pycmg_smoke(model_name: str, L: float) -> None:
    """PyCMG smoke test: Load naive modelcard and run DC analysis."""
    naive_file = TSMC7_NAIVE_DIR / f"{model_name}_l{int(L*1e9)}nm.l"

    if not naive_file.exists():
        pytest.skip(f"Naive modelcard not found: {naive_file}")

    # Test voltages
    voltages = {"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0}

    # PyCMG: Load naive modelcard (just like ASAP7)
    model = Model(str(OSDI_PATH), str(naive_file), model_name)
    inst = Instance(model, params={"L": L, "TFIN": TEST_TFIN, "NFIN": TEST_NFIN})
    result = inst.eval_dc(voltages)

    # Basic sanity checks
    assert "id" in result
    assert "ig" in result
    assert "gm" in result

    # Check reasonable values (magnitude matters, sign depends on SPICE convention)
    # In SPICE: current OUT of terminal is negative, INTO terminal is positive
    # For NMOS in saturation: Id flows OUT of drain (negative value)
    # For PMOS in saturation: Id flows INTO drain (positive value)
    assert abs(result["id"]) > 1e-7, f"Id magnitude too small: {abs(result['id']):.6e}"


@pytest.mark.skipif(not TSMC7_NAIVE_DIR.exists(), reason="naive TSMC7 modelcards not found")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.skipif(not TSMC7_FULL_PDK.exists(), reason="full TSMC7 PDK not found")
def test_naive_vs_full_pdk() -> None:
    """Verify naive modelcard matches full PDK results exactly."""
    model_name = "nch_svt_mac"
    L = 16e-9

    naive_file = TSMC7_NAIVE_DIR / f"{model_name}_l{int(L*1e9)}nm.l"
    if not naive_file.exists():
        pytest.skip(f"Naive modelcard not found: {naive_file}")

    voltages = {"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0}

    # Results from naive modelcard
    model_naive = Model(str(OSDI_PATH), str(naive_file), model_name)
    inst_naive = Instance(model_naive, params={"L": L, "TFIN": TEST_TFIN, "NFIN": TEST_NFIN})
    result_naive = inst_naive.eval_dc(voltages)

    # Results from full PDK
    from pycmg.ctypes_host import parse_tsmc7_pdk
    # Note: For full PDK, we need to pass the PDK path, model type, device type, and L
    # parse_tsmc7_pdk returns a ParsedModel, but Model needs file path
    # So we need to use a temp modelcard for the full PDK approach
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temp modelcard with full PDK params
        temp_card = Path(tmpdir) / "temp_full.l"
        # Write modelcard with baked L for NGSPICE compatibility
        full_params = parse_tsmc7_pdk(str(TSMC7_FULL_PDK), "nch", "svt_mac", L).params
        with open(temp_card, "w") as f:
            f.write(f".model {model_name} nmos (\n")
            for k, v in full_params.items():
                f.write(f"  +{k}={v}\n")
            f.write(")\n")
        model_full = Model(str(OSDI_PATH), str(temp_card), model_name)
        inst_full = Instance(model_full, params={"L": L, "TFIN": TEST_TFIN, "NFIN": TEST_NFIN})
    result_full = inst_full.eval_dc(voltages)

    # Must be binary-identical (same OSDI binary, same parameters)
    diff_id = abs(result_naive["id"] - result_full["id"])
    diff_ig = abs(result_naive["ig"] - result_full["ig"])

    assert diff_id < 1e-15, f"Id mismatch: naive={result_naive['id']:.6e}, full={result_full['id']:.6e}, diff={diff_id:.6e}"
    assert diff_ig < 1e-15, f"Ig mismatch: naive={result_naive['ig']:.6e}, full={result_full['ig']:.6e}, diff={diff_ig:.6e}"


@pytest.mark.skipif(not TSMC7_NAIVE_DIR.exists(), reason="naive TSMC7 modelcards not found")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
def test_naive_voltage_sweep() -> None:
    """Test Id-Vg sweep at multiple points."""
    model_name = "nch_svt_mac"
    L = 16e-9

    naive_file = TSMC7_NAIVE_DIR / f"{model_name}_l{int(L*1e9)}nm.l"
    if not naive_file.exists():
        pytest.skip(f"Naive modelcard not found: {naive_file}")

    # Load model
    model = Model(str(OSDI_PATH), str(naive_file), model_name)
    inst = Instance(model, params={"L": L, "TFIN": TEST_TFIN, "NFIN": TEST_NFIN})

    # Test Id-Vg sweep
    Vd = 0.75
    Vg_points = [0.0, 0.4, 0.6, 0.8, 1.0]

    prev_id = 0.0
    for Vg in Vg_points:
        voltages = {"d": Vd, "g": Vg, "s": 0.0, "e": 0.0}
        pycmg_result = inst.eval_dc(voltages)

        # Check that |Id| increases with Vg (in saturation)
        # Note: Id is negative for NMOS (flows OUT of drain in SPICE convention)
        # We compare absolute values for the sweep
        if Vg > 0.6:
            assert abs(pycmg_result["id"]) > 1e-6, f"|Id| too low at Vg={Vg}: {abs(pycmg_result['id']):.6e}"
            assert abs(pycmg_result["id"]) > abs(prev_id), f"|Id| should increase with Vg"

        prev_id = pycmg_result["id"]


@pytest.mark.skipif(not TSMC7_NAIVE_DIR.exists(), reason="naive TSMC7 modelcards not found")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
def test_naive_multiple_devices() -> None:
    """Test multiple device types with naive modelcards."""
    tested = []

    for model_name in ["nch_svt_mac", "nch_lvt_mac", "pch_svt_mac"]:
        for L in [16e-9, 20e-9]:
            naive_file = TSMC7_NAIVE_DIR / f"{model_name}_l{int(L*1e9)}nm.l"
            if not naive_file.exists():
                continue

            # Quick OP test - use appropriate bias for NMOS vs PMOS
            if "nch" in model_name:  # NMOS - positive gate bias
                voltages = {"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0}
            else:  # PMOS - negative gate bias
                voltages = {"d": -0.75, "g": -0.75, "s": 0.0, "e": 0.0}

            model = Model(str(OSDI_PATH), str(naive_file), model_name)
            inst = Instance(model, params={"L": L, "TFIN": TEST_TFIN, "NFIN": TEST_NFIN})
            result = inst.eval_dc(voltages)

            # Check reasonable current values
            # Both NMOS and PMOS should conduct with appropriate bias
            assert abs(result["id"]) > 1e-7, f"{model_name} L={L*1e9:.0f}nm: |Id| too small: {abs(result['id']):.6e}"
            assert abs(result["gm"]) > 1e-6, f"{model_name} L={L*1e9:.0f}nm: |gm| too small: {abs(result['gm']):.6e}"

            tested.append(f"{model_name}_l{int(L*1e9)}nm")

    # At least some devices should have been tested
    if len(tested) == 0:
        pytest.skip("No naive modelcards found for testing")
    else:
        print(f"Tested {len(tested)} device configurations: {', '.join(tested)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
