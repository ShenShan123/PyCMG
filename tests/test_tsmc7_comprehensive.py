"""
TSMC7 PDK Comprehensive Verification Tests

Comprehensive verification of TSMC7 PDK modelcards across:
- Multiple device types: nch_svt_mac, nch_lvt_mac, nch_ulvt_mac, pch_svt_mac
- Multiple length values: 120nm, 72nm, 36nm, 20nm, 16nm, 12nm
- Multiple bias points (DC operating points)
- NGSPICE ground truth comparison

VERIFICATION STRATEGY:
- PyCMG wraps the OSDI binary directly via ctypes (pycmg/ctypes_host.py)
- NGSPICE loads the SAME OSDI binary via .osdi command
- Tests compare PyCMG output vs NGSPICE output to ensure binary-level consistency
- Both use the exact same bsimcmg.osdi file, ensuring identical model physics

Run: pytest tests/test_tsmc7_comprehensive.py -v
Duration: ~10 minutes
Requires: NGSPICE, TSMC7 PDK modelcards
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import pytest

import pycmg
from pycmg.ctypes_host import Model, Instance, parse_tsmc7_pdk, parse_number_with_suffix

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
BUILD = ROOT / "build-deep-verify"
TSMC7_PDK_PATH = ROOT / "tech_model_cards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l"

# Tolerances (matching test_asap7.py)
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
REL_TOL = 5e-3

# Device types to test
DEVICE_TYPES = {
    "nch_svt_mac": {"type": "nch", "device": "svt_mac"},
    "nch_lvt_mac": {"type": "nch", "device": "lvt_mac"},
    "nch_ulvt_mac": {"type": "nch", "device": "ulvt_mac"},
    "pch_svt_mac": {"type": "pch", "device": "svt_mac"},
}

# Length values to test (in meters)
LENGTH_VALUES = [
    120e-9,  # 120nm (variant 2)
    72e-9,   # 72nm (variant 3)
    36e-9,   # 36nm (variant 4)
    20e-9,   # 20nm (variant 5)
    16e-9,   # 16nm (variant 6)
    12e-9,   # 12nm (variant 7)
]

# Representative bias points (not exhaustive)
BIAS_POINTS = [
    {"vd": 0.0, "vg": 0.0, "vs": 0.0, "ve": 0.0, "desc": "cutoff"},
    {"vd": 0.1, "vg": 0.0, "vs": 0.0, "ve": 0.0, "desc": "cutoff_small_vd"},
    {"vd": 0.75, "vg": 0.0, "vs": 0.0, "ve": 0.0, "desc": "cutoff_high_vd"},
    {"vd": 0.05, "vg": 0.3, "vs": 0.0, "ve": 0.0, "desc": "linear_low"},
    {"vd": 0.1, "vg": 0.6, "vs": 0.0, "ve": 0.0, "desc": "linear_mid"},
    {"vd": 0.05, "vg": 0.75, "vs": 0.0, "ve": 0.0, "desc": "linear_high_vg"},
    {"vd": 0.75, "vg": 0.75, "vs": 0.0, "ve": 0.0, "desc": "saturation"},
    {"vd": 0.1, "vg": 0.9, "vs": 0.0, "ve": 0.0, "desc": "overdrive"},
]

# Temperature test points
TEST_TEMPS = [27.0, 85.0]  # °C (representative, not full range)

# Default instance parameters
DEFAULT_TFIN = 6e-9  # 6nm fin thickness
DEFAULT_NFIN = 2.0   # 2 fins


def _make_ngspice_tsmc7_netlist(
    pdk_path: Path,
    model_name: str,
    model_type: str,
    device_type: str,
    L: float,
    TFIN: float,
    NFIN: float,
    vd: float,
    vg: float,
    vs: float,
    ve: float,
    temp_c: float,
) -> tuple[str, str]:
    """
    Create NGSPICE netlist for TSMC7 PDK verification.

    Returns:
        (netlist_content, output_csv_path)
    """
    # Determine expected model type for NGSPICE
    expected_type = "nmos" if model_type == "nch" else "pmos"

    # Extract variant number for L value
    base_name = f"{model_type}_{device_type}"

    netlist = f"* TSMC7 PDK Verification\n"
    netlist += f"* Device: {base_name}, L={L*1e9:.1f}nm\n"
    netlist += f'*include "{pdk_path}"\n'
    netlist += f".temp {temp_c}\n"
    netlist += f"Vd d 0 {vd}\n"
    netlist += f"Vg g 0 {vg}\n"
    netlist += f"Vs s 0 {vs}\n"
    netlist += f"Ve e 0 {ve}\n"

    # For NGSPICE, we use subcircuit call (TSMC7 style)
    # Subcircuit name: nch_svt_mac (no suffix)
    netlist += f"X1 d g s e {base_name} l={L} tfin={TFIN} nfin={NFIN}\n"
    netlist += ".op\n"
    netlist += ".end\n"

    out_csv = BUILD / f"tsmc7_{base_name}_L{L*1e9:.0fnm}_vd{vd:.2f}_vg{vg:.2f}.csv"

    return netlist, str(out_csv)


def _run_ngspice_with_tsmc7_pdk(
    pdk_path: Path,
    model_name: str,
    model_type: str,
    device_type: str,
    L: float,
    TFIN: float,
    NFIN: float,
    vd: float,
    vg: float,
    vs: float = 0.0,
    ve: float = 0.0,
    temp_c: float = 27.0,
) -> Dict[str, float]:
    """
    Run NGSPICE operating point analysis with full TSMC7 PDK.

    Args:
        pdk_path: Path to TSMC7 PDK file
        model_name: Model name (e.g., "nch_svt_mac")
        model_type: "nch" or "pch"
        device_type: "svt_mac", "lvt_mac", "ulvt_mac"
        L: Gate length (meters)
        TFIN: Fin thickness (meters)
        NFIN: Number of fins
        vd, vg, vs, ve: Terminal voltages (V)
        temp_c: Temperature (°C)

    Returns:
        Dictionary with NGSPICE results (id, ig, is, ie, qg, qd, qs, qb, gm, gds, gmb)
    """
    import subprocess

    netlist_content, out_csv = _make_ngspice_tsmc7_netlist(
        pdk_path=pdk_path,
        model_name=model_name,
        model_type=model_type,
        device_type=device_type,
        L=L,
        TFIN=TFIN,
        NFIN=NFIN,
        vd=vd,
        vg=vg,
        vs=vs,
        ve=ve,
        temp_c=temp_c,
    )

    # Write netlist
    net_path = BUILD / "tsmc7_ng_temp.cir"
    net_path.parent.mkdir(parents=True, exist_ok=True)
    net_path.write_text(netlist_content)

    # Create NGSPICE runner script
    runner_path = BUILD / "tsmc7_ng_runner.cir"
    runner_path.write_text(f"""
* NGSPICE runner for TSMC7 verification
.control
osdi {OSDI_PATH}
source {net_path}
set filetype=ascii
set wr_vecnames
.options saveinternals
run
wrdata {out_csv} v(g) v(d) v(s) v(e) i(vg) i(vd) i(vs) i(ve) @x1[id] @x1[ig] @x1[is] @x1[ib] @x1[qg] @x1[qd] @x1[qs] @x1[qb] @x1[gm] @x1[gds] @x1[gmbs]
.endc
.end
""")

    # Run NGSPICE
    log_path = BUILD / "tsmc7_ng.log"
    ngspice_bin = os.environ.get("NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")
    res = subprocess.run(
        [ngspice_bin, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True,
        text=True
    )

    if res.returncode != 0:
        raise RuntimeError(f"NGSPICE failed: {res.stdout}\n{res.stderr}")

    # Parse output CSV
    out_path = Path(out_csv)
    if not out_path.exists():
        raise RuntimeError(f"NGSPICE output not found: {out_csv}")

    with out_path.open() as f:
        lines = f.readlines()
        if not lines:
            raise RuntimeError("Empty NGSPICE output")

        headers = lines[0].split()
        values = [float(x) for x in lines[1].split()]

        idx_map = {name: i for i, name in enumerate(headers)}

        return {
            "id": values[idx_map["i(vd)"]],
            "ig": values[idx_map["i(vg)"]],
            "is": values[idx_map["i(vs)"]],
            "ib": values[idx_map["i(ve)"]],
            "qg": values[idx_map["@x1[qg]"]],
            "qd": values[idx_map["@x1[qd]"]],
            "qs": values[idx_map["@x1[qs]"]],
            "qb": values[idx_map["@x1[qb]"]],
            "gm": values[idx_map["@x1[gm]"]],
            "gds": values[idx_map["@x1[gds]"]],
            "gmb": values[idx_map["@x1[gmbs]"]],
        }


def _compare_results(
    label: str,
    py_result: Dict[str, float],
    ng_result: Dict[str, float],
    check_capacitances: bool = False,
) -> None:
    """
    Compare PyCMG results with NGSPICE ground truth.

    Args:
        label: Test label for error messages
        py_result: PyCMG output dictionary
        ng_result: NGSPICE output dictionary
        check_capacitances: Whether to check charge/capacitance values

    Raises:
        pytest.fail: If values differ beyond tolerances
    """
    # Compare currents
    for key in ["id", "ig", "is", "ib"]:
        py_val = py_result[key]
        ng_val = ng_result[key]
        diff = abs(py_val - ng_val)

        if diff <= ABS_TOL_I:
            continue

        denom = max(abs(ng_val), ABS_TOL_I)
        if diff / denom <= REL_TOL:
            continue

        pytest.fail(
            f"{label}@{key}: py={py_val:.6e} ng={ng_val:.6e} "
            f"diff={diff:.6e} (abs_tol={ABS_TOL_I:.3e}, rel_tol={REL_TOL:.3e})"
        )

    # Compare drain-source current (Ids = Id - Is)
    py_ids = py_result["id"] - py_result["is"]
    ng_ids = ng_result["id"] - ng_result["is"]
    diff_ids = abs(py_ids - ng_ids)

    if diff_ids > ABS_TOL_I:
        denom = max(abs(ng_ids), ABS_TOL_I)
        if diff_ids / denom > REL_TOL:
            pytest.fail(
                f"{label}@ids: py={py_ids:.6e} ng={ng_ids:.6e} "
                f"diff={diff_ids:.6e}"
            )

    # Compare charges if requested
    if check_capacitances:
        for key in ["qg", "qd", "qs", "qb"]:
            if key not in py_result or key not in ng_result:
                continue
            py_val = py_result[key]
            ng_val = ng_result[key]
            diff = abs(py_val - ng_val)

            if diff <= ABS_TOL_Q:
                continue

            denom = max(abs(ng_val), ABS_TOL_Q)
            if diff / denom <= REL_TOL:
                continue

            pytest.fail(
                f"{label}@{key}: py={py_val:.6e} ng={ng_val:.6e} "
                f"diff={diff:.6e} (abs_tol={ABS_TOL_Q:.3e}, rel_tol={REL_TOL:.3e})"
            )

    # Compare derivatives (gm, gds, gmb)
    for key in ["gm", "gds", "gmb"]:
        if key not in py_result or key not in ng_result:
            continue
        py_val = py_result[key]
        ng_val = ng_result[key]
        diff = abs(py_val - ng_val)

        if diff <= ABS_TOL_I:
            continue

        denom = max(abs(ng_val), ABS_TOL_I)
        if diff / denom <= REL_TOL:
            continue

        pytest.fail(
            f"{label}@{key}: py={py_val:.6e} ng={ng_val:.6e} "
            f"diff={diff:.6e}"
        )


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def tsmc7_pdk_path() -> Path:
    """Path to TSMC7 PDK file."""
    if not TSMC7_PDK_PATH.exists():
        pytest.skip(f"TSMC7 PDK not found: {TSMC7_PDK_PATH}")
    return TSMC7_PDK_PATH


@pytest.fixture(scope="session")
def osdi_path() -> Path:
    """Path to OSDI binary."""
    if not OSDI_PATH.exists():
        pytest.skip(f"OSDI binary not found: {OSDI_PATH}")
    return OSDI_PATH


@pytest.fixture(scope="session")
def build_dir() -> Path:
    """Build directory for temporary files."""
    BUILD.mkdir(parents=True, exist_ok=True)
    return BUILD


# =============================================================================
# Parameter Extraction Tests
# =============================================================================

@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.parametrize("device_key", ["nch_svt_mac", "nch_lvt_mac", "nch_ulvt_mac", "pch_svt_mac"])
@pytest.mark.parametrize("L", [120e-9, 72e-9, 36e-9, 20e-9, 16e-9, 12e-9])
def test_tsmc7_parameter_extraction(device_key: str, L: float) -> None:
    """Test parameter extraction for all device types and length values."""
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]

    # Extract parameters using parse_tsmc7_pdk
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)

    # Verify model name
    expected_name = f"{model_type}_{device_type}"
    assert parsed.name == expected_name

    # Verify critical parameters exist
    assert "level" in parsed.params
    assert parsed.params["level"] == 72
    assert "version" in parsed.params

    # Verify geometry parameters
    assert "eot" in parsed.params
    assert "tfin" in parsed.params
    assert "hfin" in parsed.params
    assert "xl" in parsed.params

    # Verify device-specific parameters
    if model_type == "nch":
        # NMOS-specific
        assert "nsd" in parsed.params
        assert "nbody" in parsed.params
    else:
        # PMOS-specific
        assert "nsd" in parsed.params
        assert "nbody" in parsed.params


@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.parametrize("device_key", ["nch_svt_mac", "nch_lvt_mac", "nch_ulvt_mac"])
@pytest.mark.parametrize("L", [120e-9, 72e-9, 36e-9, 20e-9, 16e-9, 12e-9])
def test_nmos_dc_evaluation(device_key: str, L: float) -> None:
    """Test DC evaluation for NMOS devices at saturation."""
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]

    # Extract model parameters
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)

    # Create model and instance
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN)

    # Run DC analysis at saturation
    result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

    # Basic sanity checks
    assert result["id"] > 0, f"{device_key} L={L*1e9:.0f}nm: drain current should be positive"
    assert abs(result["ig"]) < 1e-6, f"{device_key} L={L*1e9:.0f}nm: gate current should be small"
    assert abs(result["ie"]) < 1e-6, f"{device_key} L={L*1e9:.0f}nm: bulk current should be small"

    # Verify derivatives are reasonable
    assert result["gm"] > 0, f"{device_key} L={L*1e9:.0f}nm: transconductance should be positive"
    assert result["gds"] > 0, f"{device_key} L={L*1e9:.0f}nm: output conductance should be positive"


@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.parametrize("L", [120e-9, 72e-9, 36e-9, 20e-9, 16e-9, 12e-9])
def test_pmos_dc_evaluation(L: float) -> None:
    """Test DC evaluation for PMOS SVT device at saturation."""
    device_key = "pch_svt_mac"
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]

    # Extract model parameters
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)

    # Create model and instance
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN)

    # Run DC analysis at saturation (PMOS: negative Vgs, Vds)
    result = inst.eval_dc({"d": -0.75, "g": -0.75, "s": 0.0, "e": 0.0})

    # Basic sanity checks for PMOS (current flows into source)
    assert result["is"] < 0, f"{device_key} L={L*1e9:.0f}nm: source current should be negative for PMOS"
    assert abs(result["ig"]) < 1e-6, f"{device_key} L={L*1e9:.0f}nm: gate current should be small"
    assert abs(result["ie"]) < 1e-6, f"{device_key} L={L*1e9:.0f}nm: bulk current should be small"


# =============================================================================
# NGSPICE Verification Tests
# =============================================================================

@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
@pytest.mark.parametrize("device_key", ["nch_svt_mac", "nch_lvt_mac", "nch_ulvt_mac"])
@pytest.mark.parametrize("L", [120e-9, 72e-9, 36e-9, 20e-9, 16e-9, 12e-9])
def test_nmos_vs_ngspice_saturation(device_key: str, L: float) -> None:
    """Test NMOS devices vs NGSPICE at saturation bias point."""
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]
    model_name = f"{model_type}_{device_type}"

    # Saturation bias
    vd = 0.75
    vg = 0.75

    # Run NGSPICE
    ng_result = _run_ngspice_with_tsmc7_pdk(
        pdk_path=TSMC7_PDK_PATH,
        model_name=model_name,
        model_type=model_type,
        device_type=device_type,
        L=L,
        TFIN=DEFAULT_TFIN,
        NFIN=DEFAULT_NFIN,
        vd=vd,
        vg=vg,
        vs=0.0,
        ve=0.0,
        temp_c=27.0,
    )

    # Run PyCMG
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN)
    py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})

    # Compare results
    label = f"{device_key}_L{L*1e9:.0f}nm"
    _compare_results(label, py_result, ng_result, check_capacitances=True)


@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
@pytest.mark.parametrize("device_key", ["nch_svt_mac", "nch_lvt_mac", "nch_ulvt_mac"])
@pytest.mark.parametrize("L", [120e-9, 72e-9, 36e-9, 20e-9, 16e-9, 12e-9])
def test_nmos_vs_ngspice_linear(device_key: str, L: float) -> None:
    """Test NMOS devices vs NGSPICE at linear bias point."""
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]
    model_name = f"{model_type}_{device_type}"

    # Linear bias
    vd = 0.05
    vg = 0.75

    # Run NGSPICE
    ng_result = _run_ngspice_with_tsmc7_pdk(
        pdk_path=TSMC7_PDK_PATH,
        model_name=model_name,
        model_type=model_type,
        device_type=device_type,
        L=L,
        TFIN=DEFAULT_TFIN,
        NFIN=DEFAULT_NFIN,
        vd=vd,
        vg=vg,
        vs=0.0,
        ve=0.0,
        temp_c=27.0,
    )

    # Run PyCMG
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN)
    py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})

    # Compare results
    label = f"{device_key}_L{L*1e9:.0f}nm_linear"
    _compare_results(label, py_result, ng_result, check_capacitances=False)


@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
@pytest.mark.parametrize("L", [120e-9, 72e-9, 36e-9, 20e-9, 16e-9, 12e-9])
def test_pmos_vs_ngspice(L: float) -> None:
    """Test PMOS SVT device vs NGSPICE."""
    device_key = "pch_svt_mac"
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]
    model_name = f"{model_type}_{device_type}"

    # Saturation bias (negative for PMOS)
    vd = -0.75
    vg = -0.75

    # Run NGSPICE
    ng_result = _run_ngspice_with_tsmc7_pdk(
        pdk_path=TSMC7_PDK_PATH,
        model_name=model_name,
        model_type=model_type,
        device_type=device_type,
        L=L,
        TFIN=DEFAULT_TFIN,
        NFIN=DEFAULT_NFIN,
        vd=vd,
        vg=vg,
        vs=0.0,
        ve=0.0,
        temp_c=27.0,
    )

    # Run PyCMG
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN)
    py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})

    # Compare results
    label = f"{device_key}_L{L*1e9:.0f}nm"
    _compare_results(label, py_result, ng_result, check_capacitances=True)


@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
@pytest.mark.parametrize("device_key", ["nch_svt_mac", "nch_lvt_mac"])
@pytest.mark.parametrize("bias", BIAS_POINTS[:4])  # Test subset of bias points
def test_nmos_bias_sweep(device_key: str, bias: Dict[str, Any]) -> None:
    """Test NMOS devices across multiple bias points."""
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]
    model_name = f"{model_type}_{device_type}"

    # Use mid-range length for bias sweep
    L = 36e-9

    # Run NGSPICE
    ng_result = _run_ngspice_with_tsmc7_pdk(
        pdk_path=TSMC7_PDK_PATH,
        model_name=model_name,
        model_type=model_type,
        device_type=device_type,
        L=L,
        TFIN=DEFAULT_TFIN,
        NFIN=DEFAULT_NFIN,
        vd=bias["vd"],
        vg=bias["vg"],
        vs=bias["vs"],
        ve=bias["ve"],
        temp_c=27.0,
    )

    # Run PyCMG
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN)
    py_result = inst.eval_dc({
        "d": bias["vd"],
        "g": bias["vg"],
        "s": bias["vs"],
        "e": bias["ve"]
    })

    # Compare results
    label = f"{device_key}_L{L*1e9:.0f}nm_{bias['desc']}"
    _compare_results(label, py_result, ng_result, check_capacitances=False)


@pytest.mark.skipif(not TSMC7_PDK_PATH.exists(), reason="missing TSMC7 PDK")
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
@pytest.mark.parametrize("temp_c", TEST_TEMPS)
def test_nmos_temperature_sweep(temp_c: float) -> None:
    """Test NMOS SVT at different temperatures."""
    device_key = "nch_svt_mac"
    device_config = DEVICE_TYPES[device_key]
    model_type = device_config["type"]
    device_type = device_config["device"]
    model_name = f"{model_type}_{device_type}"

    # Use mid-range length
    L = 36e-9

    # Run NGSPICE
    ng_result = _run_ngspice_with_tsmc7_pdk(
        pdk_path=TSMC7_PDK_PATH,
        model_name=model_name,
        model_type=model_type,
        device_type=device_type,
        L=L,
        TFIN=DEFAULT_TFIN,
        NFIN=DEFAULT_NFIN,
        vd=0.75,
        vg=0.75,
        vs=0.0,
        ve=0.0,
        temp_c=temp_c,
    )

    # Run PyCMG
    parsed = parse_tsmc7_pdk(str(TSMC7_PDK_PATH), model_type, device_type, L)
    model = Model(str(OSDI_PATH), str(TSMC7_PDK_PATH), parsed.name, parsed.params)
    inst = Instance(model, L=L, TFIN=DEFAULT_TFIN, NFIN=DEFAULT_NFIN, temperature=temp_c + 273.15)
    py_result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

    # Compare results
    label = f"{device_key}_L{L*1e9:.0f}nm_T{temp_c:.0f}C"
    _compare_results(label, py_result, ng_result, check_capacitances=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
