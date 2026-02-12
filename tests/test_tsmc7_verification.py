#!/usr/bin/env python3
"""
TSMC7 16nm Full Verification Test - PyCMG vs NGSPICE

Compares PyCMG and NGSPICE outputs for TSMC7 16nm modelcard
across currents, charges, derivatives, and capacitances.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pycmg
from pycmg.ctypes_host import Model, Instance

# Import testing utilities
sys.path.insert(0, str(ROOT / "tests"))
import verify_utils as vu

# Paths
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
BUILD = ROOT / "build-deep-verify"
TSMC7_DIR = ROOT / "tech_model_cards" / "TSMC7"
TSMC7_BASE_MODELCARD = "tsmc7_simple.l"
NGSPICE_BIN = vu.NGSPICE_BIN

# Tolerances (same as ASAP7)
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
REL_TOL = 5e-3

# Test parameters for 16nm device (nominal TSMC7)
# Instance parameters for PyCMG (uppercase names per BSIM-CMG spec)
INST_PARAMS_16NM_PYCMG = {
    "L": 16e-9,
    "TFIN": 8e-9,
    "NFIN": 2.0,
}

# Parameters to bake into NGSPICE modelcard (lowercase 'l' for model parameter)
INST_PARAMS_16NM_NGSPICE = {
    "l": 16e-9,
    "TFIN": 8e-9,
    "NFIN": 2.0,
}

# Test voltages (NMOS in saturation)
VD = 0.75  # V
VG = 0.75  # V
VS = 0.0   # V
VE = 0.0   # V
TEMP_C = 27.0  # °C


def _make_tsmc7_ngspice_modelcard(src: Path, dst: Path, model_name: str,
                                  inst_params: dict) -> None:
    """Create NGSPICE-compatible modelcard with instance parameters baked in."""
    text = src.read_text()
    lines = []
    in_target = False
    found_keys = set()
    target_lower = model_name.lower()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        original_stripped = line.strip()
        is_continuation = original_stripped.startswith("+")
        stripped = original_stripped.lstrip("+").strip()

        if stripped.lower().startswith(".model"):
            if in_target:
                # Add missing parameters at end of previous model
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()

            # Check if this is our target model
            parts = stripped.split()
            if len(parts) >= 2:
                model_name_in_file = parts[1]
                if model_name_in_file.lower() == target_lower:
                    in_target = True
                    lines.append(line)
                    # Check for params on the initial .model line
                    for key, val in inst_params.items():
                        pattern = rf"{key}\s*=\s*[\d.eE+\-]+"
                        if re.search(pattern, stripped, re.IGNORECASE):
                            found_keys.add(key.upper())
                    continue

        if in_target and is_continuation:
            # Check if this line contains any of our instance params
            modified = False
            modified_line = line
            for key, val in inst_params.items():
                # Use word boundary to avoid matching substrings (e.g., "L" in "level")
                pattern = rf"\b({key})\s*=\s*[\d.eE+\-]+"
                if re.search(pattern, stripped, re.IGNORECASE):
                    # Replace the value
                    modified_line = re.sub(
                        pattern,
                        rf"\g<1> = {val}",
                        stripped,
                        flags=re.IGNORECASE
                    )
                    found_keys.add(key.upper())
                    modified = True
                    break  # Only replace one param per line

            if modified:
                lines.append(f"+ {modified_line}")
            else:
                lines.append(line)

        elif in_target and not is_continuation:
            # Next model or non-model line
            lines.append(line)
        else:
            # Not in target model
            lines.append(line)

    # Add missing params after last model
    if in_target:
        for key, val in inst_params.items():
            if key.upper() not in found_keys:
                lines.append(f"+ {key.upper()} = {val}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines))


def _run_ngspice_op_point(modelcard: Path, model_name: str,
                          vd: float, vg: float, vs: float, ve: float,
                          out_dir: Path, temp_c: float) -> dict:
    """Run NGSPICE operating point analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net = [
        "* OP point query",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vd}",
        f"Vg g 0 {vg}",
        f"Vs s 0 {vs}",
        f"Ve e 0 {ve}",
        f"N1 d g s e {model_name}",
        ".op",
        ".end",
    ]
    out_csv = out_dir / "ng_op_point.csv"
    log_path = out_dir / "ng_op_point.log"

    # Create netlist file
    net_path = out_dir / "netlist.cir"
    net_path.write_text("\n".join(net))

    # Create runner to load OSDI
    runner_path = out_dir / "runner.cir"
    runner_path.write_text(
        "* ngspice runner\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {net_path}\n"
        "set filetype=ascii\n"
        "set wr_vecnames\n"
        "run\n"
        f"wrdata {out_csv} v(g) v(d) v(s) v(e) "
        "i(vg) i(vd) i(vs) i(ve) "
        f"@n1[qg] @n1[qd] @n1[qs] @n1[qb] "
        f"@n1[gm] @n1[gds] @n1[gmbs]\n"
        ".endc\n"
        ".end\n"
    )

    subprocess.run([NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
                   check=True, capture_output=True)

    # Parse results
    headers, rows = vu.parse_wrdata(out_csv)
    if not rows:
        raise RuntimeError("empty NGSPICE op wrdata")

    row = rows[0]
    idx = {
        name: vu.col_index(headers, name)
        for name in (
            "i(vg)", "i(vd)", "i(vs)", "i(ve)",
            "@n1[qg]", "@n1[qd]", "@n1[qs]", "@n1[qb]",
            "@n1[gm]", "@n1[gds]", "@n1[gmbs]",
        )
    }

    return {
        "id": row[idx["i(vd)"]],
        "ig": row[idx["i(vg)"]],
        "is": row[idx["i(vs)"]],
        "ie": row[idx["i(ve)"]],
        "qg": row[idx["@n1[qg]"]],
        "qd": row[idx["@n1[qd]"]],
        "qs": row[idx["@n1[qs]"]],
        "qb": row[idx["@n1[qb]"]],
        "gm": row[idx["@n1[gm]"]],
        "gds": row[idx["@n1[gds]"]],
        "gmb": row[idx["@n1[gmbs]"]],
    }


def _assert_close(py_val: float, ng_val: float, label: str, abs_tol: float) -> None:
    """Assert PyCMG and NGSPICE values are within tolerance."""
    diff = abs(py_val - ng_val)
    if diff <= abs_tol:
        return
    denom = max(abs(ng_val), abs_tol)
    if diff / denom <= REL_TOL:
        return
    pytest.fail(f"{label}: py={py_val:.3e} ng={ng_val:.3e} "
                f"diff={diff:.3e} (abs_tol={abs_tol:.3e}, rel_tol={REL_TOL:.3e})")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.skipif(not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists(),
                    reason="missing TSMC7 modelcard")
def test_tsmc7_16nm_full_comparison() -> None:
    """Verify TSMC7 16nm model with PyCMG vs NGSPICE (full comparison)."""
    src_modelcard = TSMC7_DIR / TSMC7_BASE_MODELCARD

    # Create NGSPICE-compatible modelcard with instance params baked in
    out_dir = BUILD / "ngspice_eval" / "tsmc7_16nm"
    ng_modelcard = out_dir / "tsmc7_simple_ngspice.l"
    _make_tsmc7_ngspice_modelcard(src_modelcard, ng_modelcard, "nch_svt_mac",
                                   INST_PARAMS_16NM_NGSPICE)

    print("\n=== TSMC7 16nm Full Verification ===")
    print(f"Modelcard: {src_modelcard}")
    print(f"NGSPICE Modelcard: {ng_modelcard}")
    print(f"Instance params: L={INST_PARAMS_16NM_PYCMG['L']*1e9:.0f}nm, "
          f"TFIN={INST_PARAMS_16NM_PYCMG['TFIN']*1e9:.0f}nm, NFIN={INST_PARAMS_16NM_PYCMG['NFIN']}")
    print(f"Voltages: Vd={VD}V, Vg={VG}V, Vs={VS}V, Ve={VE}V")
    print(f"Temperature: {TEMP_C}°C")

    # === PyCMG Evaluation ===
    print("\n--- Running PyCMG ---")
    model = Model(str(OSDI_PATH), str(src_modelcard), "nch_svt_mac")
    inst = Instance(model, params=INST_PARAMS_16NM_PYCMG, temperature=TEMP_C + 273.15)
    py_result = inst.eval_dc({"d": VD, "g": VG, "s": VS, "e": VE})

    print(f"PyCMG Results:")
    print(f"  id  = {py_result['id']:.6e} A")
    print(f"  ig  = {py_result['ig']:.6e} A")
    print(f"  is  = {py_result['is']:.6e} A")
    print(f"  ie  = {py_result['ie']:.6e} A")
    print(f"  ids = {py_result['ids']:.6e} A")
    print(f"  qg  = {py_result['qg']:.6e} C")
    print(f"  qd  = {py_result['qd']:.6e} C")
    print(f"  qs  = {py_result['qs']:.6e} C")
    print(f"  qb  = {py_result['qb']:.6e} C")
    print(f"  gm  = {py_result['gm']:.6e} S")
    print(f"  gds = {py_result['gds']:.6e} S")
    print(f"  gmb = {py_result['gmb']:.6e} S")

    # === NGSPICE Evaluation ===
    print("\n--- Running NGSPICE ---")
    ng_result = _run_ngspice_op_point(ng_modelcard, "nch_svt_mac",
                                      VD, VG, VS, VE, out_dir, TEMP_C)

    print(f"NGSPICE Results:")
    print(f"  id  = {ng_result['id']:.6e} A")
    print(f"  ig  = {ng_result['ig']:.6e} A")
    print(f"  is  = {ng_result['is']:.6e} A")
    print(f"  ie  = {ng_result['ie']:.6e} A")
    print(f"  qg  = {ng_result['qg']:.6e} C")
    print(f"  qd  = {ng_result['qd']:.6e} C")
    print(f"  qs  = {ng_result['qs']:.6e} C")
    print(f"  qb  = {ng_result['qb']:.6e} C")
    print(f"  gm  = {ng_result['gm']:.6e} S")
    print(f"  gds = {ng_result['gds']:.6e} S")
    print(f"  gmb = {ng_result['gmb']:.6e} S")

    # === Comparison ===
    print("\n--- Comparison ---")
    _assert_close(py_result['id'], ng_result['id'], "id", ABS_TOL_I)
    print("  id  ✓")

    _assert_close(py_result['ig'], ng_result['ig'], "ig", ABS_TOL_I)
    print("  ig  ✓")

    _assert_close(py_result['is'], ng_result['is'], "is", ABS_TOL_I)
    print("  is  ✓")

    _assert_close(py_result['ie'], ng_result['ie'], "ie", ABS_TOL_I)
    print("  ie  ✓")

    # Note: NGSPICE doesn't directly output ids, but ids = id - is
    ng_ids = ng_result['id'] - ng_result['is']
    _assert_close(py_result['ids'], ng_ids, "ids", ABS_TOL_I)
    print("  ids ✓")

    _assert_close(py_result['qg'], ng_result['qg'], "qg", ABS_TOL_Q)
    print("  qg  ✓")

    _assert_close(py_result['qd'], ng_result['qd'], "qd", ABS_TOL_Q)
    print("  qd  ✓")

    _assert_close(py_result['qs'], ng_result['qs'], "qs", ABS_TOL_Q)
    print("  qs  ✓")

    _assert_close(py_result['qb'], ng_result['qb'], "qb", ABS_TOL_Q)
    print("  qb  ✓")

    _assert_close(py_result['gm'], ng_result['gm'], "gm", ABS_TOL_I)
    print("  gm  ✓")

    _assert_close(py_result['gds'], ng_result['gds'], "gds", ABS_TOL_I)
    print("  gds ✓")

    _assert_close(py_result['gmb'], ng_result['gmb'], "gmb", ABS_TOL_I)
    print("  gmb ✓")

    print("\n=== All checks passed! ===")


# === Parameter Sweep Tests ===

# Length sweep points (L parameter)
L_SWEEP = [12e-9, 16e-9, 20e-9, 24e-9]

# Fin thickness sweep points (TFIN parameter)
TFIN_SWEEP = [6e-9, 7e-9, 8e-9]

# Fin count sweep points (NFIN parameter)
NFIN_SWEEP = [1.0, 2.0, 4.0]


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.skipif(not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists(),
                    reason="missing TSMC7 modelcard")
@pytest.mark.parametrize("L_nm", L_SWEEP)
def test_tsmc7_length_sweep(L_nm) -> None:
    """Test TSMC7 with various gate lengths.

    Parametrized test sweeping L from 12nm to 24nm.
    Verifies PyCMG and NGSPICE produce consistent results.
    """
    if not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists():
        pytest.skip(f"TSMC7 modelcard not found")

    # Instance parameters with this L value
    inst_params = {
        "L": L_nm,
        "TFIN": 8e-9,
        "NFIN": 2.0,
    }

    # Modelcard parameters for NGSPICE
    ngspice_params = {
        "l": L_nm,
        "TFIN": 8e-9,
        "NFIN": 2.0,
    }

    src_modelcard = TSMC7_DIR / TSMC7_BASE_MODELCARD
    out_dir = BUILD / "ngspice_eval" / f"tsmc7_L{L_nm*1e9:.0f}nm"
    ng_modelcard = out_dir / "tsmc7_simple_ngspice.l"

    _make_tsmc7_ngspice_modelcard(src_modelcard, ng_modelcard, "nch_svt_mac",
                                   ngspice_params)

    print(f"\n=== TSMC7 {L_nm*1e9:.0f}nm Test ===")

    # === PyCMG Evaluation ===
    print("\n--- PyCMG ---")
    model = Model(str(OSDI_PATH), str(src_modelcard), "nch_svt_mac")
    inst = Instance(model, params=inst_params, temperature=TEMP_C + 273.15)
    py_result = inst.eval_dc({"d": VD, "g": VG, "s": VS, "e": VE})

    print(f"PyCMG id={py_result['id']:.3e}A ig={py_result['ig']:.3e}A is={py_result['is']:.3e}A")

    # === NGSPICE Evaluation ===
    print("\n--- NGSPICE ---")
    ng_result = _run_ngspice_op_point(ng_modelcard, "nch_svt_mac",
                                      VD, VG, VS, VE, out_dir, TEMP_C)

    print(f"NGSPICE id={ng_result['id']:.3e}A ig={ng_result['ig']:.3e}A is={ng_result['is']:.3e}A")

    # === Comparison ===
    print("\n--- Comparison ---")
    _assert_close(py_result['id'], ng_result['id'], "id", ABS_TOL_I)
    print(f"  id  ✓ (diff={abs(py_result['id'] - ng_result['id']):.3e})")

    _assert_close(py_result['ig'], ng_result['ig'], "ig", ABS_TOL_I)
    print(f"  ig  ✓ (diff={abs(py_result['ig'] - ng_result['ig']):.3e})")

    _assert_close(py_result['is'], ng_result['is'], "is", ABS_TOL_I)
    print(f"  is  ✓ (diff={abs(py_result['is'] - ng_result['is']):.3e})")

    print(f"\n=== L={L_nm*1e9:.0f}nm Test PASSED ===")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.skipif(not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists(),
                    reason="missing TSMC7 modelcard")
@pytest.mark.parametrize("tfin_nm", TFIN_SWEEP)
def test_tsmc7_tfin_sweep(tfin_nm) -> None:
    """Test TSMC7 with various fin thicknesses.

    Parametrized test sweeping TFIN from 6nm to 8nm.
    Verifies PyCMG and NGSPICE produce consistent results.
    """
    if not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists():
        pytest.skip(f"TSMC7 modelcard not found")

    # Instance parameters with this TFIN value
    inst_params = {
        "L": 16e-9,
        "TFIN": tfin_nm,
        "NFIN": 2.0,
    }

    # Modelcard parameters for NGSPICE
    ngspice_params = {
        "l": 16e-9,
        "TFIN": tfin_nm,
        "NFIN": 2.0,
    }

    src_modelcard = TSMC7_DIR / TSMC7_BASE_MODELCARD
    out_dir = BUILD / "ngspice_eval" / f"tsmc7_TFIN{tfin_nm*1e9:.0f}nm"
    ng_modelcard = out_dir / "tsmc7_simple_ngspice.l"

    _make_tsmc7_ngspice_modelcard(src_modelcard, ng_modelcard, "nch_svt_mac",
                                   ngspice_params)

    print(f"\n=== TSMC7 TFIN={tfin_nm*1e9:.0f}nm Test ===")

    # === PyCMG Evaluation ===
    print("\n--- PyCMG ---")
    model = Model(str(OSDI_PATH), str(src_modelcard), "nch_svt_mac")
    inst = Instance(model, params=inst_params, temperature=TEMP_C + 273.15)
    py_result = inst.eval_dc({"d": VD, "g": VG, "s": VS, "e": VE})

    print(f"PyCMG id={py_result['id']:.3e}A ig={py_result['ig']:.3e}A is={py_result['is']:.3e}A")

    # === NGSPICE Evaluation ===
    print("\n--- NGSPICE ---")
    ng_result = _run_ngspice_op_point(ng_modelcard, "nch_svt_mac",
                                      VD, VG, VS, VE, out_dir, TEMP_C)

    print(f"NGSPICE id={ng_result['id']:.3e}A ig={ng_result['ig']:.3e}A is={ng_result['is']:.3e}A")

    # === Comparison ===
    print("\n--- Comparison ---")
    _assert_close(py_result['id'], ng_result['id'], "id", ABS_TOL_I)
    print(f"  id  ✓ (diff={abs(py_result['id'] - ng_result['id']):.3e})")

    _assert_close(py_result['ig'], ng_result['ig'], "ig", ABS_TOL_I)
    print(f"  ig  ✓ (diff={abs(py_result['ig'] - ng_result['ig']):.3e})")

    _assert_close(py_result['is'], ng_result['is'], "is", ABS_TOL_I)
    print(f"  is  ✓ (diff={abs(py_result['is'] - ng_result['is']):.3e})")

    print(f"\n=== TFIN={tfin_nm*1e9:.0f}nm Test PASSED ===")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.skipif(not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists(),
                    reason="missing TSMC7 modelcard")
@pytest.mark.parametrize("nfin", NFIN_SWEEP)
def test_tsmc7_nfin_sweep(nfin) -> None:
    """Test TSMC7 with various fin counts.

    Parametrized test sweeping NFIN from 1 to 4.
    Verifies PyCMG and NGSPICE produce consistent results.
    """
    if not (TSMC7_DIR / TSMC7_BASE_MODELCARD).exists():
        pytest.skip(f"TSMC7 modelcard not found")

    # Instance parameters with this NFIN value
    inst_params = {
        "L": 16e-9,
        "TFIN": 8e-9,
        "NFIN": nfin,
    }

    # Modelcard parameters for NGSPICE
    ngspice_params = {
        "l": 16e-9,
        "TFIN": 8e-9,
        "NFIN": nfin,
    }

    src_modelcard = TSMC7_DIR / TSMC7_BASE_MODELCARD
    out_dir = BUILD / "ngspice_eval" / f"tsmc7_NFIN{nfin:.0f}"
    ng_modelcard = out_dir / "tsmc7_simple_ngspice.l"

    _make_tsmc7_ngspice_modelcard(src_modelcard, ng_modelcard, "nch_svt_mac",
                                   ngspice_params)

    print(f"\n=== TSMC7 NFIN={nfin:.0f} Test ===")

    # === PyCMG Evaluation ===
    print("\n--- PyCMG ---")
    model = Model(str(OSDI_PATH), str(src_modelcard), "nch_svt_mac")
    inst = Instance(model, params=inst_params, temperature=TEMP_C + 273.15)
    py_result = inst.eval_dc({"d": VD, "g": VG, "s": VS, "e": VE})

    print(f"PyCMG id={py_result['id']:.3e}A ig={py_result['ig']:.3e}A is={py_result['is']:.3e}A")

    # === NGSPICE Evaluation ===
    print("\n--- NGSPICE ---")
    ng_result = _run_ngspice_op_point(ng_modelcard, "nch_svt_mac",
                                      VD, VG, VS, VE, out_dir, TEMP_C)

    print(f"NGSPICE id={ng_result['id']:.3e}A ig={ng_result['ig']:.3e}A is={ng_result['is']:.3e}A")

    # === Comparison ===
    print("\n--- Comparison ---")
    _assert_close(py_result['id'], ng_result['id'], "id", ABS_TOL_I)
    print(f"  id  ✓ (diff={abs(py_result['id'] - ng_result['id']):.3e})")

    _assert_close(py_result['ig'], ng_result['ig'], "ig", ABS_TOL_I)
    print(f"  ig  ✓ (diff={abs(py_result['ig'] - ng_result['ig']):.3e})")

    _assert_close(py_result['is'], ng_result['is'], "is", ABS_TOL_I)
    print(f"  is  ✓ (diff={abs(py_result['is'] - ng_result['is']):.3e})")

    print(f"\n=== NFIN={nfin:.0f} Test PASSED ===")
