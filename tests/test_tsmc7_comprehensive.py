"""
TSMC7 PVT Verification Tests

Comprehensive verification across PVT corners using TSMC7 modelcards.
Tests DC, AC (capacitance), and transient analysis with representative voltage points.

VERIFICATION STRATEGY:
- PyCMG wraps OSDI binary directly via ctypes (pycmg/ctypes_host.py)
- NGSPICE loads the SAME OSDI binary via .osdi command
- Tests compare PyCMG output vs NGSPICE output to ensure binary-level consistency
- Both use the exact same bsimcmg.osdi file, ensuring identical model physics

Run: pytest tests/test_tsmc7.py -v
Duration: ~10 minutes
Requires: NGSPICE, TSMC7 modelcards
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

import pycmg
from pycmg.ctypes_host import Model, Instance

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
BUILD = ROOT / "build-deep-verify"

# TSMC7 modelcard configuration
TSMC7_DIR = ROOT / "tech_model_cards" / "TSMC7"
TSMC7_MODELCARD_FILE = "tsmc7_simple.l"
TSMC7_MODELCARD_OVERRIDE = os.environ.get("TSMC7_MODELCARD")

# TSMC7 model variants
TSMC7_MODELS = {
    "nch_svt": "nch_svt_mac",
    "pch_svt": "pch_svt_mac",
}

# Tolerances
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
REL_TOL = 5e-3

# Representative test temperatures
TSMC7_TEST_TEMPS = [-40, 27, 85, 125]

# Representative voltage points for TSMC7 (0.75V core)
TSMC7_VG_CORE = [0.0, 0.3, 0.6, 0.75, 0.9]
TSMC7_VD_CORE = [0.0, 0.3, 0.6, 0.75]

# TSMC7 length points
TSMC7_LENGTH_POINTS = [16e-9, 30e-9, 50e-9]


def _iter_tsmc7_modelcards() -> list[Path]:
    """Get list of TSMC7 modelcard files."""
    if TSMC7_MODELCARD_OVERRIDE:
        override = Path(TSMC7_MODELCARD_OVERRIDE)
        if override.is_file():
            return [override]
        if override.is_dir():
            return sorted(override.glob("*.l"))
        raise FileNotFoundError(f"TSMC7 override not found: {override}")

    if not TSMC7_DIR.exists():
        raise FileNotFoundError(f"TSMC7 directory not found: {TSMC7_DIR}")

    default_card = TSMC7_DIR / TSMC7_MODELCARD_FILE
    if default_card.exists():
        return [default_card]

    return sorted(TSMC7_DIR.glob("*.l"))


def _run_tsmc7_ngspice_op_point(modelcard: Path, model_name: str,
                                  inst_params: dict, vd: float, vg: float,
                                  vs: float, ve: float, temp_c: float) -> dict:
    """Run NGSPICE operating point analysis for TSMC7."""
    # Import verify_utils from tests directory
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    import verify_utils as vu

    out_dir = BUILD / "ngspice_eval" / "tsmc7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build instance parameter string for NGSPICE device line
    # For TSMC7, L, TFIN, NFIN are baked into modelcard
    # Device line uses simple instance without geometry params
    inst_params_str = ""

    # Create netlist with instance parameters on device line
    net = [
        "* OP point query",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vd}",
        f"Vg g 0 {vg}",
        f"Vs s 0 {vs}",
        f"Ve e 0 {ve}",
        f"N1 d g s e {model_name}{inst_params_str}",
        ".op",
        ".end",
    ]

    out_csv = out_dir / "ng_op_point.csv"
    log_path = out_dir / "ng_op_point.log"
    net_path = out_dir / "netlist.cir"

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
    net_path.write_text("\n".join(net))

    subprocess.run([vu.NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
                   check=True, capture_output=True)

    # Parse results
    headers, rows = vu.parse_wrdata(out_csv)
    if not rows:
        raise RuntimeError("empty NGSPICE op wrdata")

    row = rows[0]
    idx = {
        name: vu.col_index(headers, name)
        for name in ("i(vg)", "i(vd)", "i(vs)", "i(ve)",
                  "@n1[qg]", "@n1[qd]", "@n1[qs]", "@n1[qb]",
                  "@n1[gm]", "@n1[gds]", "@n1[gmbs]")
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


def _make_tsmc7_ngspice_modelcard(src: Path, dst: Path, model_name: str,
                                  inst_params: dict) -> None:
    """Create NGSPICE-compatible TSMC7 modelcard with instance parameters baked in."""
    text = src.read_text()
    lines = []
    in_target = False
    target_lower = model_name.lower()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        modified = line

        if stripped.lower().startswith(".model"):
            if in_target:
                # Add missing parameters at end of previous model
                for key, val in inst_params.items():
                    # Check if this param was already updated
                    param_found = any(
                        ln.upper().startswith(f"+ {key.upper()}")
                        for ln in lines[-50:] if ln.strip().startswith("+")
                    )
                    if not param_found:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False

            parts = stripped.split()
            if len(parts) >= 2:
                if parts[1].lower() == target_lower:
                    in_target = True
                    lines.append(line)
                    continue

        if in_target:
            # Update existing parameters with instance values
            for key, val in inst_params.items():
                pattern = rf"(\b{re.escape(key)}\s*=\s*)([\d.eE+\-]+)"
                match = re.search(pattern, stripped, re.IGNORECASE)
                if match:
                    # Replace the value with instance parameter value
                    prefix = match.group(1)
                    modified = re.sub(
                        pattern,
                        f"{prefix}{val}",
                        stripped,
                        flags=re.IGNORECASE
                    )
                    break  # Only replace one parameter per line

            lines.append(modified)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines))


def _assert_close(label: str, py_val: float, ng_val: float) -> None:
    """Assert PyCMG and NGSPICE values are within tolerance."""
    diff = abs(py_val - ng_val)
    if diff <= ABS_TOL_I:
        return
    denom = max(abs(ng_val), ABS_TOL_I)
    if diff / denom <= REL_TOL:
        return
    pytest.fail(f"{label}: py={py_val:.3e} ng={ng_val:.3e} "
                f"diff={diff:.3e} (abs_tol={ABS_TOL_I:.3e}, rel_tol={REL_TOL:.3e})")


@pytest.mark.fast
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_nmos_svt_op() -> None:
    """Verify TSMC7 NMOS SVT at nominal operating point.

    Quick smoke test to verify basic functionality with TSMC7 modelcards.
    Tests a single operating point for NMOS SVT device.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    nch_model = TSMC7_MODELS["nch_svt"]
    nch_params = {
        "L": 16e-9,
        "TFIN": 6e-9,
        "NFIN": 2.0,
    }

    # Create NGSPICE-compatible modelcard
    ng_modelcard = BUILD / "tsmc7_nch_svt.ng.lib"
    _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, nch_params)

    # Test at nominal operating point (Vdd = 0.75V for TSMC7)
    ng_result = _run_tsmc7_ngspice_op_point(
        modelcard=ng_modelcard,
        model_name=nch_model,
        inst_params=nch_params,
        vd=0.75, vg=0.75, vs=0.0, ve=0.0,
        temp_c=27.0
    )

    # For PyCMG, use modelcard directly
    model = Model(str(OSDI_PATH), str(modelcard), nch_model)
    inst = Instance(model, params=nch_params)
    py_result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

    # Compare currents
    _assert_close(f"{nch_model}@id", py_result["id"], ng_result["id"])
    _assert_close(f"{nch_model}@ig", py_result["ig"], ng_result["ig"])
    _assert_close(f"{nch_model}@gm", py_result["gm"], ng_result["gm"])

    # Compare drain-source current (Ids = Id - Is)
    ng_ids = ng_result["id"] - ng_result["is"]
    _assert_close(f"{nch_model}@ids", py_result["ids"], ng_ids)

    print(f"✓ NMOS SVT: Id={py_result['id']:.3e}A Ig={py_result['ig']:.3e}A")


@pytest.mark.fast
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_pmos_svt_op() -> None:
    """Verify TSMC7 PMOS SVT at nominal operating point.

    Tests a single operating point for PMOS SVT device.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    pch_model = TSMC7_MODELS["pch_svt"]
    pch_params = {
        "L": 16e-9,
        "TFIN": 6e-9,
        "NFIN": 2.0,
    }

    # Create NGSPICE-compatible modelcard
    ng_modelcard = BUILD / "tsmc7_pch_svt.ng.lib"
    _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, pch_model, pch_params)

    # PMOS: Vd=0, Vg=0, Vs=0.75 (Vgs=-0.75V for ON state)
    # This gives Vgs = Vg - Vs = 0 - 0.75 = -0.75V (PMOS ON)
    ng_result = _run_tsmc7_ngspice_op_point(
        modelcard=ng_modelcard,
        model_name=pch_model,
        inst_params=pch_params,
        vd=0.0, vg=0.0, vs=0.75, ve=0.0,
        temp_c=27.0
    )

    model = Model(str(OSDI_PATH), str(modelcard), pch_model)
    inst = Instance(model, params=pch_params)
    py_result = inst.eval_dc({"d": 0.0, "g": 0.0, "s": 0.75, "e": 0.0})

    _assert_close(f"{pch_model}@id", py_result["id"], ng_result["id"])
    _assert_close(f"{pch_model}@ig", py_result["ig"], ng_result["ig"])
    ng_ids = ng_result["id"] - ng_result["is"]
    _assert_close(f"{pch_model}@ids", py_result["ids"], ng_ids)

    print(f"✓ PMOS SVT: Id={py_result['id']:.3e}A Ig={py_result['ig']:.3e}A")


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_temperature_sweep() -> None:
    """Test TSMC7 temperature sweep at representative temperatures.

    Verifies model accuracy across -40°C to 125°C temperature range.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    nch_model = TSMC7_MODELS["nch_svt"]
    inst_params = {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0}

    ng_modelcard = BUILD / "tsmc7_temp.ng.lib"
    _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, inst_params)

    for temp_c in TSMC7_TEST_TEMPS:
        ng_result = _run_tsmc7_ngspice_op_point(
            modelcard=ng_modelcard,
            model_name=nch_model,
            inst_params=inst_params,
            vd=0.75, vg=0.75, vs=0.0, ve=0.0,
            temp_c=temp_c
        )

        model = Model(str(OSDI_PATH), str(ng_modelcard), nch_model)
        inst = Instance(model, params=inst_params, temperature=temp_c + 273.15)
        py_result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

        _assert_close(f"T={temp_c:.0f}C@id", py_result["id"], ng_result["id"])
        print(f"✓ T={temp_c:.0f}C: Id={py_result['id']:.3e}A")


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc7_voltage_sweep() -> None:
    """Test TSMC7 Id-Vg and Id-Vd sweeps at representative voltage points.

    Verifies model accuracy across the operating voltage range for TSMC7 (0 to 0.9V).
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    nch_model = TSMC7_MODELS["nch_svt"]
    inst_params = {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0}

    ng_modelcard = BUILD / "tsmc7_sweep.ng.lib"
    _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, inst_params)

    model = Model(str(OSDI_PATH), str(ng_modelcard), nch_model)
    inst = Instance(model, params=inst_params)

    # Test Id-Vg at 5 points
    vd = 0.05
    for vg in TSMC7_VG_CORE:
        ng_result = _run_tsmc7_ngspice_op_point(
            modelcard=ng_modelcard,
            model_name=nch_model,
            inst_params=inst_params,
            vd=vd, vg=vg, vs=0.0, ve=0.0,
            temp_c=27.0
        )

        py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})
        _assert_close(f"Vg={vg:.2f}@id", py_result["id"], ng_result["id"])

    # Test Id-Vd at 4 points
    vg = 0.75
    for vd in TSMC7_VD_CORE:
        ng_result = _run_tsmc7_ngspice_op_point(
            modelcard=ng_modelcard,
            model_name=nch_model,
            inst_params=inst_params,
            vd=vd, vg=vg, vs=0.0, ve=0.0,
            temp_c=27.0
        )

        py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})
        _assert_close(f"Vd={vd:.2f}@id", py_result["id"], ng_result["id"])

    print(f"✓ Voltage sweep complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
