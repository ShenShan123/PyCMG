"""
TSMC5 PVT Verification Tests

Comprehensive verification across PVT corners using TSMC5 naive modelcards.
Tests DC analysis with representative voltage points.

VERIFICATION STRATEGY:
- PyCMG wraps OSDI binary directly via ctypes (pycmg/ctypes_host.py)
- NGSPICE loads the SAME OSDI binary via .osdi command
- Tests compare PyCMG output vs NGSPICE output to ensure binary-level consistency
- Both use the exact same bsimcmg.osdi file, ensuring identical model physics

Run: pytest tests/test_tsmc5.py -v
Duration: ~10 minutes
Requires: NGSPICE, TSMC5 naive modelcards
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

# TSMC5 modelcard configuration — use naive (pre-baked) modelcards
TSMC5_NAIVE_DIR = ROOT / "tech_model_cards" / "TSMC5" / "naive"
TSMC5_MODELCARD_OVERRIDE = os.environ.get("TSMC5_MODELCARD")

NGSPICE_BIN = os.environ.get("NGSPICE_BIN",
                              "/usr/local/ngspice-45.2/bin/ngspice")

# TSMC5 naive modelcard mapping: key -> (modelcard_file, model_name)
# Naive modelcards have ONE model per file with all params baked in.
# Note: Using L=16nm for NMOS and L=20nm for PMOS for safety.
TSMC5_NMOS_SVT = {
    "file": "nch_svt_mac_l16nm.l",
    "model": "nch_svt_mac",
    "params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0},
}
TSMC5_PMOS_LVT = {
    "file": "pch_lvt_mac_l20nm.l",
    "model": "pch_lvt_mac",
    "params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0},
}

# Tolerances
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
REL_TOL = 5e-3

# Representative test temperatures
TSMC5_TEST_TEMPS = [-40, 27, 85, 125]

# Representative voltage points for TSMC5 (0.65V core)
TSMC5_VG_CORE = [0.0, 0.25, 0.5, 0.65, 0.8]
TSMC5_VD_CORE = [0.0, 0.25, 0.5, 0.65]


def _get_tsmc5_naive_modelcard(device_spec: dict) -> Path:
    """Get path to a TSMC5 naive modelcard file."""
    if TSMC5_MODELCARD_OVERRIDE:
        override = Path(TSMC5_MODELCARD_OVERRIDE)
        if override.is_file():
            return override
        raise FileNotFoundError(f"TSMC5 override not found: {override}")

    card = TSMC5_NAIVE_DIR / device_spec["file"]
    if not card.exists():
        raise FileNotFoundError(f"TSMC5 naive modelcard not found: {card}")
    return card


def _bake_inst_params_into_modelcard(src: Path, dst: Path, model_name: str,
                                      inst_params: dict) -> None:
    """Bake instance parameters into the modelcard for NGSPICE OSDI compatibility.

    NGSPICE's OSDI interface does NOT support instance-line parameter overrides
    (e.g., "N1 d g s e model L=16e-9" fails with "unknown parameter (l)").
    Instead, parameters like L, TFIN, NFIN must be set inside the .model block.

    This function reads the source modelcard, finds the target .model block,
    and appends any missing instance parameters (or overrides existing ones).
    """
    text = src.read_text()

    # Clamp EOTACC to be >= 1.1e-10 for OSDI compatibility
    def fix_eotacc(m: re.Match) -> str:
        val = float(m.group(1))
        if val <= 1.0e-10:
            return "EOTACC = 1.1e-10"
        return m.group(0)
    text = re.sub(r"EOTACC\s*=\s*([0-9eE+\-\.]+)", fix_eotacc, text, flags=re.IGNORECASE)

    lines: list[str] = []
    in_target = False
    found_keys: set[str] = set()
    target_lower = model_name.lower()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.lower().startswith(".model"):
            if in_target:
                # Append missing params at end of previous model
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()

            parts = stripped.split()
            if len(parts) >= 3 and parts[1].lower() == target_lower:
                # Force model type to bsimcmg for OSDI
                parts[2] = "bsimcmg"
                prefix = line[:line.lower().find(".model")]
                line = f"{prefix}{' '.join(parts)}"
                in_target = True

        elif in_target:
            # Detect closing ')' of .model block — insert missing params before it
            if stripped == ')':
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()
            else:
                # Override existing instance params if found
                for key, val in inst_params.items():
                    pattern = rf"(?i)\b{re.escape(key)}\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)"

                    def repl(m: re.Match, k: str = key.upper(), v: float = val) -> str:
                        found_keys.add(k)
                        return f"{k} = {v}"

                    line, _ = re.subn(pattern, repl, line)

        lines.append(line)

    # Fallback: if model block wasn't closed with ')' (unlikely for naive cards)
    if in_target:
        for key, val in inst_params.items():
            if key.upper() not in found_keys:
                lines.append(f"+ {key.upper()} = {val}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def _run_tsmc5_ngspice_op_point(modelcard: Path, model_name: str,
                                  inst_params: dict,
                                  vd: float, vg: float,
                                  vs: float, ve: float, temp_c: float) -> dict:
    """Run NGSPICE operating point analysis for TSMC5.

    Uses self-contained NGSPICE output parsing (no external dependencies).
    Instance params (L, TFIN, NFIN) are baked into the modelcard because
    NGSPICE's OSDI interface doesn't support instance-line param overrides.
    """
    out_dir = BUILD / "ngspice_eval" / "tsmc5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bake instance params into modelcard (NGSPICE OSDI can't use device-line params)
    ng_modelcard = out_dir / f"ng_{model_name}.lib"
    _bake_inst_params_into_modelcard(modelcard, ng_modelcard, model_name, inst_params)

    net = [
        "* OP point query",
        f'.include "{ng_modelcard}"',
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

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"NGSPICE failed: {res.stdout}\n{res.stderr}")

    # Self-contained wrdata parsing (no verify_utils dependency)
    with out_csv.open() as f:
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
            "ie": values[idx_map["i(ve)"]],
            "qg": values[idx_map["@n1[qg]"]],
            "qd": values[idx_map["@n1[qd]"]],
            "qs": values[idx_map["@n1[qs]"]],
            "qb": values[idx_map["@n1[qb]"]],
            "gm": values[idx_map["@n1[gm]"]],
            "gds": values[idx_map["@n1[gds]"]],
            "gmb": values[idx_map["@n1[gmbs]"]],
        }


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
def test_tsmc5_nmos_svt_op() -> None:
    """Verify TSMC5 NMOS SVT at nominal operating point.

    Quick smoke test: single OP comparison for NMOS SVT (nch_svt_mac) at L=16nm.
    Uses naive modelcard with all params pre-baked.
    """
    spec = TSMC5_NMOS_SVT
    modelcard = _get_tsmc5_naive_modelcard(spec)
    model_name = spec["model"]
    inst_params = spec["params"]

    # Test at nominal operating point (Vdd = 0.65V for TSMC5)
    ng_result = _run_tsmc5_ngspice_op_point(
        modelcard=modelcard,
        model_name=model_name,
        inst_params=inst_params,
        vd=0.65, vg=0.65, vs=0.0, ve=0.0,
        temp_c=27.0
    )

    # PyCMG — use same naive modelcard directly
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_result = inst.eval_dc({"d": 0.65, "g": 0.65, "s": 0.0, "e": 0.0})

    # Compare currents and derivatives
    _assert_close(f"{model_name}@id", py_result["id"], ng_result["id"])
    _assert_close(f"{model_name}@ig", py_result["ig"], ng_result["ig"])
    _assert_close(f"{model_name}@gm", py_result["gm"], ng_result["gm"])

    # Compare drain-source current (Ids = Id - Is)
    ng_ids = ng_result["id"] - ng_result["is"]
    _assert_close(f"{model_name}@ids", py_result["ids"], ng_ids)

    print(f"OK NMOS SVT: Id={py_result['id']:.3e}A gm={py_result['gm']:.3e}S")


@pytest.mark.fast
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc5_pmos_lvt_op() -> None:
    """Verify TSMC5 PMOS LVT at nominal operating point.

    Tests a single operating point for PMOS LVT device (pch_lvt_mac) at L=20nm.
    """
    spec = TSMC5_PMOS_LVT
    modelcard = _get_tsmc5_naive_modelcard(spec)
    model_name = spec["model"]
    inst_params = spec["params"]

    # PMOS: Vd=0, Vg=0, Vs=0.65 (Vgs = 0 - 0.65 = -0.65V, PMOS ON)
    ng_result = _run_tsmc5_ngspice_op_point(
        modelcard=modelcard,
        model_name=model_name,
        inst_params=inst_params,
        vd=0.0, vg=0.0, vs=0.65, ve=0.0,
        temp_c=27.0
    )

    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_result = inst.eval_dc({"d": 0.0, "g": 0.0, "s": 0.65, "e": 0.0})

    _assert_close(f"{model_name}@id", py_result["id"], ng_result["id"])
    _assert_close(f"{model_name}@ig", py_result["ig"], ng_result["ig"])
    ng_ids = ng_result["id"] - ng_result["is"]
    _assert_close(f"{model_name}@ids", py_result["ids"], ng_ids)

    print(f"OK PMOS LVT: Id={py_result['id']:.3e}A Ig={py_result['ig']:.3e}A")


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc5_temperature_sweep() -> None:
    """Test TSMC5 temperature sweep at representative temperatures.

    Verifies model accuracy across -40C to 125C temperature range
    using NMOS SVT naive modelcard.
    """
    spec = TSMC5_NMOS_SVT
    modelcard = _get_tsmc5_naive_modelcard(spec)
    model_name = spec["model"]
    inst_params = spec["params"]

    for temp_c in TSMC5_TEST_TEMPS:
        ng_result = _run_tsmc5_ngspice_op_point(
            modelcard=modelcard,
            model_name=model_name,
            inst_params=inst_params,
            vd=0.65, vg=0.65, vs=0.0, ve=0.0,
            temp_c=temp_c
        )

        model = Model(str(OSDI_PATH), str(modelcard), model_name)
        inst = Instance(model, params=inst_params, temperature=temp_c + 273.15)
        py_result = inst.eval_dc({"d": 0.65, "g": 0.65, "s": 0.0, "e": 0.0})

        _assert_close(f"T={temp_c:.0f}C@id", py_result["id"], ng_result["id"])
        print(f"OK T={temp_c:.0f}C: Id={py_result['id']:.3e}A")


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_tsmc5_voltage_sweep() -> None:
    """Test TSMC5 Id-Vg and Id-Vd sweeps at representative voltage points.

    Verifies model accuracy across the operating voltage range for TSMC5 (0 to 0.8V)
    using NMOS SVT naive modelcard.
    """
    spec = TSMC5_NMOS_SVT
    modelcard = _get_tsmc5_naive_modelcard(spec)
    model_name = spec["model"]
    inst_params = spec["params"]

    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)

    # Test Id-Vg at 5 points
    vd = 0.05
    for vg in TSMC5_VG_CORE:
        ng_result = _run_tsmc5_ngspice_op_point(
            modelcard=modelcard,
            model_name=model_name,
            inst_params=inst_params,
            vd=vd, vg=vg, vs=0.0, ve=0.0,
            temp_c=27.0
        )

        py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})
        _assert_close(f"Vg={vg:.2f}@id", py_result["id"], ng_result["id"])

    # Test Id-Vd at 4 points
    vg = 0.65
    for vd in TSMC5_VD_CORE:
        ng_result = _run_tsmc5_ngspice_op_point(
            modelcard=modelcard,
            model_name=model_name,
            inst_params=inst_params,
            vd=vd, vg=vg, vs=0.0, ve=0.0,
            temp_c=27.0
        )

        py_result = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})
        _assert_close(f"Vd={vd:.2f}@id", py_result["id"], ng_result["id"])

    print("OK Voltage sweep complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
