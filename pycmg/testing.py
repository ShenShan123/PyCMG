"""
PyCMG testing utilities — NGSPICE comparison helpers.

Consolidates NGSPICE runner, modelcard baking, and assertion functions
previously duplicated across test_integration.py, test_asap7.py,
test_tsmc{5,7,12,16}.py.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
BUILD = ROOT / "build-deep-verify"
NGSPICE_BIN = os.environ.get("NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")

# Tolerances — single source of truth
ABS_TOL_I = 1e-9      # Current (A)
ABS_TOL_Q = 1e-18     # Charge (C)
ABS_TOL_G = 1e-6      # Conductance (S) — Jacobian floor
REL_TOL = 5e-3        # DC/transient relative (0.5%)
REL_TOL_JAC = 1e-2    # Jacobian relative (1%) — central finite-difference


def bake_inst_params(src: Path, dst: Path, model_name: str,
                     inst_params: Dict[str, Any]) -> None:
    """Bake instance parameters into modelcard for NGSPICE OSDI compatibility.

    NGSPICE's OSDI interface cannot accept instance parameters on the device
    line (unlike HSPICE/Spectre). All geometric parameters (L, TFIN, NFIN)
    must be injected into the .model block.

    Args:
        src: Source modelcard file path
        dst: Destination path for baked modelcard
        model_name: Target .model name to modify
        inst_params: Dict of params to bake (e.g. {"L": 16e-9, "TFIN": 6e-9})
    """
    text = src.read_text()

    # Clamp EOTACC (standardized threshold: <= 1.0e-10 → 1.1e-10)
    def fix_eotacc(m: re.Match) -> str:
        val = float(m.group(1))
        if val <= 1.0e-10:
            return "EOTACC = 1.1e-10"
        return m.group(0)
    text = re.sub(r"EOTACC\s*=\s*([0-9eE+\-\.]+)", fix_eotacc, text, flags=re.IGNORECASE)

    lines: List[str] = []
    in_target = False
    found_keys: set = set()
    target_lower = model_name.lower()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.lower().startswith(".model"):
            # Close previous target block if open
            if in_target:
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()

            parts = stripped.split()
            if len(parts) >= 3 and parts[1].lower() == target_lower:
                parts[2] = "bsimcmg"
                prefix = line[:line.lower().find(".model")]
                line = f"{prefix}{' '.join(parts)}"
                in_target = True

        elif in_target:
            if stripped == ')':
                # Insert params BEFORE closing bracket
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()
            else:
                # Replace existing param values in-place
                for key, val in inst_params.items():
                    pattern = rf"(?i)\b{re.escape(key)}\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)"
                    def repl(m: re.Match, k: str = key.upper(), v: Any = val) -> str:
                        found_keys.add(k)
                        return f"{k} = {v}"
                    line, _ = re.subn(pattern, repl, line)

        lines.append(line)

    # Handle case where .model block has no closing bracket
    if in_target:
        for key, val in inst_params.items():
            if key.upper() not in found_keys:
                lines.append(f"+ {key.upper()} = {val}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def run_ngspice_op(modelcard: Path, model_name: str, inst_params: Dict[str, Any],
                   vd: float, vg: float, vs: float = 0.0, ve: float = 0.0,
                   temp_c: float = 27.0,
                   tag: str = "op") -> Dict[str, float]:
    """Run NGSPICE operating point analysis and return results.

    Args:
        modelcard: Path to modelcard file
        model_name: Model name in the modelcard
        inst_params: Instance parameters to bake
        vd, vg, vs, ve: Terminal voltages
        temp_c: Temperature in Celsius
        tag: Unique tag for output files (prevents collisions in parallel runs)

    Returns:
        Dict with keys: id, ig, is, ie, qg, qd, qs, qb, gm, gds, gmb
    """
    out_dir = BUILD / "ngspice_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bake instance params into modelcard
    ng_modelcard = out_dir / f"ng_{model_name}.lib"
    bake_inst_params(modelcard, ng_modelcard, model_name, inst_params)

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

    out_csv = out_dir / "ng_op.csv"
    net_path = out_dir / "netlist.cir"
    log_path = out_dir / "ng_op.log"
    runner_path = out_dir / "runner.cir"

    runner_path.write_text(
        "* ngspice runner\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {net_path}\n"
        "set filetype=ascii\n"
        "set wr_vecnames\n"
        ".options saveinternals\n"
        "run\n"
        f"wrdata {out_csv} v(g) v(d) v(s) v(e) "
        "i(vg) i(vd) i(vs) i(ve) "
        "@n1[qg] @n1[qd] @n1[qs] @n1[qb] "
        "@n1[gm] @n1[gds] @n1[gmbs]\n"
        ".endc\n"
        ".end\n"
    )
    net_path.write_text("\n".join(net))

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"NGSPICE failed (tag={tag}):\n{res.stdout}\n{res.stderr}\n"
            f"See log: {log_path}"
        )

    with out_csv.open() as f:
        csv_lines = f.readlines()
        if not csv_lines:
            raise RuntimeError(f"Empty NGSPICE output: {out_csv}")
        headers = csv_lines[0].split()
        values = [float(x) for x in csv_lines[1].split()]
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


def assert_close(label: str, py_val: float, ng_val: float,
                 abs_tol: Optional[float] = None,
                 rel_tol: float = REL_TOL) -> None:
    """Assert PyCMG and NGSPICE values are within tolerance.

    Auto-selects abs_tol based on label if not provided:
    - Labels containing 'q' → ABS_TOL_Q (1e-18)
    - Labels containing 'g' (conductance) → ABS_TOL_G (1e-6)
    - Default → ABS_TOL_I (1e-9)

    Args:
        label: Descriptive label for error messages
        py_val: PyCMG computed value
        ng_val: NGSPICE reference value
        abs_tol: Absolute tolerance override (auto-selected if None)
        rel_tol: Relative tolerance
    """
    if abs_tol is None:
        lbl_lower = label.lower().split("/")[-1]
        if lbl_lower.startswith("q"):
            abs_tol = ABS_TOL_Q
        elif lbl_lower.startswith("g") or lbl_lower.startswith("d("):
            abs_tol = ABS_TOL_G
        else:
            abs_tol = ABS_TOL_I

    diff = abs(py_val - ng_val)
    if diff <= abs_tol:
        return
    denom = max(abs(ng_val), abs_tol)
    if diff / denom <= rel_tol:
        return
    pytest.fail(
        f"{label} mismatch:\n"
        f"  PyCMG:   {py_val:.6e}\n"
        f"  NGSPICE: {ng_val:.6e}\n"
        f"  Diff:    {diff:.6e} (abs_tol={abs_tol:.3e}, rel_tol={rel_tol:.3e})"
    )


def run_ngspice_transient(modelcard: Path, model_name: str,
                          inst_params: Dict[str, Any], vdd: float,
                          t_step: float = 10e-12,
                          t_stop: float = 5e-9,
                          temp_c: float = 27.0,
                          tag: str = "tran") -> Dict[str, np.ndarray]:
    """Run NGSPICE transient simulation with pulse stimulus on gate.

    Returns dict mapping variable names to numpy arrays:
        'time', 'v(d)', 'v(g)', 'v(s)', 'v(e)',
        'i(vd)', 'i(vg)', 'i(vs)', 'i(ve)'
    """
    out_dir = BUILD / "ngspice_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ng_modelcard = out_dir / f"ng_{model_name}.lib"
    bake_inst_params(modelcard, ng_modelcard, model_name, inst_params)

    rise_time = 100e-12  # 100 ps
    width = 2e-9         # 2 ns
    period = 4e-9        # 4 ns

    net = [
        "* Transient test",
        f'.include "{ng_modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vdd}",
        f"Vg g 0 PULSE(0 {vdd} 500p {rise_time:.0e} {rise_time:.0e} {width:.0e} {period:.0e})",
        "Vs s 0 0",
        "Ve e 0 0",
        f"N1 d g s e {model_name}",
        f".tran {t_step:.0e} {t_stop:.0e}",
        ".end",
    ]

    raw_path = out_dir / "tran.raw"
    net_path = out_dir / "tran.cir"
    log_path = out_dir / "tran.log"
    runner_path = out_dir / "runner.cir"

    runner_path.write_text(
        "* ngspice runner\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {net_path}\n"
        "set filetype=ascii\n"
        "run\n"
        f"write {raw_path} v(d) v(g) v(s) v(e) i(vd) i(vg) i(vs) i(ve)\n"
        ".endc\n"
        ".end\n"
    )
    net_path.write_text("\n".join(net))

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"NGSPICE transient failed (tag={tag}):\n{res.stdout}\n{res.stderr}\n"
            f"See log: {log_path}"
        )

    return parse_ngspice_raw(raw_path)


def parse_ngspice_raw(raw_path: Path) -> Dict[str, np.ndarray]:
    """Parse NGSPICE ASCII raw file into dict of numpy arrays.

    Handles the standard NGSPICE raw format:
        Variables: (index, name, type)
        Values: (index followed by variable values, one per line)
    """
    with raw_path.open() as f:
        content = f.read()

    lines = content.splitlines()

    # Parse header: extract variable names
    headers: List[str] = []
    n_points = 0
    data_start = 0

    for i, line in enumerate(lines):
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":")[1].strip())
        elif line.startswith("No. Points:"):
            n_points = int(line.split(":")[1].strip())
        elif line.startswith("Variables:"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("Values:"):
                parts = lines[j].split()
                if len(parts) >= 2:
                    headers.append(parts[1])
                j += 1
            data_start = j + 1
            break

    if not headers or n_points == 0:
        raise RuntimeError(f"Could not parse NGSPICE raw file: {raw_path}")

    # Parse values — collect into list of rows, then convert to numpy
    # Format: index_number\n val1\n val2\n ... valN\n (repeated per point)
    n_vars = len(headers)
    all_rows: List[List[float]] = []
    current_row: List[float] = []

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Each value is on its own line; first value of each point
        # is preceded by the point index (e.g., "0\t1.23456e-01")
        parts = stripped.split()
        for p in parts:
            try:
                val = float(p)
                current_row.append(val)
                if len(current_row) == n_vars:
                    all_rows.append(current_row)
                    current_row = []
            except ValueError:
                pass

    if not all_rows:
        raise RuntimeError(f"No data points parsed from {raw_path}")

    data = np.array(all_rows)  # shape: (n_points, n_vars)

    result: Dict[str, np.ndarray] = {}
    for i, h in enumerate(headers):
        result[h] = data[:, i]

    return result
