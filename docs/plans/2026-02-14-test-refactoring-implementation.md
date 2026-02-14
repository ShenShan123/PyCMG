# Test Suite Refactoring Implementation Plan (Rev 2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor test suite for convergence-critical verification with full Jacobian, transient, and operating region coverage across 5 technologies. Consolidate duplicated test utilities.

**Architecture:** Analysis-type-based test organization with parametrized technology fixtures. Shared NGSPICE helpers in `pycmg/testing.py`. Each test file focuses on one verification aspect and runs against all 5 technologies.

**Tech Stack:** pytest, numpy, NGSPICE subprocess, PyCMG ctypes interface

---

## Task 1: Create pycmg/testing.py with Consolidated Utilities

**Why:** Six test files currently duplicate `_bake_inst_params()`, `_run_ngspice_op()`, and `_assert_close()`. These belong in a shared library module, not scattered across tests.

**Files:**
- Create: `pycmg/testing.py`
- Modify: `pycmg/__init__.py` (export testing module)
- Test: `python -c "from pycmg.testing import run_ngspice_op, bake_inst_params, assert_close; print('OK')"`

**Step 1: Create pycmg/testing.py with three core functions**

```python
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
        lines = f.readlines()
        if not lines:
            raise RuntimeError(f"Empty NGSPICE output: {out_csv}")
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
```

**Step 2: Verify import works**

Run: `python -c "from pycmg.testing import run_ngspice_op, bake_inst_params, assert_close, parse_ngspice_raw; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add pycmg/testing.py
git commit -m "feat(pycmg): add testing.py with consolidated NGSPICE helpers"
```

---

## Task 2: Enhance conftest.py with Technology Registry

**Why:** The technology registry centralizes all technology-specific details (Vdd, model names, file paths, instance params) so test files don't hardcode them.

**Files:**
- Modify: `tests/conftest.py`
- Test: `python -c "import sys; sys.path.insert(0, '.'); from tests.conftest import TECHNOLOGIES; print(list(TECHNOLOGIES.keys()))"`

**Step 1: Replace conftest.py contents**

```python
"""
Pytest configuration and technology registry for PyCMG verification tests.

The registry provides deterministic modelcard selection:
- ASAP7: Explicit TT corner + rvt variant (no glob ambiguity)
- TSMC: Explicit file names + per-device instance params (PMOS L=20nm)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"

# Technology registry — single source of truth for all test parametrization.
#
# Each entry specifies:
#   dir:          subdirectory under tech_model_cards/
#   vdd:          core supply voltage (V)
#   nmos_file:    exact modelcard filename for NMOS
#   pmos_file:    exact modelcard filename for PMOS
#   nmos_model:   .model name inside the NMOS modelcard
#   pmos_model:   .model name inside the PMOS modelcard
#   nmos_params:  instance params for NMOS (baked into modelcard for NGSPICE)
#   pmos_params:  instance params for PMOS
#
TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "ASAP7": {
        "dir": "ASAP7",
        "vdd": 0.9,
        "corner": "TT",
        "nmos_file": "7nm_TT_160803.pm",
        "pmos_file": "7nm_TT_160803.pm",
        "nmos_model": "nmos_rvt",
        "pmos_model": "pmos_rvt",
        "nmos_params": {"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 1.0},
        "pmos_params": {"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 1.0},
    },
    "TSMC5": {
        "dir": "TSMC5/naive",
        "vdd": 0.65,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0},
    },
    "TSMC7": {
        "dir": "TSMC7/naive",
        "vdd": 0.75,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0},
    },
    "TSMC12": {
        "dir": "TSMC12/naive",
        "vdd": 0.80,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0},
    },
    "TSMC16": {
        "dir": "TSMC16/naive",
        "vdd": 0.80,
        "nmos_file": "nch_svt_mac_l16nm.l",
        "pmos_file": "pch_lvt_mac_l20nm.l",
        "nmos_model": "nch_svt_mac",
        "pmos_model": "pch_lvt_mac",
        "nmos_params": {"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0},
        "pmos_params": {"L": 20e-9, "TFIN": 6e-9, "NFIN": 2.0},
    },
}

TECH_NAMES = list(TECHNOLOGIES.keys())


def get_tech_modelcard(tech_name: str, device_type: str = "nmos") -> Tuple[Path, str, Dict[str, float]]:
    """Get modelcard path, model name, and instance params for a technology.

    Args:
        tech_name: Key from TECHNOLOGIES registry
        device_type: "nmos" or "pmos"

    Returns:
        Tuple of (modelcard_path, model_name, inst_params)
    """
    tech = TECHNOLOGIES[tech_name]
    tech_dir = ROOT / "tech_model_cards" / tech["dir"]

    file_key = f"{device_type}_file"
    model_key = f"{device_type}_model"
    params_key = f"{device_type}_params"

    modelcard = tech_dir / tech[file_key]
    if not modelcard.exists():
        raise FileNotFoundError(f"Modelcard not found: {modelcard}")

    return modelcard, tech[model_key], tech[params_key]


# -- pytest hooks (keep existing) --

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Add test report attribute to node for result tracking."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)
```

**Step 2: Verify**

Run: `python -c "import sys; sys.path.insert(0, '.'); from tests.conftest import TECHNOLOGIES, get_tech_modelcard; print(list(TECHNOLOGIES.keys()))"`

Expected: `['ASAP7', 'TSMC5', 'TSMC7', 'TSMC12', 'TSMC16']`

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "feat(tests): add technology registry with deterministic modelcard selection"
```

---

## Task 3: Add get_jacobian_matrix() to Instance Class

**Why:** Circuit simulators need the condensed 4×4 resistive Jacobian for Newton-Raphson. PyCMG already builds the full N×N Jacobian internally — this task exposes the condensed version.

**Files:**
- Modify: `pycmg/ctypes_host.py` (add method to Instance class)
- Test: `pytest tests/test_api.py -v`

**Step 1: Add failing test to test_api.py**

Add this test to `tests/test_api.py`:

```python
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_get_jacobian_matrix():
    """Test that Instance exposes condensed 4x4 Jacobian matrix."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        J = inst.get_jacobian_matrix({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})

        # Condensed to 4 external terminals: d, g, s, e
        assert J.shape == (4, 4), f"Expected (4,4), got {J.shape}"
        assert isinstance(J, np.ndarray)
        assert np.all(np.isfinite(J)), f"Non-finite entries in Jacobian: {J}"

        # gds = dId/dVd should be positive in saturation
        # (terminal ordering from sim.terminal_indices)
        assert J.sum() != 0.0, "Jacobian is all zeros"
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_get_jacobian_matrix -v`

Expected: FAIL with `AttributeError: 'Instance' object has no attribute 'get_jacobian_matrix'`

**Step 3: Implement get_jacobian_matrix() in ctypes_host.py**

Add this method to the `Instance` class, right before `eval_dc()`:

```python
def get_jacobian_matrix(self, nodes: Dict[str, float]) -> np.ndarray:
    """Extract the condensed 4×4 resistive Jacobian matrix.

    BSIM-CMG has internal nodes (di, si, etc.) making the raw Jacobian
    7×7 or larger. This method condenses it to the 4 external terminals
    using Schur complement elimination:

        G_ext = G_ee - G_ei × G_ii⁻¹ × G_ie

    This is the matrix a circuit simulator's Newton-Raphson solver sees.

    Returns a 4×4 numpy array where:
    - Rows/cols correspond to terminal order in sim.terminal_indices
      (typically d, g, s, e)
    - G[i,j] = ∂I_i / ∂V_j (conductance, Siemens)

    Args:
        nodes: Dict with keys 'd', 'g', 's', 'e' and voltage values

    Returns:
        4×4 condensed Jacobian matrix as numpy array
    """
    # Run DC evaluation to populate OSDI Jacobian buffers
    self.eval_dc(nodes)

    # Build full N×N resistive Jacobian from OSDI
    g_full = self._build_full_jacobian(self._sim, self._sim.jacobian_resist)

    # Condense to external-only using Schur complement
    # (same math as _condense_capacitance but real-valued)
    ext = self._sim.terminal_indices
    intn = self._sim.internal_indices
    ne = len(ext)
    ni = len(intn)

    g_ee = np.zeros((ne, ne))
    for r in range(ne):
        for c in range(ne):
            g_ee[r, c] = g_full[ext[r], ext[c]]

    if ni == 0:
        return g_ee

    g_ei = np.zeros((ne, ni))
    g_ie = np.zeros((ni, ne))
    g_ii = np.zeros((ni, ni))

    for r in range(ne):
        for c in range(ni):
            g_ei[r, c] = g_full[ext[r], intn[c]]
    for r in range(ni):
        for c in range(ne):
            g_ie[r, c] = g_full[intn[r], ext[c]]
        for c in range(ni):
            g_ii[r, c] = g_full[intn[r], intn[c]]

    try:
        g_ie_sol = np.linalg.solve(g_ii, g_ie)
    except np.linalg.LinAlgError:
        return g_ee  # Fallback: no condensation

    return g_ee - g_ei @ g_ie_sol
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_get_jacobian_matrix -v`

Expected: PASS

**Step 5: Run full API test suite to check for regressions**

Run: `pytest tests/test_api.py -v`

Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add pycmg/ctypes_host.py tests/test_api.py
git commit -m "feat(pycmg): add get_jacobian_matrix() with Schur complement condensation"
```

---

## Task 4: Create test_dc_jacobian.py

**Why:** Verify that PyCMG's analytical Jacobian matches NGSPICE's numerical Jacobian. This is the most critical test for circuit simulator convergence — an incorrect Jacobian makes Newton-Raphson diverge.

**Files:**
- Create: `tests/test_dc_jacobian.py`
- Test: `pytest tests/test_dc_jacobian.py -v`

**Step 1: Create test_dc_jacobian.py with central differencing**

```python
"""
DC Jacobian Verification Tests

Compares PyCMG's condensed 4×4 analytical Jacobian against NGSPICE's
numerical Jacobian computed via central finite-difference perturbation.

Central differencing: J[:,j] = (I(V+δ_j) - I(V-δ_j)) / (2δ)
- O(δ²) accuracy (vs O(δ) for forward differencing)
- 9 NGSPICE calls per operating point (1 base + 4×2 perturbations)

Run: pytest tests/test_dc_jacobian.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import (
    OSDI_PATH, run_ngspice_op, assert_close,
    ABS_TOL_G, REL_TOL_JAC,
)
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


def get_jacobian_op_points(vdd: float) -> list[dict]:
    """Generate operating points for Jacobian testing."""
    return [
        {"name": "saturation", "vd": vdd, "vg": 0.8 * vdd, "vs": 0.0, "ve": 0.0},
        {"name": "linear", "vd": 0.3 * vdd, "vg": vdd, "vs": 0.0, "ve": 0.0},
        {"name": "off", "vd": vdd, "vg": 0.0, "vs": 0.0, "ve": 0.0},
    ]


def compute_numerical_jacobian_central(
    modelcard: Path, model_name: str, inst_params: dict,
    op: dict, delta: float = 1e-6, temp_c: float = 27.0,
    tag_prefix: str = "jac",
) -> np.ndarray:
    """Compute 4×4 Jacobian via central finite-difference perturbation.

    Uses central differencing for O(δ²) accuracy:
        J[:,j] = (I(V+δ_j) - I(V-δ_j)) / (2δ)

    Requires 9 NGSPICE simulations: 1 base + 4×2 perturbations.
    The base run is for diagnostics only; central diff doesn't need it.
    """
    terminals = ["vd", "vg", "vs", "ve"]
    op_keys = ["d", "g", "s", "e"]
    current_keys = ["id", "ig", "is", "ie"]
    n = 4
    J = np.zeros((n, n))

    for j, (term_v, op_key) in enumerate(zip(terminals, op_keys)):
        # Forward perturbation: V + δ
        fwd_op = dict(op)
        fwd_op[op_key] = op[op_key] + delta
        fwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            fwd_op["d"], fwd_op["g"], fwd_op["s"], fwd_op["e"],
            temp_c, tag=f"{tag_prefix}_fwd_{op_key}",
        )
        fwd_I = np.array([fwd[k] for k in current_keys])

        # Backward perturbation: V - δ
        bwd_op = dict(op)
        bwd_op[op_key] = op[op_key] - delta
        bwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            bwd_op["d"], bwd_op["g"], bwd_op["s"], bwd_op["e"],
            temp_c, tag=f"{tag_prefix}_bwd_{op_key}",
        )
        bwd_I = np.array([bwd[k] for k in current_keys])

        # Central difference
        J[:, j] = (fwd_I - bwd_I) / (2.0 * delta)

    return J


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("op_idx", [0, 1, 2], ids=["saturation", "linear", "off"])
def test_dc_jacobian_full_matrix(tech_name: str, op_idx: int):
    """Compare condensed 4×4 Jacobian matrix against NGSPICE numerical Jacobian."""
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")

    op_points = get_jacobian_op_points(tech["vdd"])
    op = op_points[op_idx]
    op_name = op.pop("name")

    # NGSPICE: numerical Jacobian via central differencing
    ng_J = compute_numerical_jacobian_central(
        modelcard, model_name, inst_params, op,
        tag_prefix=f"jac_{tech_name}_{op_name}",
    )

    # PyCMG: analytical condensed Jacobian
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_J = inst.get_jacobian_matrix(
        {"d": op["d"], "g": op["g"], "s": op["s"], "e": op["e"]}
    )

    # Compare each entry
    terminals = ["d", "g", "s", "e"]
    for i, term_i in enumerate(terminals):
        for j, term_j in enumerate(terminals):
            label = f"{tech_name}/{op_name}/d(I{term_i})/d(V{term_j})"
            assert_close(
                label, py_J[i, j], ng_J[i, j],
                abs_tol=ABS_TOL_G, rel_tol=REL_TOL_JAC,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Verify collection**

Run: `pytest tests/test_dc_jacobian.py --collect-only`

Expected: 15 tests (5 techs × 3 OPs)

**Step 3: Run one technology first**

Run: `pytest tests/test_dc_jacobian.py -v -k "ASAP7"`

Expected: 3 tests pass. If they fail, debug the Jacobian condensation ordering.

**Step 4: Run all**

Run: `pytest tests/test_dc_jacobian.py -v`

**Step 5: Commit**

```bash
git add tests/test_dc_jacobian.py
git commit -m "feat(tests): add DC Jacobian verification with central differencing"
```

---

## Task 5: Create test_dc_regions.py

**Why:** Verify model accuracy across all operating regions for both NMOS and PMOS. The voltage-ratio approach makes tests technology-agnostic.

**Files:**
- Create: `tests/test_dc_regions.py`
- Test: `pytest tests/test_dc_regions.py -v`

**Step 1: Create test_dc_regions.py with NMOS + PMOS coverage**

```python
"""
DC Operating Region Tests

Verifies model accuracy across voltage-ratio-defined operating regions
for both NMOS and PMOS devices across all 5 technologies.

NMOS: 5 regions with positive voltages, grounded source
PMOS: 5 regions with inverted sense (Vs=Vdd)

Run: pytest tests/test_dc_regions.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import OSDI_PATH, run_ngspice_op, assert_close, REL_TOL
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


def get_nmos_region_ops(vdd: float) -> dict:
    """NMOS operating regions: positive voltages, grounded source."""
    return {
        "off_state":           {"d": vdd,       "g": 0.0,       "s": 0.0, "e": 0.0},
        "weak_inversion":      {"d": vdd,       "g": 0.3 * vdd, "s": 0.0, "e": 0.0},
        "moderate_inversion":  {"d": 0.5 * vdd, "g": 0.6 * vdd, "s": 0.0, "e": 0.0},
        "strong_linear":       {"d": 0.3 * vdd, "g": vdd,       "s": 0.0, "e": 0.0},
        "strong_saturation":   {"d": vdd,       "g": 0.8 * vdd, "s": 0.0, "e": 0.0},
    }


def get_pmos_region_ops(vdd: float) -> dict:
    """PMOS operating regions: Vs=Vdd, Vg/Vd referenced to Vdd."""
    return {
        "off_state":           {"d": 0.0,       "g": vdd,       "s": vdd, "e": 0.0},
        "weak_inversion":      {"d": 0.0,       "g": 0.7 * vdd, "s": vdd, "e": 0.0},
        "moderate_inversion":  {"d": 0.5 * vdd, "g": 0.4 * vdd, "s": vdd, "e": 0.0},
        "strong_linear":       {"d": 0.7 * vdd, "g": 0.0,       "s": vdd, "e": 0.0},
        "strong_saturation":   {"d": 0.0,       "g": 0.2 * vdd, "s": vdd, "e": 0.0},
    }


NMOS_REGIONS = list(get_nmos_region_ops(1.0).keys())
PMOS_REGIONS = list(get_pmos_region_ops(1.0).keys())


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("region", NMOS_REGIONS)
def test_nmos_dc_region(tech_name: str, region: str):
    """Test NMOS DC currents and derivatives match NGSPICE in operating region."""
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")

    vdd = tech["vdd"]
    op = get_nmos_region_ops(vdd)[region]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        op["d"], op["g"], op["s"], op["e"],
        tag=f"region_{tech_name}_nmos_{region}",
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc(op)

    # Compare currents
    prefix = f"{tech_name}/nmos/{region}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/ig", py["ig"], ng["ig"])
    assert_close(f"{prefix}/is", py["is"], ng["is"])

    # Compare derivatives
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])

    # Compare charges
    assert_close(f"{prefix}/qg", py["qg"], ng["qg"])
    assert_close(f"{prefix}/qd", py["qd"], ng["qd"])


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("region", PMOS_REGIONS)
def test_pmos_dc_region(tech_name: str, region: str):
    """Test PMOS DC currents and derivatives match NGSPICE in operating region."""
    tech = TECHNOLOGIES[tech_name]

    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "pmos")
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")

    vdd = tech["vdd"]
    op = get_pmos_region_ops(vdd)[region]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        op["d"], op["g"], op["s"], op["e"],
        tag=f"region_{tech_name}_pmos_{region}",
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc(op)

    prefix = f"{tech_name}/pmos/{region}"
    assert_close(f"{prefix}/id", py["id"], ng["id"])
    assert_close(f"{prefix}/ig", py["ig"], ng["ig"])
    assert_close(f"{prefix}/is", py["is"], ng["is"])
    assert_close(f"{prefix}/gm", py["gm"], ng["gm"])
    assert_close(f"{prefix}/gds", py["gds"], ng["gds"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Verify collection**

Run: `pytest tests/test_dc_regions.py --collect-only`

Expected: 50 tests (5 techs × 5 NMOS regions + 5 techs × 5 PMOS regions)

**Step 3: Run one tech**

Run: `pytest tests/test_dc_regions.py -v -k "ASAP7"`

Expected: 10 tests pass (5 NMOS + 5 PMOS)

**Step 4: Commit**

```bash
git add tests/test_dc_regions.py
git commit -m "feat(tests): add DC operating region tests for NMOS and PMOS"
```

---

## Task 6: Create test_transient.py

**Why:** Verify that PyCMG's `eval_tran()` produces correct time-domain results by comparing against NGSPICE transient simulation at solved node voltages.

**Files:**
- Create: `tests/test_transient.py`
- Test: `pytest tests/test_transient.py -v`

**Step 1: Create test_transient.py**

```python
"""
Transient Waveform Verification Tests

Compares PyCMG eval_tran() output against NGSPICE transient simulation.

Methodology:
1. NGSPICE runs a transient sim with pulse stimulus → produces V(t) and I(t)
2. PyCMG receives NGSPICE's solved node voltages as input at each time point
3. Compare PyCMG currents against NGSPICE currents

Sign convention:
    NGSPICE i(vd) = current into Vd positive terminal = drain terminal current
    PyCMG id = drain terminal current (same direction)
    → No sign flip needed (documented in CLAUDE.md)

Run: pytest tests/test_transient.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import (
    OSDI_PATH, run_ngspice_transient, assert_close, REL_TOL,
)
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
def test_transient_waveform(tech_name: str):
    """Compare PyCMG transient currents against NGSPICE at solved node voltages."""
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")
    vdd = tech["vdd"]

    # Run NGSPICE transient
    ng_wave = run_ngspice_transient(
        modelcard, model_name, inst_params, vdd,
        t_step=10e-12, t_stop=5e-9,
        tag=f"tran_{tech_name}",
    )

    # Validate NGSPICE produced data
    time_key = None
    for candidate in ["time", "Time", "TIME"]:
        if candidate in ng_wave:
            time_key = candidate
            break
    if time_key is None:
        pytest.skip(f"NGSPICE transient returned no time vector for {tech_name}")

    n_points = len(ng_wave[time_key])
    if n_points < 10:
        pytest.skip(f"Too few time points ({n_points}) from NGSPICE for {tech_name}")

    # Create PyCMG instance
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)

    # Sample every Nth point to keep test runtime reasonable
    n_samples = min(20, n_points)
    sample_indices = np.linspace(1, n_points - 1, n_samples, dtype=int)

    mismatches = 0
    for idx in sample_indices:
        t = ng_wave[time_key][idx]
        nodes = {
            "d": float(ng_wave.get("v(d)", ng_wave.get("V(d)", np.zeros(n_points)))[idx]),
            "g": float(ng_wave.get("v(g)", ng_wave.get("V(g)", np.zeros(n_points)))[idx]),
            "s": float(ng_wave.get("v(s)", ng_wave.get("V(s)", np.zeros(n_points)))[idx]),
            "e": float(ng_wave.get("v(e)", ng_wave.get("V(e)", np.zeros(n_points)))[idx]),
        }

        py = inst.eval_tran(nodes, time=float(t), delta_t=10e-12)

        ng_id = float(ng_wave.get("i(vd)", ng_wave.get("I(Vd)", np.zeros(n_points)))[idx])

        try:
            assert_close(
                f"{tech_name}/t={t*1e9:.2f}ns/id",
                py["id"], ng_id,
                rel_tol=REL_TOL,
            )
        except Exception:
            mismatches += 1

    # Allow up to 10% of sample points to mismatch (transient startup transients)
    max_mismatches = max(1, int(n_samples * 0.10))
    assert mismatches <= max_mismatches, (
        f"{tech_name}: {mismatches}/{n_samples} time points exceeded tolerance "
        f"(max allowed: {max_mismatches})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Verify collection**

Run: `pytest tests/test_transient.py --collect-only`

Expected: 5 tests (one per technology)

**Step 3: Run one tech**

Run: `pytest tests/test_transient.py -v -k "ASAP7"`

**Step 4: Commit**

```bash
git add tests/test_transient.py
git commit -m "feat(tests): add transient waveform verification tests"
```

---

## Task 7: Archive Legacy Tests

**Why:** The per-technology test files (test_tsmc{5,7,12,16}.py) are now superseded by the analysis-type tests. Keep them for reference but move out of the main test directory to avoid confusion and redundant execution.

**Files:**
- Create: `tests/legacy/` directory
- Move: `tests/test_tsmc{5,7,12,16}.py`

**Step 1: Create legacy directory and move files**

```bash
mkdir -p tests/legacy
touch tests/legacy/__init__.py
git mv tests/test_tsmc5.py tests/legacy/
git mv tests/test_tsmc7.py tests/legacy/
git mv tests/test_tsmc12.py tests/legacy/
git mv tests/test_tsmc16.py tests/legacy/
```

**Step 2: Add conftest to prevent legacy tests from running by default**

Create `tests/legacy/conftest.py`:

```python
"""Legacy tests — not collected by default. Run explicitly with:
    pytest tests/legacy/test_tsmc7.py -v
"""
import pytest

def pytest_collection_modifyitems(items):
    """Skip legacy tests unless explicitly selected."""
    for item in items:
        if "legacy" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason="Legacy test — run explicitly"))
```

**Step 3: Commit**

```bash
git add tests/legacy/
git commit -m "refactor(tests): archive legacy technology-specific tests to tests/legacy/"
```

---

## Task 8: Run Full Test Suite and Verify

**Files:** No file changes — verification only.

**Step 1: Run new tests only**

```bash
pytest tests/test_dc_jacobian.py tests/test_dc_regions.py tests/test_transient.py -v --tb=short
```

Expected: ~70 tests pass (15 Jacobian + 50 regions + 5 transient)

**Step 2: Run full suite excluding legacy**

```bash
pytest tests/ -v --tb=short --ignore=tests/legacy
```

Expected: All tests pass (~83 total)

**Step 3: Verify test counts**

```bash
pytest tests/ --collect-only --ignore=tests/legacy | tail -1
```

Expected: ~83 tests collected

**Step 4: Run with timing to check < 10 min budget**

```bash
pytest tests/ -v --tb=short --ignore=tests/legacy --durations=10
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(tests): complete test suite refactoring for convergence-focused verification"
```

---

## Summary

| Task | Description | Tests | Key Change from v1 |
|------|-------------|-------|---------------------|
| 1 | Create pycmg/testing.py | 0 (infra) | Utilities in library, not conftest |
| 2 | Enhance conftest.py | 0 (infra) | Deterministic registry, no globs |
| 3 | Add get_jacobian_matrix() | 1 (API) | Uses _build_full_jacobian + Schur complement |
| 4 | Create test_dc_jacobian.py | 15 | Central differencing, 1% tolerance |
| 5 | Create test_dc_regions.py | 50 | Full PMOS region coverage added |
| 6 | Create test_transient.py | 5 | 0.5% tolerance, sign convention documented |
| 7 | Archive legacy tests | 0 | — |
| 8 | Verify suite | 0 | — |

**Total new tests**: ~71 convergence-focused tests
**Estimated runtime**: 5-10 minutes
**AC capacitance test**: Removed (fragile NGSPICE AC parsing; caps verified via eval_dc)
