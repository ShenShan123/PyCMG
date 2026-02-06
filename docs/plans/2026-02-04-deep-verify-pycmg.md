# Deep Verification + Stress Tests (pycmg) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `scripts/deep_verify.py` to support a `--backend` flag (pycmg vs osdi_eval), add pycmg evaluation path, and add stress tests comparing pycmg against osdi_eval.

**Architecture:** Keep `scripts/deep_verify.py` self-contained. Add a backend switch that routes evaluation through either the new `pycmg` in-memory path or the existing `osdi_eval` subprocess path. Add a stress test routine that randomly samples bias points and instance parameters and compares pycmg outputs to osdi_eval with existing tolerances.

**Tech Stack:** Python 3.11, PyBind11 extension `pycmg`, NGSPICE.

---

### Task 1: Add backend flag and pycmg evaluation path

**Files:**
- Modify: `scripts/deep_verify.py`
- Test: `tests/test_deep_verify_backend.py`

**Step 1: Write failing test for backend flag**

```python
from pathlib import Path
import subprocess


def test_deep_verify_backend_flag_rejected(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "deep_verify.py"
    result = subprocess.run([
        "python", str(script), "--backend", "pycmg", "--vg-start", "0", "--vg-stop", "0", "--vg-step", "1"
    ], capture_output=True, text=True)
    assert result.returncode != 2  # argparse error before flag exists
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deep_verify_backend.py -v`
Expected: FAIL (argparse does not know --backend)

**Step 3: Implement backend flag + pycmg path**

- Add `--backend {pycmg,osdi_eval}` default `pycmg`.
- Add `run_pycmg_eval(...)` using `pycmg.Model/Instance` and `eval_dc`.
- Route `compare_id_vg/compare_id_vd` to backend eval based on flag.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deep_verify_backend.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/deep_verify.py tests/test_deep_verify_backend.py
git commit -m "feat: add pycmg backend switch"
```

---

### Task 2: Add stress test comparing pycmg vs osdi_eval

**Files:**
- Modify: `scripts/deep_verify.py`
- Test: `tests/test_deep_verify_stress.py`

**Step 1: Write failing stress test**

```python
from pathlib import Path
import subprocess


def test_deep_verify_stress_runs() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "deep_verify.py"
    result = subprocess.run([
        "python", str(script), "--stress", "--stress-samples", "5"
    ], capture_output=True, text=True)
    assert result.returncode == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deep_verify_stress.py -v`
Expected: FAIL (missing --stress)

**Step 3: Implement stress test runner**

- Add `--stress` and `--stress-samples` (default small).
- Randomize Vd/Vg/Vs/Ve within 0–1.2 V and instance params within safe ranges.
- Compare pycmg vs osdi_eval for currents/charges/derivs/caps.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deep_verify_stress.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/deep_verify.py tests/test_deep_verify_stress.py
git commit -m "feat: add deep verify stress test"
```

---

### Task 3: Execute full OP/DC/AC suite with pycmg backend

**Files:**
- Modify: `scripts/deep_verify.py`

**Step 1: Run deep verification**

Run: `python scripts/deep_verify.py --backend pycmg`
Expected: PASS

**Step 2: Commit run log (optional)**

No commit required unless you want logs checked in.
