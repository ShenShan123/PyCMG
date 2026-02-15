# ASAP7 Test Script Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move NGSPICE verification and robustness test logic into `tests/`, delete redundant `scripts/` testing code, add modelcard-path support, and run full ASAP7 process-corner DC/AC/TRAN verification against NGSPICE.

**Architecture:** Extract shared verification utilities into `tests/verify_utils.py`, convert existing scripts into pytest-compatible helpers, and keep pytest tests as the entrypoints. Heavy ASAP7 corner verification is implemented as a pytest test that iterates all HSPICE modelcards and runs DC/AC/TRAN comparisons with NGSPICE.

**Tech Stack:** Python 3.13, pytest, NGSPICE, pycmg, OSDI binary.

### Task 1: Inventory And Scaffolding

**Files:**
- Create: `tests/verify_utils.py`
- Modify: `tests/conftest.py`

**Step 1: Write failing test**

Create a new test that imports `tests.verify_utils` and asserts it exposes a minimal API, e.g. `run_ngspice_op_point`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify_utils_import.py -v`
Expected: FAIL with import error.

**Step 3: Create minimal `tests/verify_utils.py`**

Add empty function stubs for the API needed by migrated tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_verify_utils_import.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/verify_utils.py tests/test_verify_utils_import.py tests/conftest.py

git commit -m "feat: scaffold verification utilities"
```

### Task 2: Migrate `scripts/deep_verify.py` Into `tests/verify_utils.py`

**Files:**
- Modify: `tests/verify_utils.py`
- Modify: `tests/test_deep_verify_backend.py`
- Modify: `tests/test_deep_verify_stress.py`
- Modify: `tests/test_pure_pycmg.py`
- Delete: `scripts/deep_verify.py`

**Step 1: Write failing test**

Update `tests/test_deep_verify_backend.py` to use `tests.verify_utils` instead of invoking `scripts/deep_verify.py` as a subprocess. Ensure it fails before implementation.

**Step 2: Implement utilities**

Move the core logic from `scripts/deep_verify.py` into `tests/verify_utils.py`:
- Constants: `ROOT`, `BUILD`, `OSDI_PATH`, `NGSPICE_BIN`.
- `ensure_modelcard`, `parse_modelcard_params`, `parse_instance_params`, `write_instance_netlist`.
- NGSPICE runner and CSV parsing helpers.
- `make_pycmg_eval` / `make_pycmg_eval_tran`.
- DC/AC/TRAN comparison functions.

Add modelcard-path support:
- All APIs accept a `modelcard` path explicitly.
- Add helper to iterate `tech_model_cards/ASAP7/*.pm`.

**Step 3: Update tests to call the new utilities**

Replace subprocess usage with direct calls. Keep behavior the same.

**Step 4: Run tests**

Run:
- `pytest tests/test_deep_verify_backend.py -v`
- `pytest tests/test_deep_verify_stress.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/verify_utils.py tests/test_deep_verify_backend.py tests/test_deep_verify_stress.py tests/test_pure_pycmg.py

git commit -m "refactor: move deep verify into tests"
```

### Task 3: Migrate `scripts/test_robustness.py` Into `tests/verify_utils.py`

**Files:**
- Modify: `tests/verify_utils.py`
- Modify: `tests/test_robustness_helpers.py`
- Delete: `scripts/test_robustness.py`

**Step 1: Write failing test**

Update `tests/test_robustness_helpers.py` to import and use `tests.verify_utils` (instead of scripts) and verify it fails before implementation.

**Step 2: Implement migration**

Move `pulse_value`, `run_pulse_test`, `run_param_sweep`, and helper functions into `tests/verify_utils.py`.

**Step 3: Run tests**

Run: `pytest tests/test_robustness_helpers.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/verify_utils.py tests/test_robustness_helpers.py

git commit -m "refactor: move robustness helpers into tests"
```

### Task 4: Add Full ASAP7 Corner Verification Test (DC/AC/TRAN)

**Files:**
- Create: `tests/test_asap7_full_verify.py`
- Modify: `tests/verify_utils.py`

**Step 1: Write failing test**

Create `tests/test_asap7_full_verify.py` that:
- Enumerates `tech_model_cards/ASAP7/*.pm`.
- For each modelcard, runs DC/AC/TRAN comparisons using `verify_utils` with NGSPICE.
- Uses a fixed set of instance params `{L, TFIN, NFIN}` and one or two operating points.

Expect FAIL due to missing APIs.

**Step 2: Implement helper APIs**

Add to `verify_utils`:
- `iter_asap7_modelcards()`.
- `run_full_corner_verify(modelcard_path, model_name)`.

Implement the per-corner DC/AC/TRAN NGSPICE comparisons using existing compare functions.

**Step 3: Run test**

Run: `pytest tests/test_asap7_full_verify.py -v`
Expected: PASS (heavy runtime)

**Step 4: Commit**

```bash
git add tests/test_asap7_full_verify.py tests/verify_utils.py

git commit -m "feat: add full ASAP7 corner verification"
```

### Task 5: Remove `scripts/` Testing Logic

**Files:**
- Delete: `scripts/deep_verify.py`
- Delete: `scripts/test_robustness.py`
- Keep: `scripts/build_osdi.sh`, `scripts/clean_osdi.sh`

**Step 1: Delete redundant scripts**

Remove the migrated files.

**Step 2: Run full test suite**

Run: `pytest`
Expected: PASS

**Step 3: Commit**

```bash
git add -u

git commit -m "chore: remove redundant test scripts"
```

### Task 6: Final Verification

**Step 1: Run full test suite**

Run: `pytest`

Expected: PASS

**Step 2: Commit (if any changes)**

```bash
git add -u

git commit -m "test: verify asap7 full corner suite"
```
