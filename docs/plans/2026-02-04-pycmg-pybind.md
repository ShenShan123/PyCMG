# PyCMG PyBind11 Extension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a high-performance PyBind11 extension (`pycmg`) that exposes the trusted C++ OSDI host for in-memory model evaluation from Python.

**Architecture:** Implement thin PyBind11 bindings over the existing `osdi_host` C++ code. Provide `Model` and `Instance` types; `Instance` binds a simulation topology at construction using instance parameters. `set_params` detects topology changes via `process_params` and either updates in-place or requires explicit rebind.

**Tech Stack:** C++17, PyBind11, CMake, Python 3.

---

### Task 1: Add PyBind11 module scaffolding

**Files:**
- Create: `cpp/pycmg_bindings.cpp`
- Modify: `CMakeLists.txt`
- Create: `pycmg/__init__.py`

**Step 1: Write the failing import test**

```python
# tests/test_pycmg_import.py
import pycmg

def test_import():
    assert hasattr(pycmg, "Model")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pycmg_import.py -v`
Expected: FAIL (module not found)

**Step 3: Implement minimal PyBind11 module**

```cpp
// cpp/pycmg_bindings.cpp (minimal)
PYBIND11_MODULE(pycmg, m) {
  m.doc() = "pycmg OSDI bindings";
}
```

Update `CMakeLists.txt` to build a Python extension target named `pycmg` and link against `osdi_host` and pybind11.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pycmg_import.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cpp/pycmg_bindings.cpp CMakeLists.txt pycmg/__init__.py tests/test_pycmg_import.py
git commit -m "feat: add pycmg module scaffold"
```

---

### Task 2: Bind Model and Instance types (construction + lifetime)

**Files:**
- Modify: `cpp/pycmg_bindings.cpp`
- Modify: `cpp/osdi_host.h`
- Modify: `cpp/osdi_host.cpp`
- Test: `tests/test_pycmg_model_instance.py`

**Step 1: Write failing test for Model/Instance construction**

```python
import pycmg

def test_model_instance_construct(tmp_path):
    # use a real modelcard/osdi from build/ in repo
    model = pycmg.Model(osdi_path="build/bsimcmg.osdi", modelcard_path="bsim-cmg-va/benchmark_test/modelcard.lib", model_name="nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    assert inst is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pycmg_model_instance.py -v`
Expected: FAIL (Model/Instance not defined)

**Step 3: Implement Model/Instance bindings**

- Expose `Model` that loads `.osdi`, parses modelcard, and stores descriptor/model.
- Expose `Instance` that:
  - accepts `params` in constructor,
  - applies params to instance data,
  - calls `process_params` to compute internal nodes,
  - binds simulation (`bind_simulation`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pycmg_model_instance.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cpp/pycmg_bindings.cpp cpp/osdi_host.h cpp/osdi_host.cpp tests/test_pycmg_model_instance.py
git commit -m "feat: bind Model/Instance"
```

---

### Task 3: Implement eval API (dc currents, charges, conductances, caps)

**Files:**
- Modify: `cpp/pycmg_bindings.cpp`
- Modify: `cpp/osdi_host.h`
- Modify: `cpp/osdi_host.cpp`
- Test: `tests/test_pycmg_eval_dc.py`

**Step 1: Write failing test for eval outputs**

```python
import pycmg

def test_eval_dc_returns_fields():
    model = pycmg.Model("build/bsimcmg.osdi", "bsim-cmg-va/benchmark_test/modelcard.lib", "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
    for key in ("id", "ig", "is", "ie", "qg", "qd", "qs", "qb", "gm", "gds", "gmb", "cgg", "cgd", "cgs", "cdg", "cdd"):
        assert key in out
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pycmg_eval_dc.py -v`
Expected: FAIL (eval not implemented)

**Step 3: Implement eval_dc**

- Use existing OSDI evaluation flags (`CALC_RESIST_*`, `CALC_REACT_*`, `CALC_OP`) and reuse `osdi_host` helpers.
- Return a dict with currents (external residuals), charges (opvars), gm/gds/gmb, and condensed caps.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pycmg_eval_dc.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cpp/pycmg_bindings.cpp cpp/osdi_host.h cpp/osdi_host.cpp tests/test_pycmg_eval_dc.py
git commit -m "feat: add eval_dc outputs"
```

---

### Task 4: Topology-change detection and rebind behavior

**Files:**
- Modify: `cpp/osdi_host.h`
- Modify: `cpp/osdi_host.cpp`
- Modify: `cpp/pycmg_bindings.cpp`
- Test: `tests/test_pycmg_rebind.py`

**Step 1: Write failing test for topology change detection**

```python
import pycmg
import pytest

def test_set_params_detects_topology_change():
    model = pycmg.Model("build/bsimcmg.osdi", "bsim-cmg-va/benchmark_test/modelcard.lib", "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    with pytest.raises(RuntimeError):
        inst.set_params({"nfin": 6})
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pycmg_rebind.py -v`
Expected: FAIL (no error raised)

**Step 3: Implement detection strategy**

- On `set_params`, apply params to instance data.
- Call `process_params` to get new internal node list.
- If internal node count differs from currently bound simulation, raise unless `allow_rebind=True`.
- If `allow_rebind=True`, rebind simulation via `bind_simulation`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pycmg_rebind.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cpp/osdi_host.h cpp/osdi_host.cpp cpp/pycmg_bindings.cpp tests/test_pycmg_rebind.py
git commit -m "feat: detect topology change on set_params"
```

---

### Task 5: Documentation and usage snippet

**Files:**
- Modify: `CLAUDE.md`
- Create: `pycmg/README.md`

**Step 1: Write usage snippet**

```markdown
```python
import pycmg
model = pycmg.Model("build/bsimcmg.osdi", "bsim-cmg-va/benchmark_test/modelcard.lib", "nmos")
inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
res = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
print(res["id"], res["cgg"])
```
```

**Step 2: Run any basic test/ import**

Run: `pytest tests/test_pycmg_import.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add CLAUDE.md pycmg/README.md
git commit -m "docs: add pycmg usage"
```
