# ASAP7 Modelcard Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend modelcard parsing to support multi-model files and BSIM-CMG via `nmos/pmos` `level=72`, with optional targeted model selection and a reproducible ASAP7 verification against NGSPICE.

**Architecture:** Keep parsing self-contained in `pycmg/ctypes_host.py` with a minimal state machine over `.model` blocks. Tests cover parsing behavior, while a standalone repro script uses existing NGSPICE helpers for one-point validation.

**Tech Stack:** Python 3.13, pytest, pycmg, NGSPICE via `scripts/deep_verify.py` helpers.

### Task 1: Add Failing Tests For Modelcard Parsing

**Files:**
- Create: `tests/test_parse_modelcard.py`

**Step 1: Write the failing test**

```python
from __future__ import annotations

import inspect
from pathlib import Path

import pycmg


def write_modelcard(path: Path, text: str) -> None:
    path.write_text(text)


def test_parse_modelcard_targets_level72_nmos(tmp_path: Path) -> None:
    card = tmp_path / "asap7.pm"
    write_modelcard(
        card,
        """
* header
.model nmos_lvt nmos level=72 l=14n tfin=7n
+ tox=1.5n
.model pmos_lvt pmos level=72 l=16n tfin=8n
+ tox=2.0n
""",
    )
    parsed = pycmg.ctypes_host.parse_modelcard(str(card), target_model_name="nmos_lvt")
    assert parsed.name == "nmos_lvt"
    assert parsed.params["l"] == pycmg.ctypes_host.parse_number_with_suffix("14n")
    assert parsed.params["tfin"] == pycmg.ctypes_host.parse_number_with_suffix("7n")
    assert parsed.params["tox"] == pycmg.ctypes_host.parse_number_with_suffix("1.5n")


def test_parse_modelcard_first_valid_when_no_target(tmp_path: Path) -> None:
    card = tmp_path / "multi.pm"
    write_modelcard(
        card,
        """
.model nmos_bad nmos level=71 l=10n
.model first_ok bsimcmg l=20n
+ tfin=9n
.model later_ok nmos level=72 l=30n
""",
    )
    parsed = pycmg.ctypes_host.parse_modelcard(str(card))
    assert parsed.name == "first_ok"
    assert parsed.params["l"] == pycmg.ctypes_host.parse_number_with_suffix("20n")
    assert parsed.params["tfin"] == pycmg.ctypes_host.parse_number_with_suffix("9n")


def test_model_init_signature_has_model_card_name() -> None:
    sig = inspect.signature(pycmg.Model.__init__)
    assert "model_card_name" in sig.parameters
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_modelcard.py -v`

Expected: FAIL because `parse_modelcard` does not accept `target_model_name` and Model lacks `model_card_name` parameter.

### Task 2: Implement `parse_modelcard` Enhancements

**Files:**
- Modify: `pycmg/ctypes_host.py`

**Step 1: Write minimal implementation**

Update signature to:

```python
def parse_modelcard(path: str, target_model_name: Optional[str] = None) -> ParsedModel:
```

Implement a state machine that:
- Parses `.model` lines and extracts name/type plus inline params.
- Accepts model if type is `bsimcmg` or (`nmos`/`pmos` and `level=72`).
- If `target_model_name` is set, select only the matching model (case-insensitive).
- If no target, select the first valid model and stop when its block ends.
- Parses inline params from `.model` and continuation lines (`+ ...`).
- Preserves existing `EOTACC` clamping behavior.

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_parse_modelcard.py -v`

Expected: PASS

### Task 3: Update `Model` Interface To Accept `model_card_name`

**Files:**
- Modify: `pycmg/ctypes_host.py`

**Step 1: Update constructor signature**

```python
class Model:
    def __init__(self, osdi_path: str, modelcard_path: str, model_name: str,
                 model_card_name: Optional[str] = None) -> None:
```

**Step 2: Pass target name to parser**

```python
if modelcard_path:
    parsed = parse_modelcard(modelcard_path, model_card_name)
    self._modelcard_params = dict(parsed.params)
```

**Step 3: Run tests**

Run: `pytest tests/test_parse_modelcard.py -v`

Expected: PASS

### Task 4: Add ASAP7 Reproduction Script With NGSPICE Check

**Files:**
- Create: `reproduce_asap7.py`

**Step 1: Write the script**

```python
from __future__ import annotations

from pathlib import Path

from scripts import deep_verify
from pycmg import ctypes_host


def main() -> None:
    modelcard = Path("/home/shenshan/pycmg-wrapper/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm")
    model_name = "nmos_lvt"
    parsed = ctypes_host.parse_modelcard(str(modelcard), target_model_name=model_name)
    print(f"model={parsed.name}")
    for key in ("l", "tfin"):
        if key in parsed.params:
            print(f"{key}={parsed.params[key]}")
        else:
            print(f"{key}=<missing>")

    inst_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}
    ng = deep_verify.run_ngspice_op_point(
        modelcard,
        model_name,
        inst_params,
        vd=0.7,
        vg=0.7,
        vs=0.0,
        ve=0.0,
        out_dir=Path("build-deep-verify") / "ngspice_eval" / "asap7_op",
        temp_c=27.0,
    )
    eval_fn = deep_verify.make_pycmg_eval(modelcard, model_name, inst_params, temp_c=27.0)
    py = eval_fn(0.7, 0.7, 0.0, 0.0)
    print(f"ngspice id={ng['id']} ig={ng['ig']} is={ng['is']} ib={ng['ib']}")
    print(f"pycmg   id={py[0]} ig={py[1]} is={py[2]} ib={py[3]}")


if __name__ == "__main__":
    main()
```

**Step 2: Run script**

Run: `python reproduce_asap7.py`

Expected: `model=nmos_lvt` with `l`/`tfin` values printed and NGSPICE vs pycmg currents shown.

### Task 5: Full Test Run

**Step 1: Run full test suite**

Run: `pytest`

Expected: PASS (with existing single skip).

### Task 6: Commit

**Step 1: Commit changes**

```bash
git add tests/test_parse_modelcard.py pycmg/ctypes_host.py reproduce_asap7.py docs/plans/2026-02-07-asap7-modelcard-support.md
# include pytest.ini/tests/conftest.py if modified in this branch

git commit -m "feat: support targeted BSIM-CMG modelcard parsing"
```
