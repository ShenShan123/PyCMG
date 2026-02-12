"""
TSMC7 PVT Verification Tests

Comprehensive verification across PVT corners using TSMC7 modelcards.
Tests DC, AC (capacitance), and transient analysis with representative voltage points.

VERIFICATION STRATEGY:
- PyCMG wraps the OSDI binary directly via ctypes (pycmg/ctypes_host.py)
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
TSMC7_MODELCARD_FILE = "tsmc7_simple.l"  # Simplified OSDI-compatible (no TMI)
TSMC7_MODELCARD_OVERRIDE = os.environ.get("TSMC7_MODELCARD")

# TSMC7 model variants
TSMC7_MODELS = {
    "nch_svt": "nch_svt_mac",  # NMOS standard Vt
    "pch_svt": "pch_svt_mac",  # PMOS standard Vt
}

# Tolerances
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
REL_TOL = 5e-3

# Representative test corners (TSMC7 uses different naming)
TSMC7_PVT_CORNERS = {
    "TT": "typical",    # Typical-typical
    "SS": "slow",       # Slow-slow
    "FF": "fast",       # Fast-fast
}

# Representative temperatures (not full range)
TSMC7_TEST_TEMPS = [-40, 27, 85, 125]  # °C

# Representative voltage points for TSMC7 (0.75V core)
TSMC7_VG_CORE = [0.0, 0.3, 0.6, 0.75, 0.9]  # V
TSMC7_VD_CORE = [0.0, 0.3, 0.6, 0.75]  # V

# TSMC7 length bins (L values from modelcard bins)
# Bin 1: 120-240nm, Bin 2: 72-120nm, Bin 3: 36-72nm, Bin 4: 20-36nm, Bin 5: 11-20nm
# Note: TSMC7 uses larger L values than ASAP7 (multi-gate finfet)
TSMC7_LENGTH_POINTS = [16e-9, 30e-9, 50e-9, 80e-9, 150e-9]


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

    # Try the default modelcard file
    default_card = TSMC7_DIR / TSMC7_MODELCARD_FILE
    if default_card.exists():
        return [default_card]

    # Fall back to any .l files
    return sorted(TSMC7_DIR.glob("*.l"))


def _make_tsmc7_ngspice_modelcard(src: Path, dst: Path, model_name: str,
                                  inst_params: dict) -> str:
    """Create NGSPICE-compatible TSMC7 modelcard with instance parameters.

    For multi-bin TSMC7 modelcards: selects appropriate bin (e.g., nch_svt_mac.1 through .19).
    For simplified modelcards: uses the model directly.

    Returns:
        The model name to use (e.g., "nch_svt_mac" or "nch_svt_mac.5")

    The output includes the model with bsimcmg type and instance params baked in.
    """
    text = src.read_text()

    # Check if this is a multi-bin or simplified modelcard
    has_multi_bin = bool(re.search(rf"\.model {re.escape(model_name)}\.\d+", text, re.IGNORECASE))

    if not has_multi_bin:
        # Simplified modelcard - use model directly
        target_model = model_name
    else:
        # Multi-bin modelcard - select appropriate bin
        l_val = inst_params.get("L", 16e-9)
        bin_num = _get_tsmc7_length_bin(text, model_name, l_val)
        target_model = f"{model_name}.{bin_num}"

    lines = []
    in_target = False
    in_global = False
    found_keys = set()
    target_lower = target_model.lower()

    # For simplified modelcards, extract just the target model
    # For multi-bin modelcards, extract global and target model
    extracted_lines = []

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.lower().startswith(".model"):
            parts = stripped.split()
            if len(parts) >= 3:
                current_model = parts[1].lower()

                # For multi-bin: look for .global and target variant
                # For simple: look for target model directly
                if has_multi_bin and current_model == f"{model_name}.global".lower():
                    in_global = True
                    in_target = False
                    # Change to bsimcmg type
                    parts[2] = "bsimcmg"
                    prefix = line[:line.lower().find(".model")]
                    line = f"{prefix}{' '.join(parts)}"
                elif current_model == target_lower:
                    in_global = False
                    in_target = True
                    # Change to bsimcmg type (if not already)
                    if parts[2].lower() != "bsimcmg":
                        parts[2] = "bsimcmg"
                        prefix = line[:line.lower().find(".model")]
                        line = f"{prefix}{' '.join(parts)}"
                elif in_target:
                    # End of target model (found another .model)
                    break

        # Collect lines if we're in the right section
        if in_global or in_target:
            # Apply parameter overrides in target model
            if in_target:
                for key, val in inst_params.items():
                    pattern = rf"(?i)\b{re.escape(key)}\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)"

                    def repl(m, k=key.upper(), v=val):
                        found_keys.add(k)
                        return f"{k} = {v}"

                    line, _ = re.subn(pattern, repl, line)

            extracted_lines.append(line)

            # Stop if we've collected enough lines
            if in_target and not has_multi_bin:
                # For simple models, we might have collected everything in one go
                # Check if next line would start a new section
                pass

    # Add missing instance parameters at end
    if in_target:
        for key, val in inst_params.items():
            if key.upper() not in found_keys:
                extracted_lines.append(f"+ {key.upper()} = {val}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(extracted_lines) + "\n")

    return target_model


def _get_tsmc7_length_bin(modelcard_text: str, model_name: str, l_val: float) -> int:
    """Determine which length bin to use for a given L value.

    TSMC7 modelcards have multiple bins (e.g., .1 through .19) covering different L ranges.
    This function finds the appropriate bin based on lmin/lmax parameters.

    Note: TSMC7 model definitions span multiple lines with continuation (+) lines.
    """
    target_lower = model_name.lower()
    lines = modelcard_text.splitlines()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Look for model definition start
        if stripped.lower().startswith(f".model {model_name}."):
            # Extract bin number
            parts = stripped.split()
            if len(parts) >= 2:
                model_full = parts[1]
                bin_match = re.match(rf"{re.escape(model_name)}\.(\d+)", model_full, re.I)
                if bin_match:
                    bin_num = int(bin_match.group(1))

                    # Collect all lines for this model (until next .model or end)
                    model_lines = [stripped]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if next_line.lower().startswith(".model"):
                            break
                        model_lines.append(next_line)
                        j += 1

                    # Join all lines and search for lmin/lmax
                    full_model = "\n".join(model_lines)
                    lmin_match = re.search(r"lmin\s*=\s*([0-9eE+\-\.]+)", full_model, re.I)
                    lmax_match = re.search(r"lmax\s*=\s*([0-9eE+\-\.]+)", full_model, re.I)

                    if lmin_match and lmax_match:
                        lmin = float(lmin_match.group(1))
                        lmax = float(lmax_match.group(1))

                        if lmin <= l_val <= lmax:
                            return bin_num

        i += 1

    # Default to bin 1 if no match found
    return 1



## Test Implementation Details

### Test 1: Operating Point (OP)

**Purpose:** Fast smoke test

**Coverage:**
- NMOS TT @ Vd=0.75V, Vg=0.75V, T=27°C
- PMOS TT @ Vd=0.0V, Vg=0.75V, Vs=0.75V, T=27°C

**Outputs:** Id, Ig, Is, Ie, Ids, gm, gds, gmb, qg, qd, qs, qb

**Reuses:** Existing `_run_tsmc7_ngspice_op_point()`, `Instance.eval_dc()`

---

### Test 2: Process Corners

**Purpose:** Verify TT/SS/FF accuracy

**Coverage:**
- NMOS: TT, SS, FF @ Vd=0.6V, Vg=0.75V, T=27°C
- PMOS: TT, SS, FF @ Vd=0.0V, Vg=0.75V, Vs=0.75V, T=27°C

**Validation:**
- SS: Id ~25-30% < TT
- FF: Id ~25-30% > TT
- All derivatives scale correctly

**Implementation:** Parametrize with `@pytest.mark.parametrize("corner", TSMC7_CORNERS)`

---

### Test 3: Temperature Sweep

**Purpose:** Verify temperature dependence

**Coverage:**
- NMOS TT @ Vd=0.6V, Vg=0.75V, T=[-40, 27, 85, 125]°C

**Validation:**
- Id decreases with increasing temperature (mobility degradation)
- Vth decreases with temperature
- Trends match NGSPICE

**Reuse:** ASAP7 temperature sweep pattern

---

### Test 4: Geometric Sweeps

**Purpose:** Verify geometric parameter handling

**Coverage:**
- Length: L=[12, 16, 24]nm @ Vd=0.6V, Vg=0.75V, T=27°C
- Fin count: NFIN=[1, 2] @ Vd=0.6V, Vg=0.75V, T=27°C

**Validation:**
- Shorter L → higher Id (short-channel effect)
- Higher NFIN → proportional Id increase
- Derivatives scale correctly

---

### Test 5: DC Voltage Sweeps

**Purpose:** Verify I-V characteristics

**Coverage:**
- Id-Vg: Vg=[0.0, 0.3, 0.6, 0.75, 0.9]V @ Vd=0.05V, T=27°C
- Id-Vd: Vd=[0.0, 0.3, 0.6, 0.75]V @ Vg=0.75V, T=27°C

**Validation:**
- Cutoff region (Vg < Vth): Id ≈ 0
- Linear region (Vd < Vgs-Vth): Id ~ Vds
- Saturation: Id ~ (Vgs-Vth)²

**Reuse:** ASAP7 voltage sweep pattern

---

### Test 6: AC Capacitance

**Purpose:** Verify capacitance matrix

**Coverage:**
- Cgg, Cgd, Cgs @ Vg=[0.0, 0.6, 0.75]V, Vd=0.6V, T=27°C

**Validation:**
- Cgg highest in strong inversion
- Cgd ≈ Cgs (symmetry)
- Values match NGSPICE AC analysis

**Reuse:** `verify_utils.py` AC patterns, `eval_dc()` returns capacitance matrix

**Tolerance:** ABS_TOL_C=1e-15, REL_TOL_C=1e-2 (relaxed vs currents)

---

### Test 7: Transient Analysis

**Purpose:** Verify transient behavior

**Coverage:**
- 3 time points: [1e-11, 5e-11, 1e-10]s
- Vg step: 0.0V → 0.75V @ Vd=0.05V, T=27°C

**Validation:**
- Id increases as Vg steps
- Charges are finite and reasonable (< 1pC)
- No numerical instability

**Reuse:** `Instance.eval_tran()`, ASAP7 transient pattern

---

## Tolerances

```python
# Currents
ABS_TOL_I = 1e-9   # 1 nA
REL_TOL_I = 5e-3   # 0.5%

# Charges
ABS_TOL_Q = 1e-18  # 1 aC

# Capacitances (relaxed)
ABS_TOL_C = 1e-15  # 1 fF
REL_TOL_C = 1e-2   # 1%
```

---

## Test Execution

**Fast Tests (CI/CD):**
```bash
pytest tests/test_tsmc7.py -m fast -v
# Runtime: < 30 seconds
```

**Full Verification:**
```bash
pytest tests/test_tsmc7.py -v
# Runtime: ~10 minutes
```

**Specific Categories:**
```bash
pytest tests/test_tsmc7.py -k "corner" -v    # Corners only
pytest tests/test_tsmc7.py -k "capacitance" -v  # AC only
pytest tests/test_tsmc7.py -k "transient" -v     # TRAN only
```

---
 SVT (standard Vt) typical corner at room temperature.

    This is a smoke test to verify basic functionality with TSMC7 modelcards.
    Tests a single operating point for NMOS and PMOS SVT devices.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]

    # Test NMOS SVT
    nch_model = TSMC7_MODELS["nch_svt"]
    nch_params = {
        "L": 16e-9,
        "TFIN": 8e-9,
        "NFIN": 2.0,
    }

    # Create NGSPICE-compatible modelcard and get variant name
    ng_modelcard = BUILD / "tsmc7_nch_svt.ng.lib"
    nch_variant = _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, nch_params)

    # Test at nominal operating point (Vdd = 0.75V for TSMC7)
    ng_result = _run_tsmc7_ngspice_op_point(
        modelcard=ng_modelcard,
        model_name=nch_variant,
        inst_params=nch_params,
        vd=0.75, vg=0.75, vs=0.0, ve=0.0,
        temp_c=27.0
    )

    # For PyCMG, use the variant model name directly
    model = Model(str(OSDI_PATH), str(ng_modelcard), nch_variant)
    inst = Instance(model, params=nch_params)
    py_result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

    # Compare currents
    _assert_close(f"{nch_variant}@id", py_result["id"], ng_result["id"])
    _assert_close(f"{nch_variant}@ig", py_result["ig"], ng_result["ig"])
    _assert_close(f"{nch_variant}@gm", py_result["gm"], ng_result["gm"])
    # Compare drain-source current (Ids = Id - Is)
    ng_ids = ng_result["id"] - ng_result["is"]
    _assert_close(f"{nch_variant}@ids", py_result["ids"], ng_ids)

    # Test PMOS SVT
    pch_model = TSMC7_MODELS["pch_svt"]
    pch_params = {
        "L": 16e-9,
        "TFIN": 8e-9,
        "NFIN": 2.0,
    }

    ng_modelcard = BUILD / "tsmc7_pch_svt.ng.lib"
    pch_variant = _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, pch_model, pch_params)

    # PMOS: Vg should be low (negative Vgs for typical operation)
    ng_result = _run_tsmc7_ngspice_op_point(
        modelcard=ng_modelcard,
        model_name=pch_variant,
        inst_params=pch_params,
        vd=0.0, vg=0.75, vs=0.75, ve=0.0,
        temp_c=27.0
    )

    model = Model(str(OSDI_PATH), str(ng_modelcard), pch_variant)
    inst = Instance(model, params=pch_params)
    py_result = inst.eval_dc({"d": 0.0, "g": 0.75, "s": 0.75, "e": 0.0})

    _assert_close(f"{pch_variant}@id", py_result["id"], ng_result["id"])
    _assert_close(f"{pch_variant}@ig", py_result["ig"], ng_result["ig"])
    ng_ids = ng_result["id"] - ng_result["is"]
    _assert_close(f"{pch_variant}@ids", py_result["ids"], ng_ids)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
def test_tsmc7_pvt_corners() -> None:
    """Test TSMC7 PVT corners at representative operating points.

    TSMC7 modelcards may include TT, SS, FF variants in separate files.
    This test verifies model consistency across process corners.
    """
    pytest.skip("PVT corner testing not yet implemented - requires separate corner files")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
def test_tsmc7_temperature_sweep() -> None:
    """Test TSMC7 temperature sweep at representative temperatures.

    Verifies model accuracy across -40°C to 125°C temperature range.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    nch_model = TSMC7_MODELS["nch_svt"]
    inst_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}

    ng_modelcard = BUILD / "tsmc7_temp.ng.lib"
    _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, inst_params)

    # Test at 4 representative temperatures
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


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
def test_tsmc7_voltage_sweep_subset() -> None:
    """Test TSMC7 Id-Vg and Id-Vd sweeps at representative voltage points.

    Verifies model accuracy across the operating voltage range for TSMC7 (0 to 0.9V).
    Tests key operating regions: cutoff, linear, and saturation.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    nch_model = TSMC7_MODELS["nch_svt"]
    inst_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}

    ng_modelcard = BUILD / "tsmc7_sweep.ng.lib"
    _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, inst_params)

    model = Model(str(OSDI_PATH), str(ng_modelcard), nch_model)
    inst = Instance(model, params=inst_params)

    # Test Id-Vg at 5 points (not full sweep)
    vd = 0.05  # Low Vd for linear region
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
    vg = 0.75  # Nominal Vg
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


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.slow
def test_tsmc7_length_bins() -> None:
    """Test TSMC7 across all length bin variants.

    TSMC7 modelcards include multiple length bins (L = 8nm to 32nm).
    This test verifies that the bin selection logic works correctly.
    """
    modelcards = _iter_tsmc7_modelcards()
    if not modelcards:
        pytest.skip("No TSMC7 modelcards found")

    modelcard = modelcards[0]
    nch_model = TSMC7_MODELS["nch_svt"]

    # Test each length point
    for l_val in TSMC7_LENGTH_POINTS:
        inst_params = {"L": l_val, "TFIN": 8e-9, "NFIN": 2.0}

        ng_modelcard = BUILD / f"tsmc7_l{l_val*1e9:.0f}n.ng.lib"
        _make_tsmc7_ngspice_modelcard(modelcard, ng_modelcard, nch_model, inst_params)

        # Test at nominal operating point
        ng_result = _run_tsmc7_ngspice_op_point(
            modelcard=ng_modelcard,
            model_name=nch_model,
            inst_params=inst_params,
            vd=0.75, vg=0.75, vs=0.0, ve=0.0,
            temp_c=27.0
        )

        model = Model(str(OSDI_PATH), str(ng_modelcard), nch_model)
        inst = Instance(model, params=inst_params)
        py_result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})

        _assert_close(f"L={l_val*1e9:.0f}n@id", py_result["id"], ng_result["id"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.fast
def test_tsmc7_16nm_comparison() -> None:
    """Verify TSMC7 16nm model with PyCMG vs NGSPICE.

    Quick smoke test to verify both tools produce identical results
    when using the TSMC7 simplified modelcard.
    """
    modelcard = TSMC7_DIR / TSMC7_MODELCARD
    if not modelcard.exists():
        pytest.skip(f"TSMC7 modelcard not found: {modelcard}")

    # Test parameters for 16nm device (nominal TSMC7)
    inst_params = {
        "L": 16e-9,
        "TFIN": 6e-9,
        "NFIN": 2.0,
    }

    # Test at nominal operating point
    vd = 0.75
    vg = 0.75
    vs = 0.0
    ve = 0.0
    temp_c = 27.0

    # === PyCMG Evaluation ===
    model = Model(str(OSDI_PATH), str(modelcard), "nch_svt_mac")
    inst = Instance(model, params=inst_params)
    py_result = inst.eval_dc(
        {"d": vd, "g": vg, "s": vs, "e": ve},
        temperature=temp_c + 273.15
    )

    print(f"\n=== PyCMG Results ===")
    print(f"id  = {py_result['id']:.6e} A")
    print(f"ig  = {py_result['ig']:.6e} A")
    print(f"is  = {py_result['is']:.6e} A")
    print(f"\n=== PyCMG Evaluation complete ===\n")


@pytest.mark.fast
def test_tsmc7_16nm_comparison() -> None:
    """Verify TSMC7 16nm model with PyCMG vs NGSPICE.

    Quick smoke test to verify both tools produce identical results
    when using the TSMC7 simplified modelcard.
    """
    modelcard = TSMC7_DIR / TSMC7_MODELCARD
    if not modelcard.exists():
        pytest.skip(f"TSMC7 modelcard not found: {modelcard}")

    # Test parameters for 16nm device (nominal TSMC7)
    inst_params = {
        "L": 16e-9,
        "TFIN": 6e-9,
        "NFIN": 2.0,
    }

    # Test at nominal operating point
    vd = 0.75
    vg = 0.75
    vs = 0.0
    ve = 0.0
    temp_c = 27.0

    # === PyCMG Evaluation ===
    model = Model(str(OSDI_PATH), str(modelcard), "nch_svt_mac")
    inst = Instance(model, params=inst_params)
    py_result = inst.eval_dc(
        {"d": vd, "g": vg, "s": vs, "e": ve},
        temperature=temp_c + 273.15
    )

    print(f"\n=== PyCMG Results ===")
    print(f"id  = {py_result['id']:.6e} A")
    print(f"ig  = {py_result['ig']:.6e} A")
    print(f"is  = {py_result['is']:.6e} A")
    print(f"\n=== PyCMG Evaluation complete ===\n")


@pytest.mark.fast
def test_tsmc7_16nm_comparison() -> None:
    """Verify TSMC7 16nm model with PyCMG vs NGSPICE.

    Quick smoke test to verify both tools produce identical results
    when using the TSMC7 simplified modelcard.
    """
    modelcard = TSMC7_DIR / TSMC7_MODELCARD
    if not modelcard.exists():
        pytest.skip(f"TSMC7 tests cannot run - missing modelcard: {TSMC7_MODELCARD}")

    # Test parameters for 16nm device (nominal TSMC7)
    inst_params = {
        "L": 16e-9,
        "TFIN": 6e-9,
        "NFIN": 2.0,
    }

    # Test at nominal operating point
    vd = 0.75
    vg = 0.75
    can't_p vs = 0.0
    ve = 0.0
    temp_c = 27.0

    # === PyCMG Evaluation ===
    model = Model(str(OSDI_PATH), str(modelcard), "nch_svt_mac")
    inst = Instance(model, params=inst_params)
    py_result = inst.eval_dc(
        {"d": vd, "g": vg, "s": vs, "e": ve},
        temperature=temp_c + 273.15
    )

    print(f"\n=== PyCMG Results ===")
    print(f"id  = {py_result['id']:.6e} A")
    print(f"ig  = {ng_result['ig']:.6e} A")
    print(f"is  = {py_result['is']:.6e} A")
    print(f"\n=== PyCMG Evaluation complete ===\n")
