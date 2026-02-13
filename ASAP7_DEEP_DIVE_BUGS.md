# ASAP7-Specific Deep Dive Bug Report

## Date: 2026-02-13

## Critical Bugs Found

### Bug #1: Parameter Storage Case Sensitivity [HIGH]
**File**: `pycmg/ctypes_host.py:399` and `pycmg/ctypes_host.py:693`

**Issue**: Parameters are stored with original case from modelcard files, causing lookup failures.

**Root Cause**:
- ASAP7 modelcards use mixed case (e.g., `EOT`, `L`, `NFIN`, `NSD`)
- `parse_modelcard()` stores with original case: `parsed_params[key] = parsed`
- `_extract_model_params()` also stores with original case: `params[key] = parsed`
- But parameter access uses `_to_lower(key)` for comparisons

**Example**:
```python
# ASAP7 file has: +EOT = 1e-009
# Stored as: {"EOT": 1e-9}
# Later lookup: params.get("eot") → None (not found!)
```

**Impact**:
- Parameters with uppercase names cannot be retrieved
- Same parameter stored multiple times with different cases
- Silent failures where parameters use default values instead of modelcard values

**Fix**: Store all parameters with lowercase keys
```python
# Before:
parsed_params[key] = parsed

# After:
parsed_params[_to_lower(key)] = parsed
```

---

### Bug #2: nfin Default Value Never Stored [HIGH]
**File**: `pycmg/ctypes_host.py:400-405`

**Issue**: The `nfin` default value (1.0) is set but never stored back to `parsed_params`.

**Code**:
```python
if _to_lower(key) == "nf":
    parsed = 1.0  # Single-fin default
    parsed_params[key] = parsed  # ✓ Stored
if _to_lower(key) == "nfin":
    parsed = 1.0  # Single-fin default
    # ✗ Never stored!
```

**Impact**:
- When `nfin` is not specified in modelcard, no default is applied
- OSDI uses wrong value (likely 0 or undefined)
- Incorrect current calculations

**Fix**: Store the `nfin` default value
```python
if _to_lower(key) == "nfin":
    parsed = 1.0  # Single-fin default
    parsed_params[_to_lower(key)] = parsed  # Store it!
```

---

### Bug #3: ASAP7 PMOS DEVTYPE Parameter [MEDIUM]
**File**: `tech_model_cards/ASAP7/README.md:128`

**Issue**: ASAP7 PMOS models exhibit inverted behavior due to missing or incorrect `DEVTYPE` parameter.

**Details**:
- BSIM-CMG uses `DEVTYPE = 1` for NMOS, `DEVTYPE = 0` for PMOS
- Standard ASAP7 files (`7nm_TT_160803.pm`) do NOT have `devtype` for NMOS
- Standard ASAP7 PMOS also missing `devtype`
- Only `7nm_TT_160803_with_devtype.pm` has explicit `devtype = 0.0` for PMOS

**Impact**:
- PMOS conducts at positive Vg instead of negative Vg
- Inverted current direction
- Incorrect circuit behavior for PMOS devices

**Workaround**:
- Use `7nm_TT_160803_with_devtype.pm` for PMOS testing
- Or add `devtype = 0.0` to PMOS models in post-processing

**Investigation Needed**:
- Determine correct DEVTYPE value for ASAP7 v107
- Check if BSIM-CMG v107 has different DEVTYPE semantics than v106.1

---

### Bug #4: Test Infrastructure Path Configuration [LOW]
**File**: `tests/test_asap7.py:34`

**Issue**: Hardcoded path doesn't match actual directory structure.

**Fix**: Changed from `asap7_pdk_r1p7/models/hspice` to `ASAP7`

---

## Test Coverage Gaps

### Missing PMOS Tests
The test file (`test_asap7.py`) only tests NMOS devices:
- `test_asap7_tt_corner`: Uses `nmos` in regex search (line 244)
- `test_asap7_pvt_corners`: Only looks for `nmos` with `lvt` (line 301)
- No PMOS verification tests exist

**Recommendation**: Add PMOS tests once DEVTYPE issue is resolved

---

## ASAP7-Specific Parameter Parsing Challenges

### 1. Scientific Notation with Spaces
ASAP7 uses: `+eot     = 1e-009          eotbox  = 1.4e-007`

The regex correctly handles this, but only if spaces are preserved.

### 2. Multi-parameter Lines
ASAP7 puts multiple parameters on one line:
```
+version = 107             bulkmod = 1               igcmod  = 1               igbmod  = 0
```

The `finditer` approach correctly handles this.

### 3. EOTACC Threshold
ASAP7 uses `eotacc = 1e-010` (exactly 1.0e-10), which is below the OSDI minimum.
The code clamps to `1.1e-10`, which is correct.

### 4. Large Exponents
ASAP7 has: `nbody = 1e+022`, `nsd = 2e+026`

The regex `[0-9eE+\-\.]+` matches these correctly, but the previous bug
where the suffix pattern was outside the capture group would have broken this.

---

## Verification Plan

1. Fix parameter storage to use lowercase keys
2. Fix nfin default value storage
3. Add PMOS tests with DEVTYPE investigation
4. Verify all ASAP7 NMOS models work correctly
5. Document DEVTYPE resolution for PMOS

---

## Priority Summary

| Bug | Priority | Impact | Fix Complexity |
|-----|----------|--------|----------------|
| #1: Case sensitivity | HIGH | Silent failures | Low (1 line) |
| #2: nfin not stored | HIGH | Wrong currents | Low (1 line) |
| #3: PMOS DEVTYPE | MEDIUM | PMOS broken | Medium (investigation) |
| #4: Test path | LOW | Tests don't run | Low (1 line) |
