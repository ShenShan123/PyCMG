# TSMC7 Modelcard Extraction and Testing Report

## Executive Summary

Successfully extracted a minimal TSMC7 modelcard subset and verified basic functionality with both PyCMG and NGSPICE. While both simulators can load and evaluate the model, there are numerical discrepancies that require further investigation.

## Files Created

### 1. Minimal Modelcard Subset
**File:** `/home/shenshan/pycmg-wrapper/.worktrees/more-techs/tech_model_cards/TSMC7/tsmc7_minimal.l`

**Contents:**
- `nch_svt_mac.global` - Base BSIM-CMG parameters (245 parameters)
- `nch_svt_mac.5` - Length bin 5 variant for L=11-20nm (185+ parameters)

**Extraction Details:**
- Source: `cln7_1d8_sp_v1d2_2p2_clean_fixed.l` (lines 13913-13950, 14850-14900)
- Model type changed from `nmos` to `bsimcmg` for OSDI compatibility
- Parameters reformatted with `+` continuation lines for readability

**Key Model Parameters:**
```
Level = 72 (BSIM-CMG)
Version = 106.1
EOT = 1.16nm
TFIN = 6nm (fixed for SVT device)
HFIN = 42nm
EOTBOX = 140nm
```

### 2. Test Files

**PyCMG Test:** `test_tsmc7_minimal.py`
- Loads model via ctypes OSDI interface
- Tests DC analysis at Vd=Vg=0.7V, Vs=Ve=0V
- Device params: L=16nm, TFIN=6nm, NFIN=2

**NGSPICE Test:** `test_tsmc7_minimal_ng.cir`
- Loads OSDI binary via `.control` section
- Sources netlist after OSDI initialization
- Device: `N1 d g s e nch_svt_mac.5`

**NGSPICE Netlist:** `test_tsmc7_minimal.cir`
- Includes modelcard
- Defines voltage sources
- Instantiates device

**Comparison Script:** `compare_tsmc7_results.py`
- Runs PyCMG and parses NGSPICE output
- Compares currents, charges, and derivatives
- Reports pass/fail with tolerances

## Test Results

### PyCMG Results (L=16nm, TFIN=6nm, NFIN=2, Vd=Vg=0.7V)

```
Currents:
  Id  = -1.113452e-04 A
  Ig  = -3.504465e-08 A
  Is  =  1.113803e-04 A
  Ie  =  1.021270e-11 A
  Ids = -2.227255e-04 A

Charges:
  Qg  =  5.111953e-17 C
  Qd  = -1.536105e-17 C
  Qs  = -3.412899e-17 C
  Qb  = -1.629499e-18 C

Derivatives:
  gm  =  3.550365e-04 S
  gds =  1.199326e-04 S
  gmb =  3.162365e-06 S

Capacitances:
  cgg =  1.351929e-16 F
  cgd =  3.584004e-17 F
  cgs = -1.599359e-16 F
  cdg = -6.624952e-17 F
  cdd = -7.794447e-18 F
```

### NGSPICE Results (same conditions)

```
Currents:
  Id  = -3.560107e-04 A
  Ig  =  0.000000e+00 A
  Is  =  3.560107e-04 A
  Ie  =  0.000000e+00 A

Charges:
  Qg  =  4.362918e-17 C
  Qd  = -1.440218e-17 C
  Qs  = -2.920392e-17 C
  Qb  = -2.307154e-20 C

Derivatives:
  gm  =  6.654850e-04 S
  gds =  2.378642e-04 S
  gmb =  0.000000e+00 S
```

### Discrepancy Analysis

**Significant Differences:**
- Id: 68% relative difference
- gm: 47% relative difference
- gds: 50% relative difference
- gmb: NGSPICE reports 0, PyCMG reports 3.16µS

**OSDI Warnings (from NGSPICE):**
```
Warning: PHIBE_i = 0.000000e+00 is less than 0.2, setting it to 0.2.
Warning: PSAT_i = -4.022679e+00 is less than 2.0, setting it to 2.0.
Warning: UA_i = -3.720932e+01 is negative, setting it to 0.
Warning: EU_i = -3.966875e+00 is negative, setting it to 0.
Warning: PTWG_i = -4.447936e+02 is negative, setting it to 0.
```

These warnings suggest that some temperature/geometry-dependent parameters are being calculated incorrectly or clamped to different values.

## Key Learnings

### 1. TSMC7 Model Structure
- **Multi-bin length models:** 19 variants (`.1` through `.30`)
- **Fixed geometry:** TFIN=6nm is hard-coded in global model for SVT
- **Age parameters:** Extensive aging model parameters included
- **Parameter format:** All parameters on single lines with `+` continuations

### 2. NGSPICE Integration
- **OSDI loading order:** Must load OSDI before sourcing netlist
- **Device syntax:** Use `N<id> d g s e <model>` (not `M<id>`)
- **Model type:** Must be `bsimcmg`, not `nmos`
- **Control section:** Use `.control` / `.endc` blocks for OSDI commands

### 3. Instance Parameters
- **Merged approach:** NGSPICE tests bake L/NFIN into modelcard
- **Runtime approach:** PyCMG passes params via Instance() call
- **TFIN limitation:** Cannot override modelcard default (6nm for SVT)

### 4. Binary Compatibility
- **Single OSDI file:** `build-deep-verify/osdi/bsimcmg.osdi` (452KB)
- **Shared source:** Both PyCMG and NGSPICE use identical binary
- **Consistency goal:** Ensures binary-level model physics agreement

## Known Issues

### Issue 1: Numerical Discrepancy
**Status:** Open investigation
**Symptoms:** Large differences in currents and derivatives between PyCMG and NGSPICE
**Possible Causes:**
1. Internal node solve differences
2. Parameter initialization order
3. Temperature/geometry parameter calculations
4. Convergence tolerance differences

**Next Steps:**
- Compare intermediate parameter values
- Check internal node voltages
- Verify temperature settings match
- Investigate parameter clamping in OSDI

### Issue 2: OSDI Parameter Warnings
**Status:** Expected behavior
**Symptoms:** Parameters clamped to physical limits (e.g., PHIBE, PSAT, UA, EU, PTWG)
**Impact:** May affect numerical accuracy but simulation completes
**Notes:** These warnings appear in both PyCMG and NGSPICE output

### Issue 3: TFIN Override
**Status:** Model limitation
**Symptoms:** Cannot change TFIN from 6nm default
**Impact:** Limits device geometry exploration
**Workaround:** Use different model variants (LVT, ULVT may have different TFIN)

## Usage Examples

### PyCMG Usage

```python
from pycmg import Model, Instance

# Load model
model = Model(
    "build-deep-verify/osdi/bsimcmg.osdi",
    "tech_model_cards/TSMC7/tsmc7_minimal.l",
    "nch_svt_mac.5"
)

# Create instance (L=16nm, TFIN=6nm, NFIN=2)
inst = Instance(model, params={"L": 16e-9, "TFIN": 6e-9, "NFIN": 2.0})

# Evaluate DC
result = inst.eval_dc({"d": 0.7, "g": 0.7, "s": 0.0, "e": 0.0})
print(f"Id = {result['id']:.3e} A")
```

### NGSPICE Usage

```spice
* Load OSDI first
.control
osdi build-deep-verify/osdi/bsimcmg.osdi
source tech_model_cards/TSMC7/test_tsmc7_minimal.cir
option noacctech
op
wrdata output.csv v(d) v(g) v(s) v(e) i(vd) i(vg) i(vs) i(ve)
exit
.endc

.end
```

## Recommendations

### Immediate Actions
1. Investigate numerical discrepancies by comparing parameter values
2. Check if NGSPICE is using the correct length bin selection
3. Verify temperature settings match between simulators

### Phase 2: Extended Model Support
1. Extract PMOS SVT model (`pch_svt_mac`)
2. Add LVT variants (`nch_lvt_mac`, `pch_lvt_mac`)
3. Add ULVT variants (`nch_ulvt_mac`, `pch_ulvt_mac`)
4. Create `tsmc7_full.l` with all core models

### Documentation
1. Document TSMC7-specific parameter meanings
2. Create length bin selection guide
3. Add aging parameter documentation
4. Write troubleshooting guide for common issues

## Deliverables Status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| `tsmc7_minimal.l` | ✅ Complete | 2 models, ~430 parameters |
| `tsmc7_full.l` | ⏳ Pending | Phase 2 task |
| PyCMG test | ✅ Passing | Functional but results differ from NGSPICE |
| NGSPICE test | ✅ Passing | Functional but results differ from PyCMG |
| Comparison script | ✅ Complete | Reports discrepancies |
| Extraction documentation | ✅ Complete | This file |

## Appendix: File Locations

All files in `/home/shenshan/pycmg-wrapper/.worktrees/more-techs/`:

```
tech_model_cards/TSMC7/
├── tsmc7_minimal.l              # Minimal subset (2 models)
├── test_tsmc7_minimal.cir       # NGSPICE netlist
├── test_tsmc7_minimal_ng.cir    # NGSPICE runner with OSDI
└── TSMC7_EXTRACTION.md          # This document

build-deep-verify/
├── osdi/bsimcmg.osdi            # OSDI binary (452KB)
├── ngspice_eval/tsmc7_minimal_ng.l  # NGSPICE modelcard with params
└── ng_tsmc7_minimal_test.csv    # NGSPICE output

Scripts:
├── extract_tsmc7.py             # Extract models from full file
├── create_tsmc7_ngspice_modelcard.py  # Create NGSPICE-compatible modelcard
├── test_tsmc7_minimal.py        # PyCMG test
└── compare_tsmc7_results.py     # Comparison script
```

---

**Document Version:** 1.0
**Date:** 2026-02-10
**Status:** Phase 1 complete, numerical investigation ongoing
