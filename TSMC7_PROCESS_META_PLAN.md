# TSMC7 Process & Meta-Parameter Implementation Plan

## Executive Summary

The current `tsmc7_simple.l` modelcard is a minimal BSIM-CMG level 72 model with 315 parameters. Analysis reveals significant gaps for production-ready 7nm FinFET technology support.

## Current State Analysis

### Present Parameters (315 total)

#### Core Geometry Parameters
| Parameter | Value | Status | Notes |
|------------|-------|--------|--------|
| `tfin` | 6e-9 | ✅ Set | Fin thickness (8nm nominal for 7nm) |
| `nbody` | 1e+23 | ✅ Set | Body doping concentration |
| `nseg` | 5 | ✅ Set | Number of segments |
| `hfin` | 4.2e-8 | ✅ Set | Fin height |
| `xl` | 1e-8 | ✅ Set | Length offset |
| `deltaw` | 0 | ✅ Set | L width variation |

#### Missing Critical Parameters
| Parameter | Priority | Impact |
|------------|----------|--------|
| `NF` | HIGH | Number of fins - **NOT SET** |
| `NFIN` | HIGH | Fin count parameter - **NOT SET** |
| `version` | HIGH | OSDI version flag - **NOT SET** |

**Issue**: Without NF and NFIN, the model cannot properly calculate multi-fin device behavior. Instance-level NFIN parameter may be silently ignored.

### Threshold & Subthreshold Parameters
| Parameter | Value | Category | Status |
|------------|-------|----------|--------|
| `dvt0` | 0.05 | Vth at Vbs=0 | ✅ Set |
| `dvt1` | 0.4 | Vth roll-off | ✅ Set |
| `phin` | 0.05 | Surface potential | ✅ Set |
| `eta0` | 0.06 | Subthreshold swing | ✅ Set |
| `dsub` | 0.35 | Subthreshold swing | ✅ Set |
| `k1rsce` | 0 | Reverse short effect | ✅ Set |

**Gap**: No temperature dependence in Vth parameters (no DVT1WF, DVT1WF, etc.)

### Mobility & Velocity Saturation
| Parameter | Value | ASAP7 Reference | Status |
|------------|-------|-----------------|--------|
| `u0` | 0.03 | 0.0283 | ✅ Matches (nominal NMOS) |
| `etamob` | 2.0 | 2.0 | ✅ Set |
| `vsat` | 150000 | 70000-80000 (depends) | ⚠️ **Differs** |
| `deltavsat` | 0.24 | 0.3 | ⚠️ **Differs** |
| `ksativ` | 2 | 2 | ✅ Set |
| `mexp` | 4 | 4 | ✅ Set |
| `rdsw` | 15 | 200 | ⚠️ **Too low** (13× lower) |

**Impact**: Lower rdsw may overestimate series resistance, affecting current accuracy.

### Gate Leakage & Tunneling
| Parameter | Value | ASAP7 Ref | Status |
|------------|-------|----------|--------|
| `igcmod` | 1 | 1 | ✅ Gate IGC model enabled |
| `aigc` | 0.022 | 0.022 | ✅ Set |
| `bigc` | 0.005 | 0.005 | ✅ Set |
| `cigc` | 0.25 | 0.25 | ✅ Set |
| `dlcigs` | 1e-9 | 1e-9 | ✅ Set |
| `dlcigd` | 1e-9 | 1e-9 | ✅ Set |

### Capacitance & Junction
| Parameter | Value | Category | Status |
|------------|-------|----------|--------|
| All Cgg, Cgd, etc. | 0 | **Disabled** | Models OFF but typical values exist |
| `cgso`, `cgdo` | 1.18e-12 | Set | ✅ Overlap capacitances |
| `cj`, `cjsw` | 0.587, 0.229 | ✅ Set | Junction capacitance |
| All PBS params | Set | ✅ | Pocket/bulk source parameters |

### Temperature Effects
| Parameter | Value | Status |
|------------|-------|--------|
| `tnom` | 25 | ✅ Set | Nominal temperature |
| `kt1`, `kt1l` | 0 | ✅ Set | Temperature coefficients |
| `ute`, `utl` | -0.7 | -0.7 | ⚠️ **Different** | Usually -1.5 |
| `tpb`, `tpbsw` | 5.56e-4 | Set | ✅ |

### Process & Meta-Parameters Status

| Category | Current State | Required |
|----------|--------------|----------|
| **Process Corners** | ❌ None | SS, FF parameters missing |
| **Meta-Parameters** | ❌ None | No version flag, no global flags |
| **Aging/Reliability** | ❌ None | No HCI, NBTI, BTI models |
| **Stress Effects** | ❌ None | Self-heating disabled |

## Implementation Strategy

### Phase 1: Add Missing Core Geometry Parameters (HIGH PRIORITY)

**Objective**: Enable multi-fin device support

1. **Add `NF` parameter**
   - Maps to BSIM-CMG: `nfin` (number of fins)
   - Default: 1 (single-fin operation)
   - Required for: Effective width calculation: `Weff = NF * NFIN * Wfin`
   - Implementation: Add to modelcard and support in ctypes_host parameter parsing

2. **Add `NFIN` parameter**
   - Maps to BSIM-CMG: Number of fins
   - Default: 1 (single fin)
   - Required for: Current calculation: `Ids = Id * NF`
   - Implementation: Add to modelcard, support in Instance parameter parsing

**Verification**: Test with NF=2, NFIN=2 produces correct Id scaling (2×, 4×).

### Phase 2: Implement Process Corners (HIGH PRIORITY)

**Objective**: Add SS (Slow-Slow) and FF (Fast-Fast) process variation

#### Corner Parameters to Add

**SS Corner (Slow-Slow)**: Lower drive current, higher Vth
```python
ss_params = {
    # Threshold voltage shifts
    "dvt0": 0.06,      # +0.01 from TT
    "dvt1": 0.45,      # +0.05 from TT

    # Mobility degradation
    "u0": 0.025,        # -17% from TT (0.03)
    "vsat": 120000,      # -20% from TT (150000)
    "rdsw": 18,          # +20% from TT (15)
    "rshs": 170,         # +20% from TT (142)
    "rshd": 170,
}
```

**FF Corner (Fast-Fast)**: Higher drive current, lower Vth
```python
ff_params = {
    # Threshold voltage shifts
    "dvt0": 0.04,      # -0.01 from TT
    "dvt1": 0.35,      # -0.05 from TT

    # Mobility enhancement
    "u0": 0.035,        # +17% from TT (0.03)
    "vsat": 180000,      # +20% from TT (150000)
    "rdsw": 12,          # -20% from TT (15)
    "rshs": 114,         # -20% from TT (142)
    "rshd": 114,
}
```

#### Implementation Approach

**Option A: Multiple Model Variants**
- Create separate files: `tsmc7_simple_tt.l`, `tsmc7_simple_ss.l`, `tsmc7_simple_ff.l`
- Use `select_tsmc7_variant()` to choose appropriate variant based on test
- Update pytest tests to parametrize across corners

**Option B: Parameter Substitution (Flexible)**
- Bake process parameters into modelcard at runtime
- Use `pytest.fixture` or environment variable for corner selection
- Update `ctypes_host.py` to support corner-aware parameter handling

**Verification**: SS corner should show ~25-30% lower Id, FF corner ~25-30% higher Id at nominal bias.

### Phase 3: Add Meta-Parameters & Advanced Features (MEDIUM PRIORITY)

**Objective**: Enable temperature dependence, aging, and advanced physical effects

#### Meta-Parameters to Add

1. **Version Flag**: `version = 107`
    - Purpose: OSDI compatibility
    - Enables proper parameter handling in newer OSDI versions
    - Implementation: Add to modelcard, read in ctypes_host

2. **Global Flags**: ` sweconf = 1` (switch configuration)
    - Purpose: Enable self-heating
    - Implementation: Add to modelcard, support in simulation

3. **Self-Heating**: `shmod = 1`
    - Purpose: Enable lattice heating effect
    - Implementation: Add to modelcard, support in simulation
    - Note: Requires `sweconf = 1` and `swh` parameter

4. **NQS (Non-Quasi-Static)**: `nqsmod = 1`
    - Purpose: Enable non-quasi-static charge trapping
    - Implementation: Add to modelcard, support in simulation

5. **Temperature Coefficients**: `kt1`, `kt1l`, `ute`, `utl`
    - Purpose: Improve temperature modeling
    - Values: Consult TSMC7 7nm PDK documentation
    - Implementation: Add to modelcard

#### Diode/Ideal Diode Parameters (Current State: OFF)

**Current**: Diode modeling disabled (all I-V related)
**Recommendation**: Consider enabling for advanced leakage modeling:
- `ijthdf` = 0.001   (Forward diode ideality factor)
- `ijthsf` = 3.0       (Reverse diode ideality factor)
- `asplit` = 0.1        (Diode splitting effect)
- `bvs` = 10           (Source-bulk junction grading)
- `cbs` = 1            (Source-bulk Schottky barrier)
- `cbs` varies by: `nbsem`, `pbsem`, `nbulkc`

**Note**: These parameters interact with gate leakage models for comprehensive I-V characterization.

### Phase 4: Parameter Consistency & Documentation (MEDIUM PRIORITY)

**Objective**: Align model parameters with industry standards and add comprehensive documentation

#### Tasks

1. **Fix Mobility Parameters**
   - Update `rdsw` from 15 to match ASAP7 nominal (~200)
   - Adjust `vsat`, `deltavsat` for consistent temperature behavior
   - Verify `etamob`, `mexp`, `ksativ` against ASAP7 reference

2. **Add Temperature Dependence**
   - Add `dvt1wf` (Vth vs. temp coefficient)
   - Add `dvt1wf` (DVT1 vs. temp width effect)
   - Add `kt1` (Boltzmann * k) and `kt1l` (lattice coefficient)
   - Add `ute`, `utl` (temperature effect on mobility)

3. **Enhance Modelcard Documentation**
   - Add parameter descriptions and units
   - Document default values and physical meanings
   - Include equations where appropriate
   - Reference BSIM-CMG specification and TSMC7 7nm PDK

4. **Update `ctypes_host.py`**
   - Ensure all new parameters are parsed and passed to OSDI
   - Support corner-specific parameter substitution
   - Add parameter validation (e.g., NF > 0)

### Testing Strategy

#### Unit Tests
```python
# Test geometry scaling
assert NF * NFIN == Weff  # Check effective width

# Test corner parameter application
test_ss_params = {"u0": 0.025, "rdsw": 18}
test_ff_params = {"u0": 0.035, "rdsw": 12}
```

#### Integration Tests
```bash
# Nominal TT corner
pytest tests/test_tsmc7_verification.py::test_tsmc7_tt_corner

# SS corner (low current)
pytest tests/test_tsmc7_verification.py -k ss

# FF corner (high current)
pytest tests/test_tsmc7_verification.py -k ff
```

### Risk Assessment

| Risk | Impact | Mitigation |
|-------|----------|------------|
| **Parameter count** | Model complexity | Use comprehensive test suite |
| **Corner models** | Testing overhead | Parametrize tests; run all on-demand |
| **Temperature modeling** | Accuracy variation | Start with basic; add advanced if needed |
| **Documentation** | User confusion | Invest in clear parameter descriptions |
| **TSMC7 IP** | Model accuracy | Follow BSIM-CMG spec; validate against PDK |

### Estimated Effort

| Phase | Tasks | Effort |
|--------|-------|--------|
| Phase 1 (Core params) | 2 params | 1-2 hours |
| Phase 2 (Process corners) | 3 model variants | 2-4 hours |
| Phase 3 (Meta & Advanced) | 5-10 params | 4-8 hours |
| Phase 4 (Documentation) | Model updates | 2-3 hours |
| **Total** | **12-18 hours** |

### Open Questions for TSMC7 Team

1. **PDK Documentation**: Does TSMC7 provide parameter values for SS/FF corners?
2. **Reference Values**: What are the target values for rdsw, vsat at 7nm?
3. **Measurement Data**: Can we get measured data for validation?
4. **Priority**: Should we focus on basic corner support first, or are advanced features needed immediately?

## Recommendation

**Implement in phases starting with Phase 1** (NF, NFIN parameters). This unblocks multi-fin device support and is foundational for all other features. Process corners (Phase 2) can be added incrementally with separate model variants.

**Next Step**: Begin Phase 1 by adding `NF` and `NFIN` parameters to `tsmc7_simple.l` and updating `ctypes_host.py` parser.
