# Test Suite Refactoring Design

**Date**: 2026-02-14
**Status**: Approved
**Priority**: Convergence-focused verification for circuit simulator integration

## Executive Summary

Refactor the test suite to focus on convergence-critical verification:
1. **DC Jacobian verification** - Full matrix comparison for Newton-Raphson convergence
2. **Transient verification** - Full waveform comparison for time-domain accuracy
3. **Operating region coverage** - Comprehensive bias condition testing
4. **Capacitance verification** - C-V characteristics for transient stability

All tests run against 5 technologies: ASAP7, TSMC5, TSMC7, TSMC12, TSMC16.

## Design Decisions

### 1. Organization: Analysis-Type-Based

Tests organized by analysis type rather than technology:
- `test_dc_jacobian.py` - Full Jacobian matrix verification
- `test_dc_regions.py` - Operating region coverage
- `test_transient.py` - Full waveform comparison
- `test_ac_capacitance.py` - AC/capacitance verification

Each file uses `@pytest.mark.parametrize` to run against all 5 technologies.

### 2. DC Jacobian: Full Matrix Comparison

Compare the complete 4×4 Jacobian matrix:
```
J = [∂Id/∂Vd  ∂Id/∂Vg  ∂Id/∂Vs  ∂Id/∂Ve]
    [∂Ig/∂Vd  ∂Ig/∂Vg  ∂Ig/∂Vs  ∂Ig/∂Ve]
    [∂Is/∂Vd  ∂Is/∂Vg  ∂Is/∂Vs  ∂Is/∂Ve]
    [∂Ib/∂Vd  ∂Ib/∂Vg  ∂Ib/∂Vs  ∂Ib/∂Ve]
```

**PyCMG**: Extract from OSDI `resist_jacobian` buffer (requires new method `Instance.get_jacobian_matrix()`)

**NGSPICE**: Compute via numerical finite-difference perturbation:
- 1 base run + 4 perturbation runs = 5 NGSPICE calls per operating point
- `delta = 1e-6` V for linear approximation

### 3. Operating Regions: Voltage-Ratio Based

Define regions by voltage ratios (no Vth estimation needed):

| Region | Vgs | Vds | Physical Meaning |
|--------|-----|-----|------------------|
| Off-state | 0.0 | Vdd | Device fully off |
| Weak-inversion | 0.3 × Vdd | Vdd | Near threshold |
| Moderate-inversion | 0.6 × Vdd | 0.5 × Vdd | Transition region |
| Strong-linear | Vdd | 0.3 × Vdd | High Vgs, low Vds |
| Strong-saturation | 0.8 × Vdd | Vdd | High Vgs, high Vds |

### 4. Transient: Full Waveform Comparison

**NGSPICE**: Run transient simulation with pulse stimulus
- Time step: 10 ps
- Duration: 10 ns (1000 points)
- Outputs: V(t), I(t), Q(t)

**PyCMG**: Call `eval_tran()` at each NGSPICE time point
- Compare Id, Ig, Is, Ib waveforms
- Compare Qg, Qd, Qs, Qb waveforms

### 5. Capacitance: AC Analysis Extraction

Extract capacitance from NGSPICE AC analysis:
```
I = jωCV → C = Im(I) / (ω × V)
```

Test C-V characteristics across bias sweeps.

## File Structure

```
tests/
├── conftest.py                 # Enhanced fixtures + technology registry
├── test_api.py                 # Keep as-is (API smoke tests)
├── test_dc_jacobian.py         # NEW: Full Jacobian matrix verification
├── test_dc_regions.py          # NEW: Operating region coverage tests
├── test_transient.py           # NEW: Full waveform comparison
├── test_ac_capacitance.py      # NEW: AC/capacitance verification
├── test_integration.py         # Keep as quick sanity check
└── legacy/                     # Move old technology-specific tests
    ├── test_tsmc5.py
    ├── test_tsmc7.py
    ├── test_tsmc12.py
    └── test_tsmc16.py
```

## Technology Registry

```python
TECHNOLOGIES = {
    "ASAP7": {"dir": "ASAP7", "vdd": 0.9, "patterns": ["*.pm"]},
    "TSMC5": {"dir": "TSMC5/naive", "vdd": 0.65, "patterns": ["nch_*.l", "pch_*.l"]},
    "TSMC7": {"dir": "TSMC7/naive", "vdd": 0.75, "patterns": ["nch_*.l", "pch_*.l"]},
    "TSMC12": {"dir": "TSMC12/naive", "vdd": 0.80, "patterns": ["nch_*.l", "pch_*.l"]},
    "TSMC16": {"dir": "TSMC16/naive", "vdd": 0.80, "patterns": ["nch_*.l", "pch_*.l"]},
}
```

## Test Coverage Summary

| Test File | Tests | Speed | Coverage |
|-----------|-------|-------|----------|
| `test_dc_jacobian.py` | 15 | Medium | Full Jacobian matrix |
| `test_dc_regions.py` | 25 | Fast | Operating regions |
| `test_transient.py` | 5 | Slow | Transient waveforms |
| `test_ac_capacitance.py` | 50 | Medium | C-V characteristics |
| `test_api.py` | 10 | Fast | API sanity |
| `test_integration.py` | 3 | Fast | Quick sanity |

**Total**: ~108 tests, estimated runtime 5-10 minutes.

## PyCMG Enhancements Required

1. **`Instance.get_jacobian_matrix()`** - Extract raw Jacobian from OSDI
2. **`Instance.eval_tran()` verification** - Ensure time-stepping is correct

## Tolerances

```python
ABS_TOL_I = 1e-9      # Current tolerance (Amps)
ABS_TOL_Q = 1e-18     # Charge tolerance (Coulombs)
ABS_TOL_G = 1e-9      # Conductance tolerance (Siemens)
REL_TOL = 5e-3        # Relative tolerance (0.5%)
```

## Success Criteria

1. All Jacobian entries match NGSPICE within tolerance
2. Transient waveforms match at all time points
3. Operating region tests pass for all 5 technologies
4. C-V characteristics match across bias range
5. Test suite runs in < 10 minutes