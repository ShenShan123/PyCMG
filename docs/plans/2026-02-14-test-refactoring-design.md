# Test Suite Refactoring Design (Rev 2)

**Date**: 2026-02-14
**Status**: Approved
**Revision**: 2 — addresses code review feedback on Jacobian implementation, AC removal, architecture

## Executive Summary

Refactor the test suite to focus on convergence-critical verification:
1. **DC Jacobian verification** — Full 4×4 condensed matrix via `_build_full_jacobian()` + condensation
2. **Transient verification** — Full waveform comparison with correct sign convention
3. **Operating region coverage** — NMOS + PMOS across voltage-ratio-defined regions
4. **Utility consolidation** — Deduplicate NGSPICE helpers into `pycmg/testing.py`

All tests run against 5 technologies: ASAP7, TSMC5, TSMC7, TSMC12, TSMC16.

**Removed**: AC capacitance extraction (NGSPICE AC raw parsing too brittle; capacitance already verified via DC `eval_dc()` outputs).

## Design Decisions

### 1. Organization: Analysis-Type-Based

Tests organized by analysis type rather than technology:
- `test_dc_jacobian.py` — Full Jacobian matrix verification
- `test_dc_regions.py` — Operating region coverage (NMOS + PMOS)
- `test_transient.py` — Full waveform comparison

Each file uses `@pytest.mark.parametrize` to run against all 5 technologies.

### 2. Utility Architecture

**Problem**: `_bake_inst_params()`, `_run_ngspice_op()`, and `_assert_close()` are duplicated across 6 test files (test_integration, test_asap7, test_tsmc{5,7,12,16}).

**Solution**: Consolidate into `pycmg/testing.py` (library code, not conftest). `conftest.py` contains only fixtures and the technology registry.

```
pycmg/testing.py          # NGSPICE runners, bake_inst_params, assert_close
tests/conftest.py          # TECHNOLOGIES registry, fixtures, tolerances
tests/test_dc_jacobian.py  # imports from both
```

### 3. DC Jacobian: Condensed Matrix via Existing Infrastructure

**Why condensation?** BSIM-CMG has internal nodes (di, si, etc.), making the raw Jacobian 7×7+. A circuit simulator sees only the 4 external terminals. Condensing to 4×4 gives the matrix that Newton-Raphson actually uses.

**PyCMG** (`Instance.get_jacobian_matrix()`):
1. Call `eval_dc()` to populate OSDI buffers
2. Use existing `_build_full_jacobian(sim, sim.jacobian_resist)` → N×N full matrix
3. Condense to 4×4 external-only using Schur complement:
   ```
   G_ext = G_ee - G_ei × G_ii⁻¹ × G_ie
   ```
   This is identical to `_condense_capacitance()` but for the real (resistive) part.

**NGSPICE**: Compute via **central** finite-difference perturbation:
- 1 base run + 8 perturbation runs (±δ for each of 4 terminals) = 9 NGSPICE calls per OP
- `delta = 1e-6` V
- `J[:,j] = (I(V+δ_j) - I(V-δ_j)) / (2δ)` — central differencing for O(δ²) accuracy

**Tolerance**: `rel_tol=0.01` (1%), `abs_tol=1e-6` S — tight enough to catch real errors, loose enough for finite-difference truncation.

### 4. Operating Regions: NMOS + PMOS

**NMOS regions** (positive voltages, grounded source):

| Region | Vgs | Vds | Physical Meaning |
|--------|-----|-----|------------------|
| Off-state | 0.0 | Vdd | Device fully off |
| Weak-inversion | 0.3 × Vdd | Vdd | Near threshold |
| Moderate-inversion | 0.6 × Vdd | 0.5 × Vdd | Transition region |
| Strong-linear | Vdd | 0.3 × Vdd | High Vgs, low Vds |
| Strong-saturation | 0.8 × Vdd | Vdd | High Vgs, high Vds |

**PMOS regions** (inverted sense: Vs=Vdd, Vg/Vd referenced to Vdd):

| Region | Vd | Vg | Vs | Physical Meaning |
|--------|----|----|----|------------------|
| Off-state | 0.0 | Vdd | Vdd | Device fully off (Vsg=0) |
| Weak-inversion | 0.0 | 0.7 × Vdd | Vdd | Near threshold |
| Moderate-inversion | 0.5 × Vdd | 0.4 × Vdd | Vdd | Transition |
| Strong-linear | 0.7 × Vdd | 0.0 | Vdd | High |Vsg|, low |Vsd| |
| Strong-saturation | 0.0 | 0.2 × Vdd | Vdd | High |Vsg|, high |Vsd| |

### 5. Transient: Full Waveform Comparison

**NGSPICE**: Run transient simulation with pulse stimulus on gate
- Time step: 10 ps, Duration: 5 ns (~500 points)
- Outputs: `v(d) v(g) v(s) v(e) i(vd) i(vg) i(vs) i(ve)`

**PyCMG**: Call `eval_tran()` at each NGSPICE time point using NGSPICE's solved node voltages as input.

**Sign convention**: NGSPICE `i(vd)` = current into positive terminal of voltage source Vd. This equals the current flowing into the drain terminal. PyCMG `id` = drain terminal current (same direction). **No sign flip needed** for OP comparison (documented in CLAUDE.md).

**Tolerance**: `rel_tol=0.005` (0.5%), `abs_tol=1e-9` A — same as DC since we use NGSPICE's solved voltages.

### 6. Technology Registry with Corner Selection

```python
TECHNOLOGIES = {
    "ASAP7": {
        "dir": "ASAP7",
        "vdd": 0.9,
        "corner": "TT",
        "modelcard_file": "7nm_TT_160803.pm",
        "nmos_model": "nmos_rvt",
        "pmos_model": "pmos_rvt",
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
    # TSMC7, TSMC12, TSMC16 follow same pattern
}
```

Key differences from v1:
- **ASAP7**: Explicit corner ("TT") and model name ("nmos_rvt") — no ambiguous glob
- **TSMC**: Explicit file names and per-device instance params (PMOS uses L=20nm)
- **No glob patterns** — everything is deterministic

## File Structure

```
pycmg/
├── testing.py                  # NEW: Consolidated NGSPICE helpers
tests/
├── conftest.py                 # Enhanced: technology registry + fixtures
├── test_api.py                 # Keep as-is (API smoke tests)
├── test_dc_jacobian.py         # NEW: Full Jacobian matrix verification
├── test_dc_regions.py          # NEW: Operating region coverage (NMOS+PMOS)
├── test_transient.py           # NEW: Full waveform comparison
├── test_integration.py         # Keep as quick sanity check
└── legacy/                     # Move old technology-specific tests
    ├── test_tsmc5.py
    ├── test_tsmc7.py
    ├── test_tsmc12.py
    └── test_tsmc16.py
```

## Test Coverage Summary

| Test File | Tests | Speed | Coverage |
|-----------|-------|-------|----------|
| `test_dc_jacobian.py` | 15 | Medium | 4×4 condensed Jacobian (5 techs × 3 OPs) |
| `test_dc_regions.py` | 50 | Fast | 5 NMOS regions + 5 PMOS regions × 5 techs |
| `test_transient.py` | 5 | Slow | Transient waveforms (5 techs) |
| `test_api.py` | ~10 | Fast | API sanity |
| `test_integration.py` | ~3 | Fast | Quick sanity |

**Total**: ~83 tests, estimated runtime 5-10 minutes.

## PyCMG Enhancements Required

1. **`Instance.get_jacobian_matrix()`** — Condensed 4×4 resistive Jacobian via `_build_full_jacobian()` + Schur complement
2. **`Instance.eval_tran()` verification** — Confirm time-stepping produces correct results

## Tolerances

```python
# DC and transient (exact voltage inputs → tight tolerance)
ABS_TOL_I = 1e-9      # Current (A)
ABS_TOL_Q = 1e-18     # Charge (C)
REL_TOL = 5e-3        # Relative (0.5%)

# Jacobian (central finite-difference → slightly looser)
ABS_TOL_G = 1e-6      # Conductance (S) — floor for near-zero entries
REL_TOL_JAC = 1e-2    # Jacobian relative (1%)
```

## Success Criteria

1. All 4×4 condensed Jacobian entries match NGSPICE numerical Jacobian within 1%
2. Transient waveforms match at all sampled time points within 0.5%
3. Operating region tests pass for all 5 technologies (NMOS + PMOS)
4. Test suite runs in < 10 minutes
5. Zero duplicated utility code across test files
