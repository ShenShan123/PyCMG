# CLAUDE.md - BSIM-CMG Python Model Interface & Verification

## Project Overview
Develop a standalone Python interface for the BSIM-CMG Verilog-A model using OpenVAF/OSDI.

### Verification Strategy
**PyCMG** wraps the OSDI binary directly via ctypes (`pycmg/core.py`, `pycmg/model.py`), while **NGSPICE** loads the SAME OSDI binary via the `.osdi` command. Tests compare PyCMG output vs NGSPICE output to ensure:

1. **Binary-level consistency**: Both use the identical `bsimcmg.osdi` file
2. **Ctypes wrapper correctness**: Verifies proper OSDI function calls
3. **Numerical accuracy**: Direct comparison of currents, charges, derivatives
4. **Full model coverage**: DC, AC (capacitance), and transient analysis

The OSDI binary is the single source of truth for all model physics calculations.

## Environment & Tools
* **OpenVAF Compiler:** `/usr/local/bin/openvaf`
* **NGSPICE Simulator:** `/usr/local/ngspice-45.2/bin/ngspice`
* **Build System:** CMake / Make
* **Python Interface:** ctypes (no C++ compilation needed)
* **Environment Overrides:**
    * `NGSPICE_BIN` to point at a custom NGSPICE binary.
    * `ASAP7_MODELCARD` to point ASAP7 verification at a file or directory.

## Directory Structure
```
pycmg-wrapper/
├── bsim-cmg-va/              # Verilog-A source files
│   ├── code/                 # Main model Verilog-A files
│   ├── README.txt           # Original BSIM-CMG documentation
│   ├── *.pdf                 # Technical manuals (3 PDFs)
├── pycmg/                    # Python package
│   ├── __init__.py          # Public API exports
│   ├── osdi_types.py        # OSDI constants, ctypes structures, function type declarations
│   ├── core.py              # OsdiLibrary, OsdiModel, OsdiInstance, OsdiSimulation, AlignedBuffer
│   ├── parser.py            # parse_modelcard, parse_number_with_suffix, ParsedModel, parse_tsmc_pdk
│   ├── model.py             # Model, Instance (public API), eval_dc, eval_tran, get_jacobian_matrix
│   ├── tech.py              # Technology registry (TECH_REGISTRY, DeviceConfig, TechConfig, resolve_modelcard)
│   ├── sweep.py             # Sweep engine (SweepConfig, sweep_dc, generate_dataset, to_csv)
│   └── sensitivity.py       # OAT sensitivity analysis (compute_sensitivity, enumerate_model_params)
├── tests/                    # Test suite (280 tests)
│   ├── __init__.py          # Package init
│   ├── conftest.py          # Tiered technology registry (5 base + 16 Vt variants = 21 total)
│   ├── helpers.py           # NGSPICE runner helpers, comparison functions, modelcard baking
│   ├── test_api.py          # Public API tests (smoke, basic functionality)
│   ├── test_ac_caps.py      # AC capacitance verification vs NGSPICE
│   ├── test_body_bias.py    # Body bias (Ve != 0) verification vs NGSPICE
│   ├── test_dc_jacobian.py  # DC Jacobian verification vs NGSPICE (NMOS+PMOS)
│   ├── test_dc_regions.py   # DC operating region tests vs NGSPICE (NMOS+PMOS)
│   ├── test_nfin_scaling.py # NFIN scaling sanity tests (PyCMG-only)
│   ├── test_temperature.py  # Temperature verification vs NGSPICE
│   ├── test_transient.py    # Transient waveform verification vs NGSPICE (NMOS+PMOS)
│   └── test_vt_variants.py  # Core Vt variant DC verification (lvt/slvt/sram/ulvt/elvt/hvt/lnvt)
├── scripts/                  # Utility scripts
│   ├── generate_training_data.py # CLI for training data generation
│   ├── sensitivity_analysis.py   # CLI for process parameter sensitivity analysis
│   └── generate_naive_tsmc.py   # Generalized TSMC naive modelcard generator
├── modelcards/               # Technology model cards
│   ├── ASAP7/               # ASAP7 PDK model files
│   ├── TSMC5/               # TSMC5 model files
│   │   └── naive/           # Pre-baked naive modelcards
│   ├── TSMC7/               # TSMC7 model files
│   │   └── naive/           # Pre-baked naive modelcards
│   ├── TSMC12/              # TSMC12 model files
│   │   └── naive/           # Pre-baked naive modelcards
│   └── TSMC16/              # TSMC16 model files
│       └── naive/           # Pre-baked naive modelcards
├── build/                    # Build artifacts (generated)
│   ├── osdi/                # Compiled .osdi files
│   └── ngspice_eval/        # Verification outputs
└── CLAUDE.md                 # This file
```

### Module Organization
* **`pycmg/osdi_types.py`**: OSDI constants, ctypes structure definitions, function type declarations
  - OSDI ABI constants and enums
  - Ctypes structure wrappers for OSDI descriptors
  - Function pointer type declarations

* **`pycmg/core.py`**: Low-level OSDI interface
  - `OsdiLibrary`: OSDI shared library loader
  - `OsdiModel`: Model descriptor wrapper
  - `OsdiInstance`: Instance descriptor wrapper
  - `OsdiSimulation`: Simulation state manager
  - `AlignedBuffer`: Memory-aligned buffer for OSDI data
  - `apply_param()`: Parameter application helper

* **`pycmg/parser.py`**: Modelcard and parameter parsing
  - `parse_modelcard()`: Modelcard parser with unit suffix support
  - `parse_number_with_suffix()`: SPICE number parsing (e.g., "1n" -> 1e-9)
  - `ParsedModel`: Parsed model data container
  - `parse_tsmc_pdk()`: TSMC PDK parser (NFIN-aware variant selection)
  - `VariantInfo`: Dataclass for parsed PDK variant metadata
  - `_scan_all_variants()`: Scan all numbered variants from a TSMC PDK file
  - `scan_pdk_geometry_combos()`: Enumerate PDK-defined (L, NFIN) sweep points

* **`pycmg/model.py`**: Public API (Model, Instance)
  - `Model`: OSDI model wrapper (public API)
  - `Instance`: Device instance with DC/TRAN evaluation
  - `eval_dc()`: DC operating point evaluation
  - `eval_tran()`: Transient evaluation
  - `get_jacobian_matrix()`: Jacobian extraction

* **`pycmg/sensitivity.py`**: OAT sensitivity analysis
  - `enumerate_model_params()`: Discover all real-valued model-level parameters from OSDI descriptor
  - `compute_sensitivity()`: Central-difference perturbation analysis at representative bias points
  - `rank_parameters()`: Rank parameters by aggregate sensitivity per I-V/Q-V/C-V category
  - `format_sensitivity_table()`: Terminal-friendly output formatting
  - `SensitivityResult`: Result container with sensitivities, rankings, bias points

* **`tests/helpers.py`**: Verification and testing utilities
  - NGSPICE runner helpers
  - Comparison functions (DC, AC, TRAN)
  - Technology modelcard handling
  - Modelcard baking for NGSPICE

* **`tests/conftest.py`**: Tiered technology registry
  - `TECHNOLOGIES` dict (Tier 1): 5 base technologies (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16)
  - `CORE_VT_VARIANTS` dict (Tier 2): 16 additional Vt flavors (lvt, slvt, sram, ulvt, elvt, hvt, lnvt)
  - `ALL_TECHNOLOGIES` dict: Union of Tier 1 + Tier 2 (21 total entries)
  - `get_tech_modelcard()`: Retrieves modelcard path, model name, and instance params from ALL_TECHNOLOGIES
  - `TECH_NAMES` / `CORE_VT_NAMES` / `ALL_TECH_NAMES`: Lists for test parametrization

* **`tests/`**: Test suite (280 tests total)
  - `test_api.py`: Quick smoke tests for public API (no NGSPICE comparison)
  - `test_dc_jacobian.py`: DC Jacobian verification, NMOS+PMOS across all 5 base technologies
  - `test_dc_regions.py`: DC operating region tests (off/linear/saturation), NMOS+PMOS across all 5 base technologies
  - `test_transient.py`: Transient waveform verification, NMOS+PMOS across all 5 base technologies
  - `test_ac_caps.py`: AC capacitance verification (cgg, cgd, cgs, cdg, cdd) vs NGSPICE
  - `test_body_bias.py`: Body bias (Ve != 0) verification across all 5 base technologies
  - `test_temperature.py`: Temperature verification (-40C, 85C, 125C) vs NGSPICE
  - `test_nfin_scaling.py`: NFIN scaling sanity tests (PyCMG-only)
  - `test_vt_variants.py`: Core Vt variant DC verification, NMOS+PMOS across 16 Vt flavors (96 tests)

## PyCMG Output Coverage

PyCMG provides comprehensive model outputs covering currents, derivatives, charges, and capacitances. All outputs are verified against NGSPICE using the exact same OSDI binary.

### Supported Outputs (17 total)

| Category | Outputs | Description |
|----------|---------|-------------|
| **Currents** | `id`, `ig`, `is`, `ie`, `ids` | Terminal currents + drain-source current (Id-Is) |
| **Derivatives** | `gm`, `gds`, `gmb` | Transconductance, output conductance, bulk transconductance |
| **Charges** | `qg`, `qd`, `qs`, `qb` | Gate, drain, source, bulk charges |
| **Capacitances** | `cgg`, `cgd`, `cgs`, `cdg`, `cdd` | Capacitance matrix (condensed) |

### Key Features

- **ids**: Drain-source current computed as `Id - Is` for common-source configuration
- **All outputs verified** against NGSPICE ground truth using same OSDI binary
- **Capacitance condensation**: Full internal capacitance matrix reduced to terminal terminals
- **Full coverage**: 17/17 critical model outputs implemented and tested

### Return Values

**DC Analysis** (`Instance.eval_dc()`):
```python
result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
# Returns: id, ig, is, ie, ids, qg, qd, qs, qb, gm, gds, gmb, cgg, cgd, cgs, cdg, cdd
```

**Transient Analysis** (`Instance.eval_tran()`):
```python
result = inst.eval_tran({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0}, time=1e-9, delta_t=1e-12)
# Returns: id, ig, is, ie, ids, qg, qd, qs, qb
```

## Implementation Workflow

### 1. Model Compilation (OpenVAF)

The Verilog-A source must be compiled to OSDI format using OpenVAF.

**Prerequisites:**
- OpenVAF compiler (v23.5.0+): Install from https://github.com/ngspice/openvaf
- CMake (v3.20+)

**Build Methods:**

**Option A: Manual CMake build (Recommended)**
```bash
# Create build directory
mkdir -p build
cd build

# Configure CMake
cmake ..

# Build OSDI model
cmake --build . --target osdi
```

**Option B: Direct OpenVAF compilation**
```bash
# Compile Verilog-A directly without CMake
openvaf -I bsim-cmg-va/code -o bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
```

**Verification:**
- Ensure output file exists: `build/osdi/bsimcmg.osdi`
- File should be a shared object: `file build/osdi/bsimcmg.osdi`
- Typical size: ~2-3 MB

**Constraint:** Ensure the output is a standard `.osdi` file compatible with NGSPICE and the PyCMG ctypes host.

### 2. Python Interface Layer (ctypes-based OSDI host)
* **A) Model Card Parser:**
    * Read `.lib`, `.l`, etc., files.
    * Extract global model parameters (e.g., `EOT`, `CIGC`).
    * Handle unit conversion (e.g., `15n` -> `1.5e-8`).
    * Apply default values for parameters undefined in the card (relying on OSDI defaults).
* **B) Netlist Parameter Extraction:**
    * Parse instance lines from SPICE netlists (e.g., `X1 ...` in `.cir` or `.sp`).
    * Extract instance-specific geometric parameters (e.g., `L`, `TFIN`, `NFIN`).
* **C) Simulation Conditions:**
    * Parse `.dc` or `.tran` commands to generate input voltage vectors ($V_d, V_g, V_s, V_e$) and temperature settings.
* **Execution:** Pass combined Model Params + Instance Params + Voltage Vectors to the OSDI binary via ctypes.

### 3. Verification (NGSPICE Ground Truth)
* **Configuration:**
    * NGSPICE must load the **exact same** `.osdi` file generated in Step 1 using the `.osdi` command. Do NOT use `.hdl`.
    * Do not allow NGSPICE to re-compile the Verilog-A source; it must use the pre-compiled binary to ensure binary-level consistency.
* **Procedure:**
    1.  Run NGSPICE on test netlists to generate `.csv` output.
    2.  Run the Python Model Interface using identical parameters and voltage vectors.
    3.  Compare currents ($I_d, I_g$) and Derivatives ($g_m, g_{ds}$) numerically.
    4.  Assert accuracy within accepted tolerance (e.g., `ABS_TOL_I=1e-9`, `REL_TOL=5e-3`).
* **Test Strategy:**
    * **Technology Registry** (`tests/conftest.py`): Tiered parametrization
      - Tier 1 (`TECHNOLOGIES`): 5 base technologies (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16)
      - Tier 2 (`CORE_VT_VARIANTS`): 16 Vt variants (ASAP7 lvt/slvt/sram, TSMC ulvt/elvt/hvt/lnvt)
      - Each entry has vdd, modelcard paths, model names, instance params
    * **DC Jacobian tests** (`tests/test_dc_jacobian.py`): Verify DC derivatives vs NGSPICE
      - Tests all 5 base technologies using Tier 1 registry
      - Covers gm, gds, gmb derivatives
    * **DC Region tests** (`tests/test_dc_regions.py`): DC operating region verification
      - Tests all 5 base technologies using Tier 1 registry
      - Covers subthreshold, linear, saturation regions
    * **Transient tests** (`tests/test_transient.py`): Transient waveform verification
      - Tests all 5 base technologies using Tier 1 registry
      - Covers charge/ discharge waveforms
    * **Vt Variant tests** (`tests/test_vt_variants.py`): Cross-Vt DC verification
      - Tests all 16 Vt variants using Tier 2 registry
      - Covers saturation, linear, subthreshold regions for NMOS+PMOS
    * **API tests** (`tests/test_api.py`): Quick smoke tests
      - Basic functionality verification
      - No NGSPICE comparison (fast execution)

## Development Rules
1.  **No Circuit Solvers:** The Python code must not contain KCL/KVL solvers or circuit simulation logic. It is strictly a Model Evaluator ($V \to I, Q, Jacobian$).
2.  **Source of Truth:** The OSDI binary is the single source of truth for physics calculations.
3.  **Data Flow:**
    * *Input:* Text (Netlists/Model Cards) -> Python Parsers -> Float Values.
    * *Compute:* Float Values -> ctypes Host (`pycmg/core.py`) -> OSDI Binary.
    * *Output:* OSDI Results (Values + Derivatives) -> Numpy Arrays -> Verification.

## Other Tips in This Project
* **Start every complex task in plan mode:**
    * Pour your energy into the plan for 1-shot the implementation.
    * The moment something goes sideways, just switch back to plan mode and re-plan. Don't keep pushing.
    * Enter plan mode for verification steps, not just for the build.
* **Update CLAUDE.md:**
    * After every correction, update your CLAUDE.md so you don't make that mistake again.
* **Never be lazy:**
    * Never be lazy in writing the code and running tests.
    * Do NOT use any simplifed equations or self-defined CMG models as reference, ALWAYS use simulation results as ground truth for comparison.
* Use subagents.
    * Use a second agent to review the plan as a staff engineer.
    * If you want to try multiple solutions, use multiple subagents, git commit to different branches. Roll back and to the main branch and create new branch when the subagent find it's a dead end.
* Enable the "Explanatory" or "Learning" output style in /config to explain the *why* behind its changes.

## Design Principles & Known Constraints

### TSMC PDK Variant Selection
- TSMC PDK variants are indexed by **both** L range (lmin/lmax) **and** NFIN range (nfinmin/nfinmax). Any function that selects a variant must match on both dimensions. `_scan_all_variants()` is the single source of truth for parsing variant metadata.
- **Model name matching must be exact-word, never substring.** `.model nch_svt_mac.1` must not match `.model nch_svt_mac.10`. Always split the line and compare `parts[1] == model_name`, never use `model_name in line`.

### Capacitance Sign Convention
- The OSDI reactive Jacobian (dQ/dV) uses **Y-matrix convention** where off-diagonal entries are negative. SPICE capacitance variables (cgd, cgs, cdg) use the **opposite** sign. Off-diagonal entries must be negated when extracting from the condensed matrix; diagonal entries (cgg, cdd) need no sign flip.

### NGSPICE OSDI Limitations
- **No instance-line parameters**: NGSPICE OSDI cannot accept instance parameters on the device line (e.g., `N1 d g s e model L=16e-9` fails silently). All geometric parameters must be **baked into the `.model` block**.
- **Multi-model files**: When a modelcard contains multiple `.model` blocks, `Model()` must pass `model_name` to `parse_modelcard(target=...)` so the correct block is parsed.
- **TSMC PDK sentinel values**: TSMC PDKs use `-999*10^n` as "use default" markers. These are filtered during naive modelcard generation using arithmetic mantissa detection (not string matching, since `str(float)` switches to scientific notation for large values).

### OSDI Parameter Access Flags
- Use `ACCESS_FLAG_SET` (1), **not** `ACCESS_FLAG_READ` (0), for reading model parameter values from the OSDI buffer. `ACCESS_FLAG_READ` returns null for parameters not explicitly written. Ensure `enumerate_model_params()` is called AFTER creating a baseline `Instance`.
- For instance-level opvar reads, use `ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE` (= 5). Note that `ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE` equals `ACCESS_FLAG_INSTANCE` (= 4) since `ACCESS_FLAG_READ = 0` — a bitwise OR with zero is a no-op.

### Extended Voltage Range Design
- Extended voltage sweep uses `voltage_scale` multiplier (default 1.0) relative to VDD.
- The dense region around Vth must use `±0.15*vdd` (nominal VDD), NOT `±0.15*v_max`. Threshold voltage is a physical property independent of sweep range.
- When `voltage_scale=2.0`, PMOS gets negative gate voltages (deep accumulation regime). BSIM-CMG handles this correctly.

### OSDI Interaction Rules
- OSDI init out-of-bounds errors should be treated as warnings (matching NGSPICE behavior), not fatal.
- Some OSDI params are integer-typed; read/write using `PARA_TY_INT` to avoid garbage values.
- Do not pass `prev_solve` to OSDI unless it is explicitly initialized; uninitialized `prev_solve` breaks DC/AC comparisons.
- **Always check `EVAL_RET_FLAG_FATAL` (bit 1) on the return value of `eval()` / `eval_with_time()`.** If set, residual and Jacobian buffers contain undefined values — raise an error rather than reading garbage.
- **Always null-guard `info.errors` before iterating** in `_check_init_result`. A malformed OSDI binary may set `num_errors > 0` with a null pointer.
- **Always check `solve_internal_nodes` return value.** `eval_dc` raises `RuntimeError` on convergence failure — callers that sweep bias points (data generation) must catch this to avoid garbage data (id=40 kA, NaN derivatives). `eval_tran` uses a relaxed tolerance (1e-3 vs 1e-9) and warns instead of raising, because the circuit-level NR provides outer convergence.
- **NFIN=1 causes convergence failure** for certain TSMC process variants (e.g., tsmc5:ulvt, tsmc16:lnvt) where BSIM-CMG instance parameters (ETA0_i, U0_i, UA_i) become negative. When `solve_internal_nodes` diverges, the internal drain node drifts to ~40 V (200 iterations × 0.2 V clamp), producing `id ≈ 40 kA` and NaN for all derivatives. Avoid NFIN=1 in data generation; use NFIN ≥ 2.

### ctypes Buffer Safety
- **Jacobian arrays must never be reallocated after `bind_simulation`.** `bind_simulation` stores raw C pointers into the OSDI instance buffer; reallocating the backing arrays creates dangling pointers. `build_jacobian()` must reuse/zero existing arrays in-place when the size hasn't changed.
- **Keep-alive references for ctypes arrays must be stored as object attributes** (e.g., `sim._keep_alive`), not bare local variables like `_ = (...)`. The `_` convention is CPython-specific and may be optimized away by other implementations.
- **After `set_params` triggers a rebind**, all transient state (`_has_prev_solve`, `_has_prev_q`, `_prev_q*`) must be reset. Stale charge history from a previous geometry produces incorrect `dQ/dt`.

### Instance / Model Isolation
- **`model_overrides` writes to a shared `OsdiModel` buffer.** Creating multiple Instances from the same Model with different `model_overrides` will silently corrupt earlier Instances. For per-instance process variation, create a separate `Model()` per override set.
- **Device polarity must come from `DeviceConfig.inst_params["DEVTYPE"]`**, not from substring matching on device names. `DEVTYPE=1` is NMOS, `DEVTYPE=0` is PMOS.
- **Cache terminal index lookups** (`_term_g`, `_term_d`, `_term_s`) after `bind_simulation`. Do not re-scan `terminal_indices` on every `eval_dc` call.

### Newton-Raphson Limiting (`_pnjlim`)
- Always guard `math.log(vnew / vt)` against `vnew <= 0`. Fall back to `vcrit` when the argument would be non-positive.

## Gap Checklist (Inventory vs Workflow)
- OSDI build pipeline: CMake builds `.osdi` via OpenVAF.
- Python ctypes host: `pycmg/core.py` (low-level OSDI) + `pycmg/model.py` (public API: `Model`, `Instance`, `eval_dc`, `eval_tran`).
- OSDI type definitions: `pycmg/osdi_types.py` provides constants and ctypes structure definitions.
- Modelcard parsing: `pycmg/parser.py` includes SPICE-compatible parser with unit suffix support.
- Verification utilities: `tests/helpers.py` provides NGSPICE comparison helpers.
- Technology registry: `tests/conftest.py` tiered registry with 21 entries (5 base + 16 Vt variants).
- DC Jacobian tests: `tests/test_dc_jacobian.py` NMOS+PMOS across all 5 base technologies.
- DC Region tests: `tests/test_dc_regions.py` NMOS+PMOS across all 5 base technologies, includes gmb.
- Transient tests: `tests/test_transient.py` NMOS+PMOS across all 5 base technologies.
- AC Capacitance tests: `tests/test_ac_caps.py` NMOS across all 5 base technologies.
- Body bias tests: `tests/test_body_bias.py` NMOS+PMOS across all 5 base technologies.
- Temperature tests: `tests/test_temperature.py` NMOS+PMOS at -40C, 85C, 125C (ASAP7).
- NFIN scaling tests: `tests/test_nfin_scaling.py` NMOS+PMOS scaling sanity (ASAP7, PyCMG-only).
- Vt variant tests: `tests/test_vt_variants.py` NMOS+PMOS across 16 Vt flavors (3 regions each).
- API tests: `tests/test_api.py` quick smoke tests (no NGSPICE).
- Environment override: set `ASAP7_MODELCARD` to a file or directory to redirect ASAP7 inputs.
- C++ OSDI host: removed (was `cpp/osdi_host.cpp`); Python uses ctypes directly via `pycmg/core.py`.
- Naive modelcard generation: `scripts/generate_naive_tsmc.py` generates all Vt flavors from raw PDKs. `generate_naive_tsmc_modelcard` delegates to `parse_tsmc_pdk` for the merge step — do not duplicate the global+variant merge logic.
- Extended voltage range: `SweepConfig.voltage_scale` (default 1.0, use 2.0 for 2*VDD) for NN simulator convergence training.
- PDK geometry sweep: `SweepConfig.sweep_geometry` (default True) enumerates PDK-defined (L, NFIN) combinations from variant bin boundaries.
- PDK introspection: `scan_pdk_geometry_combos()` returns all (lmin, nfin) sweep points; `_scan_all_variants()` returns parsed variant metadata.
- NFIN-aware modelcard generation: `resolve_modelcard()` accepts NFIN to select correct NFIN group variant; cache includes NFIN in filename.
- Sensitivity analysis: `pycmg/sensitivity.py` with OAT perturbation, `scripts/sensitivity_analysis.py` CLI.
- Sensitivity tests: `tests/test_sensitivity.py` (7 tests).
- **Not yet covered**: I/O voltage-domain devices (1.2V/1.8V), PVT corners (SS/FF), RF variants.

## Technology Modelcard Verification

All verification tests use the tiered technology registry in `tests/conftest.py`. Tier 1 (`TECHNOLOGIES`) covers 5 base technologies for comprehensive analysis types. Tier 2 (`CORE_VT_VARIANTS`) extends coverage to 16 additional Vt flavors for DC verification.

### Tier 1: Base Technology Registry (5 entries)

| Technology | Vdd | NMOS Modelcard | PMOS Modelcard | NMOS L | PMOS L |
|------------|-----|----------------|----------------|--------|--------|
| ASAP7 | 0.9V | 7nm_TT_160803.pm (rvt) | 7nm_TT_160803.pm (rvt) | 7nm | 7nm |
| TSMC5 | 0.65V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |
| TSMC7 | 0.75V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |
| TSMC12 | 0.80V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |
| TSMC16 | 0.80V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |

### Tier 2: Core Vt Variants (16 entries)

| Variant | Tech | Vt Flavor | Vdd | NMOS Model | PMOS Model |
|---------|------|-----------|-----|------------|------------|
| ASAP7_lvt | ASAP7 | Low Vt | 0.9V | nmos_lvt | pmos_lvt |
| ASAP7_slvt | ASAP7 | Super-Low Vt | 0.9V | nmos_slvt | pmos_slvt |
| ASAP7_sram | ASAP7 | SRAM | 0.9V | nmos_sram | pmos_sram |
| TSMC5_lvt | TSMC5 | Low Vt | 0.65V | nch_lvt_mac | pch_lvt_mac |
| TSMC5_ulvt | TSMC5 | Ultra-Low Vt | 0.65V | nch_ulvt_mac | pch_ulvt_mac |
| TSMC5_elvt | TSMC5 | Extreme-Low Vt | 0.65V | nch_elvt_mac | pch_elvt_mac |
| TSMC7_lvt | TSMC7 | Low Vt | 0.75V | nch_lvt_mac | pch_lvt_mac |
| TSMC7_ulvt | TSMC7 | Ultra-Low Vt | 0.75V | nch_ulvt_mac | pch_ulvt_mac |
| TSMC12_lvt | TSMC12 | Low Vt | 0.80V | nch_lvt_mac | pch_lvt_mac |
| TSMC12_hvt | TSMC12 | High Vt | 0.80V | nch_hvt_mac | pch_hvt_mac |
| TSMC12_ulvt | TSMC12 | Ultra-Low Vt | 0.80V | nch_ulvt_mac | pch_ulvt_mac |
| TSMC12_lnvt | TSMC12 | Low-Noise Vt | 0.80V | nch_lnvt_mac | pch_lnvt_mac |
| TSMC16_lvt | TSMC16 | Low Vt | 0.80V | nch_lvt_mac | pch_lvt_mac |
| TSMC16_hvt | TSMC16 | High Vt | 0.80V | nch_hvt_mac | pch_hvt_mac |
| TSMC16_ulvt | TSMC16 | Ultra-Low Vt | 0.80V | nch_ulvt_mac | pch_ulvt_mac |
| TSMC16_lnvt | TSMC16 | Low-Noise Vt | 0.80V | nch_lnvt_mac | pch_lnvt_mac |

### Verification Test Types

| Test File | Coverage | Description |
|-----------|----------|-------------|
| `test_dc_jacobian.py` | 5 base techs, NMOS+PMOS | DC derivatives (gm, gds, gmb) vs NGSPICE |
| `test_dc_regions.py` | 5 base techs, NMOS+PMOS | DC operating regions (off/linear/saturation) vs NGSPICE |
| `test_transient.py` | 5 base techs, NMOS+PMOS | Transient charge/discharge waveforms vs NGSPICE |
| `test_ac_caps.py` | 5 base techs, NMOS+PMOS | AC capacitances (cgg, cgd, cgs, cdg, cdd) vs NGSPICE |
| `test_body_bias.py` | 5 base techs, NMOS+PMOS | Body bias (Ve != 0) verification vs NGSPICE |
| `test_temperature.py` | ASAP7, NMOS+PMOS | Temperature (-40C, 85C, 125C) verification vs NGSPICE |
| `test_nfin_scaling.py` | ASAP7, NMOS+PMOS | NFIN scaling sanity (PyCMG-only, no NGSPICE) |
| `test_vt_variants.py` | 16 Vt variants, NMOS+PMOS | DC sat/linear/subthreshold vs NGSPICE (96 tests) |
| `test_api.py` | Smoke only | Basic functionality, no NGSPICE |

### Key Implementation Details

- **Modelcard baking**: `_bake_inst_params_into_modelcard()` in `tests/helpers.py` injects instance params (L, TFIN, NFIN, DEVTYPE) before the closing `)` of the `.model` block
- **NGSPICE OSDI limitation**: Cannot accept instance params on device line; must be in `.model` block
- **Tolerances**: ABS_TOL_I=1e-9, ABS_TOL_Q=1e-18, ABS_TOL_C=1e-18 (capacitance), REL_TOL=5e-3, REL_TOL_CAP=1e-2 (1% for capacitance)
- **DEVTYPE injection**: Automatic injection of devtype=1.0 (NMOS) or devtype=0.0 (PMOS) for models missing this parameter
- **Sentinel filtering**: TSMC PDK sentinel values (-999*10^n) filtered during naive modelcard generation
- **Tiered registry design**: Tier 1 entries use shared param templates with `.copy()` to prevent cross-contamination; Tier 2 uses helper functions (`_asap7_entry`, `_tsmc_entry`) for consistency
