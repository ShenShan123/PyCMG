# CLAUDE.md - BSIM-CMG Python Model Interface & Verification

## Project Overview
Develop a standalone Python interface for the BSIM-CMG Verilog-A model using OpenVAF/OSDI.

### Verification Strategy
**PyCMG** wraps the OSDI binary directly via ctypes (`pycmg/ctypes_host.py`), while **NGSPICE** loads the SAME OSDI binary via the `.osdi` command. Tests compare PyCMG output vs NGSPICE output to ensure:

1. **Binary-level consistency**: Both use the identical `bsimcmg.osdi` file
2. **Ctypes wrapper correctness**: Verifies proper OSDI function calls
3. **Numerical accuracy**: Direct comparison of currents, charges, derivatives
4. **Full model coverage**: DC, AC (capacitance), and transient analysis

The OSDI binary is the single source of truth for all model physics calculations.

## Environment & Tools
* **OpenVAF Compiler:** `/usr/local/bin/openvaf`
* **NGSPICE Simulator:** `/usr/local/ngspice-45.2/bin/ngspice`
* **Build System:** CMake / Make
* **Python Bindings:** PyBind11
* **Environment Overrides:**
    * `NGSPICE_BIN` to point at a custom NGSPICE binary.
    * `ASAP7_MODELCARD` to point ASAP7 verification at a file or directory.

## Directory Structure
```
pycmg-wrapper/
├── bsim-cmg-va/              # Verilog-A source files
│   ├── *.va                  # Main model files
│   └── benchmark_test/       # SPICE netlists and model cards for verification
├── pycmg/                    # Python package
│   ├── __init__.py          # Public API exports
│   ├── ctypes_host.py       # Core OSDI interface (Model, Instance, eval_dc, eval_tran)
│   └── testing.py           # Verification utilities (moved from tests/verify_utils.py)
├── cpp/                      # C++ OSDI host
│   ├── osdi_host.h          # Header file
│   ├── osdi_host.cpp        # Core host implementation
│   ├── osdi_cli.cpp         # CLI tool (osdi_eval)
│   └── pycmg_bindings.cpp   # PyBind11 bindings (optional, not currently used)
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── test_api.py          # Public API tests (smoke, basic functionality)
│   ├── test_asap7.py        # ASAP7 verification (optimized PVT corners)
│   ├── test_integration.py  # NGSPICE comparison tests (representative subset)
│   ├── test_tsmc5.py        # TSMC5 verification (naive modelcards)
│   ├── test_tsmc7.py        # TSMC7 verification (naive modelcards)
│   ├── test_tsmc12.py       # TSMC12 verification (naive modelcards)
│   └── test_tsmc16.py       # TSMC16 verification (naive modelcards)
├── scripts/                  # Utility scripts
│   ├── generate_naive_tsmc7.py  # TSMC7-specific naive modelcard generator
│   └── generate_naive_tsmc.py   # Generalized TSMC naive modelcard generator
├── tech_model_cards/         # Technology model cards
│   ├── ASAP7/               # ASAP7 PDK model files
│   ├── TSMC5/               # TSMC5 model files
│   │   └── naive/           # Pre-baked naive modelcards
│   ├── TSMC7/               # TSMC7 model files
│   │   └── naive/           # Pre-baked naive modelcards
│   ├── TSMC12/              # TSMC12 model files
│   │   └── naive/           # Pre-baked naive modelcards
│   └── TSMC16/              # TSMC16 model files
│       └── naive/           # Pre-baked naive modelcards
├── build-deep-verify/        # Build artifacts (generated)
│   ├── osdi/                # Compiled .osdi files
│   └── ngspice_eval/        # Verification outputs
├── main.py                   # CLI entrypoint for quick test execution
└── README.md                 # This file
```

### Module Organization
* **`pycmg/ctypes_host.py`**: Core ctypes-based OSDI interface
  - `Model`: OSDI model wrapper
  - `Instance`: Device instance with DC/TRAN evaluation
  - `parse_modelcard()`: Modelcard parser with unit suffix support
  - `parse_number_with_suffix()`: SPICE number parsing (e.g., "1n" -> 1e-9)

* **`pycmg/testing.py`**: Verification and testing utilities
  - NGSPICE runner helpers
  - Comparison functions (DC, AC, TRAN)
  - ASAP7 modelcard handling
  - Stress testing utilities

* **`tests/`**: Test suite
  - `test_api.py`: Quick smoke tests for public API
  - `test_asap7.py**: Full ASAP7 verification (TT, SS, FF corners at representative temps)
  - `test_integration.py`: NGSPICE ground truth comparison (limited sweep points)

## PyCMG Output Coverage

PyCMG provides comprehensive model outputs covering currents, derivatives, charges, and capacitances. All outputs are verified against NGSPICE using the exact same OSDI binary.

### Supported Outputs (18 total)

| Category | Outputs | Description |
|----------|----------|-------------|
| **Currents** | `id`, `ig`, `is`, `ie`, `ids` | Terminal currents + drain-source current (Id-Is) |
| **Derivatives** | `gm`, `gds`, `gmb` | Transconductance, output conductance, bulk transconductance |
| **Charges** | `qg`, `qd`, `qs`, `qb` | Gate, drain, source, bulk charges |
| **Capacitances** | `cgg`, `cgd`, `cgs`, `cdg`, `cdd` | Capacitance matrix (condensed) |

### Key Features

- **ids**: Drain-source current computed as `Id - Is` for common-source configuration
- **All outputs verified** against NGSPICE ground truth using same OSDI binary
- **Capacitance condensation**: Full internal capacitance matrix reduced to terminal terminals
- **Full coverage**: 18/18 critical model outputs implemented and tested

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
- C++ compiler with C++17 support
- PyBind11 (for build system, not required for ctypes interface)

**Build Methods:**

**Option A: Using the build script (Recommended)**
```bash
# From project root
./build_osdi.sh
```

**Option B: Manual CMake build**
```bash
# Install pybind11 if not present
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11

# Create build directory
mkdir -p build-deep-verify
cd build-deep-verify

# Configure CMake (finds pybind11 automatically)
cmake ..

# Or specify pybind11 path explicitly
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..

# Build OSDI model
cmake --build . --target osdi
```

**Option C: Direct OpenVAF compilation**
```bash
# Compile Verilog-A directly without CMake
openvaf -I bsim-cmg-va/code -o bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
```

**Verification:**
- Ensure output file exists: `build-deep-verify/osdi/bsimcmg.osdi`
- File should be a shared object: `file build-deep-verify/osdi/bsimcmg.osdi`
- Typical size: ~2-3 MB

**Constraint:** Ensure the output is a standard `.osdi` file compatible with both the custom C++ host and NGSPICE.

### 2. C++ OSDI Host (PyBind11 Wrapper)
* **Dynamic Loading:** Load the compiled `.osdi` library using `dlopen`.
* **Symbol Binding:** Map OSDI standard functions (`create`, `set_param`, `evaluate`).
* **Derivative Extraction:**
    * Directly access the Jacobian matrix returned by the OSDI `evaluate` function.
    * **Strict Rule:** Do not calculate derivatives numerically (finite difference) in Python. Map specific Jacobian indices to model outputs (Gm, Gds, Gmb, etc.).
* **Memory Management:** Handle instance creation and destruction cleanly.

### 3. Python Interface Layer
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
* **Execution:** Pass combined Model Params + Instance Params + Voltage Vectors to the C++ core.

### 4. Verification (NGSPICE Ground Truth)
* **Configuration:**
    * NGSPICE must load the **exact same** `.osdi` file generated in Step 1 using the `.osdi` command. Do NOT use `.hdl`.
    * Do not allow NGSPICE to re-compile the Verilog-A source; it must use the pre-compiled binary to ensure binary-level consistency.
* **Procedure:**
    1.  Run NGSPICE on test netlists to generate `.csv` output.
    2.  Run the Python Model Interface using identical parameters and voltage vectors.
    3.  Compare currents ($I_d, I_g$) and Derivatives ($g_m, g_{ds}$) numerically.
    4.  Assert accuracy within accepted tolerance (e.g., `ABS_TOL_I=1e-9`, `REL_TOL=5e-3`).
* **Test Strategy:**
    * **ASAP7 tests** (`tests/test_asap7.py`): Primary verification across PVT corners
      * TT (typical), SS (slow), FF (fast) corners
      * Representative temperatures: -40°C, 27°C, 85°C, 125°C
      * DC, AC (capacitance), and TRAN verification
    * **TSMC FinFET tests** (`tests/test_tsmc{5,7,12,16}.py`): Multi-node verification
      * TSMC5 (5nm, Vdd=0.65V), TSMC7 (7nm, Vdd=0.75V), TSMC12 (12nm, Vdd=0.80V), TSMC16 (16nm, Vdd=0.80V)
      * NMOS SVT and PMOS LVT operating points per node
      * Temperature sweeps (-40°C to 125°C) and voltage sweeps
      * Uses naive modelcards generated by `scripts/generate_naive_tsmc.py`
      * Modelcard baking for NGSPICE OSDI compatibility
    * **Integration tests** (`tests/test_integration.py`): NGSPICE comparison
      * Limited voltage sweep points (not exhaustive)
      * Focus on critical operating regions
    * **API tests** (`tests/test_api.py`): Quick smoke tests
      * Basic functionality verification
      * No NGSPICE comparison (fast execution)

## Development Rules
1.  **No Circuit Solvers:** The Python code must not contain KCL/KVL solvers or circuit simulation logic. It is strictly a Model Evaluator ($V \to I, Q, Jacobian$).
2.  **Source of Truth:** The OSDI binary is the single source of truth for physics calculations.
3.  **Data Flow:**
    * *Input:* Text (Netlists/Model Cards) -> Python Parsers -> Float Values.
    * *Compute:* Float Values -> C++ Wrapper -> OSDI Binary.
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

## Lessons from Bugs (Keep Coming)

### Multi-Technology Verification & NGSPICE OSDI Limitations (2026-02-14)

- **NGSPICE OSDI does NOT support instance-line parameters**: Unlike HSPICE or Spectre, NGSPICE's OSDI interface cannot accept instance parameters on the device line (e.g., `N1 d g s e model L=16e-9` fails silently). All geometric parameters (L, TFIN, NFIN) must be **baked into the `.model` block** in the modelcard file.

- **Modelcard baking for NGSPICE**: The `_bake_inst_params_into_modelcard()` function in `test_tsmc7.py` inserts instance params before the closing `)` of the `.model` block. Critical: detect `stripped == ')'` to insert BEFORE the bracket, not after.

- **PMOS DEVTYPE in multi-model files**: When a modelcard contains multiple `.model` blocks (e.g., NMOS + PMOS in one file), `Model()` must pass `model_name` to `parse_modelcard(target=...)` so the correct block is parsed. Otherwise PMOS inherits DEVTYPE=1 from the first (NMOS) model, causing inverted behavior.

- **TSMC7 PMOS L=16nm NGSPICE convergence failure**: At L=16nm, TSMC7 PMOS naive modelcards have binning parameters that produce invalid `PDIBL2_i=-0.118`, causing NGSPICE "Timestep too small" DC convergence failure. PyCMG single-shot evaluation doesn't fail (no iterative solver), making comparison impossible. **Workaround**: Use L=20nm or larger for PMOS verification.

- **Stale test files with `sys.exit(1)`**: Module-level `sys.exit(1)` calls in test files crash pytest collection for the entire `tests/` directory. Clean up stale/scratch test files before running `pytest tests/`.

- **TSMC7 naive modelcards**: Use `nch_svt_mac_l16nm.l` (NMOS) and `pch_lvt_mac_l20nm.l` (PMOS). These contain pre-baked geometric params but require additional instance-param injection for NGSPICE compatibility.

- **TSMC PDK sentinel values**: TSMC PDKs use `-999*10^n` (e.g., `cth0 = -99900000000.0`) as "use default" markers. These extreme values cause OSDI "Parameter CTH0 is out of bounds!" errors during init. **Fix**: `scripts/generate_naive_tsmc.py` filters sentinel values (abs > 1e9 and string starts with "999") during naive modelcard generation. TSMC5 was the only node affected (CTH0 sentinel); TSMC7/12/16 had no sentinels.

- **Multi-node naive modelcard generation**: `scripts/generate_naive_tsmc.py` supports all 4 TSMC FinFET nodes (TSMC5/7/12/16) with `--tech`, `--pdk`, `--output`, `--devices`, `--lengths` arguments. Uses `_extract_model_params()` and `_find_length_variant()` from `pycmg.ctypes_host` to merge `.global` + variant parameters.

### ASAP7 Deep Dive Analysis (2026-02-13 Round 3)
- **Critical parameter storage bug**: Both `parse_modelcard()` and `_extract_model_params()` stored parameters with original case (e.g., "EOT", "L", "NFIN") instead of lowercase. This caused parameter lookup failures when the code tried to access them using lowercase comparisons. Fixed by storing all parameters as lowercase: `parsed_params[_to_lower(key)] = parsed`.
- **nfin default value bug**: The `nfin` default value (1.0) was set but never stored back to `parsed_params` because the code had a double-assignment pattern that left the last conditional branch without a storage statement. Fixed by using a single assignment at the end after all conditionals.
- **ASAP7 path configuration**: Test file had hardcoded path `asap7_pdk_r1p7/models/hspice` but actual directory is `ASAP7`. Fixed by updating path.
- **ASAP7 PMOS DEVTYPE issue RESOLVED**: PMOS models exhibited inverted behavior (conducted at positive Vg) due to missing `devtype` parameter. Standard ASAP7 files omit this parameter. Fixed by auto-injecting `devtype=0.0` for PMOS and `devtype=1.0` for NMOS in `parse_modelcard()` and `_extract_model_params()`. Original modelcard files remain unmodified.
- **Test infrastructure gap**: ASAP7 tests only verify NMOS devices; PMOS verification tests can now be added since DEVTYPE issue is resolved.

### ASAP7 PMOS DEVTYPE Auto-Injection (2026-02-13 Round 4)

- **DEVTYPE auto-injection**: BSIM-CMG v107 uses integer parameter `DEVTYPE = 1` for NMOS (ntype) and `DEVTYPE = 0` for PMOS (ptype) to distinguish device types. Standard ASAP7 modelcards omit this parameter, causing PMOS models to conduct at positive Vg instead of negative Vg (inverted behavior).

- **Solution**: Implemented automatic DEVTYPE injection in both `parse_modelcard()` and `_extract_model_params()` functions:

  - Detects if `devtype` is missing from parsed parameters

  - For PMOS models: injects `devtype = 0.0`

  - For NMOS models: injects `devtype = 1.0`

- **Implementation**: Applied to both ASAP7 and TSMC7 parsing functions for consistency

- **Result**: Original ASAP7 modelcard files remain unmodified; PMOS models now work correctly without manual workarounds

- **Verification**: All 16 API and ASAP7 tests pass; DEVTYPE injection verified via Python test


### Modelcard Parsing & Parameter Handling (2026-02-13 Round 1)
- **Double assignment bug in `_parse_params()`**: The original code had `parsed_params[key] = parsed` followed by conditional blocks that modified `parsed` without storing back. This caused `nfin` defaults to never be applied. Fixed by using `if-elif-elif` chain with single assignment at end.
- **SPICE suffix capture**: When updating regex patterns, ensure the `[a-zA-Z]*` suffix pattern remains INSIDE the value capture group, otherwise suffixes like `n`, `p`, `u` are lost during parsing.
- **Scientific notation regex**: The pattern `[0-9eE+\-\.]+` was fragile because it matched `+` and `-` in any position. Use `[0-9]*\.?[0-9]+(?:[eE][+\-]?[0-9]+)?` for proper scientific notation.
- **EOTACC clamping inconsistency**: Different thresholds were used in `parse_modelcard()` vs `_make_ngspice_modelcard()`. Standardized to `<= 1.0e-10` → `1.1e-10` across all locations (Python, C++ CLI, C++ bindings).
- **Parameter validation**: Added checks for NaN, inf, and inappropriate negative values in `OsdiModel.set_param()` to prevent silent corruption.

### Case Sensitivity & Parameter Storage (2026-02-13 Round 2)
- **Case-insensitive parameter storage**: Both `parse_modelcard()` and `_extract_model_params()` were storing parameters with original case from files, but used `_to_lower()` for comparisons. This caused lookup failures. Fixed by storing all parameters as lowercase keys: `params[_to_lower(key)] = parsed`.
- **TSMC7 PDK robustness**: Added explicit `.global` variant handling in `_find_length_variant()` with warning messages for unexpected non-numeric suffixes. Improved error messages when `.global` model is missing.

### Testing & Verification (2026-02-13 Round 1)
- **Assertion tolerance selection**: The `_assert_close()` function was using `ABS_TOL_I` (1e-9) for ALL parameters, but charges need `ABS_TOL_Q` (1e-18). Added auto-selection based on parameter name.
- **Temperature list completeness**: Test documentation mentioned -40°C but `TEST_TEMPS` list was missing it. Added -40.0°C for comprehensive temperature coverage.
- **Model file naming**: PVT_CORNERS dict used hard-coded `.pm` extensions that didn't match actual files. Changed to base patterns for glob matching.

### Documentation (2026-02-13 Round 2)
- **Temperature units documentation**: Added comprehensive docstrings explaining that ALL temperatures in the module are in KELVIN. Provided conversion formula `temp_K = temp_C + 273.15` and practical examples for common temperatures (-40°C, 27°C, 85°C, 125°C).
- **Accessible documentation**: Users can now access via `help(pycmg.ctypes_host)`, `help(Model)`, `help(Instance)`, etc.

### Code Quality (2026-02-13 Round 1 & 2)
- **Duplicate code removal**: Removed 33 lines of duplicate code in `_find_length_variant()` that was processing variants twice.
- **Error handling**: Added helpful error messages in `parse_tsmc7_pdk()` when `.global` model is missing, with diagnostic information.

### Earlier Bugs
- Modelcard parsing must handle spaced `PARAM = VALUE` and exponent `1e+22`; otherwise key params (NBODY/NSD/NSEG/GEOMOD) silently default and mismatch ngspice.
- OSDI init out-of-bounds errors should be treated as warnings (matching ngspice behavior), not fatal.
- Some OSDI params are integer-typed; read/write using `PARA_TY_INT` to avoid garbage values.
- Internal-node DC solve must use residuals/Jacobian with cleared buffers; once params are correct, residuals match ngspice currents.
- Do not pass `prev_solve` to OSDI unless it is explicitly initialized; uninitialized `prev_solve` breaks DC/AC comparisons.
- Stress tests must align NGSPICE sign conventions: compare `i(vx)` directly to pycmg currents (no sign flip) for OP.

## Gap Checklist (Inventory vs Workflow)
- ✅ OSDI build pipeline: CMake builds `.osdi` via OpenVAF.
- ✅ C++ OSDI host: implemented in `cpp/osdi_host.cpp`.
- ✅ Python ctypes host: `pycmg/ctypes_host.py` exposes `Model`, `Instance`, `eval_dc`, `eval_tran`.
- ✅ Modelcard parsing: `pycmg/ctypes_host.py` includes SPICE-compatible parser with unit suffix support.
- ✅ Verification utilities: `pycmg/testing.py` provides NGSPICE comparison helpers.
- ✅ ASAP7 verification: `tests/test_asap7.py` runs DC/AC/TRAN across PVT corners.
- ✅ TSMC7 verification: `tests/test_tsmc7.py` runs NMOS/PMOS with naive modelcards.
- ✅ TSMC5 verification: `tests/test_tsmc5.py` runs NMOS/PMOS with naive modelcards.
- ✅ TSMC12 verification: `tests/test_tsmc12.py` runs NMOS/PMOS with naive modelcards.
- ✅ TSMC16 verification: `tests/test_tsmc16.py` runs NMOS/PMOS with naive modelcards.
- ✅ Environment override: set `ASAP7_MODELCARD` to a file or directory to redirect ASAP7 inputs.
- ⚠️ PyBind11 layer: `cpp/pycmg_bindings.cpp` exists but ctypes implementation is currently used.

## Technology Modelcard Verification

### ASAP7 (7nm PDK)
- **Status**: ✅ Fully verified
- **Test File**: `tests/test_asap7.py`
- **Coverage**: TT, SS, FF corners at -40°C, 27°C, 85°C, 125°C
- **Outputs**: All 18 model outputs (currents, derivatives, charges)
- **Result**: PyCMG and NGSPICE produce binary-identical results within specified tolerances

### TSMC7 (Taiwan Semiconductor 7nm)
- **Status**: ✅ Verified (2026-02-14)
- **Test File**: `tests/test_tsmc7.py`
- **Modelcards**: `tech_model_cards/TSMC7/naive/` directory
  - NMOS: `nch_svt_mac_l16nm.l` (L=16nm, TFIN=6nm, NFIN=2)
  - PMOS: `pch_lvt_mac_l20nm.l` (L=20nm to avoid NGSPICE convergence failure)
- **Coverage**: 4 tests covering NMOS SVT, PMOS LVT, temperature sweep, voltage sweep
- **Test Results**: All 4 tests pass

| Test | Description | Result |
|------|-------------|--------|
| `test_tsmc7_nmos_svt_op` | NMOS operating point at L=16nm | ✅ Pass |
| `test_tsmc7_pmos_lvt_op` | PMOS operating point at L=20nm | ✅ Pass |
| `test_tsmc7_temperature_sweep` | -40°C to 125°C | ✅ Pass |
| `test_tsmc7_voltage_sweep` | Vg sweep 0V to 0.8V | ✅ Pass |

**Key Implementation Details**:
- **Modelcard baking**: `_bake_inst_params_into_modelcard()` injects instance params (L, TFIN, NFIN) before the closing `)` of the `.model` block
- **NGSPICE OSDI limitation**: Cannot accept instance params on device line; must be in `.model` block
- **PMOS L=16nm caveat**: Invalid `PDIBL2_i=-0.118` from binning causes NGSPICE convergence failure; use L≥20nm
- **Tolerances**: ABS_TOL_I=1e-9, ABS_TOL_Q=1e-18, REL_TOL=5e-3

### TSMC5 (Taiwan Semiconductor 5nm)
- **Status**: ✅ Verified (2026-02-14)
- **Test File**: `tests/test_tsmc5.py`
- **Modelcards**: `tech_model_cards/TSMC5/naive/` directory
  - NMOS: `nch_svt_mac_l16nm.l` (L=16nm, TFIN=6nm, NFIN=2)
  - PMOS: `pch_lvt_mac_l20nm.l` (L=20nm, TFIN=6nm, NFIN=2)
- **Coverage**: 4 tests (NMOS SVT OP, PMOS LVT OP, temperature sweep, voltage sweep)
- **Vdd**: 0.65V (5nm core voltage)
- **Sentinel Fix**: CTH0=-99900000000.0 filtered during naive modelcard generation

| Test | Description | Result |
|------|-------------|--------|
| `test_tsmc5_nmos_svt_op` | NMOS operating point at L=16nm | ✅ Pass |
| `test_tsmc5_pmos_lvt_op` | PMOS operating point at L=20nm | ✅ Pass |
| `test_tsmc5_temperature_sweep` | -40°C to 125°C | ✅ Pass |
| `test_tsmc5_voltage_sweep` | Vg/Vd sweep at Vdd=0.65V | ✅ Pass |

### TSMC12 (Taiwan Semiconductor 12nm)
- **Status**: ✅ Verified (2026-02-14)
- **Test File**: `tests/test_tsmc12.py`
- **Modelcards**: `tech_model_cards/TSMC12/naive/` directory
  - NMOS: `nch_svt_mac_l16nm.l` (L=16nm, TFIN=6nm, NFIN=2)
  - PMOS: `pch_lvt_mac_l20nm.l` (L=20nm, TFIN=6nm, NFIN=2)
- **Coverage**: 4 tests (NMOS SVT OP, PMOS LVT OP, temperature sweep, voltage sweep)
- **Vdd**: 0.80V (12nm core voltage)

| Test | Description | Result |
|------|-------------|--------|
| `test_tsmc12_nmos_svt_op` | NMOS operating point at L=16nm | ✅ Pass |
| `test_tsmc12_pmos_lvt_op` | PMOS operating point at L=20nm | ✅ Pass |
| `test_tsmc12_temperature_sweep` | -40°C to 125°C | ✅ Pass |
| `test_tsmc12_voltage_sweep` | Vg/Vd sweep at Vdd=0.80V | ✅ Pass |

### TSMC16 (Taiwan Semiconductor 16nm)
- **Status**: ✅ Verified (2026-02-14)
- **Test File**: `tests/test_tsmc16.py`
- **Modelcards**: `tech_model_cards/TSMC16/naive/` directory
  - NMOS: `nch_svt_mac_l16nm.l` (L=16nm, TFIN=6nm, NFIN=2)
  - PMOS: `pch_lvt_mac_l20nm.l` (L=20nm, TFIN=6nm, NFIN=2)
- **Coverage**: 4 tests (NMOS SVT OP, PMOS LVT OP, temperature sweep, voltage sweep)
- **Vdd**: 0.80V (16nm core voltage)

| Test | Description | Result |
|------|-------------|--------|
| `test_tsmc16_nmos_svt_op` | NMOS operating point at L=16nm | ✅ Pass |
| `test_tsmc16_pmos_lvt_op` | PMOS operating point at L=20nm | ✅ Pass |
| `test_tsmc16_temperature_sweep` | -40°C to 125°C | ✅ Pass |
| `test_tsmc16_voltage_sweep` | Vg/Vd sweep at Vdd=0.80V | ✅ Pass |
