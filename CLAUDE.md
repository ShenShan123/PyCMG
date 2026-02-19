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
│   ├── code/                 # Main model Verilog-A files
│   ├── README.txt           # Original BSIM-CMG documentation
│   ├── *.pdf                 # Technical manuals (3 PDFs)
├── pycmg/                    # Python package
│   ├── __init__.py          # Public API exports
│   ├── ctypes_host.py       # Core OSDI interface (Model, Instance, eval_dc, eval_tran)
│   └── testing.py           # Verification utilities
├── cpp/                      # C++ OSDI host (reference, not used by Python ctypes)
│   ├── osdi_host.h          # Header file
│   ├── osdi_host.cpp        # Core host implementation
│   ├── osdi_cli.cpp         # CLI inspector tool
│   └── osdi_eval.cpp        # CLI evaluator tool
├── tests/                    # Test suite (138 tests)
│   ├── __init__.py          # Package init
│   ├── conftest.py          # Technology registry (5 techs: ASAP7, TSMC5, TSMC7, TSMC12, TSMC16)
│   ├── test_api.py          # Public API tests (smoke, basic functionality)
│   ├── test_ac_caps.py      # AC capacitance verification vs NGSPICE
│   ├── test_body_bias.py    # Body bias (Ve != 0) verification vs NGSPICE
│   ├── test_dc_jacobian.py  # DC Jacobian verification vs NGSPICE (NMOS+PMOS)
│   ├── test_dc_regions.py   # DC operating region tests vs NGSPICE (NMOS+PMOS)
│   ├── test_nfin_scaling.py # NFIN scaling sanity tests (PyCMG-only)
│   ├── test_temperature.py  # Temperature verification vs NGSPICE
│   └── test_transient.py    # Transient waveform verification vs NGSPICE (NMOS+PMOS)
├── scripts/                  # Utility scripts
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
└── CLAUDE.md                 # This file
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
  - Technology modelcard handling
  - Stress testing utilities

* **`tests/conftest.py`**: Technology registry
  - `TECHNOLOGIES` dict: 5 technologies (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16)
  - `get_tech_modelcard()`: Retrieves modelcard path, model name, and instance params
  - Provides parametrization for all verification tests

* **`tests/`**: Test suite (138 tests total)
  - `test_api.py`: Quick smoke tests for public API (no NGSPICE comparison)
  - `test_dc_jacobian.py`: DC Jacobian verification, NMOS+PMOS across all 5 technologies
  - `test_dc_regions.py`: DC operating region tests, NMOS+PMOS across all 5 technologies
  - `test_transient.py`: Transient waveform verification, NMOS+PMOS across all 5 technologies
  - `test_ac_caps.py`: AC capacitance verification (cgg, cgd, cgs, cdg, cdd) vs NGSPICE
  - `test_body_bias.py`: Body bias (Ve != 0) verification across all 5 technologies
  - `test_temperature.py`: Temperature verification (-40C, 85C, 125C) vs NGSPICE
  - `test_nfin_scaling.py`: NFIN scaling sanity tests (PyCMG-only)

## PyCMG Output Coverage

PyCMG provides comprehensive model outputs covering currents, derivatives, charges, and capacitances. All outputs are verified against NGSPICE using the exact same OSDI binary.

### Supported Outputs (18 total)

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

**Option A: Manual CMake build (Recommended)**
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

**Option B: Direct OpenVAF compilation**
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
    * **Technology Registry** (`tests/conftest.py`): Centralized parametrization across 5 technologies
      - ASAP7, TSMC5, TSMC7, TSMC12, TSMC16
      - Each tech has vdd, modelcard paths, instance params
    * **DC Jacobian tests** (`tests/test_dc_jacobian.py`): Verify DC derivatives vs NGSPICE
      - Tests all 5 technologies using the registry
      - Covers gm, gds, gmb derivatives
    * **DC Region tests** (`tests/test_dc_regions.py`): DC operating region verification
      - Tests all 5 technologies using the registry
      - Covers subthreshold, linear, saturation regions
    * **Transient tests** (`tests/test_transient.py`): Transient waveform verification
      - Tests all 5 technologies using the registry
      - Covers charge/ discharge waveforms
    * **API tests** (`tests/test_api.py`): Quick smoke tests
      - Basic functionality verification
      - No NGSPICE comparison (fast execution)

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

### Capacitance Sign Convention in _condense_caps() (2026-02-19)

- **Bug**: Off-diagonal capacitances (cgd, cgs, cdg) returned by `_condense_caps()` in `pycmg/ctypes_host.py` had the wrong sign, causing mismatches against NGSPICE `@n1[cXX]` operating-point variables.

- **Root cause**: The OSDI reactive Jacobian (dQ/dV) uses **Y-matrix convention**, where off-diagonal entries are negative (e.g., `dQg/dVd < 0`). However, SPICE capacitance variables like `@n1[cgd]` use the **opposite sign convention** for off-diagonals — they report `cgd = -dQg/dVd > 0`. The `_condense_caps()` function was extracting raw matrix entries without applying this sign flip.

- **Fix**: Negate off-diagonal entries when extracting from the condensed capacitance matrix:
  ```python
  # Diagonal: no negation
  caps["cgg"] = float(c_condensed[g, g])
  caps["cdd"] = float(c_condensed[d, d])
  # Off-diagonal: negate to match SPICE convention
  caps["cgd"] = -float(c_condensed[g, d])
  caps["cgs"] = -float(c_condensed[g, s])
  caps["cdg"] = -float(c_condensed[d, g])
  ```

- **Lesson**: When extracting small-signal parameters from OSDI Jacobian matrices, always verify sign conventions against NGSPICE. The OSDI binary returns raw matrix entries in Y-matrix convention; SPICE tools may present them with different signs. Diagonal elements (cgg, cdd) are always positive and need no sign flip. Off-diagonal elements (cgd, cgs, cdg) require negation to match SPICE convention.

- **Testing**: Added `test_ac_caps.py` with `run_ngspice_ac()` helper in `pycmg/testing.py` to verify all 5 capacitance elements across all 5 technologies.

### PMOS Transient Netlist Generation (2026-02-19)

- **Issue**: `run_ngspice_transient()` only generated NMOS-style netlists (drain at Vdd, source at 0V, gate pulse from 0 to Vdd). PMOS requires inverted biasing: drain at 0V, source at Vdd, gate pulse from Vdd to 0V.

- **Fix**: Added `device_type: str = "nmos"` parameter to `run_ngspice_transient()`. When `device_type="pmos"`, the netlist swaps drain/source voltage sources and inverts the gate pulse direction.

### Multi-Technology Verification & NGSPICE OSDI Limitations (2026-02-14)

- **NGSPICE OSDI does NOT support instance-line parameters**: Unlike HSPICE or Spectre, NGSPICE's OSDI interface cannot accept instance parameters on the device line (e.g., `N1 d g s e model L=16e-9` fails silently). All geometric parameters (L, TFIN, NFIN) must be **baked into the `.model` block** in the modelcard file.

- **Modelcard baking for NGSPICE**: The `_bake_inst_params_into_modelcard()` function in `pycmg/testing.py` inserts instance params before the closing `)` of the `.model` block. Critical: detect `stripped == ')'` to insert BEFORE the bracket, not after.

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

- **Implementation**: Applied to both ASAP7 and TSMC parsing functions for consistency

- **Result**: Original ASAP7 modelcard files remain unmodified; PMOS models now work correctly without manual workarounds

- **Verification**: DEVTYPE injection verified via Python tests; all verification tests use the technology registry


### Modelcard Parsing & Parameter Handling (2026-02-13 Round 1)
- **Double assignment bug in `_parse_params()`**: The original code had `parsed_params[key] = parsed` followed by conditional blocks that modified `parsed` without storing back. This caused `nfin` defaults to never be applied. Fixed by using `if-elif-elif` chain with single assignment at end.
- **SPICE suffix capture**: When updating regex patterns, ensure the `[a-zA-Z]*` suffix pattern remains INSIDE the value capture group, otherwise suffixes like `n`, `p`, `u` are lost during parsing.
- **Scientific notation regex**: The pattern `[0-9eE+\-\.]+` was fragile because it matched `+` and `-` in any position. Use `[0-9]*\.?[0-9]+(?:[eE][+\-]?[0-9]+)?` for proper scientific notation.
- **EOTACC clamping inconsistency**: Different thresholds were used in `parse_modelcard()` vs `_make_ngspice_modelcard()`. Standardized to `<= 1.0e-10` -> `1.1e-10` across all locations (Python, C++ CLI, C++ bindings).
- **Parameter validation**: Added checks for NaN, inf, and inappropriate negative values in `OsdiModel.set_param()` to prevent silent corruption.

### Case Sensitivity & Parameter Storage (2026-02-13 Round 2)
- **Case-insensitive parameter storage**: Both `parse_modelcard()` and `_extract_model_params()` were storing parameters with original case from files, but used `_to_lower()` for comparisons. This caused lookup failures. Fixed by storing all parameters as lowercase keys: `params[_to_lower(key)] = parsed`.
- **TSMC7 PDK robustness**: Added explicit `.global` variant handling in `_find_length_variant()` with warning messages for unexpected non-numeric suffixes. Improved error messages when `.global` model is missing.

### Testing & Verification (2026-02-13 Round 1)
- **Assertion tolerance selection**: The `_assert_close()` function was using `ABS_TOL_I` (1e-9) for ALL parameters, but charges need `ABS_TOL_Q` (1e-18). Added auto-selection based on parameter name.
- **Temperature list completeness**: Test documentation mentioned -40C but `TEST_TEMPS` list was missing it. Added -40.0C for comprehensive temperature coverage.
- **Model file naming**: PVT_CORNERS dict used hard-coded `.pm` extensions that didn't match actual files. Changed to base patterns for glob matching.

### Documentation (2026-02-13 Round 2)
- **Temperature units documentation**: Added comprehensive docstrings explaining that ALL temperatures in the module are in KELVIN. Provided conversion formula `temp_K = temp_C + 273.15` and practical examples for common temperatures (-40C, 27C, 85C, 125C).
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
- OSDI build pipeline: CMake builds `.osdi` via OpenVAF.
- C++ OSDI host: implemented in `cpp/osdi_host.cpp`.
- Python ctypes host: `pycmg/ctypes_host.py` exposes `Model`, `Instance`, `eval_dc`, `eval_tran`.
- Modelcard parsing: `pycmg/ctypes_host.py` includes SPICE-compatible parser with unit suffix support.
- Verification utilities: `pycmg/testing.py` provides NGSPICE comparison helpers.
- Technology registry: `tests/conftest.py` defines 5 technologies (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16).
- DC Jacobian tests: `tests/test_dc_jacobian.py` NMOS+PMOS across all 5 technologies.
- DC Region tests: `tests/test_dc_regions.py` NMOS+PMOS across all 5 technologies, includes gmb.
- Transient tests: `tests/test_transient.py` NMOS+PMOS across all 5 technologies.
- AC Capacitance tests: `tests/test_ac_caps.py` NMOS across all 5 technologies.
- Body bias tests: `tests/test_body_bias.py` NMOS+PMOS across all 5 technologies.
- Temperature tests: `tests/test_temperature.py` NMOS+PMOS at -40C, 85C, 125C (ASAP7).
- NFIN scaling tests: `tests/test_nfin_scaling.py` NMOS+PMOS scaling sanity (ASAP7, PyCMG-only).
- API tests: `tests/test_api.py` quick smoke tests (no NGSPICE).
- Environment override: set `ASAP7_MODELCARD` to a file or directory to redirect ASAP7 inputs.
- C++ OSDI host: `cpp/osdi_host.cpp` exists as reference; Python uses ctypes directly.

## Technology Modelcard Verification

All verification tests use the centralized technology registry in `tests/conftest.py`, which parametrizes tests across all 5 technologies (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16) with consistent test coverage.

### Technology Registry Coverage

| Technology | Vdd | NMOS Modelcard | PMOS Modelcard | NMOS L | PMOS L |
|------------|-----|----------------|----------------|--------|--------|
| ASAP7 | 0.9V | 7nm_TT_160803.pm | 7nm_TT_160803.pm | 7nm | 7nm |
| TSMC5 | 0.65V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |
| TSMC7 | 0.75V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |
| TSMC12 | 0.80V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |
| TSMC16 | 0.80V | nch_svt_mac_l16nm.l | pch_lvt_mac_l20nm.l | 16nm | 20nm |

### Verification Test Types

| Test File | Coverage | Description |
|-----------|----------|-------------|
| `test_dc_jacobian.py` | All 5 techs, NMOS+PMOS | DC derivatives (gm, gds, gmb) vs NGSPICE |
| `test_dc_regions.py` | All 5 techs, NMOS+PMOS | DC operating regions + gmb verification vs NGSPICE |
| `test_transient.py` | All 5 techs, NMOS+PMOS | Transient charge/discharge waveforms vs NGSPICE |
| `test_ac_caps.py` | All 5 techs, NMOS | AC capacitances (cgg, cgd, cgs, cdg, cdd) vs NGSPICE |
| `test_body_bias.py` | All 5 techs, NMOS+PMOS | Body bias (Ve != 0) verification vs NGSPICE |
| `test_temperature.py` | ASAP7, NMOS+PMOS | Temperature (-40C, 85C, 125C) verification vs NGSPICE |
| `test_nfin_scaling.py` | ASAP7, NMOS+PMOS | NFIN scaling sanity (PyCMG-only, no NGSPICE) |
| `test_api.py` | Smoke only | Basic functionality, no NGSPICE |

### Key Implementation Details

- **Modelcard baking**: `_bake_inst_params_into_modelcard()` in `pycmg/testing.py` injects instance params (L, TFIN, NFIN, DEVTYPE) before the closing `)` of the `.model` block
- **NGSPICE OSDI limitation**: Cannot accept instance params on device line; must be in `.model` block
- **PMOS L=16nm caveat**: For TSMC nodes, invalid binning parameters at L=16nm cause NGSPICE convergence failure; use L=20nm for PMOS
- **Tolerances**: ABS_TOL_I=1e-9, ABS_TOL_Q=1e-18, ABS_TOL_C=1e-18 (capacitance), REL_TOL=5e-3, REL_TOL_CAP=1e-2 (1% for capacitance)
- **DEVTYPE injection**: Automatic injection of devtype=1.0 (NMOS) or devtype=0.0 (PMOS) for models missing this parameter
- **Sentinel filtering**: TSMC PDK sentinel values (-999*10^n) filtered during naive modelcard generation
