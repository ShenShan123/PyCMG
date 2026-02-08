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
│   └── test_integration.py  # NGSPICE comparison tests (representative subset)
├── tech_model_cards/         # Technology model cards
│   └── asap7_pdk_r1p7/      # ASAP7 PDK model files
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
- ✅ Environment override: set `ASAP7_MODELCARD` to a file or directory to redirect ASAP7 inputs.
- ⚠️ PyBind11 layer: `cpp/pycmg_bindings.cpp` exists but ctypes implementation is currently used.
