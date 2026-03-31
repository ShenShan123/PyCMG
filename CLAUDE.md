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
├── tests/                    # Test suite (333 tests)
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
│   ├── test_transient_vt.py # Transient Vt variant verification vs NGSPICE (NMOS+PMOS)
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

* **`tests/`**: Test suite (333 tests total)
  - `test_api.py`: Quick smoke tests for public API (no NGSPICE comparison)
  - `test_dc_jacobian.py`: DC Jacobian verification, NMOS+PMOS across all 5 base technologies
  - `test_dc_regions.py`: DC operating region tests, NMOS+PMOS across all 5 base technologies
  - `test_transient.py`: Transient waveform verification, NMOS+PMOS across all 5 base technologies
  - `test_transient_vt.py`: Transient Vt variant verification, NMOS+PMOS across 16 Vt flavors (32 tests)
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
    * **Transient Vt tests** (`tests/test_transient_vt.py`): Transient waveform verification for Vt variants
      - Tests all 16 Vt variants using Tier 2 registry
      - Same methodology as test_transient.py (sequential stepping, quasi-steady comparison)
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

## Lessons from Bugs (Keep Coming)

### TSMC PMOS NFIN-Group Variant Selection Bug (2026-03-31)

- **Bug**: `_find_length_variant()` in `pycmg/parser.py` only matched on L (lmin/lmax), ignoring NFIN (nfinmin/nfinmax). TSMC PDKs organize variants as a 2D grid of (L_bin x NFIN_group). The function returned the first L match, which for PMOS was the highest NFIN group (e.g., nfinmin=20 for TSMC7 pch), causing wrong binning coefficients when actual NFIN was small (1-3).

- **Root cause**: TSMC7 pch_lvt_mac has 36 variants: 6 L bins x 6 NFIN groups. Multiple variants share the same lmin/lmax but have different nfinmin/nfinmax. Without NFIN matching, variant 5 (nfinmin=20) was always selected for L=16nm, even when NFIN=1 needed variant 35 (nfinmin=1).

- **Fix**: Added `NFIN: Optional[float]` parameter to `_find_length_variant()` and threaded it through `parse_tsmc_pdk()` → `generate_naive_tsmc_modelcard()` → `resolve_modelcard()`. Added `_scan_all_variants()` helper and `scan_pdk_geometry_combos()` for PDK introspection. Replaced `SweepConfig.l_multipliers`/`nfins` with `sweep_geometry: bool` flag that enumerates PDK-defined (lmin, nfinmin) and (lmin, nfinmax) pairs.

- **Design**: Gate lengths and NFIN ranges are binned into PDK-defined discrete values. The sweep now uses these discrete (L, NFIN) combinations directly from the PDK variant structure, not arbitrary user-defined multipliers. For ASAP7 (no binning), TSMC7's NFIN boundaries are used as reference, matched per device type.

- **Lesson**: TSMC PDK variants are indexed by BOTH L range AND NFIN range. Any function that selects a variant must match on both dimensions, not just L. The `_scan_all_variants()` function now provides a single point of truth for parsing variant metadata.

### Capacitance Sign Convention in _condense_caps() (2026-02-19)

- **Bug**: Off-diagonal capacitances (cgd, cgs, cdg) returned by `_condense_caps()` in `pycmg/model.py` (formerly `pycmg/ctypes_host.py`) had the wrong sign, causing mismatches against NGSPICE `@n1[cXX]` operating-point variables.

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

- **Testing**: Added `test_ac_caps.py` with `run_ngspice_ac()` helper in `tests/helpers.py` (formerly `pycmg/testing.py`) to verify all 5 capacitance elements across all 5 technologies.

### PMOS Transient Netlist Generation (2026-02-19)

- **Issue**: `run_ngspice_transient()` only generated NMOS-style netlists (drain at Vdd, source at 0V, gate pulse from 0 to Vdd). PMOS requires inverted biasing: drain at 0V, source at Vdd, gate pulse from Vdd to 0V.

- **Fix**: Added `device_type: str = "nmos"` parameter to `run_ngspice_transient()`. When `device_type="pmos"`, the netlist swaps drain/source voltage sources and inverts the gate pulse direction.

### Multi-Technology Verification & NGSPICE OSDI Limitations (2026-02-14)

- **NGSPICE OSDI does NOT support instance-line parameters**: Unlike HSPICE or Spectre, NGSPICE's OSDI interface cannot accept instance parameters on the device line (e.g., `N1 d g s e model L=16e-9` fails silently). All geometric parameters (L, TFIN, NFIN) must be **baked into the `.model` block** in the modelcard file.

- **Modelcard baking for NGSPICE**: The `_bake_inst_params_into_modelcard()` function in `tests/helpers.py` (formerly `pycmg/testing.py`) inserts instance params before the closing `)` of the `.model` block. Critical: detect `stripped == ')'` to insert BEFORE the bracket, not after.

- **PMOS DEVTYPE in multi-model files**: When a modelcard contains multiple `.model` blocks (e.g., NMOS + PMOS in one file), `Model()` must pass `model_name` to `parse_modelcard(target=...)` so the correct block is parsed. Otherwise PMOS inherits DEVTYPE=1 from the first (NMOS) model, causing inverted behavior.

- **TSMC7 PMOS L=16nm NGSPICE convergence failure**: At L=16nm, TSMC7 PMOS naive modelcards have binning parameters that produce invalid `PDIBL2_i=-0.118`, causing NGSPICE "Timestep too small" DC convergence failure. PyCMG single-shot evaluation doesn't fail (no iterative solver), making comparison impossible. **Workaround**: Use L=20nm or larger for PMOS verification.

- **Stale test files with `sys.exit(1)`**: Module-level `sys.exit(1)` calls in test files crash pytest collection for the entire `tests/` directory. Clean up stale/scratch test files before running `pytest tests/`.

- **TSMC7 naive modelcards**: Use `nch_svt_mac_l16nm.l` (NMOS) and `pch_lvt_mac_l20nm.l` (PMOS). These contain pre-baked geometric params but require additional instance-param injection for NGSPICE compatibility.

- **TSMC PDK sentinel values**: TSMC PDKs use `-999*10^n` (e.g., `cth0 = -99900000000.0`) as "use default" markers. These extreme values cause OSDI "Parameter CTH0 is out of bounds!" errors during init. **Fix**: `scripts/generate_naive_tsmc.py` filters sentinel values (abs > 1e9 and string starts with "999") during naive modelcard generation. TSMC5 was the only node affected (CTH0 sentinel); TSMC7/12/16 had no sentinels.

- **Multi-node naive modelcard generation**: `scripts/generate_naive_tsmc.py` supports all 4 TSMC FinFET nodes (TSMC5/7/12/16) with `--tech`, `--pdk`, `--output`, `--devices`, `--lengths` arguments. Uses `_extract_model_params()` and `_find_length_variant()` from `pycmg.parser` (formerly `pycmg.ctypes_host`) to merge `.global` + variant parameters.

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
- **Accessible documentation**: Users can now access via `help(pycmg.model)`, `help(pycmg.core)`, `help(Model)`, `help(Instance)`, etc.

### Code Quality (2026-02-13 Round 1 & 2)
- **Duplicate code removal**: Removed 33 lines of duplicate code in `_find_length_variant()` that was processing variants twice.
- **Error handling**: Added helpful error messages in `parse_tsmc7_pdk()` when `.global` model is missing, with diagnostic information.

### OSDI Parameter Access Flags for Reading Model Values (2026-03-31)

- **Bug**: `enumerate_model_params()` in `pycmg/sensitivity.py` returned an empty list despite the model having 230+ parameters. The function used `ACCESS_FLAG_READ` (0) to obtain pointers to model parameter values.

- **Root cause**: `ACCESS_FLAG_READ` (0) returns null pointers for model-level parameters that haven't been explicitly written through `apply_param()`. The `Model` constructor stores modelcard params in a Python dict (`_modelcard_params`) but does NOT apply them to the OSDI buffer — that only happens during `Instance.__init__()`. Even after Instance creation populates the buffer, `ACCESS_FLAG_READ` may still return null for some parameters.

- **Fix**: Use `ACCESS_FLAG_SET` (1) instead of `ACCESS_FLAG_READ` (0) for reading model parameter values from the OSDI buffer. `ACCESS_FLAG_SET` is what `apply_param()` and `OsdiModel.set_param()` use, and it reliably returns valid pointers. The returned pointer can be dereferenced for reading just as well as writing.

- **Lesson**: When reading OSDI model parameters via `desc.access()`, always use `ACCESS_FLAG_SET` (1), not `ACCESS_FLAG_READ` (0). Also, ensure `enumerate_model_params()` is called AFTER creating a baseline `Instance`, which triggers `apply_param()` for all modelcard values and populates the OSDI model buffer.

### TSMC5 PMOS LVT Binning Parameter Failure (2026-03-31)

- **Bug**: TSMC5 `pmos_lvt` produces NaN for all outputs at all tested gate lengths (min_l and 2x min_l). Sensitivity analysis returned all-zero rankings.

- **Root cause**: The TSMC5 PMOS LVT naive modelcard generates invalid binning-derived internal parameters (similar to TSMC7 PMOS L=16nm PDIBL2 issue). Unlike single-shot evaluation failures that are silent, this affects the entire model — no valid bias point works.

- **Workaround**: Use `pmos_svt` or `pmos_elvt` for TSMC5 PMOS sensitivity analysis instead of `pmos_lvt`. The sensitivity code now emits a warning when all baseline evaluations return NaN, guiding the user to try a different device or L multiplier.

- **Lesson**: Not all TSMC naive modelcard + device + length combinations produce valid models. Before running sensitivity analysis or data generation on a new device, verify baseline evaluation works with a quick smoke test. The `scripts/sensitivity_analysis.py` tool now handles NaN baselines gracefully (warns instead of silently producing zeros).

### Extended Voltage Range Design (2026-03-31)

- **Design decision**: Extended voltage sweep uses a `voltage_scale` multiplier (default 1.0) rather than separate `vg_max`/`vd_max` parameters. This is cleaner because the extended range is always relative to the technology's VDD.

- **Key insight**: The dense region around Vth must remain at `±0.15*vdd` (nominal VDD), NOT `±0.15*v_max`. Threshold voltage is a physical property that doesn't change when you extend the sweep range. Only the sparse grid upper bound and Vd upper bound use `v_max = vdd * voltage_scale`.

- **PMOS handling**: When `voltage_scale=2.0`, `build_nodes()` produces negative gate voltages for PMOS (e.g., `Vg = vdd - 2*vdd = -vdd`). This is physically valid (deep accumulation regime) and BSIM-CMG handles it correctly. No special PMOS treatment needed.

### Earlier Bugs
- Modelcard parsing must handle spaced `PARAM = VALUE` and exponent `1e+22`; otherwise key params (NBODY/NSD/NSEG/GEOMOD) silently default and mismatch ngspice.
- OSDI init out-of-bounds errors should be treated as warnings (matching ngspice behavior), not fatal.
- Some OSDI params are integer-typed; read/write using `PARA_TY_INT` to avoid garbage values.
- Internal-node DC solve must use residuals/Jacobian with cleared buffers; once params are correct, residuals match ngspice currents.
- Do not pass `prev_solve` to OSDI unless it is explicitly initialized; uninitialized `prev_solve` breaks DC/AC comparisons.
- Stress tests must align NGSPICE sign conventions: compare `i(vx)` directly to pycmg currents (no sign flip) for OP.

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
- Transient Vt tests: `tests/test_transient_vt.py` NMOS+PMOS across 16 Vt flavors.
- AC Capacitance tests: `tests/test_ac_caps.py` NMOS across all 5 base technologies.
- Body bias tests: `tests/test_body_bias.py` NMOS+PMOS across all 5 base technologies.
- Temperature tests: `tests/test_temperature.py` NMOS+PMOS at -40C, 85C, 125C (ASAP7).
- NFIN scaling tests: `tests/test_nfin_scaling.py` NMOS+PMOS scaling sanity (ASAP7, PyCMG-only).
- Vt variant tests: `tests/test_vt_variants.py` NMOS+PMOS across 16 Vt flavors (3 regions each).
- API tests: `tests/test_api.py` quick smoke tests (no NGSPICE).
- Environment override: set `ASAP7_MODELCARD` to a file or directory to redirect ASAP7 inputs.
- C++ OSDI host: removed (was `cpp/osdi_host.cpp`); Python uses ctypes directly via `pycmg/core.py`.
- Naive modelcard generation: `scripts/generate_naive_tsmc.py` generates all Vt flavors from raw PDKs.
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

### Verification Test Types (266 tests total)

| Test File | Coverage | Description |
|-----------|----------|-------------|
| `test_dc_jacobian.py` | 5 base techs, NMOS+PMOS | DC derivatives (gm, gds, gmb) vs NGSPICE |
| `test_dc_regions.py` | 5 base techs, NMOS+PMOS | DC operating regions + gmb verification vs NGSPICE |
| `test_transient.py` | 5 base techs, NMOS+PMOS | Transient charge/discharge waveforms vs NGSPICE |
| `test_transient_vt.py` | 16 Vt variants, NMOS+PMOS | Transient waveforms vs NGSPICE (32 tests) |
| `test_ac_caps.py` | 5 base techs, NMOS | AC capacitances (cgg, cgd, cgs, cdg, cdd) vs NGSPICE |
| `test_body_bias.py` | 5 base techs, NMOS+PMOS | Body bias (Ve != 0) verification vs NGSPICE |
| `test_temperature.py` | ASAP7, NMOS+PMOS | Temperature (-40C, 85C, 125C) verification vs NGSPICE |
| `test_nfin_scaling.py` | ASAP7, NMOS+PMOS | NFIN scaling sanity (PyCMG-only, no NGSPICE) |
| `test_vt_variants.py` | 16 Vt variants, NMOS+PMOS | DC sat/linear/subthreshold vs NGSPICE (96 tests) |
| `test_api.py` | Smoke only | Basic functionality, no NGSPICE |

### Key Implementation Details

- **Modelcard baking**: `_bake_inst_params_into_modelcard()` in `tests/helpers.py` injects instance params (L, TFIN, NFIN, DEVTYPE) before the closing `)` of the `.model` block
- **NGSPICE OSDI limitation**: Cannot accept instance params on device line; must be in `.model` block
- **PMOS L=16nm caveat**: For TSMC nodes, invalid binning parameters at L=16nm cause NGSPICE convergence failure; use L=20nm for PMOS
- **Tolerances**: ABS_TOL_I=1e-9, ABS_TOL_Q=1e-18, ABS_TOL_C=1e-18 (capacitance), REL_TOL=5e-3, REL_TOL_CAP=1e-2 (1% for capacitance)
- **DEVTYPE injection**: Automatic injection of devtype=1.0 (NMOS) or devtype=0.0 (PMOS) for models missing this parameter
- **Sentinel filtering**: TSMC PDK sentinel values (-999*10^n) filtered during naive modelcard generation
- **Tiered registry design**: Tier 1 entries use shared param templates with `.copy()` to prevent cross-contamination; Tier 2 uses helper functions (`_asap7_entry`, `_tsmc_entry`) for consistency
