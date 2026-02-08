# PyCMG Wrapper - BSIM-CMG Python Model Interface

A standalone Python interface for the BSIM-CMG Verilog-A model using OpenVAF/OSDI, with comprehensive NGSPICE-backed verification.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Verification](#verification)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

PyCMG provides a Python interface to evaluate BSIM-CMG compact models through compiled OSDI binaries. It supports:

- **DC Analysis**: Steady-state I-V characterization
- **AC Analysis**: Small-signal capacitance extraction
- **Transient Analysis**: Time-domain simulation
- **NGSPICE Verification**: Automated comparison against NGSPICE ground truth

The interface is ideal for:
- Circuit simulation tools integration
- Device characterization workflows
- Model validation and verification
- Educational and research applications

## Requirements

### Essential Tools
- **OpenVAF**: `/usr/local/bin/openvaf`
- **NGSPICE**: `/usr/local/ngspice-45.2/bin/ngspice` (for verification)
- **CMake**: ≥ 3.15
- **Python**: ≥ 3.10
- **Compiler**: GCC or Clang with C++17 support

### Python Dependencies
```bash
pip install numpy pytest
```

### Environment Variables (Optional)
```bash
# Override default NGSPICE binary
export NGSPICE_BIN=/path/to/ngspice

# Override ASAP7 modelcard location
export ASAP7_MODELCARD=/path/to/modelcards
```

## Quick Start

### 1. Build the OSDI Model

```bash
# From project root
mkdir -p build-deep-verify
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

This generates `build-deep-verify/osdi/bsimcmg.osdi`.

### 2. Basic Python Usage

```python
from pycmg import Model, Instance

# Create model from OSDI binary and modelcard
model = Model(
    osdi_path="build-deep-verify/osdi/bsimcmg.osdi",
    modelcard_path="bsim-cmg-va/benchmark_test/modelcard.nmos",
    model_name="nmos1"
)

# Create device instance with geometry parameters
inst = Instance(
    model=model,
    params={
        "L": 16e-9,     # Channel length: 16 nm
        "TFIN": 8e-9,   # Fin thickness: 8 nm
        "NFIN": 2.0,    # Number of fins: 2
    },
    temperature=300.15  # Temperature: 300.15 K (27°C)
)

# Evaluate at operating point
result = inst.eval_dc({
    "d": 0.5,   # Drain voltage: 0.5 V
    "g": 0.8,   # Gate voltage: 0.8 V
    "s": 0.0,   # Source voltage: 0 V
    "e": 0.0,   # Bulk voltage: 0 V
})

# Access results
print(f"Drain current: {result['id']:.3e} A")
print(f"Transconductance: {result['gm']:.3e} S")
print(f"Gate capacitance: {result['cgg']:.3e} F")
```

### 3. Run Verification Tests

```bash
# Quick smoke tests (no NGSPICE required)
pytest tests/test_api.py -v

# Full ASAP7 verification (NGSPICE required)
pytest tests/test_asap7.py -v

# Integration tests with NGSPICE comparison
pytest tests/test_integration.py -v
```

## Installation

### Prerequisites

**Essential Tools:**
1. **OpenVAF**: Verilog-A to OSDI compiler
   ```bash
   # Download from https://github.com/ngspice/openvaf/releases
   # Or build from source
   git clone https://github.com/ngspice/openvaf.git
   cd openvaf
   cargo build --release
   sudo cp target/release/openvaf /usr/local/bin/
   ```

2. **CMake**: ≥ 3.15
   ```bash
   sudo apt-get install cmake  # Ubuntu/Debian
   ```

3. **Python**: ≥ 3.10 with numpy
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pytest
   ```

**Optional (for verification):**
- NGSPICE: ≥ 45.2
  ```bash
   # Download from http://ngspice.sourceforge.net/
   # Or use package manager
   sudo apt-get install ngspice
  ```

### Clone and Build

**Method 1: Using the build script (Recommended)**
```bash
# Clone repository
git clone <repository-url>
cd pycmg-wrapper

# Run the automated build script
chmod +x build_osdi.sh
./build_osdi.sh
```

**Method 2: Manual CMake build**
```bash
# Step 1: Install dependencies
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11

# Step 2: Create build directory
mkdir -p build-deep-verify
cd build-deep-verify

# Step 3: Configure CMake
# Option A: Let CMake find pybind11 automatically
cmake ..

# Option B: Specify pybind11 path explicitly
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..

# Step 4: Build OSDI model
cmake --build . --target osdi

# Step 5: (Optional) Build verification tools
cmake --build . --target osdi_eval
```

**Method 3: Direct OpenVAF compilation**
```bash
# Compile without CMake (minimal build)
openvaf -I bsim-cmg-va/code -o bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va

# Move to expected location
mkdir -p build-deep-verify/osdi
mv bsimcmg.osdi build-deep-verify/osdi/
```

### Verify Installation

```bash
# Check OSDI binary exists and is valid
ls -lh build-deep-verify/osdi/bsimcmg.osdi
file build-deep-verify/osdi/bsimcmg.osdi
# Should show: "ELF 64-bit LSB shared object" or similar

# Test Python import
python -c "from pycmg import Model; print('✓ PyCMG imported successfully')"

# Quick smoke test
pytest tests/test_api.py::test_parse_number_with_suffix -v
```

### Troubleshooting Installation

**OpenVAF not found:**
```bash
# Check if openvaf is in PATH
which openvaf

# If not found, add to PATH or install
export PATH="/path/to/openvaf:$PATH"
```

**CMake pybind11 error:**
```bash
# Install pybind11 using pip with Tsinghua mirror (China)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11

# Find pybind11 cmake directory
python -m pybind11 --cmakedir

# Pass to CMake explicitly
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..
```

**Missing bsim-cmg-va directory:**
```bash
# If using a git worktree, create symlink to parent repo
ln -s /path/to/pycmg-wrapper/bsim-cmg-va bsim-cmg-va

# Or copy from parent repository
cp -r /path/to/pycmg-wrapper/bsim-cmg-va .
```

**Build succeeds but tests fail:**
```bash
# Verify OSDI binary is valid
nm build-deep-verify/osdi/bsimcmg.osdi | grep osdi

# Check OpenVAF version compatibility
openvaf --version  # Should be v23.5.0 or later
```

## Usage

### Creating a Model

```python
from pycmg import Model

# Method 1: From OSDI binary + modelcard
model = Model(
    osdi_path="path/to/model.osdi",
    modelcard_path="path/to/modelcard.lib",
    model_name="modelname"  # Optional if only one model in file
)

# Method 2: Using modelcard parser
from pycmg import parse_modelcard
parsed = parse_modelcard("path/to/modelcard.lib", target_model_name="nmos1")
print(f"Parsed {len(parsed.params)} parameters")
```

### Creating an Instance

```python
from pycmg import Instance

# Basic instance
inst = Instance(
    model=model,
    params={
        "L": 20e-9,    # Length (m)
        "TFIN": 10e-9, # Fin thickness (m)
        "NFIN": 3.0,   # Number of fins
        "NRS": 1.0,    # Source resistance multiplier
        "NRD": 1.0,    # Drain resistance multiplier
    },
    temperature=300.15  # Kelvin (default: 300.15 K)
)

# Update parameters after creation
inst.set_params({"NFIN": 5.0}, allow_rebind=True)
```

### DC Analysis

```python
# Single operating point
result = inst.eval_dc({
    "d": 0.5,  # Drain voltage
    "g": 0.8,  # Gate voltage
    "s": 0.0,  # Source voltage
    "e": 0.0,  # Bulk voltage
})

# Available outputs:
# - id, ig, is, ie, ids: Terminal currents (A)
#   ids = id - is (drain-source current for common-source configuration)
# - qg, qd, qs, qb: Terminal charges (C)
# - gm, gds, gmb: Transconductances (S)
# - cgg, cgd, cgs, cdg, cdd: Capacitances (F)
```

### Voltage Sweep

```python
import numpy as np

# Id-Vg sweep
vg_values = np.linspace(0, 1.2, 13)
id_values = []
for vg in vg_values:
    result = inst.eval_dc({"d": 0.05, "g": vg, "s": 0.0, "e": 0.0})
    id_values.append(result["id"])

import matplotlib.pyplot as plt
plt.plot(vg_values, id_values)
plt.xlabel("Vg (V)")
plt.ylabel("Id (A)")
plt.show()
```

### Transient Analysis

```python
# Time-domain simulation
time = 1e-9        # Current time (s)
dt = 1e-12         # Time step (s)
nodes = {
    "d": 0.05,
    "g": 0.8,
    "s": 0.0,
    "e": 0.0,
}

result = inst.eval_tran(nodes, time, dt)

# Transient outputs include currents and charges
# Note: Capacitive currents are automatically computed
```

### Temperature Sweep

```python
temperatures = [223.15, 273.15, 323.15, 373.15, 398.15]  # K
id_at_temp = []
for temp_k in temperatures:
    inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0},
                    temperature=temp_k)
    result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
    id_at_temp.append(result["id"])
```

## Running Tests

### Test Organization

Tests are organized by purpose and speed:

| Test Suite | Purpose | Duration | NGSPICE Required |
|------------|---------|----------|-----------------|
| `test_api.py` | Quick API validation | ~5 seconds | No |
| `test_integration.py` | NGSPICE ground truth comparison | ~30 seconds | Yes |
| `test_asap7.py` | Full PVT verification | ~5 minutes | Yes |

### Running Specific Tests

```bash
# Run all tests (slowest)
pytest tests/ -v

# Quick smoke test only
pytest tests/test_api.py -v

# Integration tests with NGSPICE
pytest tests/test_integration.py -v

# ASAP7 verification (comprehensive)
pytest tests/test_asap7.py -v

# Run specific test function
pytest tests/test_api.py::test_eval_dc_smoke -v

# Run with coverage
pytest tests/ --cov=pycmg --cov-report=html
```

### Test Configuration

Tests automatically skip if required artifacts are missing:

```bash
# Missing OSDI binary → tests skipped
# Missing modelcards → tests skipped
# Missing NGSPICE → only test_api.py runs
```

### Using main.py CLI

```bash
# Quick test execution
python main.py test api           # Quick smoke tests
python main.py test integration   # NGSPICE comparison
python main.py test asap7         # Full ASAP7 verification
python main.py test all           # Run all tests
```

## Project Structure

```
pycmg-wrapper/
├── bsim-cmg-va/              # BSIM-CMG Verilog-A source
│   ├── code/                 # Verilog-A source files
│   │   ├── bsimcmg_main.va  # Main model entry point
│   │   ├── bsimcmg.va       # Core model definitions
│   │   ├── bsimcmg_body.include      # Body charge model
│   │   ├── bsimcmg_nqsmod3.va        # Non-quasi-static model
│   │   └── *.include                  # Additional model components
│   └── benchmark_test/       # Example modelcards and netlists
│       ├── modelcard.nmos   # NMOS example parameters
│       └── modelcard.pmos   # PMOS example parameters
├── pycmg/                    # Python package
│   ├── __init__.py          # Public API
│   ├── ctypes_host.py       # OSDI interface (Model, Instance)
│   └── testing.py           # Verification utilities
├── cpp/                      # C++ OSDI host
│   ├── osdi_host.h
│   ├── osdi_host.cpp        # Core host implementation
│   └── osdi_cli.cpp         # CLI tool (osdi_eval)
├── tests/                    # Test suite
│   ├── conftest.py          # Shared fixtures
│   ├── test_api.py          # Quick API tests
│   ├── test_integration.py  # NGSPICE comparison
│   └── test_asap7.py        # ASAP7 PVT verification
├── build-deep-verify/        # Build artifacts (generated)
│   └── osdi/
│       └── bsimcmg.osdi     # Compiled OSDI binary
├── build_osdi.sh             # Automated build script
├── main.py                   # CLI entrypoint
├── CMakeLists.txt            # CMake build configuration
├── CLAUDE.md                 # Development guidelines
└── README.md                 # This file
```

### Verilog-A Source Organization

The BSIM-CMG model is distributed across multiple Verilog-A files:

| File | Purpose |
|------|---------|
| `bsimcmg_main.va` | Main entry point, includes all sub-modules |
| `bsimcmg.va` | Core BSIM-CMG model definitions |
| `bsimcmg_body.include` | Body charge and current models |
| `bsimcmg_nqsmod3.va` | Non-quasi-static (NQS) charge model |
| `bsimcmg_cfringe.include` | Fringe capacitance model |
| `bsimcmg_quasi_static_cv.include` | Quasi-static C-V model |
| `bsimcmg_rdsmod.include` | Drain-source resistance model |
| `common_defs.include` | Common parameter definitions |

**Note:** The `bsimcmg_main.va` file includes all other components using `.include` directives. OpenVAF processes these includes during compilation to generate the final OSDI binary.

## API Reference

### `Model`

```python
Model(osdi_path: str, modelcard_path: str, model_name: str = None)
```

Load an OSDI compiled model with parameters from a modelcard.

**Parameters:**
- `osdi_path`: Path to `.osdi` binary file
- `modelcard_path`: Path to SPICE modelcard (`.lib`, `.l`, `.scn`)
- `model_name`: Model name within modelcard (optional if only one)

**Methods:**
- None (model is a container for parameters)

### `Instance`

```python
Instance(model: Model, params: dict, temperature: float = 300.15)
```

Create a device instance with geometry parameters.

**Parameters:**
- `model`: `Model` object
- `params`: Dictionary of instance parameters (L, TFIN, NFIN, etc.)
- `temperature`: Temperature in Kelvin (default: 300.15 K = 27°C)

**Methods:**

```python
eval_dc(nodes: dict) -> dict
```
Evaluate DC operating point.

**Parameters:**
- `nodes`: Dictionary {"d": vd, "g": vg, "s": vs, "e": ve}

**Returns:** Dictionary with keys:
- Currents: `id`, `ig`, `is`, `ie`, `ids` (A)
  - `ids` = `id` - `is` (drain-source current for common-source configuration)
- Charges: `qg`, `qd`, `qs`, `qb` (C)
- Derivatives: `gm`, `gds`, `gmb` (S)
- Capacitances: `cgg`, `cgd`, `cgs`, `cdg`, `cdd` (F)

```python
eval_tran(nodes: dict, time: float, dt: float,
          prev_state: list = None) -> dict
```
Evaluate transient response.

**Parameters:**
- `nodes`: Terminal voltages
- `time`: Current time (s)
- `dt`: Time step (s)
- `prev_state`: Previous state vector (optional, for internal continuity)

**Returns:** Dictionary with currents and charges

```python
set_params(params: dict, allow_rebind: bool = False)
```
Update instance parameters. May require rebind if topology changes.

### Utility Functions

```python
parse_modelcard(path: str, target_model_name: str = None) -> ParsedModel
```
Parse a SPICE modelcard file.

**Returns:** `ParsedModel` with `name` and `params` (dict)

```python
parse_number_with_suffix(token: str) -> float
```
Parse SPICE number with suffix (e.g., "1n" → 1e-9, "2meg" → 2e6).

## Verification

### Verification Strategy

**PyCMG** and **NGSPICE** both use the **exact same OSDI binary** for model evaluation:

```
┌─────────────────────────────────────────────────────────────┐
│                    bsimcmg.osdi                              │
│              (Compiled BSIM-CMG Model)                       │
└───────────────────┬────────────────────┬────────────────────┘
                    │                    │
         ┌──────────▼─────────┐  ┌──────▼──────────────┐
         │   PyCMG Wrapper    │  │    NGSPICE          │
         │   (ctypes)         │  │    (.osdi command)  │
         └──────────┬─────────┘  └──────┬──────────────┘
                    │                    │
              eval_dc()            OP analysis
              eval_tran()          AC analysis
                                   Transient analysis
                    └────────┬───────────┘
                             │
                      Comparison
                   (PyCMG == NGSPICE?)
```

This verification approach ensures:
1. **Binary-level consistency**: Both paths use identical OSDI binary
2. **No recompilation differences**: Same physics, same parameters
3. **Ctypes wrapper correctness**: Verifies PyCMG properly calls OSDI functions
4. **Numerical accuracy**: Direct comparison of currents, charges, derivatives

### Test Coverage

- **test_api.py**: API smoke tests (no NGSPICE comparison)
- **test_integration.py**: PyCMG vs NGSPICE direct comparison
- **test_asap7.py**: Comprehensive PVT verification across ASAP7 modelcards

All tests verify that PyCMG output matches NGSPICE output within tight tolerances.

### Tolerances

- **Absolute current tolerance**: `ABS_TOL_I = 1e-9` A
- **Absolute charge tolerance**: `ABS_TOL_Q = 1e-18` C
- **Absolute capacitance tolerance**: `ABS_TOL_C = 1e-18` F
- **Relative tolerance**: `REL_TOL = 5e-3` (0.5%)

### Running Custom Verification

```python
from pycmg import Model, Instance
from pycmg.testing import run_ngspice_op_point

# Setup
model = Model("path/to/osdi", "path/to/modelcard", "nmos1")
inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

# Get NGSPICE ground truth
ng_result = run_ngspice_op_point(
    modelcard_path="path/to/modelcard",
    model_name="nmos1",
    inst_params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0},
    vd=0.5, vg=0.8, vs=0.0, ve=0.0,
    out_dir="output_dir",
    temp_c=27.0
)

# Get PyCMG result
py_result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})

# Compare
assert abs(py_result["id"] - ng_result["id"]) < 1e-9
```

## Advanced Usage

### Using C++ Host Directly

The `cpp/osdi_cli.cpp` provides a command-line interface:

```bash
# Evaluate operating point
./build-deep-verify/osdi_eval \
    --osdi build-deep-verify/osdi/bsimcmg.osdi \
    --modelcard bsim-cmg-va/benchmark_test/modelcard.nmos \
    --node d=0.5 \
    --node g=0.8 \
    --node s=0.0 \
    --node e=0.0 \
    --param NFIN=2 \
    --print-charges \
    --print-cap \
    --print-derivs

# List available nodes
./build-deep-verify/osdi_eval --osdi build-deep-verify/osdi/bsimcmg.osdi --list-nodes

# List available parameters
./build-deep-verify/osdi_eval --osdi build-deep-verify/osdi/bsimcmg.osdi --list-params
```

### Custom Modelcards

```python
# Override parameters during modelcard creation
from pycmg import Model, Instance
from pycmg.testing import make_ngspice_modelcard

make_ngspice_modelcard(
    src_path="original.lib",
    dst_path="custom.lib",
    model_name="nmos1",
    overrides={"EOT": "1.5n", "TOXP": "2.0n"}
)

model = Model("path/to/osdi", "custom.lib", "nmos1")
```

### ASAP7 Modelcards

```python
import os
os.environ["ASAP7_MODELCARD"] = "/path/to/asap7/models"

# Test will automatically use the override
# pytest tests/test_asap7.py -v
```

### Internal Node Analysis

```python
from pycmg import Model, Instance

model = Model(...)
inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

# Check if device has internal nodes
n_internal = inst.internal_node_count()
print(f"Internal nodes: {n_internal}")

# Query state variables
n_states = inst.state_count()
print(f"State variables: {n_states}")
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'pycmg'`
```bash
# Ensure you're in the project root
cd pycmg-wrapper
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "from pycmg import Model; print('OK')"
```

**Issue**: `OSError: cannot open shared object file`
```bash
# Rebuild the OSDI binary
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

**Issue**: NGSPICE tests fail with "ngspice: not found"
```bash
# Set custom NGSPICE path
export NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice
pytest tests/test_integration.py -v
```

**Issue**: Tests skip with "missing OSDI build artifact"
```bash
# Build the OSDI binary
mkdir -p build-deep-verify
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

**Issue**: Modelcard parsing fails with "parameter not found"
```python
# Enable verbose parsing
from pycmg import parse_modelcard
parsed = parse_modelcard("path/to/card", target_model_name="nmos1")
print(f"Found {len(parsed.params)} parameters")
print(f"Model name: {parsed.name}")
```

### Debug Mode

Enable verbose OSDI logging:

```python
import sys
from pycmg import Model, Instance

# The OSDI library will log to stderr
model = Model(...)
inst = Instance(model, ...)
result = inst.eval_dc(...)  # Check stderr for OSDI messages
```

### Verification Debugging

```bash
# Run pytest with verbose output
pytest tests/test_integration.py -vvs

# Run single test with pdb
pytest tests/test_api.py::test_eval_dc_smoke --pdb
```

## Performance Tips

1. **Reuse instances**: Create one `Instance` and call `eval_dc()` multiple times
2. **Avoid rebinds**: Use `allow_rebind=False` when possible for faster parameter updates
3. **Batch evaluations**: Vectorize sweeps using numpy
4. **Skip verification**: Use `test_api.py` for rapid iteration during development

## Contributing

See `CLAUDE.md` for development guidelines.

## License

[Specify your license here]

## Citation

If you use this tool in research, please cite:
```
[Your citation information]
```
