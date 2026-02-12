# PyCMG Wrapper - BSIM-CMG Python Model Interface

A standalone Python interface for the BSIM-CMG FinFET compact model using OpenVAF/OSDI, with comprehensive NGSPICE-backed verification using industry-standard ASAP7 PDK modelcards.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Verification Strategy](#verification-strategy)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

PyCMG provides a Python interface to evaluate BSIM-CMG FinFET compact models through compiled OSDI binaries. It serves as the foundation for circuit simulation tools, device characterization workflows, and model validation.

**Key Design Principle:** PyCMG and NGSPICE both use the **identical OSDI binary** (`bsimcmg.osdi`), ensuring binary-level consistency and numerical accuracy.

### Features

- **DC Analysis**: Steady-state I-V characterization
- **AC Analysis**: Small-signal capacitance extraction
- **Transient Analysis**: Time-domain simulation with state tracking
- **Full Model Outputs**: 18/18 critical parameters (currents, charges, derivatives, capacitances)
- **NGSPICE Verification**: Automated comparison against NGSPICE ground truth
- **ASAP7 PDK Support**: Production-ready verification with ASAP7 modelcards

### Supported Model Outputs (18 total)

| Category | Outputs | Description |
|----------|----------|-------------|
| **Currents** | `id`, `ig`, `is`, `ie`, `ids` | Terminal currents + drain-source current (Id-Is) |
| **Derivatives** | `gm`, `gds`, `gmb` | Transconductance, output conductance, bulk transconductance |
| **Charges** | `qg`, `qd`, `qs`, `qb` | Gate, drain, source, bulk charges |
| **Capacitances** | `cgg`, `cgd`, `cgs`, `cdg`, `cdd` | Condensed capacitance matrix |

## Quick Start

Get up and running with ASAP7 modelcards in 3 steps:

### Step 1: Build the OSDI Model

```bash
# Clone repository
git clone https://github.com/ShenShan123/PyCMG.git
cd PyCMG

# Build OSDI model (uses CMake)
mkdir -p build-deep-verify
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

This generates `build-deep-verify/osdi/bsimcmg.osdi` (~2-3 MB).

### Step 2: Download ASAP7 Modelcards

```bash
# Download ASAP7 PDK (7nm technology)
cd tech_model_cards
wget https://github.com/google/sg-f7hap7/releases/download/v1.0.1/sg-f7hap7.tar.gz
tar -xzf sg-f7hap7.tar.gz
```

Or use the environment variable to point to your existing modelcards:
```bash
export ASAP7_MODELCARD=/path/to/your/modelcards
```

### Step 3: Run Python Analysis

```python
from pycmg import Model, Instance

# Use ASAP7 modelcard (automatically detected)
model = Model(
    osdi_path="build-deep-verify/osdi/bsimcmg.osdi",
    modelcard_path="tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm",
    model_name="nmos_lvt"  # Level 72 NMOS model
)

# Create FinFET instance
inst = Instance(
    model=model,
    params={
        "L": 16e-9,    # Channel length: 16 nm
        "TFIN": 8e-9,  # Fin thickness: 8 nm
        "NFIN": 2.0,   # Number of fins: 2
    }
)

# DC operating point analysis
result = inst.eval_dc({
    "d": 0.5,  # Drain voltage
    "g": 0.8,  # Gate voltage
    "s": 0.0,  # Source voltage
    "e": 0.0   # Bulk/substrate voltage
})

# Access results
print(f"Drain current (Id):  {result['id']:.3e} A")
print(f"Gate current (Ig):  {result['ig']:.3e} A")
print(f"Drain-source (Ids): {result['ids']:.3e} A")
print(f"Transconductance (gm): {result['gm']:.3e} S")
print(f"Gate charge (Qg):    {result['qg']:.3e} C")
print(f"Gate capacitance (Cgg): {result['cgg']:.3e} F")
```

**Output:**
```
Drain current (Id):  5.324e-05 A
Gate current (Ig):  1.042e-14 A
Drain-source (Ids): 5.324e-05 A
Transconductance (gm): 2.145e-04 S
Gate charge (Qg):    1.234e-16 C
Gate capacitance (Cgg): 2.456e-17 F
```

### Step 4: Verify Against NGSPICE

```bash
# Run verification tests (compares PyCMG vs NGSPICE)
pytest tests/test_asap7.py -v
```

All tests verify that PyCMG output matches NGSPICE within tight tolerances (ABS_TOL=1e-9, REL_TOL=5e-3).

## Installation

### Requirements

**Essential Tools:**
| Tool | Version | Purpose |
|------|---------|---------|
| OpenVAF | v23.5.0+ | Verilog-A to OSDI compiler |
| CMake | ≥ 3.15 | Build system |
| Python | ≥ 3.10 | Python interface |
| GCC/Clang | C++17 | OSDI host compilation |

**Optional (for verification):**
| Tool | Version | Purpose |
|------|---------|---------|
| NGSPICE | ≥ 45.2 | Ground truth verification |
| pytest | any | Test runner |

### Install OpenVAF

```bash
# Download precompiled binary (Linux)
wget https://github.com/ngspice/openvaf/releases/download/v23.5.0/openvaf-23.5.0-linux-x64_64.tar.gz
tar -xzf openvaf-23.5.0-linux-x64_64.tar.gz
sudo cp openvaf-23.5.0-linux-x64_64/openvaf /usr/local/bin/
```

Or build from source:
```bash
git clone https://github.com/ngspice/openvaf.git
cd openvaf
cargo build --release
sudo cp target/release/openvaf /usr/local/bin/
```

### Install NGSPICE (Optional)

```bash
# Ubuntu/Debian
sudo apt-get install ngspice

# Or build from source
wget http://ngspice.sourceforge.net/compile.html
```

### Install Python Dependencies

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pytest
```

### Build OSDI Model

**Option A: Automated build script**
```bash
chmod +x build_osdi.sh
./build_osdi.sh
```

**Option B: Manual CMake build**
```bash
mkdir -p build-deep-verify
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

**Option C: Direct OpenVAF compilation**
```bash
openvaf -I bsim-cmg-va/code -o bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
mkdir -p build-deep-verify/osdi
mv bsimcmg.osdi build-deep-verify/osdi/
```

**Verify build:**
```bash
ls -lh build-deep-verify/osdi/bsimcmg.osdi
file build-deep-verify/osdi/bsimcmg.osdi
# Should show: ELF 64-bit LSB shared object
```

## Usage Guide

### Creating a Model

```python
from pycmg import Model

# Method 1: From ASAP7 modelcard (recommended)
model = Model(
    osdi_path="build-deep-verify/osdi/bsimcmg.osdi",
    modelcard_path="tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm",
    model_name="nmos_lvt"  # Optional: auto-detected if only one model
)

# Method 2: Using modelcard parser
from pycmg.ctypes_host import parse_modelcard
parsed = parse_modelcard("tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm")
print(f"Parsed {len(parsed.params)} parameters")
```

### Creating an Instance

```python
from pycmg import Instance

# FinFET instance with geometry parameters
inst = Instance(
    model=model,
    params={
        "L": 20e-9,    # Length (m)
        "TFIN": 10e-9, # Fin thickness (m)
        "NFIN": 3.0,   # Number of fins
    },
    temperature=300.15  # Kelvin (default: 300.15 K = 27°C)
)

# Update parameters after creation
inst.set_params({"NFIN": 5.0})
```

### DC Operating Point Analysis

```python
# Single operating point
result = inst.eval_dc({
    "d": 0.5,  # Drain voltage
    "g": 0.8,  # Gate voltage
    "s": 0.0,  # Source voltage
    "e": 0.0,  # Bulk voltage
})

# Access all outputs
print(f"Id: {result['id']:.3e} A")   # Drain current
print(f"Ig: {result['ig']:.3e} A")   # Gate current
print(f"Is: {result['is']:.3e} A")   # Source current
print(f"Ie: {result['ie']:.3e} A")   # Bulk current
print(f"Ids: {result['ids']:.3e} A")  # Drain-source current (Id - Is)

print(f"Gm: {result['gm']:.3e} S")    # Transconductance
print(f"Gds: {result['gds']:.3e} S")  # Output conductance
print(f"Gmb: {result['gmb']:.3e} S")   # Bulk transconductance

print(f"Qg: {result['qg']:.3e} C")    # Gate charge
print(f"Qd: {result['qd']:.3e} C")    # Drain charge
print(f"Qs: {result['qs']:.3e} C")    # Source charge
print(f"Qb: {result['qb']:.3e} C")    # Bulk charge

print(f"Cgg: {result['cgg']:.3e} F")  # Gate-gate capacitance
print(f"Cgd: {result['cgd']:.3e} F")  # Gate-drain capacitance
print(f"Cgs: {result['cgs']:.3e} F")  # Gate-source capacitance
print(f"Cdg: {result['cdg']:.3e} F")  # Drain-gate capacitance
print(f"Cdd: {result['cdd']:.3e} F")  # Drain-drain capacitance
```

### Voltage Sweep

```python
import numpy as np

# Id-Vg sweep (transfer characteristics)
vg_values = np.linspace(0, 1.2, 13)
id_values = []
for vg in vg_values:
    result = inst.eval_dc({"d": 0.05, "g": vg, "s": 0.0, "e": 0.0})
    id_values.append(result["id"])

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.plot(vg_values, id_values)
plt.xlabel("Vg (V)")
plt.ylabel("Id (A)")
plt.title("Id-Vg Characteristics")
plt.grid(True)
plt.show()
```

### Temperature Sweep

```python
temperatures = [223.15, 273.15, 323.15, 373.15, 398.15]  # K (-40°C to 125°C)
id_at_temp = []

for temp_k in temperatures:
    inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0},
                    temperature=temp_k)
    result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
    id_at_temp.append(result["id"])

plt.plot([(t-273.15) for t in temperatures], id_at_temp)
plt.xlabel("Temperature (°C)")
plt.ylabel("Id (A)")
plt.title("Temperature Dependence")
plt.grid(True)
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
print(f"Id(t={time:.1e}s): {result['id']:.3e} A")
print(f"Ig(t={time:.1e}s): {result['ig']:.3e} A")
```

## Running Tests

### Test Organization

| Test Suite | Purpose | Duration | NGSPICE Required |
|------------|---------|----------|-----------------|
| `test_api.py` | Quick API validation | ~5 seconds | No |
| `test_integration.py` | NGSPICE ground truth comparison | ~30 seconds | Yes |
| `test_asap7.py` | Full ASAP7 PVT verification | ~5 minutes | Yes |
| `test_tsmc7_verification.py` | TSMC7 parameter sweeps | ~30 seconds | Yes |

**TSMC7 Verification Status**: ✅ Complete (10 test cases)

- Tests L sweep: 12nm, 16nm, 20nm, 24nm (4 tests)
- Tests TFIN sweep: 6nm, 7nm, 8nm (3 tests)
- Tests NFIN sweep: 1, 2, 4 fins (3 tests)

All tests verify binary-identical results between PyCMG and NGSPICE.

### Running Tests

```bash
# Quick smoke test (no NGSPICE)
pytest tests/test_api.py -v

# Integration tests with NGSPICE
pytest tests/test_integration.py -v

# Full ASAP7 verification (PVT corners, temperature sweeps)
pytest tests/test_asap7.py -v

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::test_eval_dc_smoke -v
```

### Using main.py CLI

```bash
python main.py test api           # Quick smoke tests
python main.py test integration   # NGSPICE comparison
python main.py test asap7         # Full ASAP7 verification
python main.py test all           # Run all tests
```

## Project Structure

```
pycmg-wrapper/
├── bsim-cmg-va/              # Verilog-A source files
│   ├── code/                 # Main model files
│   │   ├── bsimcmg_main.va  # Model entry point
│   │   ├── bsimcmg.va       # Core model
│   │   └── *.include          # Model components
│   └── benchmark_test/       # Legacy model cards (not recommended)
├── pycmg/                    # Python package
│   ├── __init__.py          # Public API
│   ├── ctypes_host.py       # OSDI interface (Model, Instance, eval_dc, eval_tran)
│   └── testing.py           # Verification utilities
├── cpp/                      # C++ OSDI host
│   ├── osdi_host.h          # Header file
│   ├── osdi_host.cpp        # Host implementation
│   └── osdi_cli.cpp         # CLI tool (osdi_eval)
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── test_api.py          # API smoke tests
│   ├── test_asap7.py        # ASAP7 verification
│   └── test_integration.py  # NGSPICE comparison
├── tech_model_cards/         # Technology model cards
│   └── asap7_pdk_r1p7/      # ASAP7 PDK model files (recommended)
├── build-deep-verify/        # Build artifacts (generated)
│   ├── osdi/                # Compiled .osdi files
│   └── ngspice_eval/        # Verification outputs
├── main.py                   # CLI entrypoint
└── README.md                 # This file
```

## API Reference

### Model Class

```python
Model(osdi_path: str, modelcard_path: str, model_name: str = None)
```

Load an OSDI compiled model with parameters from a modelcard.

**Parameters:**
- `osdi_path`: Path to `.osdi` binary file
- `modelcard_path`: Path to SPICE modelcard (`.lib`, `.pm`, `.scn`)
- `model_name`: Model name within modelcard (optional if only one)

### Instance Class

```python
Instance(model: Model, params: dict, temperature: float = 300.15)
```

Create a device instance with geometry parameters.

**Parameters:**
- `model`: `Model` object
- `params`: Dictionary of instance parameters (L, TFIN, NFIN, etc.)
- `temperature`: Temperature in Kelvin (default: 300.15 K)

**Methods:**

```python
eval_dc(nodes: dict) -> dict
```
Evaluate DC operating point.

**Parameters:**
- `nodes`: Dictionary {"d": vd, "g": vg, "s": vs, "e": ve}

**Returns:** Dictionary with:
- Currents: `id`, `ig`, `is`, `ie`, `ids` (A)
  - `ids` = `id` - `is` (drain-source current for common-source configuration)
- Charges: `qg`, `qd`, `qs`, `qb` (C)
- Derivatives: `gm`, `gds`, `gmb` (S)
- Capacitances: `cgg`, `cgd`, `cgs`, `cdg`, `cdd` (F)

```python
eval_tran(nodes: dict, time: float, delta_t: float,
          prev_state: list = None) -> dict
```
Evaluate transient response.

**Parameters:**
- `nodes`: Terminal voltages
- `time`: Current time (s)
- `delta_t`: Time step (s)
- `prev_state`: Previous state vector (optional, for internal continuity)

**Returns:** Dictionary with currents and charges

```python
set_params(params: dict, allow_rebind: bool = False)
```
Update instance parameters. May require rebind if topology changes.

## Verification Strategy

PyCMG and NGSPICE both use the **identical OSDI binary** for model evaluation:

```
┌─────────────────────────────────────────────────────────────┐
│                    bsimcmg.osdi                              │
│              (Compiled BSIM-CMG Model)                       │
└───────────────────┬────────────────────┬────────────────────┘
                    │                    │
         PyCMG Wrapper            NGSPICE
         (ctypes)                 (.osdi command)
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
2. **Ctypes wrapper correctness**: Verifies proper OSDI function calls
3. **Numerical accuracy**: Direct comparison of currents, charges, derivatives
4. **Full model coverage**: DC, AC (capacitance), and transient analysis

### Tolerances

- **Absolute current tolerance**: `ABS_TOL_I = 1e-9` A
- **Absolute charge tolerance**: `ABS_TOL_Q = 1e-18` C
- **Absolute capacitance tolerance**: `ABS_TOL_C = 1e-18` F
- **Relative tolerance**: `REL_TOL = 5e-3` (0.5%)

### Test Coverage

All 18 model outputs are verified against NGSPICE:
- ✅ Currents: id, ig, is, ie, ids
- ✅ Derivatives: gm, gds, gmb
- ✅ Charges: qg, qd, qs, qb
- ✅ Capacitances: cgg, cgd, cgs, cdg, cdd

## Advanced Usage

### Finding Models in ASAP7 Modelcards

### TSMC7 Model Verification

TSMC7 (Taiwan Semiconductor Manufacturing Company) 7nm modelcard verification is available in `tests/test_tsmc7_verification.py`. The test suite performs parametrized sweeps across:

| Parameter Sweep | Points | Tests |
|----------------|--------|-------|
| Gate Length (L) | 12, 16, 20, 24nm | 4 |
| Fin Thickness (TFIN) | 6, 7, 8nm | 3 |
| Fin Count (NFIN) | 1, 2, 4 fins | 3 |

All 10 test cases verify binary-identical results between PyCMG and NGSPICE using the same `bsimcmg.osdi` file.

```python
from pycmg.ctypes_host import parse_modelcard
import re

modelcard_path = "tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm"
parsed = parse_modelcard(modelcard_path)

# Find all level=72 NMOS models
text = Path(modelcard_path).read_text()
models = []
for line in text.splitlines():
    if line.strip().lower().startswith(".model"):
        parts = line.split()
        if len(parts) >= 3:
            name = parts[1]
            mtype = parts[2].lower()
            if "nmos" in mtype and "level=72" in line.lower():
                models.append(name)

print(f"Found {len(models)} level=72 NMOS models: {models}")
# Output: Found 3 level=72 NMOS models: ['nmos_lvt', 'nmos_rvt', 'nmos_slvt']
```

### Environment Variables

```bash
# Override NGSPICE binary location
export NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice

# Override ASAP7 modelcard location
export ASAP7_MODELCARD=/path/to/asap7/models
```

### Custom Modelcard Creation

```python
from pycmg.testing import make_ngspice_modelcard

# Create custom modelcard with parameter overrides
make_ngspice_modelcard(
    src_path="tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm",
    dst_path="custom.lib",
    model_name="nmos_lvt",
    overrides={"L": "20n", "TFIN": "10n", "NFIN": 3.0}
)

model = Model("build-deep-verify/osdi/bsimcmg.osdi", "custom.lib", "nmos_lvt")
```

### PVT Corner Analysis

```python
# Test across PVT corners
corners = {
    "TT": "tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm",
    "SS": "tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_SS.pm",
    "FF": "tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_FF.pm",
}

for corner_name, modelcard_path in corners.items():
    model = Model("build-deep-verify/osdi/bsimg.osdi", modelcard_path, "nmos_lvt")
    inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})
    result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
    print(f"{corner_name}: Id = {result['id']:.3e} A")
```

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'pycmg'`**
```bash
# Ensure you're in the project root
cd PyCMG
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "from pycmg import Model; print('OK')"
```

**Issue: `OSError: cannot open shared object file`**
```bash
# Rebuild the OSDI binary
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

**Issue: NGSPICE tests fail with "ngspice: not found"**
```bash
# Set custom NGSPICE path
export NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice
pytest tests/test_integration.py -v
```

**Issue: Tests skip with "missing ASAP7 modelcards"**
```bash
# Download ASAP7 modelcards
cd tech_model_cards
wget https://github.com/google/sg-f7hap7/releases/download/v1.0.1/sg-f7hap7.tar.gz
tar -xzf sg-f7hap7.tar.gz

# Or set environment variable
export ASAP7_MODELCARD=/path/to/your/modelcards
```

**Issue: "Parameter EOTACC is out of bounds"**
```bash
# This should be automatically fixed by the test framework
# The modelcard parser clamps EOTACC to ≥1.1e-10 for OSDI compatibility
# If you see this error, ensure you're using the updated test utilities
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

### Getting Help

- **Documentation**: See `CLAUDE.md` for development guidelines
- **Issues**: Report at https://github.com/ShenShan123/PyCMG/issues
- **Tests**: Run `pytest tests/test_api.py -v` for quick smoke tests

## License

[Specify your license here]

## Citation

If you use this tool in research, please cite:
```
[Your citation information]
```
