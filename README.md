# PyCMG Wrapper - BSIM-CMG Python Model Interface

A standalone Python interface for the BSIM-CMG FinFET compact model using OpenVAF/OSDI, with comprehensive NGSPICE-backed verification using industry-standard ASAP7 PDK modelcards.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Model Usage](#model-usage)
  - [DC Analysis](#dc-analysis)
  - [Transient Analysis](#transient-analysis)
  - [Jacobian Matrix](#jacobian-matrix)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Verification Strategy](#verification-strategy)

## Overview

PyCMG provides a Python interface to evaluate BSIM-CMG FinFET compact models through compiled OSDI binaries. It serves as a foundation for circuit simulation tools, device characterization workflows, and model validation.

### Features

- **DC Analysis**: Steady-state I-V characterization
- **AC Analysis**: Small-signal capacitance extraction
- **Transient Analysis**: Time-domain simulation with state tracking
- **Full Model Outputs**: 18/18 critical parameters (currents, charges, derivatives, capacitances)
- **NGSPICE Verification**: Automated comparison against NGSPICE ground truth
- **Multi-Technology Support**: ASAP7, TSMC5/7/12/16 PDK support

### Supported Technologies

| Technology | Node | Status | Vdd |
|------------|------|--------|-----|
| ASAP7 | 7nm | Production-ready | 0.90V |
| TSMC5 | 5nm | Verified | 0.65V |
| TSMC7 | 7nm | Verified | 0.75V |
| TSMC12 | 12nm | Verified | 0.80V |
| TSMC16 | 16nm | Verified | 0.80V |

## Quick Start

### Step 1: Build OSDI Model
```bash
# From project root
mkdir -p build-deep-verify && cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

Alternatively, compile directly with OpenVAF:
```bash
openvaf -I bsim-cmg-va/code -o bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
```

### Step 2: Run Tests
```bash
# Quick API test (no NGSPICE required)
pytest tests/test_api.py -v

# Full test suite (NGSPICE required)
pytest tests/ -v
```

## Installation

### Prerequisites

- Python 3.8+
- OpenVAF compiler (v23.5.0+)
- NGSPICE (v45+, for verification)
- CMake (v3.20+)

### Build OSDI Binary

```bash
# Using CMake (recommended)
mkdir -p build-deep-verify && cd build-deep-verify
cmake ..
cmake --build . --target osdi

# Or compile directly with OpenVAF
openvaf -I bsim-cmg-va/code -o bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
```

The output will be at `build-deep-verify/osdi/bsimcmg.osdi`.

### Install Python Dependencies

```bash
pip install numpy pytest
```

## Usage Guide

### Model Usage

The core interface consists of two main classes: `Model` and `Instance`.

```python
from pathlib import Path
from pycmg import Model, Instance

# Paths
osdi_path = Path("build-deep-verify/osdi/bsimcmg.osdi")
modelcard_path = Path("tech_model_cards/ASAP7/7nm_TT_160803.pm")

# Load model from modelcard
model = Model(
    osdi_path=str(osdi_path),
    modelcard_path=str(modelcard_path),
    model_name="nmos_lvt"  # Optional: specific model in multi-model file
)

# Create device instance with geometry parameters
inst = Instance(model, params={
    "L": 21e-9,      # Gate length (m)
    "TFIN": 14e-9,   # Fin thickness (m)
    "NFIN": 2.0,     # Number of fins
})
```

### DC Analysis

Evaluate DC operating point with terminal voltages:

```python
# Set terminal voltages (in Volts)
nodes = {"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0}  # Vd, Vg, Vs, Ve (extended/bulk)

# Evaluate DC operating point
result = inst.eval_dc(nodes)

# Access results
print(f"Drain current: {result['id']:.3e} A")
print(f"Transconductance: {result['gm']:.3e} S")
print(f"Output conductance: {result['gds']:.3e} S")

# All available outputs:
# Currents: id, ig, is, ie, ids
# Charges: qg, qd, qs, qb
# Derivatives: gm, gds, gmb
# Capacitances: cgg, cgd, cgs, cdg, cdd
```

### Transient Analysis

Evaluate transient behavior with time stepping:

```python
# Transient evaluation with time and delta_t
result = inst.eval_tran(
    nodes={"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0},
    time=1e-9,      # Current time point (s)
    delta_t=1e-12   # Time step (s)
)

# Transient outputs: id, ig, is, ie, ids, qg, qd, qs, qb
print(f"Drain current: {result['id']:.3e} A")
print(f"Gate charge: {result['qg']:.3e} C")
```

### Jacobian Matrix

Extract the condensed 4x4 Jacobian matrix (dI/dV):

```python
import numpy as np

# Get Jacobian matrix at operating point
J = inst.get_jacobian_matrix({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})

# J is a 4x4 numpy array with terminals ordered as [d, g, s, e]
# J[i,j] = dI_terminal_i / dV_terminal_j

print(f"Jacobian shape: {J.shape}")  # (4, 4)
print(f"gds (dId/dVd): {J[0, 0]:.3e} S")
```

### Temperature Sweep

Evaluate at different temperatures:

```python
# Temperature is in Kelvin
for temp_c in [-40, 27, 85, 125]:
    temp_k = temp_c + 273.15
    inst = Instance(model, params={"L": 21e-9, "TFIN": 14e-9, "NFIN": 2.0},
                    temperature=temp_k)
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
    print(f"T={temp_c}C: Id={result['id']:.3e} A")
```

### Parameter Updates

Modify instance parameters after creation:

```python
inst.set_params({"NFIN": 4.0}, allow_rebind=False)
result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
```

## Running Tests

### Test Categories

| Test File | Description | NGSPICE |
|-----------|-------------|---------|
| `test_api.py` | API smoke tests | No |
| `test_dc_jacobian.py` | DC Jacobian verification | Yes |
| `test_dc_regions.py` | DC operating region tests | Yes |
| `test_transient.py` | Transient waveform verification | Yes |

### Quick Tests (No NGSPICE)

```bash
# API smoke tests
pytest tests/test_api.py -v
```

### Integration Tests (NGSPICE Required)

```bash
# DC Jacobian verification
pytest tests/test_dc_jacobian.py -v

# DC operating region tests
pytest tests/test_dc_regions.py -v

# Transient verification
pytest tests/test_transient.py -v

# All tests
pytest tests/ -v
```

### Environment Variables

```bash
# Custom NGSPICE binary
export NGSPICE_BIN=/path/to/ngspice

# Custom ASAP7 modelcard
export ASAP7_MODELCARD=/path/to/modelcard.pm
```

## Project Structure

```
pycmg-wrapper/
├── pycmg/                    # Python package
│   ├── __init__.py          # Public API exports
│   ├── ctypes_host.py       # Core OSDI interface
│   └── testing.py           # Verification utilities
├── tests/                    # Test suite
│   ├── conftest.py          # Technology registry (ASAP7, TSMC5/7/12/16)
│   ├── test_api.py          # API smoke tests
│   ├── test_dc_jacobian.py  # DC Jacobian verification
│   ├── test_dc_regions.py   # DC operating region tests
│   └── test_transient.py    # Transient waveform verification
├── tech_model_cards/         # Technology model cards
│   ├── ASAP7/               # ASAP7 model files
│   ├── TSMC5/               # TSMC5 model files
│   ├── TSMC7/               # TSMC7 model files
│   ├── TSMC12/              # TSMC12 model files
│   └── TSMC16/              # TSMC16 model files
├── cpp/                      # C++ OSDI host (reference implementation)
├── bsim-cmg-va/             # Verilog-A source and documentation
├── scripts/                  # Utility scripts
│   └── generate_naive_tsmc.py
├── build-deep-verify/        # Build artifacts
│   └── osdi/bsimcmg.osdi    # Compiled OSDI binary
├── CMakeLists.txt            # Build system
└── main.py                   # CLI entrypoint
```

## API Reference

### `Model`

```python
Model(osdi_path: str, modelcard_path: str, model_name: str = None)
```

Load a BSIM-CMG model from modelcard file.

- `osdi_path`: Path to compiled `.osdi` binary
- `modelcard_path`: Path to SPICE modelcard file
- `model_name`: Specific model to load (for multi-model files)

### `Instance`

```python
Instance(model: Model, params: dict, temperature: float = 300.15)
```

Create a device instance with geometry parameters.

- `model`: Loaded Model object
- `params`: Instance parameters (L, TFIN, NFIN, etc.)
- `temperature`: Temperature in Kelvin (default: 27°C)

#### Methods

- `eval_dc(nodes: dict) -> dict`: DC operating point evaluation
- `eval_tran(nodes: dict, time: float, delta_t: float) -> dict`: Transient evaluation
- `get_jacobian_matrix(nodes: dict) -> np.ndarray`: Extract 4x4 Jacobian
- `set_params(params: dict, allow_rebind: bool = False)`: Update parameters

### Utility Functions

```python
from pycmg.ctypes_host import parse_modelcard, parse_number_with_suffix

# Parse SPICE number with suffix
parse_number_with_suffix("16n")  # -> 16e-9
parse_number_with_suffix("1.5meg")  # -> 1.5e6

# Parse modelcard file
parsed = parse_modelcard("modelcard.pm", target_model_name="nmos_lvt")
print(parsed.name)  # Model name
print(parsed.params)  # Dict of parameters
```

## Verification Strategy

PyCMG wraps the OSDI binary directly via ctypes, while NGSPICE loads the SAME OSDI binary via the `.osdi` command. Tests compare PyCMG output vs NGSPICE output to ensure:

1. **Binary-level consistency**: Both use the identical `bsimcmg.osdi` file
2. **Ctypes wrapper correctness**: Verifies proper OSDI function calls
3. **Numerical accuracy**: Direct comparison of currents, charges, derivatives
4. **Full model coverage**: DC, AC (capacitance), and transient analysis

The OSDI binary is the single source of truth for all model physics calculations.

### Tolerances

| Parameter | Absolute | Relative |
|-----------|----------|----------|
| Current (A) | 1e-9 | 0.5% |
| Charge (C) | 1e-18 | 0.5% |
| Conductance (S) | 1e-6 | 1% |

## License

This project is provided for educational and research purposes. The BSIM-CMG model is licensed separately by the BSIM Group at UC Berkeley.
