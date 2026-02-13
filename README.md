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
- [Lessons from Bugs](#lessons-learned)

## Overview

PyCMG provides a Python interface to evaluate BSIM-CMG FinFET compact models through compiled OSDI binaries. It serves as a foundation for circuit simulation tools, device characterization workflows, and model validation.

### Features

- **DC Analysis**: Steady-state I-V characterization
- **AC Analysis**: Small-signal capacitance extraction
- **Transient Analysis**: Time-domain simulation with state tracking
- **Full Model Outputs**: 18/18 critical parameters (currents, charges, derivatives, capacitances)
- **NGSPICE Verification**: Automated comparison against NGSPICE ground truth
- **ASAP7 PDK Support**: Production-ready verification with ASAP7 modelcards
- **TSMC7 PDK Support**: Full TSMC7 PDK support with automatic variant selection

### Supported Technologies

- **ASAP7**: 7nm PDK from Arizona State University (production-ready)
- **TSMC7**: 7nm PDK from Taiwan Semiconductor Manufacturing Company (production-ready)

## Quick Start

Get up and running with ASAP7 or TSMC7 modelcards in 3 steps:

### Step 1: Build OSDI Model
```bash
# From project root
./build_osdi.sh
```

### Step 2: Download Modelcards

#### Option A: ASAP7 (Arizona State University)
```bash
# Clone repository
git clone https://github.com/ShenShan123/PyCMG.git
cd PyCMG

# Download ASAP7 PDK
cd tech_model_cards/asap7_pdk_r1p7/
wget https://github.com/google/sg-f7hap7/releases/download/v1.0.1/sg-f7hap7.tar.gz
tar -xzf sg-f7hap7.tar.gz
cd asap7_pdk_r1p7
# Set environment variable
export ASAP7_MODELCARD=$(pwd)/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT.pm
```

#### Option B: TSMC7 (Taiwan Semiconductor)
```bash
# TSMC7 PDK is already included in this repository
# No download needed - use existing files in tech_model_cards/TSMC7/
export TSMC7_MODELCARD=$(pwd)/tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l
```

### Step 3: Run Python Analysis
```bash
# Quick API test (no NGSPICE - ~5 seconds)
python main.py test api

# Run integration tests with NGSPICE (~30 seconds)
pytest tests/test_integration.py -v

# Or run specific test
pytest tests/test_integration.py -v -k "test_nmos_svtt_smoke"
```

### Step 4: Verify Results
Check output for:
- ✅ Correct number of parameters extracted
- ✅ Reasonable current values in saturation region
- ✅ Binary-identical results with NGSPICE
