# PyCMG -- BSIM-CMG Training Data Generator for Neural Network Compact Models

PyCMG is a Python ctypes wrapper around the BSIM-CMG OSDI binary, purpose-built for generating training data for neural network compact models. It evaluates the full BSIM-CMG FinFET model (currents, charges, derivatives, capacitances) across systematic voltage, geometry, and temperature sweeps, producing ready-to-train CSV datasets verified against NGSPICE.

## What is PyCMG?

PyCMG loads a compiled BSIM-CMG `.osdi` binary via ctypes (no C++ compilation needed) and calls the model's evaluation functions directly. Given terminal voltages, geometry parameters, and temperature, it returns 17 model outputs (5 currents, 4 charges, 3 derivatives, 5 capacitances). The sweep engine drives this evaluator across PDK-defined geometry combinations (L, NFIN bin boundaries), temperatures, and voltage grids to produce million-row datasets for ML training.

## Quick Start

### 1. Build the OSDI Binary

```bash
mkdir -p build && cd build && cmake .. && cmake --build . --target osdi && cd ..
```

### 2. Generate Your First Dataset

```bash
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7
# Output: training_data/ASAP7_dc.csv (~300MB, 1.5M rows)
```

### 3. Load into Your ML Framework

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("training_data/ASAP7_dc.csv")
inputs = df[["L", "NFIN", "TFIN", "temp_K", "Vg", "Vd", "Vs", "Ve"]].values
outputs = df[["ids", "gm", "gds", "cgg", "cgd", "cgs"]].values

class MosfetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

loader = DataLoader(MosfetDataset(inputs, outputs), batch_size=1024, shuffle=True)
```

## Installation

### Prerequisites

- **Python 3.8+** with NumPy
- **OpenVAF compiler** (v23.5.0+) -- compiles Verilog-A to OSDI
- **CMake** (v3.20+) -- build system
- **NGSPICE** (v45+) -- optional, for verification tests only

### Building the OSDI Binary

**Option A: CMake (recommended)**

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --target osdi
```

**Option B: Direct OpenVAF**

```bash
mkdir -p build/osdi
openvaf -I bsim-cmg-va/code -o build/osdi/bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
```

Verify the output exists: `build/osdi/bsimcmg.osdi` (should be ~2-3 MB shared object).

### Install Python Dependencies

```bash
pip install numpy pytest
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NGSPICE_BIN` | Path to custom NGSPICE binary | `/usr/local/ngspice-45.2/bin/ngspice` |
| `ASAP7_MODELCARD` | Override ASAP7 modelcard path (file or directory) | `modelcards/ASAP7/7nm_TT_160803.pm` |

## Generating Training Data

### CLI Usage

The main entry point is `scripts/generate_training_data.py`.

```bash
# All 5 technologies, all 42 devices, PDK-defined geometry sweep
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi

# Single technology
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech TSMC7

# Specific devices with glob patterns
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_*

# Custom voltage grid and temperature
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 \
    --temps -40 27 85 125 \
    --vg-points 80 --vd-points 80

# Quick test (single default geometry, minimal grid)
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_rvt \
    --no-sweep-geometry --temps 27 \
    --vg-points 5 --vd-points 5

# List available devices
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi --list-devices
```

By default, `--sweep-geometry` is enabled: the sweep enumerates all PDK-defined (L, NFIN) bin boundary combinations from the technology modelcard. Use `--no-sweep-geometry` for a single (min_l, 1) point per device.

### Python API: One-Liner

```python
from pycmg import generate_dataset

paths = generate_dataset(
    osdi_path="build/osdi/bsimcmg.osdi",
    techs=["ASAP7"],
    output_dir="./training_data",
)
# Returns: ["/.../training_data/ASAP7_dc.csv"]
```

### Python API: Composable Pipeline

```python
from pycmg.sweep import SweepConfig, sweep_dc, to_csv

config = SweepConfig(
    techs=["ASAP7", "TSMC7"],
    sweep_geometry=True,              # use PDK-defined (L, NFIN) combos
    temperatures=[300.15, 358.15],    # 27C, 85C in Kelvin
    vg_points=80,
    vd_points=80,
)

result = sweep_dc("build/osdi/bsimcmg.osdi", config, verbose=2)
paths = to_csv(result, "./training_data", split_by="tech")
```

### PDK-Defined Geometry Sweep

TSMC PDKs organize model variants as a 2D grid of (L_bin x NFIN_group), where each combination has specifically fitted binning coefficients. The sweep engine reads these bin boundaries directly from the PDK file:

```python
from pycmg import scan_pdk_geometry_combos

# Enumerate all (L, NFIN) sweep points for TSMC7 PMOS LVT
combos = scan_pdk_geometry_combos(
    "modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l",
    "pch_lvt_mac",
)
# Returns 42 (L, NFIN) pairs: 6 L bins x 7 unique NFIN boundaries
# [(8e-9, 1.0), (8e-9, 2.0), ..., (1.2e-7, 24.888)]
```

For each variant, the sweep generates two points using `{nfinmin, nfinmax}` boundaries. This ensures correct binning coefficients are used for every (L, NFIN) combination.

### Process Variation

Sweep model parameters like EOT (oxide thickness) or TOXP alongside the voltage grid:

```bash
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 \
    --process-var eot=0.9e-9,1.0e-9,1.1e-9 \
    --process-var toxp=1.8e-9,2.1e-9
```

Process variation parameters are passed as `model_overrides` to `Instance`, overriding the modelcard value for each combination in the Cartesian product. The resulting CSV includes extra columns for each varied parameter.

### Extended Voltage Range (2*VDD)

For NN models used in circuit simulators, the Newton-Raphson solver may temporarily evaluate voltages beyond the nominal VDD. Training data covering `[0, 2*VDD]` prevents the NN from extrapolating in these regions:

```bash
# Extend voltage sweep to 2x VDD
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_rvt \
    --voltage-scale 2.0

# Or via Python API
from pycmg import generate_dataset
paths = generate_dataset(
    osdi_path="build/osdi/bsimcmg.osdi",
    techs=["ASAP7"],
    voltage_scale=2.0,
)
```

The dense region around Vth keeps the same width regardless of `voltage_scale` -- only the sparse grid and Vd grid extend to `VDD * voltage_scale`.

### Sensitivity Analysis: Finding Dominant Process Parameters

Before modeling process variation, identify which parameters matter most. The sensitivity analysis tool perturbs each BSIM-CMG model parameter independently and ranks them by influence on I-V, Q-V, and C-V characteristics:

```bash
# Identify top 9 process parameters for ASAP7 NMOS
python scripts/sensitivity_analysis.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --device nmos_rvt

# Custom: TSMC5, 10% perturbation, top 15
python scripts/sensitivity_analysis.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech TSMC5 --device nmos_svt \
    --delta 0.10 --top-n 15

# Save full results to CSV
python scripts/sensitivity_analysis.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --device nmos_rvt \
    --output sensitivity_results.csv
```

**Example output** (TSMC5 nmos_svt):

```
=== I-V Sensitivity (top 9) ===
Rank  Parameter      ids       gm        gds       gmb       Score
1     phig           2.04e+01  3.16e+03  3.21e+03  5.13e+03  1.15e+04
2     easub          1.91e+01  2.42e+03  2.44e+03  3.23e+03  8.10e+03
3     nu0            2.13e+00  2.82e+00  2.83e+00  2.85e+00  1.06e+01
...
```

**Python API:**

```python
from pycmg import compute_sensitivity

result = compute_sensitivity(
    osdi_path="build/osdi/bsimcmg.osdi",
    modelcard_path="modelcards/ASAP7/7nm_TT_160803.pm",
    model_name="nmos_rvt",
    inst_params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0},
    vdd=0.9,
    device_type="nmos",
    delta_fraction=0.05,  # 5% perturbation
    top_n=9,
)

# Top 9 parameters for each category
print(result.rankings["iv"])  # I-V: ['phig', 'easub', ...]
print(result.rankings["qv"])  # Q-V: ['phig', 'easub', ...]
print(result.rankings["cv"])  # C-V: ['phig', 'easub', ...]

# Full sensitivity data per parameter per output
print(result.sensitivities["phig"])  # {'ids': 94.02, 'gm': 29.70, ...}
```

The analysis evaluates at 4 representative bias points (subthreshold, linear, saturation, strong inversion) using central-difference perturbation and normalized sensitivity.

## Supported Technologies

| Technology | Node | Vdd | TFIN | Vt Flavors | Devices |
|------------|------|-----|------|------------|---------|
| ASAP7 | 7nm | 0.90V | 6.5nm | rvt, lvt, slvt, sram | 8 |
| TSMC5 | 5nm | 0.65V | 6.0nm | svt, lvt, ulvt, elvt | 8 |
| TSMC7 | 7nm | 0.75V | 6.0nm | svt, lvt, ulvt | 6 |
| TSMC12 | 12nm | 0.80V | 6.0nm | svt, lvt, hvt, ulvt, lnvt | 10 |
| TSMC16 | 16nm | 0.80V | 6.0nm | svt, lvt, hvt, ulvt, lnvt | 10 |
| **Total** | | | | | **42** |

Each "device" is an NMOS/PMOS pair for a given Vt flavor. For example, ASAP7 has 4 flavors x 2 polarities = 8 devices: `nmos_rvt`, `pmos_rvt`, `nmos_lvt`, `pmos_lvt`, etc.

Gate lengths and NFIN ranges are defined by PDK bin boundaries. The sweep engine reads these directly from the TSMC PDK files (discrete lmin values and nfinmin/nfinmax groups). For ASAP7, TSMC7's NFIN boundaries are used as reference.

## Output Format

### CSV Schema

Each row is one DC operating point. The base schema has 28 columns (more if process variation is enabled):

| Group | Columns | Unit |
|-------|---------|------|
| Identity | tech, device | -- |
| Geometry | L, NFIN, TFIN, temp_K | m, --, m, K |
| Voltage | Vg, Vd, Vs, Ve, Vth | V |
| Currents | id, ig, is, ie, ids | A |
| Charges | qg, qd, qs, qb | C |
| Derivatives | gm, gds, gmb | S |
| Capacitances | cgg, cgd, cgs, cdg, cdd | F |

When `--process-var` is used, the varied parameter columns (e.g., `eot`, `toxp`) are inserted between the geometry and voltage groups, sorted alphabetically.

### Dataset Size Estimates

With default settings (`sweep_geometry=True`, 5 temperatures, 50x50 voltage grid), data size depends on each technology's PDK variant structure. TSMC nodes typically have 25-42 geometry combos per device, ASAP7 has 6-7 (one L, TSMC7 NFIN boundaries).

### Non-Uniform Voltage Sampling

The Vg grid uses threshold-aware non-uniform sampling to concentrate points near the subthreshold-to-saturation transition region, which is critical for NN training accuracy:

- **Dense region** (default 60% of points): +/- 0.15*Vdd centered on Vth
- **Sparse region** (remaining 40%): uniformly distributed across [0, Vdd]
- **Vd grid**: uniform across [0, Vdd]

Threshold voltage (Vth) is auto-detected per device configuration via the peak-gm method before building the grid.

### Loading Data for Training

**Pandas:**

```python
import pandas as pd

df = pd.read_csv("training_data/ASAP7_dc.csv")
nmos = df[df["device"].str.startswith("nmos")]
print(f"NMOS rows: {len(nmos)}, columns: {list(df.columns)}")
```

**PyTorch Dataset:**

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MosfetDataset(Dataset):
    def __init__(self, csv_path, input_cols, output_cols):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[input_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[output_cols].values, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

ds = MosfetDataset(
    "training_data/ASAP7_dc.csv",
    input_cols=["L", "NFIN", "TFIN", "temp_K", "Vg", "Vd", "Vs", "Ve"],
    output_cols=["ids", "gm", "gds", "cgg", "cgd", "cgs"],
)
loader = DataLoader(ds, batch_size=1024, shuffle=True)
```

**TensorFlow:**

```python
import tensorflow as tf
import pandas as pd

df = pd.read_csv("training_data/ASAP7_dc.csv")
inputs = df[["L", "NFIN", "TFIN", "temp_K", "Vg", "Vd", "Vs", "Ve"]].values
outputs = df[["ids", "gm", "gds", "cgg", "cgd", "cgs"]].values

dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
dataset = dataset.shuffle(10000).batch(1024).prefetch(tf.data.AUTOTUNE)
```

## Advanced Usage

### Single-Point Evaluation

For interactive exploration or custom sweep logic, use the `Model` and `Instance` API directly:

```python
from pycmg import Model, Instance

model = Model(
    osdi_path="build/osdi/bsimcmg.osdi",
    modelcard_path="modelcards/ASAP7/7nm_TT_160803.pm",
    model_name="nmos_rvt",
)

inst = Instance(model, params={"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 2.0})

result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
print(f"Ids = {result['ids']:.3e} A")
print(f"gm  = {result['gm']:.3e} S")
print(f"gds = {result['gds']:.3e} S")
print(f"cgg = {result['cgg']:.3e} F")
```

All 17 outputs are available: `id`, `ig`, `is`, `ie`, `ids`, `qg`, `qd`, `qs`, `qb`, `gm`, `gds`, `gmb`, `cgg`, `cgd`, `cgs`, `cdg`, `cdd`.

### Temperature Sweeps

Temperature is specified in Kelvin. Convert from Celsius: `temp_K = temp_C + 273.15`.

```python
for temp_c in [-40, 27, 85, 125]:
    temp_k = temp_c + 273.15
    inst = Instance(model, params={"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 2.0},
                    temperature=temp_k)
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
    print(f"T={temp_c:4d}C: Ids={result['ids']:.3e} A, gm={result['gm']:.3e} S")
```

### Body Bias Effects

Apply body bias via the `e` (extended/bulk) terminal:

```python
for ve in [0.0, 0.1, 0.2, 0.3]:
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": ve})
    print(f"Ve={ve:.1f}V: Ids={result['ids']:.3e} A, gmb={result['gmb']:.3e} S")
```

In sweep mode, use `--ve-values 0.0 0.1 0.2 0.3` on the CLI.

### NFIN Scaling

```python
for nfin in [2, 3, 5, 10]:
    inst = Instance(model, params={"L": 7e-9, "TFIN": 6.5e-9, "NFIN": float(nfin)})
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
    print(f"NFIN={nfin}: Ids={result['ids']:.3e} A")
```

> **Note:** NFIN=1 causes convergence failures for certain TSMC process variants (e.g., tsmc5:ulvt, tsmc16:lnvt) where BSIM-CMG parameters go negative. Use NFIN >= 2 for reliable results. See [Known Limitations](#known-limitations).

### Custom Voltage Grids

Build your own voltage grid for finer control:

```python
from pycmg.sweep import build_voltage_grid, find_threshold, build_nodes

vth = find_threshold(inst, vdd=0.9, device_type="nmos")

# Standard range [0, VDD]
vg_arr, vd_arr = build_voltage_grid(vdd=0.9, vth_mag=vth, vg_points=100, vd_points=100)

# Extended range [0, 2*VDD] for simulator convergence training
vg_arr, vd_arr = build_voltage_grid(vdd=0.9, vth_mag=vth, vg_points=100, vd_points=100,
                                     voltage_scale=2.0)

for vg in vg_arr:
    for vd in vd_arr:
        nodes = build_nodes(vg, vd, 0.0, 0.9, "nmos")
        result = inst.eval_dc(nodes)
```

### PMOS Conventions

PMOS devices use source-at-Vdd convention. In magnitude space (used by the sweep engine), all voltages are positive. The `build_nodes()` function handles the polarity mapping:

```python
# NMOS: source at ground
nodes_nmos = build_nodes(vg_mag=0.5, vd_mag=0.7, ve_mag=0.0, vdd=0.9, device_type="nmos")
# -> {"g": 0.5, "d": 0.7, "s": 0.0, "e": 0.0}

# PMOS: source at Vdd, voltages reflected
nodes_pmos = build_nodes(vg_mag=0.5, vd_mag=0.7, ve_mag=0.0, vdd=0.9, device_type="pmos")
# -> {"g": 0.4, "d": 0.2, "s": 0.9, "e": 0.9}
```

### Transient Analysis

```python
result = inst.eval_tran(
    nodes={"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0},
    time=1e-9,
    delta_t=1e-12,
)
# Returns: id, ig, is, ie, ids, qg, qd, qs, qb
```

### Jacobian Matrix

Extract the condensed 4x4 Jacobian (dI/dV):

```python
J = inst.get_jacobian_matrix({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
# J is a 4x4 numpy array, terminals ordered as [d, g, s, e]
# J[i,j] = dI_terminal_i / dV_terminal_j
```

## Verification Against NGSPICE

### Strategy

PyCMG wraps the OSDI binary directly via ctypes, while NGSPICE loads the **same** OSDI binary via the `.osdi` command. Tests compare PyCMG output vs NGSPICE output to verify:

1. **Binary-level consistency**: Both use the identical `bsimcmg.osdi` file
2. **Ctypes wrapper correctness**: Proper OSDI function call sequences
3. **Numerical accuracy**: Direct comparison of currents, charges, derivatives
4. **Full model coverage**: DC, AC (capacitance), and transient analysis

The OSDI binary is the single source of truth for all model physics calculations.

### Tolerances

| Parameter | Absolute Tolerance | Relative Tolerance |
|-----------|--------------------|--------------------|
| Current (A) | 1e-9 | 0.5% |
| Charge (C) | 1e-18 | 0.5% |
| Conductance (S) | 1e-6 | 1% |
| Capacitance (F) | 1e-18 | 1% |

### Running the Test Suite

```bash
# Quick smoke tests (no NGSPICE required)
pytest tests/test_api.py -v

# Technology registry tests (no NGSPICE required)
pytest tests/test_tech.py -v

# Base technology verification (5 techs, NGSPICE required)
pytest tests/test_dc_jacobian.py tests/test_dc_regions.py tests/test_transient.py -v

# Vt variant verification (16 additional Vt flavors)
pytest tests/test_vt_variants.py -v

# Full suite (280 tests)
pytest tests/ -v
```

| Test File | Tests | Description | NGSPICE |
|-----------|-------|-------------|---------|
| `test_api.py` | 20 | API smoke tests | No |
| `test_tech.py` | 13 | Technology registry and config tests | No |
| `test_sweep.py` | 27 | Sweep engine tests | No |
| `test_nfin_scaling.py` | 2 | NFIN scaling sanity | No |
| `test_sensitivity.py` | 7 | Sensitivity analysis tests | No |
| `test_dc_jacobian.py` | 30 | DC Jacobian (gm, gds, gmb) vs NGSPICE | Yes |
| `test_dc_regions.py` | 30 | DC operating regions (off/linear/saturation) vs NGSPICE | Yes |
| `test_transient.py` | 10 | Transient waveforms vs NGSPICE | Yes |
| `test_ac_caps.py` | 15 | AC capacitances vs NGSPICE | Yes |
| `test_body_bias.py` | 20 | Body bias verification vs NGSPICE | Yes |
| `test_temperature.py` | 10 | Temperature sweep vs NGSPICE | Yes |
| `test_vt_variants.py` | 96 | Vt variant DC verification vs NGSPICE | Yes |

## Project Structure

```
pycmg-wrapper/
├── pycmg/                        # Python package
│   ├── __init__.py              # Public API exports (Model, Instance, generate_dataset, ...)
│   ├── osdi_types.py            # OSDI constants, ctypes structures, function types
│   ├── core.py                  # Low-level OSDI interface (OsdiLibrary, OsdiModel, OsdiInstance)
│   ├── parser.py                # Modelcard parsing, PDK introspection (scan_pdk_geometry_combos)
│   ├── model.py                 # Public API (Model, Instance, eval_dc, eval_tran)
│   ├── tech.py                  # Technology registry (TECH_REGISTRY, DeviceConfig, TechConfig)
│   ├── sweep.py                 # Sweep engine (generate_dataset, SweepConfig, sweep_dc, to_csv)
│   └── sensitivity.py           # Sensitivity analysis (compute_sensitivity, SensitivityResult)
├── tests/                        # Test suite (280 tests)
│   ├── conftest.py              # Tiered technology registry (21 entries)
│   ├── helpers.py               # NGSPICE runner helpers, comparison functions
│   ├── test_api.py              # API smoke tests
│   ├── test_tech.py             # Technology registry tests
│   ├── test_sweep.py            # Sweep engine tests (incl. voltage_scale)
│   ├── test_sensitivity.py      # Sensitivity analysis tests
│   ├── test_dc_jacobian.py      # DC Jacobian verification
│   ├── test_dc_regions.py       # DC operating region tests
│   ├── test_transient.py        # Transient waveform verification
│   ├── test_ac_caps.py          # AC capacitance verification
│   ├── test_body_bias.py        # Body bias verification
│   ├── test_temperature.py      # Temperature sweep verification
│   ├── test_nfin_scaling.py     # NFIN scaling sanity
│   └── test_vt_variants.py      # Vt variant DC verification
├── scripts/                      # CLI utilities
│   ├── generate_training_data.py # Training data generation CLI
│   ├── sensitivity_analysis.py  # Process parameter sensitivity CLI
│   └── generate_naive_tsmc.py   # Naive TSMC modelcard generator
├── modelcards/                   # Technology model cards
│   ├── ASAP7/                   # ASAP7 PDK model files
│   ├── TSMC5/                   # TSMC 5nm model files
│   ├── TSMC7/                   # TSMC 7nm model files
│   ├── TSMC12/                  # TSMC 12nm model files
│   └── TSMC16/                  # TSMC 16nm model files
├── bsim-cmg-va/                  # Verilog-A source and documentation
│   └── code/                    # BSIM-CMG Verilog-A source files
├── build/                        # Build artifacts (generated)
│   ├── osdi/bsimcmg.osdi       # Compiled OSDI binary
│   └── modelcards/              # Cached generated TSMC modelcards
└── CMakeLists.txt                # Build system
```

## API Reference

### pycmg.sweep

**`generate_dataset(osdi_path, techs, devices, output_dir, ...)`** -- Convenience wrapper. Builds a `SweepConfig`, runs `sweep_dc()`, writes CSVs via `to_csv()`. Returns list of output file paths.

**`SweepConfig`** -- Dataclass configuring the full sweep: `techs`, `devices`, `sweep_geometry` (bool, default True), `temperatures`, `vg_points`, `vd_points`, `ve_values`, `process_vars`, `dense_ratio`, `voltage_scale`.

**`SweepResult`** -- Container with `columns` (ordered column names), `data` (list of rows), `metadata` (timing, counts).

**`sweep_dc(osdi_path, config, verbose)`** -- Core sweep loop. When `sweep_geometry=True`, iterates technologies x devices x PDK-defined (L, NFIN) combos x temperatures x process combos x voltage grid. Returns `SweepResult`.

**`to_csv(results, output_dir, split_by)`** -- Writes `SweepResult` to CSV files. `split_by` controls grouping: `"tech"` (default), `"device"`, or `"none"`.

**`build_voltage_grid(vdd, vth_mag, vg_points, vd_points, dense_ratio, voltage_scale)`** -- Non-uniform Vg + uniform Vd grid builder. `voltage_scale` extends the grid to `vdd * voltage_scale` (default 1.0).

**`find_threshold(inst, vdd, device_type, n_coarse)`** -- Peak-gm threshold detection.

**`build_nodes(vg_mag, vd_mag, ve_mag, vdd, device_type)`** -- Magnitude-space to terminal-voltage mapping.

### pycmg.model

**`Model(osdi_path, modelcard_path, model_name)`** -- Loads BSIM-CMG model from OSDI binary + modelcard file.

**`Instance(model, params, temperature, model_overrides)`** -- Device instance with geometry. `params` sets instance parameters (L, TFIN, NFIN). `temperature` in Kelvin (default: 300.15). `model_overrides` overrides modelcard parameters (for process variation).

- `eval_dc(nodes) -> dict` -- DC operating point. Returns 17 outputs. Raises `RuntimeError` if internal node NR fails to converge (e.g., NFIN=1 with certain TSMC variants).
- `eval_tran(nodes, time, delta_t) -> dict` -- Transient evaluation. Returns 9 outputs. Warns (instead of raising) on internal node convergence failure; the circuit-level NR provides outer convergence.
- `get_jacobian_matrix(nodes) -> np.ndarray` -- 4x4 condensed Jacobian (dI/dV).
- `set_params(params, allow_rebind)` -- Update instance parameters.

### pycmg.tech

**`TECH_REGISTRY`** -- Dict mapping technology names to `TechConfig` objects. 5 technologies, 42 devices total.

**`TechConfig`** -- Technology node config: `name`, `vdd`, `tfin`, `devices` (dict of `DeviceConfig`), `pdk_path`.

**`DeviceConfig`** -- Single device config: `model_name`, `inst_params`, `modelcard`, `pdk_device`, `get_min_l()`, `get_geometry_combos()`.

**`resolve_modelcard(device, tech, L, NFIN=None)`** -- Returns modelcard path. For ASAP7, returns the static file. For TSMC, generates a naive modelcard from the PDK on-the-fly and caches it under `build/modelcards/`. When `NFIN` is provided, selects the correct NFIN-group variant.

**`get_tech_config(name)`** / **`list_techs()`** -- Registry lookup helpers.

### pycmg.sensitivity

**`compute_sensitivity(osdi_path, modelcard_path, model_name, inst_params, vdd, device_type, temperature, delta_fraction, top_n, verbose)`** -- OAT sensitivity analysis. Perturbs each real-valued model parameter by `+/- delta_fraction` and measures normalized output change at 4 representative bias points. Returns `SensitivityResult`.

**`SensitivityResult`** -- Container with `param_names`, `sensitivities` (per-param per-output normalized sensitivity), `rankings` (per-category top-N lists: `"iv"`, `"qv"`, `"cv"`), `bias_points`, `delta_fraction`.

**`enumerate_model_params(desc, model)`** -- Discovers all real-valued model-level parameters from the OSDI descriptor. Returns list of `ParamInfo(index, name, value)`.

**`rank_parameters(sensitivities, categories, top_n)`** -- Ranks parameters by aggregate sensitivity within each output category.

**`format_sensitivity_table(result, category)`** -- Formats a ranked sensitivity table for terminal output.

### pycmg.parser

**`parse_modelcard(path, target_model_name)`** -- Parses a SPICE `.model` block. Returns `ParsedModel` with `name` and `params` dict.

**`parse_number_with_suffix(s)`** -- Parses SPICE numbers with engineering suffixes (e.g., `"16n"` -> `16e-9`, `"1.5meg"` -> `1.5e6`).

**`scan_pdk_geometry_combos(path, base_name)`** -- Enumerates PDK-defined (L, NFIN) sweep points for a TSMC device. For each variant, returns `(lmin, nfinmin)` and `(lmin, nfinmax)`. Sorted and deduplicated.

## Known Limitations

### NFIN=1 Convergence Failures

BSIM-CMG computes NFIN-dependent instance parameters (ETA0_i, U0_i, UA_i) that can become negative at NFIN=1 for certain process variants. The OSDI binary warns but does not abort. The internal node Newton-Raphson then diverges monotonically (0.2 V/step × 200 iterations → ~40 V internal drain), producing `id ≈ 40 kA` and `NaN` for all derivatives.

**Affected variants** (known): `tsmc5:ulvt`, `tsmc16:lnvt` at NFIN=1. Other techs (ASAP7, TSMC7, TSMC12) and NFIN ≥ 2 are unaffected.

**Behavior**: `eval_dc` raises `RuntimeError` when internal NR fails to converge. Callers that sweep bias points should catch this exception. `eval_tran` warns instead of raising.

**Recommendation**: Use NFIN ≥ 2 for data generation and sweeps. NFIN=1 single-fin devices are an edge case rarely used in real designs.

## License

This project is provided for educational and research purposes. The BSIM-CMG model is licensed separately by the BSIM Group at UC Berkeley.
