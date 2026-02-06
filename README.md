# PyCMG (Python Model Evaluator for BSIM‑CMG via OSDI)

PyCMG is a **Python‑first model evaluator** for BSIM‑CMG using OSDI. It does **not** solve circuits (no KCL/KVL). It maps terminal voltages to currents, charges, and Jacobians so you can generate clean device‑level data for ML, characterization, or custom simulators.

## What this is (and isn’t)
- ✅ **Model evaluator**: given terminal voltages → returns Id/Ig/Is/Ie, Qg/Qd/Qs/Qb, gm/gds/gmb, condensed capacitances.
- ✅ **Validated** against NGSPICE using provided scripts.
- ❌ **Not a circuit solver**: no nodal analysis, no transient integration beyond model evaluation.

## Environment setup (Conda recommended)
Create and use the **pycmg-pybind** environment (Python 3.11):

```bash
conda create -n pycmg-pybind python=3.11
conda activate pycmg-pybind
```

Install Python dependencies via pip (Tsinghua mirror):

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pytest
```

## Prerequisites
- **Python**: 3.11+ (tested)
- **OSDI model binary**: `build-deep-verify/osdi/bsimcmg.osdi`
- **Model cards**: `bsim-cmg-va/benchmark_test/modelcard.nmos` and `.pmos`

If the OSDI binary is missing, build it using your existing OpenVAF/OSDI build pipeline.

## Quickstart
```python
import pycmg

# Paths
osdi_path = "build-deep-verify/osdi/bsimcmg.osdi"
modelcard = "bsim-cmg-va/benchmark_test/modelcard.nmos"

# Load model and instance
model = pycmg.Model(osdi_path, modelcard, "nmos1")
inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0})

# DC evaluation
out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
print(out["id"], out["qg"], out["cgg"])  # currents / charges / caps

# Transient evaluation (QS path)
out_tran = inst.eval_tran({"d": 0.05, "g": 1.0, "s": 0.0, "e": 0.0}, t=1e-9, delta_t=1e-12)
print(out_tran["id"], out_tran["qg"])
```

## Detailed usage
### Model / Instance
```python
model = pycmg.Model(osdi_path, modelcard, "nmos1")
inst = pycmg.Instance(
    model,
    params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
    temperature=300.15,  # Kelvin
)
```

### Update instance parameters
```python
# Safe for topology changes when allow_rebind=True
inst.set_params({"L": 2.0e-8, "NFIN": 4.0}, allow_rebind=True)
```

### DC evaluation
```python
out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
```

### Transient evaluation (QS)
```python
out_tran = inst.eval_tran({"d": 0.05, "g": 1.0, "s": 0.0, "e": 0.0}, t=1e-9, delta_t=1e-12)
```

### Output keys
`eval_dc` returns (subset shown):
- **Currents**: `id`, `ig`, `is`, `ie`
- **Charges**: `qg`, `qd`, `qs`, `qb`
- **Conductances**: `gm`, `gds`, `gmb`
- **Condensed caps**: `cgg`, `cgd`, `cgs`, `cdg`, `cdd`

`eval_tran` returns similar keys (QS charge‑based current integration).

## Verification
All verification is **against NGSPICE ground truth**.

```bash
# Deep verification: DC/AC/Q across multiple corners
PYTHONPATH=. python scripts/deep_verify.py --backend pycmg

# Robustness checks: pulse stability, param sensitivity, threading
PYTHONPATH=. python scripts/test_robustness.py --all
```

## Notes & pitfalls
- **Temperature**: `pycmg.Instance(..., temperature=K)` expects **Kelvin**.
- **Model name**: ensure the modelcard `.model` name matches what you pass to `Model`.
- **OSDI path**: must point to a valid `.osdi` file.
- **No circuit solving**: results are per‑device, per‑bias only.

## Project layout (user‑relevant)
- `pycmg/` – Python interface and ctypes host
- `scripts/deep_verify.py` – NGSPICE validation harness
- `scripts/test_robustness.py` – robustness tests for data generation
- `bsim-cmg-va/benchmark_test/` – reference modelcards

## License
Project‑specific; see repository policy.
