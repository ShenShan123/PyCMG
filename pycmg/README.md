# pycmg

PyBind11 bindings for the BSIM-CMG OSDI host.

## Usage

```python
import pycmg

model = pycmg.Model("build/osdi/bsimcmg.osdi",
                    "bsim-cmg-va/benchmark_test/modelcard.nmos",
                    "nmos")
inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})

out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
print(out["id"], out["cgg"])
```

## Build

```bash
cmake -S . -B build-pycmg \
  -DPython_EXECUTABLE=/opt/miniconda3/bin/python3.13 \
  -DPython_ROOT_DIR=/opt/miniconda3 \
  -Dpybind11_DIR=$(python3.13 -m pybind11 --cmakedir)
cmake --build build-pycmg --target _pycmg
```
