# PyCMG Wrapper

PyCMG provides a Python interface to the BSIM-CMG Verilog-A model using OpenVAF/OSDI, with NGSPICE-backed verification.

## Requirements
- OpenVAF: `/usr/local/bin/openvaf`
- NGSPICE: `/usr/local/ngspice-45.2/bin/ngspice`
- CMake + Make
- Python 3.10+ (PyBind11 for the extension)

## Environment overrides
- `NGSPICE_BIN`: override the NGSPICE binary path (used by `tests/verify_utils.py`).
- `ASAP7_MODELCARD`: point ASAP7 verification at a single modelcard file or a directory of `.pm` files.

## Build artifacts
Verification builds into `build-deep-verify/` (OSDI library, `osdi_eval`, and `_pycmg`). The build is created on demand by `tests/verify_utils.py`.

## Entry point: `main.py`
`main.py` is the primary entry for verification tests and data collection.

### Run tests
```bash
python main.py test comprehensive
python main.py test repro
python main.py test asap7-full
python main.py test asap7-pvt
python main.py test all
```

### Collect data (deep verify / ASAP7)
```bash
python main.py collect deep-verify --temps 27 --tran
python main.py collect asap7 --temp-c 27.0
```

## Direct pytest
```bash
pytest tests/ -v
```

Note: `pytest tests` is long-running because ASAP7 and NGSPICE sweeps execute many cases. Expect on the order of tens of minutes.
