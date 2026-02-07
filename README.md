# PyCMG Wrapper

PyCMG provides a Python interface to the BSIM-CMG Verilog-A model using OpenVAF/OSDI, with NGSPICE-backed verification.

## Requirements
- OpenVAF: `/usr/local/bin/openvaf`
- NGSPICE: `/usr/local/ngspice-45.2/bin/ngspice`
- CMake + Make
- Python 3.10+ (PyBind11 for the extension)

## Entry Point
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
