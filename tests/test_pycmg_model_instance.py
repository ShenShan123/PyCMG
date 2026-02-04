from pathlib import Path

import pycmg


def test_model_instance_construct():
    root = Path(__file__).resolve().parents[1]
    osdi_path = root / "build" / "osdi" / "bsimcmg.osdi"
    modelcard = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
    model = pycmg.Model(str(osdi_path), str(modelcard), "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    assert inst is not None
