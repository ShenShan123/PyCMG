from pathlib import Path

import pycmg


def test_eval_dc_returns_fields():
    root = Path(__file__).resolve().parents[1]
    osdi_path = root / "build" / "osdi" / "bsimcmg.osdi"
    modelcard = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
    model = pycmg.Model(str(osdi_path), str(modelcard), "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
    for key in (
        "id",
        "ig",
        "is",
        "ie",
        "qg",
        "qd",
        "qs",
        "qb",
        "gm",
        "gds",
        "gmb",
        "cgg",
        "cgd",
        "cgs",
        "cdg",
        "cdd",
    ):
        assert key in out
