from pathlib import Path
from typing import Dict

import pycmg


def test_eval_dc_returns_fields() -> None:
    root: Path = Path(__file__).resolve().parents[1]
    osdi_path: Path = root / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    modelcard: Path = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
    model = pycmg.Model(str(osdi_path), str(modelcard), "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    out: Dict[str, float] = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
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
