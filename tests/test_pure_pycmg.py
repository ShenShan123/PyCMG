from pathlib import Path
import re
import subprocess

import pytest


def test_parse_number_with_suffix() -> None:
    from pycmg import ctypes_host

    parse = ctypes_host.parse_number_with_suffix
    assert parse("1") == 1.0
    assert parse("1u") == pytest.approx(1e-6)
    assert parse("2n") == pytest.approx(2e-9)
    assert parse("3meg") == pytest.approx(3e6)
    assert parse("4t") == pytest.approx(4e12)


def test_eval_dc_smoke() -> None:
    from pycmg import ctypes_host

    root = Path(__file__).resolve().parents[1]
    osdi_path = root / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    modelcard = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"

    if not osdi_path.exists():
        pytest.skip("missing OSDI build artifact")
    if not modelcard.exists():
        pytest.skip("missing modelcard")

    model = ctypes_host.Model(str(osdi_path), str(modelcard), "nmos1")
    inst = ctypes_host.Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})
    out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})

    for key in ("id", "ig", "is", "ie", "qg", "qd", "qs", "qb", "gm", "gds", "gmb",
                "cgg", "cgd", "cgs", "cdg", "cdd"):
        assert key in out
        assert isinstance(out[key], float)


def test_eval_tran_smoke() -> None:
    from pycmg import ctypes_host

    root = Path(__file__).resolve().parents[1]
    osdi_path = root / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    modelcard = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"

    if not osdi_path.exists():
        pytest.skip("missing OSDI build artifact")
    if not modelcard.exists():
        pytest.skip("missing modelcard")

    model = ctypes_host.Model(str(osdi_path), str(modelcard), "nmos1")
    inst = ctypes_host.Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})
    out = inst.eval_tran({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0}, 1e-12, 1e-12)

    for key in ("id", "ig", "is", "ie", "qg", "qd", "qs", "qb"):
        assert key in out
        assert isinstance(out[key], float)


def test_eval_dc_matches_osdi_eval() -> None:
    from pycmg import ctypes_host

    root = Path(__file__).resolve().parents[1]
    osdi_path = root / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
    modelcard = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
    osdi_eval = root / "build-deep-verify" / "osdi_eval"

    if not osdi_path.exists():
        pytest.skip("missing OSDI build artifact")
    if not modelcard.exists():
        pytest.skip("missing modelcard")
    if not osdi_eval.exists():
        pytest.skip("missing osdi_eval binary")

    cmd = [
        str(osdi_eval),
        "--osdi", str(osdi_path),
        "--modelcard", str(modelcard),
        "--node", "d=0.05",
        "--node", "g=0.8",
        "--node", "s=0.0",
        "--node", "e=0.0",
        "--param", "NFIN=2",
        "--print-charges",
        "--print-cap",
        "--print-derivs",
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    id_val = ig_val = is_val = ie_val = None
    for line in res.stdout.splitlines():
        if line.startswith("Id="):
            m = re.findall(r"Id=([eE0-9+\-\.]+)\s+Ig=([eE0-9+\-\.]+)\s+Is=([eE0-9+\-\.]+)\s+Ie=([eE0-9+\-\.]+)", line)
            if m:
                id_val = float(m[0][0])
                ig_val = float(m[0][1])
                is_val = float(m[0][2])
                ie_val = float(m[0][3])
    if id_val is None:
        pytest.fail("failed to parse osdi_eval output")

    model = ctypes_host.Model(str(osdi_path), str(modelcard), "nmos1")
    inst = ctypes_host.Instance(model, params={"NFIN": 2.0})
    out = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})

    assert out["id"] == pytest.approx(id_val, rel=5e-3, abs=1e-9)
    assert out["ig"] == pytest.approx(ig_val, rel=5e-3, abs=1e-9)
    assert out["is"] == pytest.approx(is_val, rel=5e-3, abs=1e-9)
    assert out["ie"] == pytest.approx(ie_val, rel=5e-3, abs=1e-9)
