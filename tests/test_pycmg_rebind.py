from pathlib import Path
from typing import List, Optional, Tuple

import pycmg


def test_set_params_detects_topology_change() -> None:
    root: Path = Path(__file__).resolve().parents[1]
    osdi_path: Path = root / "build" / "osdi" / "bsimcmg.osdi"
    modelcard: Path = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
    model = pycmg.Model(str(osdi_path), str(modelcard), "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    base = inst.internal_node_count()

    candidates: List[Tuple[str, float]] = [
        ("nqsmod", 1.0),
        ("shmod", 1.0),
        ("rgatemod", 1.0),
    ]

    changed: Optional[Tuple[str, float]] = None
    for name, value in candidates:
        try:
            inst.set_params({name: value}, allow_rebind=True)
        except RuntimeError:
            continue
        if inst.internal_node_count() != base:
            changed = (name, value)
            break

    if changed is None:
        import pytest

        pytest.skip("No topology-changing params in current build; enable NQSMOD/SHMOD to test.")

    name, value = changed
    inst2 = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    import pytest

    with pytest.raises(RuntimeError):
        inst2.set_params({name: value})


def test_internal_node_count_exposed() -> None:
    root: Path = Path(__file__).resolve().parents[1]
    osdi_path: Path = root / "build" / "osdi" / "bsimcmg.osdi"
    modelcard: Path = root / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
    model = pycmg.Model(str(osdi_path), str(modelcard), "nmos")
    inst = pycmg.Instance(model, params={"l": 2e-8, "tfin": 1e-8, "nfin": 5})
    count = inst.internal_node_count()
    assert isinstance(count, int)
    assert count >= 0
