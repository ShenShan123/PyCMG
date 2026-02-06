import pycmg


def test_import() -> None:
    assert hasattr(pycmg, "Model")
