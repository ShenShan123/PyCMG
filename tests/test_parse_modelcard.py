from __future__ import annotations

import inspect
from pathlib import Path

import pycmg


def write_modelcard(path: Path, text: str) -> None:
    path.write_text(text)


def test_parse_modelcard_targets_level72_nmos(tmp_path: Path) -> None:
    card = tmp_path / "asap7.pm"
    write_modelcard(
        card,
        """
* header
.model nmos_lvt nmos level=72 l=14n tfin=7n
+ tox=1.5n
.model pmos_lvt pmos level=72 l=16n tfin=8n
+ tox=2.0n
""",
    )
    parsed = pycmg.ctypes_host.parse_modelcard(str(card), target_model_name="nmos_lvt")
    assert parsed.name == "nmos_lvt"
    assert parsed.params["l"] == pycmg.ctypes_host.parse_number_with_suffix("14n")
    assert parsed.params["tfin"] == pycmg.ctypes_host.parse_number_with_suffix("7n")
    assert parsed.params["tox"] == pycmg.ctypes_host.parse_number_with_suffix("1.5n")


def test_parse_modelcard_first_valid_when_no_target(tmp_path: Path) -> None:
    card = tmp_path / "multi.pm"
    write_modelcard(
        card,
        """
.model nmos_bad nmos level=71 l=10n
.model first_ok bsimcmg l=20n
+ tfin=9n
.model later_ok nmos level=72 l=30n
""",
    )
    parsed = pycmg.ctypes_host.parse_modelcard(str(card))
    assert parsed.name == "first_ok"
    assert parsed.params["l"] == pycmg.ctypes_host.parse_number_with_suffix("20n")
    assert parsed.params["tfin"] == pycmg.ctypes_host.parse_number_with_suffix("9n")


def test_model_init_signature_has_model_card_name() -> None:
    sig = inspect.signature(pycmg.Model.__init__)
    assert "model_card_name" in sig.parameters
