"""Technology registry for PyCMG training data generation.

Provides DeviceConfig, TechConfig, and TECH_REGISTRY covering all supported
FinFET technology nodes (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16) with their
device variants (rvt, lvt, slvt, svt, hvt, ulvt, elvt, lnvt, sram).

Usage::

    from pycmg.tech import TECH_REGISTRY, get_tech_config, list_techs

    tech = get_tech_config("TSMC7")
    dev = tech.get_device("nmos_svt")
    print(dev.pdk_device)  # "nch_svt_mac"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DeviceConfig:
    """Configuration for a single device within a technology node.

    Attributes:
        model_name: SPICE model name, e.g. "nmos_rvt", "nch_svt_mac".
        inst_params: Static instance parameters (TFIN, DEVTYPE only).
            L and NFIN are excluded because they are swept during data generation.
        modelcard: Path to a static modelcard file relative to project root.
            Used for ASAP7 which has pre-built modelcard files.
            None for TSMC nodes (modelcards are generated from PDK on the fly).
        pdk_device: Device name within the TSMC PDK file, e.g. "nch_svt_mac".
            None for ASAP7 (no PDK parsing needed).
        _min_l: Per-device minimum channel length, auto-detected and cached.
            Populated by Task 3 (get_min_l); defaults to None.
    """

    model_name: str
    inst_params: Dict[str, float]
    modelcard: Optional[str] = None
    pdk_device: Optional[str] = None
    _min_l: Optional[float] = None


@dataclass
class TechConfig:
    """Configuration for a technology node.

    Attributes:
        name: Technology identifier, e.g. "ASAP7", "TSMC7".
        vdd: Core supply voltage in volts.
        tfin: Fin height/thickness in meters (e.g. 6.5e-9 for ASAP7).
        devices: Mapping from canonical device name to DeviceConfig.
            Canonical names follow the pattern: {nmos|pmos}_{vt_flavor}.
        pdk_path: Path to the TSMC PDK file relative to project root.
            None for ASAP7 (uses static modelcards instead).
    """

    name: str
    vdd: float
    tfin: float
    devices: Dict[str, DeviceConfig] = field(default_factory=dict)
    pdk_path: Optional[str] = None

    def list_devices(self) -> List[str]:
        """Return sorted list of canonical device names."""
        return sorted(self.devices.keys())

    def get_device(self, name: str) -> DeviceConfig:
        """Look up a device by canonical name.

        Args:
            name: Canonical device name, e.g. "nmos_rvt", "pmos_svt".

        Returns:
            The corresponding DeviceConfig.

        Raises:
            KeyError: If the device name is not found.
        """
        if name not in self.devices:
            available = ", ".join(sorted(self.devices.keys()))
            raise KeyError(
                f"Device '{name}' not found in {self.name}. "
                f"Available: {available}"
            )
        return self.devices[name]


# ---------------------------------------------------------------------------
# Helper functions to reduce repetition in registry construction
# ---------------------------------------------------------------------------

_ASAP7_MODELCARD = "modelcards/ASAP7/7nm_TT_160803.pm"


def _asap7_device(model_name: str, *, is_pmos: bool) -> DeviceConfig:
    """Create a DeviceConfig for an ASAP7 device."""
    return DeviceConfig(
        model_name=model_name,
        inst_params={"TFIN": 6.5e-9, "DEVTYPE": 0 if is_pmos else 1},
        modelcard=_ASAP7_MODELCARD,
        pdk_device=None,
    )


def _tsmc_device(
    pdk_device: str, *, tfin: float, is_pmos: bool
) -> DeviceConfig:
    """Create a DeviceConfig for a TSMC device."""
    return DeviceConfig(
        model_name=pdk_device,
        inst_params={"TFIN": tfin, "DEVTYPE": 0 if is_pmos else 1},
        modelcard=None,
        pdk_device=pdk_device,
    )


def _asap7_devices(vt_flavors: List[str]) -> Dict[str, DeviceConfig]:
    """Build device dict for ASAP7 with given Vt flavors."""
    devices: Dict[str, DeviceConfig] = {}
    for vt in vt_flavors:
        devices[f"nmos_{vt}"] = _asap7_device(f"nmos_{vt}", is_pmos=False)
        devices[f"pmos_{vt}"] = _asap7_device(f"pmos_{vt}", is_pmos=True)
    return devices


def _tsmc_devices(
    vt_flavors: List[str], *, tfin: float
) -> Dict[str, DeviceConfig]:
    """Build device dict for a TSMC node with given Vt flavors.

    Uses the naming convention: nch_{vt}_mac / pch_{vt}_mac for PDK device
    names, and nmos_{vt} / pmos_{vt} for canonical device names.
    """
    devices: Dict[str, DeviceConfig] = {}
    for vt in vt_flavors:
        pdk_n = f"nch_{vt}_mac"
        pdk_p = f"pch_{vt}_mac"
        devices[f"nmos_{vt}"] = _tsmc_device(pdk_n, tfin=tfin, is_pmos=False)
        devices[f"pmos_{vt}"] = _tsmc_device(pdk_p, tfin=tfin, is_pmos=True)
    return devices


# ---------------------------------------------------------------------------
# TECH_REGISTRY: master technology registry
# ---------------------------------------------------------------------------

TECH_REGISTRY: Dict[str, TechConfig] = {
    "ASAP7": TechConfig(
        name="ASAP7",
        vdd=0.9,
        tfin=6.5e-9,
        devices=_asap7_devices(["rvt", "lvt", "slvt", "sram"]),
        pdk_path=None,
    ),
    "TSMC5": TechConfig(
        name="TSMC5",
        vdd=0.65,
        tfin=6e-9,
        devices=_tsmc_devices(["svt", "lvt", "ulvt", "elvt"], tfin=6e-9),
        pdk_path="modelcards/TSMC5/cln5_1d2_sp_v1d2_2p2.l",
    ),
    "TSMC7": TechConfig(
        name="TSMC7",
        vdd=0.75,
        tfin=6e-9,
        devices=_tsmc_devices(["svt", "lvt", "ulvt"], tfin=6e-9),
        pdk_path="modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l",
    ),
    "TSMC12": TechConfig(
        name="TSMC12",
        vdd=0.80,
        tfin=6e-9,
        devices=_tsmc_devices(
            ["svt", "lvt", "hvt", "ulvt", "lnvt"], tfin=6e-9
        ),
        pdk_path="modelcards/TSMC12/cln12ffcll_1d8_sp_v1d0_2p4.l",
    ),
    "TSMC16": TechConfig(
        name="TSMC16",
        vdd=0.80,
        tfin=6e-9,
        devices=_tsmc_devices(
            ["svt", "lvt", "hvt", "ulvt", "lnvt"], tfin=6e-9
        ),
        pdk_path="modelcards/TSMC16/crn16ffcll_1d8_sp_v1d0_2p1.l",
    ),
}


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def list_techs() -> List[str]:
    """Return sorted list of all registered technology names."""
    return sorted(TECH_REGISTRY.keys())


def get_tech_config(name: str) -> TechConfig:
    """Look up a technology by name.

    Args:
        name: Technology name, e.g. "ASAP7", "TSMC7".

    Returns:
        The corresponding TechConfig.

    Raises:
        KeyError: If the technology name is not found.
    """
    if name not in TECH_REGISTRY:
        available = ", ".join(sorted(TECH_REGISTRY.keys()))
        raise KeyError(
            f"Technology '{name}' not found. Available: {available}"
        )
    return TECH_REGISTRY[name]
