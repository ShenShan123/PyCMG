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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pycmg.parser import parse_number_with_suffix

# Project root for resolving relative modelcard/pdk paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Regex for key=value assignments in SPICE model blocks
_ASSIGN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?[a-zA-Z]*)"
)


def _resolve_path(rel_path: str) -> Path:
    """Resolve a path that may be relative to project root or absolute."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def _parse_modelcard_l(modelcard_path: str) -> float:
    """Extract L from the first .model block in a modelcard file.

    Used for ASAP7-style modelcards where L is a parameter inside the
    model block (on continuation lines starting with '+').

    Args:
        modelcard_path: Path to the modelcard file (relative or absolute).

    Returns:
        Gate length L in meters.

    Raises:
        RuntimeError: If no L parameter is found in any .model block.
    """
    path = _resolve_path(modelcard_path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    in_model = False
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.lower().startswith(".model"):
            in_model = True
            continue
        if in_model and stripped.startswith("+"):
            content = stripped[1:]
            for m in _ASSIGN_RE.finditer(content):
                key = m.group(1).lower()
                if key == "l":
                    return parse_number_with_suffix(m.group(2))
        elif in_model and not stripped.startswith("+"):
            # End of model block continuation — only check the first block
            break

    raise RuntimeError(
        f"No L parameter found in modelcard: {modelcard_path}"
    )


def _scan_pdk_device_min_l(pdk_path: str, device_name: str) -> float:
    """Scan a TSMC PDK file for the minimum lmin across all numbered variants.

    TSMC PDK files contain numbered model bins like ``nch_svt_mac.1``,
    ``nch_svt_mac.2``, etc.  Each bin has ``lmin`` and ``lmax`` parameters
    defining its length range.  This function finds the global minimum
    ``lmin`` across all numbered variants for the given device.

    The ``lmin`` parameter may appear on the ``.model`` line itself or on
    continuation lines (starting with ``+``).

    Args:
        pdk_path: Path to the TSMC PDK file (relative or absolute).
        device_name: PDK device name, e.g. ``"nch_svt_mac"``.

    Returns:
        Minimum gate length in meters.

    Raises:
        RuntimeError: If no numbered variants are found for the device.
    """
    path = _resolve_path(pdk_path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    prefix = f"{device_name.lower()}."
    all_lmin: list[float] = []
    idx = 0

    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()

        if not stripped or stripped.startswith("*"):
            idx += 1
            continue

        if stripped.lower().startswith(".model"):
            parts = stripped.split()
            if len(parts) >= 3:
                model_name = parts[1].lower()
                if model_name.startswith(prefix):
                    suffix = model_name[len(prefix):]
                    if suffix.isdigit():
                        # Collect all lines of this model block
                        block_text = stripped
                        idx += 1
                        while idx < len(lines):
                            cont = lines[idx].strip()
                            if not cont or cont.startswith("*"):
                                idx += 1
                                continue
                            if cont.startswith("+"):
                                block_text += " " + cont[1:]
                                idx += 1
                                continue
                            break
                        # Extract lmin from the block
                        for m in _ASSIGN_RE.finditer(block_text):
                            if m.group(1).lower() == "lmin":
                                all_lmin.append(
                                    parse_number_with_suffix(m.group(2))
                                )
                                break
                        continue  # idx already advanced

        idx += 1

    if not all_lmin:
        raise RuntimeError(
            f"No numbered variants found for device '{device_name}' "
            f"in PDK file: {pdk_path}"
        )
    return min(all_lmin)


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

    def get_min_l(self, pdk_path: str | None = None) -> float:
        """Return minimum gate length for this device, auto-detecting if needed.

        For TSMC devices (pdk_device is set), scans the PDK file for the
        smallest ``lmin`` across all numbered length variants.  For ASAP7
        devices (modelcard is set), parses L from the first ``.model`` block.

        The result is cached in ``_min_l`` after the first call.

        Args:
            pdk_path: Path to TSMC PDK file, required for TSMC devices.
                Ignored for ASAP7 devices (which use modelcard instead).

        Returns:
            Minimum gate length in meters.

        Raises:
            RuntimeError: If min_l cannot be determined (no modelcard,
                no pdk_device, or missing pdk_path for TSMC device).
        """
        if self._min_l is not None:
            return self._min_l
        if self.pdk_device is not None and pdk_path is not None:
            self._min_l = _scan_pdk_device_min_l(pdk_path, self.pdk_device)
        elif self.modelcard is not None:
            self._min_l = _parse_modelcard_l(self.modelcard)
        else:
            raise RuntimeError(
                f"Cannot detect min_l for {self.model_name}: "
                f"no modelcard or pdk_device+pdk_path available"
            )
        return self._min_l


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
