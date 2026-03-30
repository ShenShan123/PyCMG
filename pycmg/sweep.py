"""Sweep building blocks for PyCMG training data generation.

Provides voltage grid construction, threshold detection, device resolution,
and configuration dataclasses for systematic parameter sweeps across
technology nodes and device variants.
"""

from __future__ import annotations

import fnmatch
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from pycmg.model import Instance
    from pycmg.tech import TechConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_KEYS: List[str] = [
    "id", "ig", "is", "ie", "ids",
    "qg", "qd", "qs", "qb",
    "gm", "gds", "gmb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

GEOM_COLUMNS: List[str] = ["tech", "device", "L", "NFIN", "TFIN", "temp_K"]

VOLTAGE_COLUMNS: List[str] = ["Vg", "Vd", "Vs", "Ve", "Vth"]


def build_all_columns(process_keys: List[str]) -> List[str]:
    """Build the full ordered column list for sweep output.

    Column order: geometry -> sorted process vars -> voltages -> outputs.

    Args:
        process_keys: Process variation parameter names (e.g., ["eot", "toxp"]).

    Returns:
        Ordered list of column names.
    """
    return GEOM_COLUMNS + sorted(process_keys) + VOLTAGE_COLUMNS + OUTPUT_KEYS


# ---------------------------------------------------------------------------
# Voltage helpers
# ---------------------------------------------------------------------------


def build_nodes(
    vg_mag: float,
    vd_mag: float,
    ve_mag: float,
    vdd: float,
    device_type: str,
) -> Dict[str, float]:
    """Convert magnitude-space voltages to actual terminal voltages.

    In magnitude space, all voltages are positive and represent the
    *magnitude* of the bias relative to the source terminal. This function
    maps them to actual node voltages depending on device polarity.

    Args:
        vg_mag: Gate voltage magnitude (0 to Vdd).
        vd_mag: Drain voltage magnitude (0 to Vdd).
        ve_mag: Bulk/body voltage magnitude (typically 0).
        vdd: Supply voltage.
        device_type: ``"nmos"`` or ``"pmos"``.

    Returns:
        Dict with keys ``"g"``, ``"d"``, ``"s"``, ``"e"`` mapped to voltages.
    """
    if device_type == "nmos":
        return {"g": vg_mag, "d": vd_mag, "s": 0.0, "e": ve_mag}
    else:
        # PMOS: source at Vdd, drain/gate/bulk reflected
        return {
            "g": vdd - vg_mag,
            "d": vdd - vd_mag,
            "s": vdd,
            "e": vdd - ve_mag,
        }


def build_voltage_grid(
    vdd: float,
    vth_mag: float,
    vg_points: int = 50,
    vd_points: int = 50,
    dense_ratio: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a non-uniform Vg grid (dense near threshold) and uniform Vd grid.

    The Vg grid concentrates points around the threshold voltage to better
    capture the subthreshold-to-saturation transition, which is critical for
    neural network training accuracy.

    Args:
        vdd: Supply voltage (upper bound for both grids).
        vth_mag: Threshold voltage magnitude for dense region centering.
        vg_points: Total target number of Vg points.
        vd_points: Number of Vd points (uniform).
        dense_ratio: Fraction of vg_points allocated to the dense region.

    Returns:
        Tuple of ``(vg_array, vd_array)``, both sorted with no duplicates.
    """
    # Dense region: +/- 0.15*Vdd around Vth, clipped to [0, Vdd]
    dense_lo = max(0.0, vth_mag - 0.15 * vdd)
    dense_hi = min(vdd, vth_mag + 0.15 * vdd)

    n_dense = int(vg_points * dense_ratio)
    n_sparse = vg_points - n_dense

    # Dense points in the threshold region
    vg_dense = np.linspace(dense_lo, dense_hi, n_dense)

    # Sparse points across the full range
    vg_sparse = np.linspace(0.0, vdd, n_sparse)

    # Merge, sort, deduplicate
    vg_all = np.concatenate([vg_dense, vg_sparse])
    vg_all = np.sort(vg_all)
    vg_all = np.unique(vg_all)

    # Vd: uniform
    vd_all = np.linspace(0.0, vdd, vd_points)

    return vg_all, vd_all


# ---------------------------------------------------------------------------
# Threshold detection
# ---------------------------------------------------------------------------


def find_threshold(
    inst: Instance,
    vdd: float,
    device_type: str = "nmos",
    n_coarse: int = 30,
) -> float:
    """Find threshold voltage via peak-gm method (coarse sweep).

    Sweeps Vg_mag from 0 to Vdd at Vd = Vdd/2 in magnitude space,
    evaluates DC at each point, and returns the Vg_mag where |gm| is
    maximum. This is a standard approximation of Vth used in compact
    model characterization.

    Args:
        inst: Initialized PyCMG Instance.
        vdd: Supply voltage.
        device_type: ``"nmos"`` or ``"pmos"``.
        n_coarse: Number of uniform sweep points.

    Returns:
        Threshold voltage magnitude (positive float).
    """
    vg_sweep = np.linspace(0.0, vdd, n_coarse)
    vd_mag = vdd / 2.0

    best_vg = 0.0
    best_gm = 0.0

    for vg_mag in vg_sweep:
        nodes = build_nodes(vg_mag, vd_mag, 0.0, vdd, device_type)
        result = inst.eval_dc(nodes)
        gm_abs = abs(result["gm"])
        if gm_abs > best_gm:
            best_gm = gm_abs
            best_vg = float(vg_mag)

    return best_vg


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_devices(
    device_filter: Dict[str, List[str]] | None,
    tech_name: str,
    tech_config: TechConfig,
) -> List[str]:
    """Resolve which devices to sweep for a given technology.

    Supports explicit device names and glob patterns (e.g., ``"nmos_*"``).
    Missing devices are skipped with a warning rather than raising an error.

    Args:
        device_filter: Optional mapping from tech name to device name patterns.
            ``None`` means all devices in the technology.
        tech_name: Technology identifier (e.g., ``"ASAP7"``).
        tech_config: Technology configuration with device registry.

    Returns:
        Sorted list of resolved canonical device names.
    """
    available = sorted(tech_config.devices.keys())

    if device_filter is None:
        return available

    patterns = device_filter.get(tech_name)
    if patterns is None:
        return available

    resolved: list[str] = []
    for pattern in patterns:
        # Check if it's a glob pattern
        if any(c in pattern for c in ("*", "?", "[", "]")):
            matches = [d for d in available if fnmatch.fnmatch(d, pattern)]
            if not matches:
                warnings.warn(
                    f"Pattern '{pattern}' matched no devices in {tech_name}. "
                    f"Available: {', '.join(available)}",
                    stacklevel=2,
                )
            resolved.extend(matches)
        else:
            # Exact name
            if pattern in tech_config.devices:
                resolved.append(pattern)
            else:
                warnings.warn(
                    f"Device '{pattern}' not found in {tech_name}. "
                    f"Available: {', '.join(available)}",
                    stacklevel=2,
                )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in resolved:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Configuration for a full training-data sweep.

    Attributes:
        techs: Technology names to sweep (e.g., ``["ASAP7", "TSMC7"]``).
        devices: Optional per-tech device filter (supports glob patterns).
            ``None`` means all devices in each technology.
        l_multipliers: Gate length multipliers applied to min_l.
        nfins: Fin count values to sweep.
        temperatures: Operating temperatures in Kelvin.
        vg_points: Number of gate voltage grid points.
        vd_points: Number of drain voltage grid points.
        ve_values: Body bias voltage magnitudes to sweep.
        process_vars: Optional process variation parameters to sweep.
            Maps parameter name to list of values.
        dense_ratio: Fraction of vg_points in the threshold-dense region.
        n_coarse: Number of points for coarse threshold detection.
    """

    techs: List[str]
    devices: Dict[str, List[str]] | None = None
    l_multipliers: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    nfins: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    temperatures: List[float] = field(
        default_factory=lambda: [233.15, 273.15, 300.15, 358.15, 398.15]
    )
    vg_points: int = 50
    vd_points: int = 50
    ve_values: List[float] = field(default_factory=lambda: [0.0])
    process_vars: Dict[str, List[float]] | None = None
    dense_ratio: float = 0.6
    n_coarse: int = 30


@dataclass
class SweepResult:
    """Container for sweep output data.

    Attributes:
        columns: Ordered column names matching ``build_all_columns()`` output.
        data: List of rows, each row is a list of ``str | float`` values.
        metadata: Arbitrary metadata dict (e.g., sweep config, timing info).
    """

    columns: List[str]
    data: List[list]  # str | float
    metadata: Dict[str, object]
