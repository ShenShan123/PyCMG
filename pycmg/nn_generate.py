"""Generate NN training data (.npz) via PyCMG BSIM-CMG sweeps.

Replaces nn_model/data/generate.py. Phase D rewrite (BSIM-AR plan) adds:

- D1 Temperature sweep (default {-25, 27, 125} °C). T is now stored
  per-sample in column 2 of the geometry array.
- D2 Vbs is now a sampled input axis (was hardcoded to 0).
- D3 LHS over (Vgs, Vds, Vbs) at [0, 2]·VDD (NMOS) or [-2, 0]·VDD (PMOS),
  replacing the old 71×71 Vg×Vd grid. The wider box covers Newton-Raphson
  overshoot at LEVEL=74 (CLAUDE.md NN Rule #3). The paper trains on
  [0, 1]·VDD; we widen for the circuit-simulator use case.
- D6 NFIN=1 is no longer pre-filtered. ``_create_model_and_instance``
  now does a smoke-test eval at zero bias and skips bins where the
  initial OSDI solve diverges (e.g. tsmc5:ulvt NFIN=1).
- ``generate_one_bin`` is a top-level worker function that takes a fully
  serializable bin spec, so the parent script can dispatch bins across a
  multiprocessing Pool (D4 in scripts/generate_nn_data.py).

Geometry array layout (N, 15) — unchanged column count, T now varies:
    [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]

Inputs array layout (N, 4):
    [Vd, Vg, Vs, Vb] in source-relative frame (Vs always 0).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sweep import NN_OUTPUT_COLUMNS, find_threshold
from .model import Model, Instance
from .nn_config import (
    OSDI_PATH,
    NNTechConfig,
    ProcessParams,
    PROCESS_PARAM_NAMES,
    TECH_CONFIGS,
    OUTPUT_COLUMNS,
    extract_process_params,
)


# ── Sample-class codes (B1 metadata: tags every row's source) ────────────────
# Each row in the dataset gets one of these int8 codes so downstream
# consumers (B2 slope loss, B6 universal overlay) can subset rows by
# origin without re-running data generation.
SAMPLE_CLASS_NAMES: Tuple[str, ...] = (
    "anchor",     # 0  deterministic corner anchors
    "vds_zero",   # 1  Vds=0 boundary line (for Id(Vds=0)=0 enforcement)
    "subthresh",  # 2  subthreshold-to-transition region densification
    "small_vds",  # 3  small |Vds| linear-region densification
    "grid",       # 4  base hybrid uniform grid + jitter
    "hot",        # 5  hot-region densification (high-Vgs / high-Vds plateau)
    "lhs",        # 6  legacy LHS (only used when --sampler=lhs)
    # v5 plan §4 additions:
    "inv_trip",   # 7  inverter trip-point overlay (Vth-centered band)
    "overshoot",  # 8  NR-overshoot densification (|V| > VDD region)
    "vbs_lhs",    # 9  Vbs LHS jitter on the Vgs/Vds grid
)
SAMPLE_CLASS_CODES: Dict[str, int] = {
    name: i for i, name in enumerate(SAMPLE_CLASS_NAMES)
}


# ── Defaults ──────────────────────────────────────────────────────────────────

# D1: temperature sweep (paper §3 — full operating range, in Kelvin).
DEFAULT_TEMPERATURES_K: Tuple[float, ...] = (
    248.15,   # -25 °C
    300.15,   #  27 °C
    398.15,   # 125 °C
)

# D3: voltage box widening factor. paper uses 1.0 (i.e. [0, 1]·VDD).
# v5p (V5'): revert B2 box-factor change. Restore V4 B1 default 2.0.
# Phase A+B evidence (results/v5_v4_vs_phaseA_vs_phaseAB_2026_05_08.md)
# showed B2's 1.5 box + overshoot overlay regressed TSMC7/12/16 to
# NR-runaway 10^12 V. Reverting to V4 B1 box; the ``overshoot`` sample
# class is also disabled below (DEFAULT_OVERSHOOT_PER_AXIS=0).
DEFAULT_VOLTAGE_BOX_FACTOR: float = 2.0

# D3: per-bin LHS sample budget. paper uses ~5K/bin for the [0, 1]·VDD
# range; doubled for [0, 2]·VDD to keep the same density.
DEFAULT_LHS_SAMPLES_PER_BIN: int = 5000

# B1 (v5 plan §4-B1): hybrid uniform-grid sampler defaults.
# Replaces LHS for the bulk of the per-bin samples. The grid is
# strictly more uniform than LHS in the high-current corner that
# dominates the verifier metric (D1 finding: hot region holds 3.07 %
# of LHS samples but 16× the verifier-weighted error mass).
DEFAULT_GRID_PER_AXIS: int = 30        # 30 × 30 = 900 (Vgs, Vds) points
DEFAULT_VBS_LEVELS: int = 5            # {0, ±0.25, ±0.5}·VDD
DEFAULT_HOT_PER_AXIS: int = 12         # 12 × 12 hot-region densification
DEFAULT_JITTER_SIGMA_FRAC: float = 0.05  # σ = 0.05·VDD on each axis
DEFAULT_SAMPLER: str = "grid"          # "grid" | "lhs"

# v5p (V5'): all three Phase B overlays default off. inv_trip is
# turned back on per-bin for TSMC5 only via the gate inside
# generate_one_bin; B2 (overshoot) and B3 (vbs_lhs) stay off.
DEFAULT_INV_TRIP: bool = False
DEFAULT_OVERSHOOT_PER_AXIS: int = 0    # B2 off (caused TSMC7/12/16 regression)
DEFAULT_N_VBS_LHS: int = 0             # B3 off (caused TSMC7/12/16 regression)


# ── Bin spec (used by parallel worker) ───────────────────────────────────────

@dataclass
class BinSpec:
    """Fully serializable spec for one (tech, variant, L, NFIN, T) bin.

    A self-contained unit of work that can be sent across a
    multiprocessing.Pool boundary. The worker function rebuilds Model
    and Instance from this spec, evaluates n_samples bias points, and
    returns a partial dataset dict.
    """
    tech_name: str            # key into TECH_CONFIGS
    device_type: str          # "nmos" | "pmos"
    variant: str
    L: float
    NFIN: float
    temperature_k: float
    vdd: float
    n_lhs_samples: int
    voltage_box_factor: float
    seed: int                 # per-bin RNG seed for LHS reproducibility
    sampler: str = DEFAULT_SAMPLER  # "grid" | "lhs" (B1)
    grid_per_axis: int = DEFAULT_GRID_PER_AXIS
    vbs_levels: int = DEFAULT_VBS_LEVELS
    hot_per_axis: int = DEFAULT_HOT_PER_AXIS
    jitter_sigma_frac: float = DEFAULT_JITTER_SIGMA_FRAC
    # v5 plan §4 additions:
    enable_inv_trip: bool = DEFAULT_INV_TRIP
    overshoot_per_axis: int = DEFAULT_OVERSHOOT_PER_AXIS
    n_vbs_lhs: int = DEFAULT_N_VBS_LHS


# ── Model + instance with smoke test (D6) ────────────────────────────────────

def _create_model_and_instance(
    tech: NNTechConfig,
    device_type: str,
    variant: str,
    L: float,
    NFIN: float,
    temperature_k: float,
) -> Optional[Tuple[Model, Instance, ProcessParams]]:
    """Create a Model + Instance for a specific (L, NFIN, T) bin and
    extract process params.

    Each (L, NFIN) bin gets its own ``Model`` because ``model_overrides``
    writes to a shared ``OsdiModel`` buffer (CLAUDE.md "Instance / Model
    Isolation" constraint).

    D6: this function also performs an immediate smoke-test eval at zero
    bias. If the OSDI internal-node solve diverges (the typical NFIN=1
    failure mode for tsmc5:ulvt and tsmc16:lnvt), it returns ``None`` so
    the caller can skip the entire bin without spamming per-point
    warnings.

    Returns:
        (model, instance, proc) tuple, or ``None`` if the bin is unstable.
    """
    model_name = tech.get_model_name(device_type, variant)
    modelcard_path = tech.resolve_modelcard(device_type, variant, L, NFIN)

    try:
        model = Model(
            osdi_path=OSDI_PATH,
            modelcard_path=modelcard_path,
            model_name=model_name,
            model_card_name=model_name,
        )
    except Exception as exc:
        print(f"  SKIP {tech.name}:{variant} L={L*1e9:.1f}nm "
              f"NFIN={NFIN:.0f} T={temperature_k:.0f}K (model build): {exc}")
        return None

    proc = extract_process_params(model.modelcard_params)
    devtype = 1 if device_type == "nmos" else 0

    try:
        inst = Instance(
            model=model,
            params={"L": L, "NFIN": NFIN, "TFIN": tech.tfin, "DEVTYPE": devtype},
            temperature=temperature_k,
        )
    except Exception as exc:
        print(f"  SKIP {tech.name}:{variant} L={L*1e9:.1f}nm "
              f"NFIN={NFIN:.0f} T={temperature_k:.0f}K (instance build): {exc}")
        return None

    # D6 smoke test: one zero-bias eval. If this fails, the whole bin
    # is unstable for this tech/variant/L/NFIN — skip cleanly.
    smoke = eval_single_point(inst, 0.0, 0.0, 0.0, 0.0, _silent=True)
    if smoke is None:
        print(f"  SKIP {tech.name}:{variant} L={L*1e9:.1f}nm "
              f"NFIN={NFIN:.0f} T={temperature_k:.0f}K (smoke-test diverged)")
        return None

    return model, inst, proc


def eval_single_point(
    inst: Instance,
    vd: float,
    vg: float,
    vs: float = 0.0,
    vb: float = 0.0,
    *,
    _silent: bool = False,
) -> Optional[Dict[str, float]]:
    """Evaluate one bias point. Returns None on failure or non-physical result."""
    try:
        result = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": vb})
        out = {k: result[k] for k in NN_OUTPUT_COLUMNS}
        if any(math.isnan(v) or math.isinf(v) for v in out.values()):
            return None
        if abs(out["id"]) > 1.0:
            return None
        return out
    except Exception as exc:
        if not _silent:
            print(f"  WARNING: eval_dc failed at "
                  f"Vd={vd:.3f} Vg={vg:.3f} Vb={vb:.3f}: {exc}")
        return None


# ── Sampling (D2 + D3) ───────────────────────────────────────────────────────

def _sample_lhs_voltages(
    n_samples: int,
    vdd: float,
    is_pmos: bool,
    voltage_box_factor: float,
    seed: int,
) -> np.ndarray:
    """Latin Hypercube samples of (Vg, Vd, Vbs) over the training box.

    NMOS box:  [0, voltage_box_factor·VDD]^3
    PMOS box:  [-voltage_box_factor·VDD, 0]^3 (source-relative frame)

    Returns:
        (n_samples, 3) array of (vg, vd, vbs) bias points.
    """
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=n_samples)  # shape (n, 3) ∈ [0, 1]
    samples = samples * (voltage_box_factor * vdd)
    if is_pmos:
        samples = -samples
    return samples


def _vbs_levels(vdd: float, n_levels: int) -> np.ndarray:
    """Return ``n_levels`` symmetric Vbs levels in NMOS-positive convention.

    {0, ±0.25, ±0.5}·VDD, truncated/extended for ``n_levels`` other than 5.
    The caller mirrors through origin for PMOS via the ``is_pmos`` flag in
    :func:`_sample_hybrid_grid_voltages`.
    """
    base = [0.0, 0.25 * vdd, -0.25 * vdd, 0.5 * vdd, -0.5 * vdd]
    if n_levels >= len(base):
        # Pad symmetrically with 0.75/-0.75 if requested.
        extra = [0.75 * vdd, -0.75 * vdd]
        full = base + extra
        return np.array(full[:n_levels], dtype=np.float64)
    return np.array(base[:n_levels], dtype=np.float64)


def _sample_hybrid_grid_voltages(
    vdd: float,
    is_pmos: bool,
    voltage_box_factor: float,
    seed: int,
    *,
    n_grid_per_axis: int = DEFAULT_GRID_PER_AXIS,
    n_vbs_levels: int = DEFAULT_VBS_LEVELS,
    n_hot_per_axis: int = DEFAULT_HOT_PER_AXIS,
    jitter_sigma_frac: float = DEFAULT_JITTER_SIGMA_FRAC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid uniform-grid + jitter sampler with hot-region densification.

    Replaces LHS for the bulk per-bin samples (B1 of the v5 plan).

    Layout:

    - **Base grid** (``n_grid_per_axis × n_grid_per_axis × n_vbs_levels``):
      Uniform 2D grid in (Vgs, Vds) over ``[0, voltage_box_factor·VDD]``
      crossed with ``n_vbs_levels`` Vbs levels {0, ±0.25, ±0.5}·VDD,
      with N(0, ``jitter_sigma_frac``·VDD) Gaussian jitter on each
      axis. Defaults: 30 × 30 × 5 = 4500 samples.

    - **Hot densification** (``n_hot_per_axis²·n_vbs_levels``):
      A second uniform grid over the saturation plateau hot region —
      Vgs ∈ [0.5·VDD, VDD] × Vds ∈ [0.4·VDD, VDD] — at ≈ 1.6× the
      base grid density (per-bin verifier-weighted hot-region NRMSE
      lives here; D1 diagnostic). Defaults: 12 × 12 × 5 = 720 samples.

    Per-axis jitter clips back into the box so no negative-coord
    samples leak. PMOS mirrors the entire (Vgs, Vds, Vbs) point cloud
    through the origin (source-relative frame).

    Returns:
        samples: (N, 3) float64 array of (vg, vd, vbs) in NMOS-positive
                 convention pre-mirror; PMOS rows come back negated.
        sample_classes: (N,) int8 array with values
                        ``SAMPLE_CLASS_CODES['grid']`` for the base
                        grid rows and ``SAMPLE_CLASS_CODES['hot']``
                        for the hot rows.
    """
    rng = np.random.default_rng(seed)
    box_max_pos = voltage_box_factor * vdd
    sigma = jitter_sigma_frac * vdd
    vbs_levels = _vbs_levels(vdd, n_vbs_levels)

    # Base uniform grid over [0, box_max_pos]^2 (Vgs, Vds) × Vbs levels.
    vg_grid = np.linspace(0.0, box_max_pos, n_grid_per_axis)
    vd_grid = np.linspace(0.0, box_max_pos, n_grid_per_axis)
    G, D, B = np.meshgrid(vg_grid, vd_grid, vbs_levels, indexing="ij")
    base = np.stack([G.ravel(), D.ravel(), B.ravel()], axis=1)
    base_jit = rng.normal(0.0, sigma, size=base.shape)
    base = base + base_jit
    # Clip Vgs, Vds to the [0, box_max_pos] training box; allow Vbs to
    # exceed its discrete level set (jitter dispersion is desirable
    # there too) but cap at the same box for safety.
    base[:, 0] = np.clip(base[:, 0], 0.0, box_max_pos)
    base[:, 1] = np.clip(base[:, 1], 0.0, box_max_pos)
    base[:, 2] = np.clip(base[:, 2], -box_max_pos, box_max_pos)
    base_classes = np.full(base.shape[0],
                           SAMPLE_CLASS_CODES["grid"], dtype=np.int8)

    # Hot-region densification: doubled grid density on the saturation
    # plateau (Vgs ∈ [0.5,1]·VDD, Vds ∈ [0.4,1]·VDD).
    if n_hot_per_axis > 0:
        vg_hot = np.linspace(0.5 * vdd, vdd, n_hot_per_axis)
        vd_hot = np.linspace(0.4 * vdd, vdd, n_hot_per_axis)
        GH, DH, BH = np.meshgrid(vg_hot, vd_hot, vbs_levels, indexing="ij")
        hot = np.stack([GH.ravel(), DH.ravel(), BH.ravel()], axis=1)
        # Use a tighter jitter inside the hot region so points stay
        # inside; sigma half of base.
        hot_jit = rng.normal(0.0, 0.5 * sigma, size=hot.shape)
        hot = hot + hot_jit
        hot[:, 0] = np.clip(hot[:, 0], 0.0, box_max_pos)
        hot[:, 1] = np.clip(hot[:, 1], 0.0, box_max_pos)
        hot[:, 2] = np.clip(hot[:, 2], -box_max_pos, box_max_pos)
        hot_classes = np.full(hot.shape[0],
                              SAMPLE_CLASS_CODES["hot"], dtype=np.int8)
        samples = np.concatenate([base, hot], axis=0)
        classes = np.concatenate([base_classes, hot_classes], axis=0)
    else:
        samples = base
        classes = base_classes

    if is_pmos:
        samples = -samples

    return samples, classes


def _anchor_points(
    vdd: float,
    is_pmos: bool,
) -> List[Tuple[float, float, float]]:
    """Deterministic anchor bias points appended to every (L, NFIN, T) bin.

    The list is intentionally small — LHS already covers the box uniformly,
    so anchors only need to nail the cutoff corners and the
    operating-point center where the model is most sensitive.
    """
    s = -1.0 if is_pmos else 1.0
    half = s * 0.5 * vdd
    rail = s * vdd

    return [
        (0.0,  0.0,  0.0),    # zero bias (deep cutoff)
        (rail, 0.0,  0.0),    # off, max Vds
        (0.0,  rail, 0.0),    # max Vgs, Vds=0
        (rail, rail, 0.0),    # max Vgs, max Vds (saturation rail)
        (half, half, 0.0),    # mid-supply
        (rail, half, 0.0),    # high Vgs, mid Vds
        (half, rail, 0.0),    # mid Vgs, high Vds
        (rail, rail, half),   # forward body bias near rail
        (half, half, rail),   # mid op-pt with full body bias
    ]


def _vds_zero_line_points(
    vdd: float,
    is_pmos: bool,
) -> List[Tuple[float, float, float]]:
    """Dense samples along Vds=0 to enforce the Id(Vds=0)=0 boundary.

    Returns (vg, vd=0, vbs) tuples spanning the full Vg range at Vds=0.
    60 points per bin (20 Vg x 3 Vbs).
    """
    s = -1.0 if is_pmos else 1.0
    vg_steps = np.linspace(0, s * 2.0 * vdd, 20)
    vbs_steps = [0.0, s * 0.25 * vdd, s * 0.5 * vdd]
    return [(float(vg), 0.0, float(vbs)) for vg in vg_steps for vbs in vbs_steps]


def _subthreshold_transition_points(
    vdd: float,
    is_pmos: bool,
) -> List[Tuple[float, float, float]]:
    """Dense sweep in the subthreshold-to-transition Vgs region.

    Covers Vgs from 0 to 60 %% VDD crossed with Vds from 0 to VDD.
    This is where the inverter VTC slope is steepest and sign accuracy
    matters most.  300 points per bin (30 Vg x 10 Vd).
    """
    s = -1.0 if is_pmos else 1.0
    vg_steps = np.linspace(0, s * 0.6 * vdd, 30)
    vd_steps = np.linspace(0, s * vdd, 10)
    return [(float(vg), float(vd), 0.0) for vg in vg_steps for vd in vd_steps]


def _small_vds_points(
    vdd: float,
    is_pmos: bool,
) -> List[Tuple[float, float, float]]:
    """Dense samples at small |Vds| where the analytical Vds correction is active.

    Improves intrinsic NN accuracy in the linear/triode region so the
    inference-time correction has less work to do.
    120 points per bin (8 Vds x 15 Vg).
    """
    s = -1.0 if is_pmos else 1.0
    vds_vals = [s * v for v in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]]
    vg_steps = np.linspace(0, s * vdd, 15)
    return [(float(vg), float(vd), 0.0) for vd in vds_vals for vg in vg_steps]


def _inv_trip_points(
    vth_mag: float,
    vdd: float,
    is_pmos: bool,
) -> List[Tuple[float, float, float]]:
    """v5 plan §4-B1 inverter trip-point overlay.

    A 25 × 9 × 3 (Vg × Vd × Vbs) grid centred on Vth (signed for the
    polarity) covering the inverter switching band:

        Vg  in [Vth − 0.10, Vth + 0.15]   (in NMOS-positive convention,
                                           or Vth in PMOS-negative)
        Vd  in [0.30·VDD, 0.70·VDD]       (signed)
        Vbs in {0, +0.25·VDD, -0.25·VDD}  (signed)

    675 samples per bin. Tagged with sample_class="inv_trip".
    """
    s = -1.0 if is_pmos else 1.0
    vth_signed = s * abs(vth_mag)
    vg_lo = vth_signed - s * 0.10
    vg_hi = vth_signed + s * 0.15
    vg_steps = np.linspace(vg_lo, vg_hi, 25)
    vd_steps = np.linspace(s * 0.30 * vdd, s * 0.70 * vdd, 9)
    vbs_levels = [0.0, s * 0.25 * vdd, -s * 0.25 * vdd]
    return [
        (float(vg), float(vd), float(vbs))
        for vg in vg_steps
        for vd in vd_steps
        for vbs in vbs_levels
    ]


def _sample_overshoot_voltages(
    vdd: float,
    is_pmos: bool,
    *,
    n_per_axis: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """v5 plan §4-B2 NR-overshoot densification.

    Dense uniform grid in ``(Vgs, Vds) ∈ [VDD, 1.6·VDD]^2`` with
    ``Vbs = 0``. ``n_per_axis × n_per_axis`` samples (default 400).
    Smooth-joins the in-distribution box with the Phase A tanh
    rail-restoring inference glue.

    PMOS mirrors through the origin via the standard ``is_pmos`` flag.

    Returns:
        samples: (N, 3) array of (vg, vd, vbs).
        classes: (N,) int8 array tagged ``overshoot``.
    """
    vg_grid = np.linspace(vdd, 1.6 * vdd, n_per_axis)
    vd_grid = np.linspace(vdd, 1.6 * vdd, n_per_axis)
    G, D = np.meshgrid(vg_grid, vd_grid, indexing="ij")
    samples = np.stack(
        [G.ravel(), D.ravel(), np.zeros(G.size, dtype=np.float64)], axis=1
    )
    if is_pmos:
        samples = -samples
    classes = np.full(samples.shape[0],
                      SAMPLE_CLASS_CODES["overshoot"], dtype=np.int8)
    return samples, classes


def _sample_vbs_lhs_voltages(
    vdd: float,
    is_pmos: bool,
    seed: int,
    *,
    n_samples: int = 600,
    grid_per_axis: int = DEFAULT_GRID_PER_AXIS,
    voltage_box_factor: float = DEFAULT_VOLTAGE_BOX_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """v5 plan §4-B3 LHS Vbs jitter on the existing (Vgs, Vds) grid.

    Holds (Vgs, Vds) on the same uniform grid as the bulk sampler and
    jitters ``Vbs ~ U(-0.5·VDD, +0.5·VDD)`` once per (Vg, Vd) point.
    ``n_samples`` (Vg, Vd) points are drawn uniformly with replacement
    from the lattice (the grid has ``grid_per_axis^2`` cells). Targets
    the on-grid Vbs overfit observed in the SML §3.3 TSMC7 NMOS DC
    L-vs-S inversion.

    Returns:
        samples: (n_samples, 3) array of (vg, vd, vbs).
        classes: (n_samples,) int8 array tagged ``vbs_lhs``.
    """
    rng = np.random.default_rng(seed)
    box_max_pos = voltage_box_factor * vdd

    # Uniform random selection from the (Vgs, Vds) lattice.
    vg_grid = np.linspace(0.0, box_max_pos, grid_per_axis)
    vd_grid = np.linspace(0.0, box_max_pos, grid_per_axis)
    vg_idx = rng.integers(0, grid_per_axis, size=n_samples)
    vd_idx = rng.integers(0, grid_per_axis, size=n_samples)
    vg = vg_grid[vg_idx]
    vd = vd_grid[vd_idx]

    # LHS-style stratified Vbs in [-0.5, +0.5]·VDD.
    from scipy.stats.qmc import LatinHypercube
    lhs = LatinHypercube(d=1, seed=seed).random(n=n_samples).ravel()
    vbs = (lhs - 0.5) * vdd  # uniform on [-0.5*VDD, +0.5*VDD]

    samples = np.stack([vg, vd, vbs], axis=1)
    if is_pmos:
        samples = -samples
    classes = np.full(samples.shape[0],
                      SAMPLE_CLASS_CODES["vbs_lhs"], dtype=np.int8)
    return samples, classes


# ── Per-bin worker (D4 — picklable for multiprocessing) ──────────────────────

def generate_one_bin(spec: BinSpec) -> Optional[Dict[str, np.ndarray]]:
    """Generate samples for one (tech, variant, L, NFIN, T) bin.

    Top-level (not nested) so it can be dispatched to a
    multiprocessing.Pool. Returns ``None`` for unstable bins.

    The bulk-sample block is selected by ``spec.sampler``:
        - ``"grid"`` (default, B1): hybrid uniform grid + jitter + hot
          densification (``_sample_hybrid_grid_voltages``).
        - ``"lhs"`` (legacy): Latin Hypercube (``_sample_lhs_voltages``)
          with ``spec.n_lhs_samples`` points.

    Every kept row is tagged with one of ``SAMPLE_CLASS_CODES`` so
    downstream loss/data-aug code can subset rows by origin without
    re-running data generation.
    """
    tech = TECH_CONFIGS[spec.tech_name]
    is_pmos = spec.device_type == "pmos"

    built = _create_model_and_instance(
        tech, spec.device_type, spec.variant,
        spec.L, spec.NFIN, spec.temperature_k,
    )
    if built is None:
        return None
    _model, inst, proc = built

    geo = np.array(
        [spec.NFIN, spec.L, spec.temperature_k] + proc.as_array(),
        dtype=np.float64,
    )

    inputs: List[np.ndarray] = []
    geometry: List[np.ndarray] = []
    outputs: List[np.ndarray] = []
    classes: List[int] = []
    failed = 0

    def _eval_and_keep(vg: float, vd: float, vbs: float, klass: int) -> bool:
        nonlocal failed
        result = eval_single_point(inst, vd=float(vd), vg=float(vg),
                                   vs=0.0, vb=float(vbs))
        if result is None:
            failed += 1
            return False
        inputs.append(np.array([vd, vg, 0.0, vbs]))
        geometry.append(geo.copy())
        outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))
        classes.append(klass)
        return True

    # Anchor points first (cheap, deterministic).
    cls_anchor = SAMPLE_CLASS_CODES["anchor"]
    for vg, vd, vbs in _anchor_points(spec.vdd, is_pmos):
        _eval_and_keep(vg, vd, vbs, cls_anchor)

    # Dense targeted points: each region has its own sample-class code so
    # downstream filters can keep / drop them independently.
    for _gen_fn, _klass in (
        (_vds_zero_line_points, SAMPLE_CLASS_CODES["vds_zero"]),
        (_subthreshold_transition_points, SAMPLE_CLASS_CODES["subthresh"]),
        (_small_vds_points, SAMPLE_CLASS_CODES["small_vds"]),
    ):
        for vg, vd, vbs in _gen_fn(spec.vdd, is_pmos):
            _eval_and_keep(vg, vd, vbs, _klass)

    # v5 plan §4-B1: inverter trip-point overlay. Vth_n / Vth_p is
    # determined per-bin via the peak-gm coarse sweep (find_threshold)
    # so the band tracks the actual modelcard variant.
    # v5p (V5'): gated to TSMC5 only — the single tech where the overlay
    # had proven leverage (DN inv-tran 16.90 % → 0.92 % PASS).
    if spec.enable_inv_trip and spec.tech_name == "tsmc5":
        try:
            vth_mag = find_threshold(inst, spec.vdd,
                                     device_type=spec.device_type)
        except Exception as exc:
            print(f"  WARNING: find_threshold failed for "
                  f"{spec.tech_name}:{spec.variant} L={spec.L*1e9:.1f}nm "
                  f"NFIN={spec.NFIN:.0f}: {exc!r}; skipping inv_trip overlay")
            vth_mag = None
        if vth_mag is not None:
            cls_inv = SAMPLE_CLASS_CODES["inv_trip"]
            for vg, vd, vbs in _inv_trip_points(vth_mag, spec.vdd, is_pmos):
                _eval_and_keep(vg, vd, vbs, cls_inv)

    # v5 plan §4-B2: NR-overshoot densification — dense grid past the
    # 1.5·VDD bulk box, in [VDD, 1.6·VDD]^2 with Vbs=0.
    if spec.overshoot_per_axis > 0:
        os_xyz, os_classes = _sample_overshoot_voltages(
            vdd=spec.vdd,
            is_pmos=is_pmos,
            n_per_axis=spec.overshoot_per_axis,
        )
        for (vg, vd, vbs), klass in zip(os_xyz, os_classes):
            _eval_and_keep(vg, vd, vbs, int(klass))

    # v5 plan §4-B3: Vbs LHS jitter — 600 LHS samples / bin.
    if spec.n_vbs_lhs > 0:
        vl_xyz, vl_classes = _sample_vbs_lhs_voltages(
            vdd=spec.vdd,
            is_pmos=is_pmos,
            seed=spec.seed + 1009,    # decorrelate from bulk grid jitter
            n_samples=spec.n_vbs_lhs,
            grid_per_axis=spec.grid_per_axis,
            voltage_box_factor=spec.voltage_box_factor,
        )
        for (vg, vd, vbs), klass in zip(vl_xyz, vl_classes):
            _eval_and_keep(vg, vd, vbs, int(klass))

    # Bulk samples — dispatch on sampler.
    if spec.sampler == "grid":
        bulk_xyz, bulk_classes = _sample_hybrid_grid_voltages(
            vdd=spec.vdd,
            is_pmos=is_pmos,
            voltage_box_factor=spec.voltage_box_factor,
            seed=spec.seed,
            n_grid_per_axis=spec.grid_per_axis,
            n_vbs_levels=spec.vbs_levels,
            n_hot_per_axis=spec.hot_per_axis,
            jitter_sigma_frac=spec.jitter_sigma_frac,
        )
        for (vg, vd, vbs), klass in zip(bulk_xyz, bulk_classes):
            _eval_and_keep(vg, vd, vbs, int(klass))
    elif spec.sampler == "lhs":
        lhs = _sample_lhs_voltages(
            n_samples=spec.n_lhs_samples,
            vdd=spec.vdd,
            is_pmos=is_pmos,
            voltage_box_factor=spec.voltage_box_factor,
            seed=spec.seed,
        )
        cls_lhs = SAMPLE_CLASS_CODES["lhs"]
        for vg, vd, vbs in lhs:
            _eval_and_keep(vg, vd, vbs, cls_lhs)
    else:
        raise ValueError(
            f"Unknown sampler {spec.sampler!r} (expected 'grid' or 'lhs')"
        )

    if not inputs:
        return None

    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "geometry": np.asarray(geometry, dtype=np.float64),
        "outputs": np.asarray(outputs, dtype=np.float64),
        "sample_class": np.asarray(classes, dtype=np.int8),
        "n_kept": len(inputs),
        "n_failed": failed,
    }


# ── Bin enumeration ──────────────────────────────────────────────────────────

def enumerate_bins(
    tech: NNTechConfig,
    device_type: str,
    *,
    variant_names: Optional[List[str]] = None,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES_K,
    n_lhs_samples: int = DEFAULT_LHS_SAMPLES_PER_BIN,
    voltage_box_factor: float = DEFAULT_VOLTAGE_BOX_FACTOR,
    base_seed: int = 42,
    sampler: str = DEFAULT_SAMPLER,
    grid_per_axis: int = DEFAULT_GRID_PER_AXIS,
    vbs_levels: int = DEFAULT_VBS_LEVELS,
    hot_per_axis: int = DEFAULT_HOT_PER_AXIS,
    jitter_sigma_frac: float = DEFAULT_JITTER_SIGMA_FRAC,
    enable_inv_trip: bool = DEFAULT_INV_TRIP,
    overshoot_per_axis: int = DEFAULT_OVERSHOOT_PER_AXIS,
    n_vbs_lhs: int = DEFAULT_N_VBS_LHS,
) -> List[BinSpec]:
    """Enumerate every (variant, L, NFIN, T) bin spec for a tech/polarity."""
    variants = variant_names or tech.variant_names
    if not variants:
        raise ValueError(
            f"No variants for {tech.name}. Available: {tech.variant_names}"
        )

    bins: List[BinSpec] = []
    counter = 0
    for variant in variants:
        for L, NFIN in tech.get_geometry_combos(device_type, variant):
            for T in temperatures:
                bins.append(BinSpec(
                    tech_name=tech.name.lower(),
                    device_type=device_type,
                    variant=variant,
                    L=float(L),
                    NFIN=float(NFIN),
                    temperature_k=float(T),
                    vdd=tech.vdd,
                    n_lhs_samples=n_lhs_samples,
                    voltage_box_factor=voltage_box_factor,
                    # Stable per-bin seed: deterministic across runs.
                    seed=base_seed + counter,
                    sampler=sampler,
                    grid_per_axis=grid_per_axis,
                    vbs_levels=vbs_levels,
                    hot_per_axis=hot_per_axis,
                    jitter_sigma_frac=jitter_sigma_frac,
                    enable_inv_trip=enable_inv_trip,
                    overshoot_per_axis=overshoot_per_axis,
                    n_vbs_lhs=n_vbs_lhs,
                ))
                counter += 1
    return bins


# ── Top-level dataset assembly ───────────────────────────────────────────────

def _assemble(
    bin_results: List[Optional[Dict[str, np.ndarray]]],
    metadata: Dict,
    verbose: bool,
) -> Dict[str, np.ndarray]:
    inputs_list, geo_list, out_list, cls_list = [], [], [], []
    n_kept_total = 0
    n_failed_total = 0
    n_bins_kept = 0
    n_bins_dropped = 0

    for r in bin_results:
        if r is None:
            n_bins_dropped += 1
            continue
        n_bins_kept += 1
        n_kept_total += int(r["n_kept"])
        n_failed_total += int(r["n_failed"])
        inputs_list.append(r["inputs"])
        geo_list.append(r["geometry"])
        out_list.append(r["outputs"])
        # Older bin results (pre-B1) don't carry sample_class; tag them
        # all as "lhs" so the assembled array still has length N.
        if "sample_class" in r:
            cls_list.append(r["sample_class"])
        else:
            cls_list.append(
                np.full(int(r["n_kept"]),
                        SAMPLE_CLASS_CODES["lhs"], dtype=np.int8)
            )

    if not inputs_list:
        raise RuntimeError(
            "No samples generated — every bin failed. Check the OSDI binary "
            "and modelcards."
        )

    inputs = np.concatenate(inputs_list, axis=0)
    geometry = np.concatenate(geo_list, axis=0)
    outputs = np.concatenate(out_list, axis=0)
    sample_class = np.concatenate(cls_list, axis=0).astype(np.int8)

    if verbose:
        print(f"\nDataset assembled: "
              f"{n_kept_total:,} kept, {n_failed_total:,} failed, "
              f"{n_bins_kept} bins kept, {n_bins_dropped} bins dropped")
        print(f"Shapes -- inputs: {inputs.shape}, geometry: {geometry.shape}, "
              f"outputs: {outputs.shape}, sample_class: {sample_class.shape}")
        if geometry.size:
            t_unique = np.unique(geometry[:, 2])
            print(f"Temperatures (K): {t_unique.tolist()}")
        # Per-class breakdown for visibility.
        for name, code in SAMPLE_CLASS_CODES.items():
            n_cls = int(np.sum(sample_class == code))
            if n_cls:
                print(f"  sample_class[{name}]={code}: {n_cls:,}")

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
        "sample_class": sample_class,
        "metadata": metadata,
    }


def _run_bins(
    bins: List[BinSpec],
    n_workers: int,
    verbose: bool,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """Run all bins, optionally in parallel via multiprocessing.Pool."""
    if n_workers <= 1:
        results: List[Optional[Dict[str, np.ndarray]]] = []
        t0 = time.time()
        for i, spec in enumerate(bins, 1):
            r = generate_one_bin(spec)
            results.append(r)
            if verbose and (i % 5 == 0 or i == len(bins)):
                kept = r["n_kept"] if r is not None else 0
                print(f"  [{i}/{len(bins)}] {spec.tech_name}:{spec.variant} "
                      f"L={spec.L*1e9:.1f}nm NFIN={spec.NFIN:.0f} "
                      f"T={spec.temperature_k:.0f}K -> "
                      f"{kept} pts  (elapsed {time.time()-t0:.0f}s)")
        return results

    # Parallel path. The OSDI library is loaded per worker on first
    # call, so each worker pays a one-time ~2 s init cost; this is
    # negligible vs the per-bin eval cost.
    from multiprocessing import get_context
    ctx = get_context("spawn")
    if verbose:
        print(f"Dispatching {len(bins)} bins across {n_workers} workers")
    t0 = time.time()
    results: List[Optional[Dict[str, np.ndarray]]] = [None] * len(bins)
    with ctx.Pool(processes=n_workers) as pool:
        for i, (idx, r) in enumerate(
            zip(range(len(bins)), pool.imap(generate_one_bin, bins)), 1
        ):
            results[idx] = r
            if verbose and (i % max(n_workers, 1) == 0 or i == len(bins)):
                kept = r["n_kept"] if r is not None else 0
                spec = bins[idx]
                print(f"  [{i}/{len(bins)}] {spec.tech_name}:{spec.variant} "
                      f"L={spec.L*1e9:.1f}nm NFIN={spec.NFIN:.0f} "
                      f"T={spec.temperature_k:.0f}K -> "
                      f"{kept} pts  (elapsed {time.time()-t0:.0f}s)")
    return results


# ── Public API ───────────────────────────────────────────────────────────────

def generate_dataset(
    tech: NNTechConfig,
    device_type: str,
    *,
    variant_names: Optional[List[str]] = None,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES_K,
    n_lhs_samples: int = DEFAULT_LHS_SAMPLES_PER_BIN,
    voltage_box_factor: float = DEFAULT_VOLTAGE_BOX_FACTOR,
    n_workers: int = 1,
    seed: int = 42,
    verbose: bool = True,
    sampler: str = DEFAULT_SAMPLER,
    grid_per_axis: int = DEFAULT_GRID_PER_AXIS,
    vbs_levels: int = DEFAULT_VBS_LEVELS,
    hot_per_axis: int = DEFAULT_HOT_PER_AXIS,
    jitter_sigma_frac: float = DEFAULT_JITTER_SIGMA_FRAC,
    enable_inv_trip: bool = DEFAULT_INV_TRIP,
    overshoot_per_axis: int = DEFAULT_OVERSHOOT_PER_AXIS,
    n_vbs_lhs: int = DEFAULT_N_VBS_LHS,
) -> Dict[str, np.ndarray]:
    """Generate training data for one tech/polarity across all bins.

    Args:
        tech: Technology configuration (vdd, variant names, fallback L/NFIN).
        device_type: ``"nmos"`` or ``"pmos"``.
        variant_names: Subset of variants; ``None`` = all.
        temperatures: Temperatures in Kelvin (D1 default = paper sweep).
        n_lhs_samples: LHS samples per (variant, L, NFIN, T) bin.
        voltage_box_factor: Multiplier on VDD for the (Vg, Vd, Vbs) box (D3).
        n_workers: ``1`` = serial; ``>1`` = ``multiprocessing.Pool`` (D4).
        seed: Base RNG seed; per-bin seeds derive from this monotonically.
        verbose: Per-bin progress prints.

    Returns:
        Dict with ``inputs`` (N,4), ``geometry`` (N,15), ``outputs`` (N,13),
        and ``metadata``.
    """
    if device_type not in {"nmos", "pmos"}:
        raise ValueError(f"device_type must be nmos|pmos, got {device_type!r}")

    bins = enumerate_bins(
        tech, device_type,
        variant_names=variant_names,
        temperatures=temperatures,
        n_lhs_samples=n_lhs_samples,
        voltage_box_factor=voltage_box_factor,
        base_seed=seed,
        sampler=sampler,
        grid_per_axis=grid_per_axis,
        vbs_levels=vbs_levels,
        hot_per_axis=hot_per_axis,
        jitter_sigma_frac=jitter_sigma_frac,
        enable_inv_trip=enable_inv_trip,
        overshoot_per_axis=overshoot_per_axis,
        n_vbs_lhs=n_vbs_lhs,
    )

    if verbose:
        if sampler == "grid":
            n_bulk = (grid_per_axis * grid_per_axis * vbs_levels
                      + hot_per_axis * hot_per_axis * vbs_levels)
            print(f"\n{tech.name} {device_type}: {len(bins)} bins "
                  f"(sampler=grid, {len(bins) * n_bulk:,} bulk "
                  f"+ ~{len(bins) * 489} targeted) "
                  f"[T sweep {len(temperatures)} pts, "
                  f"box={voltage_box_factor}·VDD]")
        else:
            print(f"\n{tech.name} {device_type}: {len(bins)} bins "
                  f"(sampler=lhs, {len(bins) * n_lhs_samples:,} LHS "
                  f"+ ~{len(bins) * 489} targeted) "
                  f"[T sweep {len(temperatures)} pts, "
                  f"box={voltage_box_factor}·VDD]")

    results = _run_bins(bins, n_workers=n_workers, verbose=verbose)

    metadata = {
        "tech_name": tech.name,
        "device_type": device_type,
        "vdd": tech.vdd,
        "temperatures_k": np.array(temperatures, dtype=np.float64),
        "voltage_box_factor": voltage_box_factor,
        "output_columns": np.array(NN_OUTPUT_COLUMNS),
        "variants": np.array(variant_names or tech.variant_names),
        "sampler": sampler,
        "grid_per_axis": grid_per_axis,
        "vbs_levels": vbs_levels,
        "hot_per_axis": hot_per_axis,
        "jitter_sigma_frac": jitter_sigma_frac,
        "sample_class_names": np.array(SAMPLE_CLASS_NAMES),
        "enable_inv_trip": np.bool_(enable_inv_trip),
        "overshoot_per_axis": int(overshoot_per_axis),
        "n_vbs_lhs": int(n_vbs_lhs),
    }
    return _assemble(results, metadata, verbose=verbose)


def generate_universal_dataset(
    device_type: str,
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES_K,
    n_lhs_samples: int = DEFAULT_LHS_SAMPLES_PER_BIN,
    voltage_box_factor: float = DEFAULT_VOLTAGE_BOX_FACTOR,
    n_workers: int = 1,
    seed: int = 42,
    verbose: bool = True,
    sampler: str = DEFAULT_SAMPLER,
    grid_per_axis: int = DEFAULT_GRID_PER_AXIS,
    vbs_levels: int = DEFAULT_VBS_LEVELS,
    hot_per_axis: int = DEFAULT_HOT_PER_AXIS,
    jitter_sigma_frac: float = DEFAULT_JITTER_SIGMA_FRAC,
    enable_inv_trip: bool = DEFAULT_INV_TRIP,
    overshoot_per_axis: int = DEFAULT_OVERSHOOT_PER_AXIS,
    n_vbs_lhs: int = DEFAULT_N_VBS_LHS,
    exclude_techs: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Concatenate per-tech datasets across all 5 technologies and variants.

    Bins from every tech are flattened into a single bin list and run
    through one ``_run_bins`` call so the multiprocessing pool can keep
    every worker busy across tech boundaries.
    """
    excl = {t.lower() for t in (exclude_techs or [])}
    all_bins: List[BinSpec] = []
    for _name, tech in TECH_CONFIGS.items():
        if _name.lower() in excl:
            if verbose:
                print(f"\n[skip] {tech.name.upper()} excluded by --exclude-techs")
            continue
        if verbose:
            print(f"\n{'='*60}\n  {tech.name.upper()} -- "
                  f"{len(tech.variant_names)} variants, VDD={tech.vdd}V"
                  f"\n{'='*60}")
        all_bins.extend(enumerate_bins(
            tech, device_type,
            temperatures=temperatures,
            n_lhs_samples=n_lhs_samples,
            voltage_box_factor=voltage_box_factor,
            base_seed=seed + len(all_bins),
            sampler=sampler,
            grid_per_axis=grid_per_axis,
            vbs_levels=vbs_levels,
            hot_per_axis=hot_per_axis,
            jitter_sigma_frac=jitter_sigma_frac,
            enable_inv_trip=enable_inv_trip,
            overshoot_per_axis=overshoot_per_axis,
            n_vbs_lhs=n_vbs_lhs,
        ))

    if verbose:
        if sampler == "grid":
            n_bulk = (grid_per_axis * grid_per_axis * vbs_levels
                      + hot_per_axis * hot_per_axis * vbs_levels)
            print(f"\nUniversal {device_type}: total {len(all_bins)} bins, "
                  f"sampler=grid, ~{len(all_bins) * n_bulk:,} bulk samples")
        else:
            print(f"\nUniversal {device_type}: total {len(all_bins)} bins, "
                  f"sampler=lhs, ~{len(all_bins) * n_lhs_samples:,} LHS samples")

    results = _run_bins(all_bins, n_workers=n_workers, verbose=verbose)

    included_techs = [n for n in TECH_CONFIGS.keys() if n.lower() not in excl]
    metadata = {
        "tech_name": "universal",
        "device_type": device_type,
        "vdd": 0.0,
        "temperatures_k": np.array(temperatures, dtype=np.float64),
        "voltage_box_factor": voltage_box_factor,
        "output_columns": np.array(NN_OUTPUT_COLUMNS),
        "variants": np.array(included_techs),
        "sampler": sampler,
        "grid_per_axis": grid_per_axis,
        "vbs_levels": vbs_levels,
        "hot_per_axis": hot_per_axis,
        "jitter_sigma_frac": jitter_sigma_frac,
        "sample_class_names": np.array(SAMPLE_CLASS_NAMES),
        "enable_inv_trip": np.bool_(enable_inv_trip),
        "overshoot_per_axis": int(overshoot_per_axis),
        "n_vbs_lhs": int(n_vbs_lhs),
        "excluded_techs": np.array(sorted(excl)),
    }
    return _assemble(results, metadata, verbose=verbose)
