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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sweep import NN_OUTPUT_COLUMNS
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


# ── Defaults ──────────────────────────────────────────────────────────────────

# D1: temperature sweep (paper §3 — full operating range, in Kelvin).
DEFAULT_TEMPERATURES_K: Tuple[float, ...] = (
    248.15,   # -25 °C
    300.15,   #  27 °C
    398.15,   # 125 °C
)

# D3: voltage box widening factor. paper uses 1.0 (i.e. [0, 1]·VDD).
# We default to 2.0 because the BSIMAR model is consumed inside an NR
# solver at LEVEL=74 and that solver routinely overshoots ±VDD before
# converging (CLAUDE.md NN Rule #3). [0, 2]·VDD covers one full VDD of
# overshoot headroom on each side and removes inference-time
# extrapolation garbage. This is an explicit override of the paper.
DEFAULT_VOLTAGE_BOX_FACTOR: float = 2.0

# D3: per-bin LHS sample budget. paper uses ~5K/bin for the [0, 1]·VDD
# range; doubled for [0, 2]·VDD to keep the same density.
DEFAULT_LHS_SAMPLES_PER_BIN: int = 5000


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


# ── Per-bin worker (D4 — picklable for multiprocessing) ──────────────────────

def generate_one_bin(spec: BinSpec) -> Optional[Dict[str, np.ndarray]]:
    """Generate samples for one (tech, variant, L, NFIN, T) bin.

    Top-level (not nested) so it can be dispatched to a
    multiprocessing.Pool. Returns ``None`` for unstable bins.
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
    failed = 0

    # Anchor points first (cheap, deterministic).
    for vg, vd, vbs in _anchor_points(spec.vdd, is_pmos):
        result = eval_single_point(inst, vd=vd, vg=vg, vs=0.0, vb=vbs)
        if result is None:
            failed += 1
            continue
        inputs.append(np.array([vd, vg, 0.0, vbs]))
        geometry.append(geo.copy())
        outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))

    # LHS samples in the wide voltage box.
    lhs = _sample_lhs_voltages(
        n_samples=spec.n_lhs_samples,
        vdd=spec.vdd,
        is_pmos=is_pmos,
        voltage_box_factor=spec.voltage_box_factor,
        seed=spec.seed,
    )
    for vg, vd, vbs in lhs:
        result = eval_single_point(inst, vd=float(vd), vg=float(vg),
                                   vs=0.0, vb=float(vbs))
        if result is None:
            failed += 1
            continue
        inputs.append(np.array([vd, vg, 0.0, vbs]))
        geometry.append(geo.copy())
        outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))

    if not inputs:
        return None

    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "geometry": np.asarray(geometry, dtype=np.float64),
        "outputs": np.asarray(outputs, dtype=np.float64),
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
                ))
                counter += 1
    return bins


# ── Top-level dataset assembly ───────────────────────────────────────────────

def _assemble(
    bin_results: List[Optional[Dict[str, np.ndarray]]],
    metadata: Dict,
    verbose: bool,
) -> Dict[str, np.ndarray]:
    inputs_list, geo_list, out_list = [], [], []
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

    if not inputs_list:
        raise RuntimeError(
            "No samples generated — every bin failed. Check the OSDI binary "
            "and modelcards."
        )

    inputs = np.concatenate(inputs_list, axis=0)
    geometry = np.concatenate(geo_list, axis=0)
    outputs = np.concatenate(out_list, axis=0)

    if verbose:
        print(f"\nDataset assembled: "
              f"{n_kept_total:,} kept, {n_failed_total:,} failed, "
              f"{n_bins_kept} bins kept, {n_bins_dropped} bins dropped")
        print(f"Shapes -- inputs: {inputs.shape}, geometry: {geometry.shape}, "
              f"outputs: {outputs.shape}")
        if geometry.size:
            t_unique = np.unique(geometry[:, 2])
            print(f"Temperatures (K): {t_unique.tolist()}")

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
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
    )

    if verbose:
        print(f"\n{tech.name} {device_type}: {len(bins)} bins "
              f"({len(bins) * n_lhs_samples:,} expected LHS samples + anchors) "
              f"[T sweep {len(temperatures)} pts, box={voltage_box_factor}·VDD]")

    results = _run_bins(bins, n_workers=n_workers, verbose=verbose)

    metadata = {
        "tech_name": tech.name,
        "device_type": device_type,
        "vdd": tech.vdd,
        "temperatures_k": np.array(temperatures, dtype=np.float64),
        "voltage_box_factor": voltage_box_factor,
        "output_columns": np.array(NN_OUTPUT_COLUMNS),
        "variants": np.array(variant_names or tech.variant_names),
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
) -> Dict[str, np.ndarray]:
    """Concatenate per-tech datasets across all 5 technologies and variants.

    Bins from every tech are flattened into a single bin list and run
    through one ``_run_bins`` call so the multiprocessing pool can keep
    every worker busy across tech boundaries.
    """
    all_bins: List[BinSpec] = []
    for _name, tech in TECH_CONFIGS.items():
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
        ))

    if verbose:
        print(f"\nUniversal {device_type}: total {len(all_bins)} bins, "
              f"~{len(all_bins) * n_lhs_samples:,} expected LHS samples")

    results = _run_bins(all_bins, n_workers=n_workers, verbose=verbose)

    metadata = {
        "tech_name": "universal",
        "device_type": device_type,
        "vdd": 0.0,
        "temperatures_k": np.array(temperatures, dtype=np.float64),
        "voltage_box_factor": voltage_box_factor,
        "output_columns": np.array(NN_OUTPUT_COLUMNS),
        "variants": np.array(list(TECH_CONFIGS.keys())),
    }
    return _assemble(results, metadata, verbose=verbose)
