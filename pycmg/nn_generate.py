"""Generate NN training data (.npz) via PyCMG BSIM-CMG sweeps.

Replaces nn_model/data/generate.py. Uses sweep.py primitives for threshold
detection and voltage grid construction.

Key design: enumerates PDK-legal (L, NFIN) combos per variant and extracts
process parameters on-the-fly from the resolved modelcard for each bin.

Dataset coverage:
  Geometric:          (L, NFIN) from PDK bin boundaries (TSMC) or fallback list (ASAP7)
  Operating cond.:    Vd, Vg, Vs=0, Vb=0 (source-relative); T = tech.temperature
  Process params:     12 params per (L, NFIN) bin, extracted from modelcard on-the-fly
  Voltage range:      NMOS Vg/Vd in [-VDD, 2*VDD]; PMOS in [-2*VDD, +VDD]
  Outputs (13):       id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd

Geometry array layout (N, 15):
    [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sweep import (
    find_threshold,
    build_voltage_grid,
    build_nodes,
    NN_OUTPUT_COLUMNS,
    save_npz,
)
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


def _create_model_and_instance(
    tech: NNTechConfig,
    device_type: str,
    variant: str,
    L: float,
    NFIN: float,
) -> Tuple[Model, Instance, ProcessParams]:
    """Create a Model + Instance for a specific (L, NFIN) bin and extract process params.

    Each (L, NFIN) bin gets its own Model (because model_overrides write to a
    shared OsdiModel buffer -- reusing across bins would corrupt earlier instances).

    Returns:
        (model, instance, process_params) tuple.
    """
    model_name = tech.get_model_name(device_type, variant)
    modelcard_path = tech.resolve_modelcard(device_type, variant, L, NFIN)

    model = Model(
        osdi_path=OSDI_PATH,
        modelcard_path=modelcard_path,
        model_name=model_name,
        model_card_name=model_name,
    )

    proc = extract_process_params(model.modelcard_params)

    devtype = 1 if device_type == "nmos" else 0
    inst = Instance(
        model=model,
        params={"L": L, "NFIN": NFIN, "TFIN": tech.tfin, "DEVTYPE": devtype},
        temperature=tech.temperature,
    )

    return model, inst, proc


def eval_single_point(
    inst: Instance,
    vd: float,
    vg: float,
    vs: float = 0.0,
    vb: float = 0.0,
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
    except Exception as e:
        print(f"  WARNING: eval_dc failed at Vd={vd:.3f} Vg={vg:.3f}: {e}")
        return None


def generate_dataset(
    tech: NNTechConfig,
    device_type: str,
    variant_names: Optional[List[str]] = None,
    verbose: bool = True,
    vg_points: int = 71,
    vd_points: int = 71,
    dense_ratio: float = 0.6,
    n_dense_mid: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate training data for one tech/polarity across all variants and legal (L, NFIN) bins.

    For each variant, enumerates PDK-legal (L, NFIN) combos via
    NNTechConfig.get_geometry_combos(). For each combo, resolves the
    NFIN-aware modelcard and extracts process params on-the-fly.

    Args:
        tech: Technology configuration (VDD, variant names).
        device_type: "nmos" or "pmos".
        variant_names: Subset of variants; None = all.
        verbose: Print per-(L, NFIN) progress.
        vg_points: Total Vg grid points.
        vd_points: Vd grid points.
        dense_ratio: Fraction of vg_points in dense Vth region.
        n_dense_mid: Extra dense points near mid-supply.

    Returns:
        Dict with "inputs" (N,4), "geometry" (N,15), "outputs" (N,13), "metadata".
    """
    vdd = tech.vdd
    is_pmos = device_type == "pmos"

    v_min = -2.0 * vdd if is_pmos else -vdd
    voltage_scale = 1.0 if is_pmos else 2.0

    variants_to_use = variant_names or tech.variant_names
    if not variants_to_use:
        raise ValueError(f"No variants for {tech.name}. Available: {tech.variant_names}")

    all_inputs: List[np.ndarray] = []
    all_geometry: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []
    total_pts = 0
    failed_pts = 0

    for variant_name in variants_to_use:
        combos = tech.get_geometry_combos(device_type, variant_name)

        if verbose:
            print(f"\n--- {tech.name} {device_type} variant={variant_name} "
                  f"({len(combos)} geometry combos) ---")

        for L, NFIN in combos:
            try:
                _model, inst, proc = _create_model_and_instance(
                    tech, device_type, variant_name, L, NFIN,
                )
            except Exception as e:
                if verbose:
                    print(f"  SKIP L={L*1e9:.1f}nm NFIN={NFIN:.0f}: {e}")
                continue

            geo = np.array([NFIN, L, tech.temperature] + proc.as_array())

            vth_mag = find_threshold(inst, vdd, device_type)
            vth_center = -vth_mag if is_pmos else vth_mag

            vg_arr, vd_arr = build_voltage_grid(
                vdd=vdd, vth_mag=vth_mag,
                vg_points=vg_points, vd_points=vd_points,
                dense_ratio=dense_ratio, voltage_scale=voltage_scale,
                v_min=v_min, n_dense_mid=n_dense_mid, vth_center=vth_center,
            )

            t0 = time.time()
            bin_pts = 0

            for vg in vg_arr:
                for vd in vd_arr:
                    result = eval_single_point(inst, vd, vg, 0.0, 0.0)
                    if result is None:
                        failed_pts += 1
                        continue
                    all_inputs.append(np.array([vd, vg, 0.0, 0.0]))
                    all_geometry.append(geo.copy())
                    all_outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))
                    bin_pts += 1

            # Zero-bias anchor (3x weight)
            result = eval_single_point(inst, 0.0, 0.0)
            if result is not None:
                out_arr = np.array([result[k] for k in NN_OUTPUT_COLUMNS])
                for _ in range(3):
                    all_inputs.append(np.array([0.0, 0.0, 0.0, 0.0]))
                    all_geometry.append(geo.copy())
                    all_outputs.append(out_arr.copy())
                    bin_pts += 1

            # Deep cutoff anchors
            if is_pmos:
                cutoff_vg = [0.0, 0.05, 0.1]
                cutoff_vd = [0.0, -vdd / 2, -vdd]
            else:
                cutoff_vg = [-0.1, -0.05, 0.0]
                cutoff_vd = [0.0, vdd / 2, vdd]
            for vg_c in cutoff_vg:
                for vd_c in cutoff_vd:
                    result = eval_single_point(inst, vd_c, vg_c)
                    if result is not None:
                        all_inputs.append(np.array([vd_c, vg_c, 0.0, 0.0]))
                        all_geometry.append(geo.copy())
                        all_outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))
                        bin_pts += 1

            total_pts += bin_pts
            if verbose:
                elapsed = max(time.time() - t0, 0.001)
                print(f"  L={L*1e9:.1f}nm NFIN={NFIN:.0f}: {bin_pts} pts in {elapsed:.1f}s "
                      f"(vth={vth_mag:.3f}V, PHIG={proc.phig:.4f}, {bin_pts/elapsed:.0f} pts/s)")

    inputs = np.array(all_inputs, dtype=np.float64)
    geometry = np.array(all_geometry, dtype=np.float64)
    outputs = np.array(all_outputs, dtype=np.float64)

    if verbose:
        print(f"\nTotal: {total_pts} pts, {failed_pts} failed")
        print(f"Shapes -- inputs: {inputs.shape}, geometry: {geometry.shape}, "
              f"outputs: {outputs.shape}")
        print(f"\nProcess parameter ranges across all (L, NFIN) bins:")
        for i, pname in enumerate(PROCESS_PARAM_NAMES):
            col = geometry[:, 3 + i]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 5:
                print(f"  {pname:>6s}: {unique_vals}")
            else:
                print(f"  {pname:>6s}: [{col.min():.4e}, {col.max():.4e}] ({len(unique_vals)} unique)")

    unique_L = np.unique(geometry[:, 1])

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
        "metadata": {
            "tech_name": tech.name,
            "device_type": device_type,
            "vdd": tech.vdd,
            "L_values": unique_L,
            "temperature": tech.temperature,
            "output_columns": np.array(NN_OUTPUT_COLUMNS),
            "variants": np.array(variants_to_use),
        },
    }


def generate_universal_dataset(
    device_type: str,
    verbose: bool = True,
    vg_points: int = 71,
    vd_points: int = 71,
    dense_ratio: float = 0.6,
    n_dense_mid: int = 0,
) -> Dict[str, np.ndarray]:
    """Concatenate per-tech datasets across all 5 technologies and all variants."""
    all_inputs, all_geometry, all_outputs = [], [], []

    for tech_name, tech in TECH_CONFIGS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {tech_name.upper()} -- {len(tech.variant_names)} variants, VDD={tech.vdd}V")
            print(f"{'='*60}")
        data = generate_dataset(tech, device_type, verbose=verbose,
                                vg_points=vg_points, vd_points=vd_points,
                                dense_ratio=dense_ratio, n_dense_mid=n_dense_mid)
        all_inputs.append(data["inputs"])
        all_geometry.append(data["geometry"])
        all_outputs.append(data["outputs"])

    inputs = np.concatenate(all_inputs, axis=0)
    geometry = np.concatenate(all_geometry, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Universal {device_type.upper()}: {inputs.shape[0]:,} total points")
        unique_L = np.unique(geometry[:, 1])
        print(f"Unique L values: {[f'{l*1e9:.1f}nm' for l in unique_L]}")
        unique_NFIN = np.unique(geometry[:, 0])
        print(f"Unique NFIN values: {unique_NFIN.astype(int).tolist()}")
        print(f"{'='*60}")

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
        "metadata": {
            "tech_name": "universal",
            "device_type": device_type,
            "vdd": 0.0,
            "L_values": np.unique(geometry[:, 1]),
            "temperature": 300.15,
            "output_columns": np.array(NN_OUTPUT_COLUMNS),
            "variants": np.array(list(TECH_CONFIGS.keys())),
        },
    }
