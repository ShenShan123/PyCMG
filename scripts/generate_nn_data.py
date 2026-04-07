#!/usr/bin/env python3
"""Generate NN training data (.npz) from PyCMG BSIM-CMG sweeps.

Usage (from PyCMG root):
    python scripts/generate_nn_data.py --device both --universal
    python scripts/generate_nn_data.py --device nmos --tech asap7
    python scripts/generate_nn_data.py --device both --universal --n-workers 8
    python scripts/generate_nn_data.py --device both --universal \
        --finetune-size 8000

Phase D rewrite features:
    --n-workers N           : parallelize bins via multiprocessing.Pool (D4)
    --temperatures T1,T2..  : Kelvin temperature sweep (D1)
    --n-lhs-samples N       : LHS sample budget per bin (D3)
    --voltage-box-factor F  : Vg/Vd/Vbs box width in units of VDD (D3)
    --finetune-size N       : also write a stratified finetune split (D8)

Output goes to --data-dir (default: ../../bsimar/data/datasets/).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import (
    DEFAULT_LHS_SAMPLES_PER_BIN,
    DEFAULT_TEMPERATURES_K,
    DEFAULT_VOLTAGE_BOX_FACTOR,
    generate_dataset,
    generate_universal_dataset,
)
from pycmg.sweep import save_npz


def _default_data_dir() -> Path:
    pycmg_root = Path(__file__).resolve().parents[1]
    project_root = pycmg_root.parents[1]
    return project_root / "external_compact_models" / "bsimar" / "data" / "datasets"


def _parse_temperatures(arg: str) -> tuple:
    return tuple(float(x.strip()) for x in arg.split(",") if x.strip())


def _save_finetune_split(
    full: dict,
    out_path: Path,
    n_samples: int,
    seed: int,
) -> None:
    """Write a stratified random subset of `full` for paper-style fine-tune.

    "Stratified" here means a uniform random shuffle (the LHS sampler
    already produced uniform coverage; further stratification by
    (variant, T) bin is overkill for an N=1k–8k subset). Reuses the
    same metadata as the parent dataset.
    """
    n_total = full["inputs"].shape[0]
    n = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)[:n]

    save_npz(
        full["inputs"][idx],
        full["geometry"][idx],
        full["outputs"][idx],
        out_path,
        metadata=full["metadata"],
    )
    print(f"  Fine-tune split: {n:,} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NN training data (.npz) from PyCMG BSIM-CMG"
    )
    parser.add_argument("--device", choices=["nmos", "pmos", "both"], default="nmos")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()) + ["all"],
                        default="asap7")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names (default: all)")
    parser.add_argument("--universal", action="store_true",
                        help="Generate universal dataset across all techs/variants")

    # D1
    parser.add_argument(
        "--temperatures", type=_parse_temperatures,
        default=DEFAULT_TEMPERATURES_K,
        help="Comma-separated temperatures in Kelvin (default: -25, 27, 125 °C)",
    )

    # D3
    parser.add_argument("--n-lhs-samples", type=int,
                        default=DEFAULT_LHS_SAMPLES_PER_BIN,
                        help="LHS samples per (variant, L, NFIN, T) bin")
    parser.add_argument("--voltage-box-factor", type=float,
                        default=DEFAULT_VOLTAGE_BOX_FACTOR,
                        help="Voltage box width in units of VDD "
                             "(2.0 = [0, 2]·VDD; covers NR overshoot)")

    # D4
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel worker count (1 = serial)")
    parser.add_argument("--seed", type=int, default=42)

    # D8
    parser.add_argument("--finetune-size", type=int, default=0,
                        help="If >0, also write finetune_<base>.npz with a "
                             "stratified random subset of N samples (D8)")

    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Output directory for .npz files")
    args = parser.parse_args()

    data_dir = args.data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    devices = ["nmos", "pmos"] if args.device == "both" else [args.device]

    common_kw = dict(
        temperatures=args.temperatures,
        n_lhs_samples=args.n_lhs_samples,
        voltage_box_factor=args.voltage_box_factor,
        n_workers=args.n_workers,
        seed=args.seed,
        verbose=True,
    )

    if args.universal:
        for device_type in devices:
            data = generate_universal_dataset(device_type, **common_kw)
            out = data_dir / f"universal_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"])
            if args.finetune_size > 0:
                ft_out = data_dir / f"finetune_universal_{device_type}.npz"
                _save_finetune_split(
                    data, ft_out, args.finetune_size, seed=args.seed,
                )
        return

    techs = list(TECH_CONFIGS.values()) if args.tech == "all" \
        else [TECH_CONFIGS[args.tech]]
    variant_names = None if args.variants == "all" \
        else [v.strip() for v in args.variants.split(",")]

    for tech in techs:
        for device_type in devices:
            data = generate_dataset(
                tech, device_type,
                variant_names=variant_names,
                **common_kw,
            )
            out = data_dir / f"{tech.name.lower()}_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"])
            if args.finetune_size > 0:
                ft_out = data_dir / f"finetune_{tech.name.lower()}_{device_type}.npz"
                _save_finetune_split(
                    data, ft_out, args.finetune_size, seed=args.seed,
                )


if __name__ == "__main__":
    main()
