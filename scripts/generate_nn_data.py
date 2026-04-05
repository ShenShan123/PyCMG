#!/usr/bin/env python3
"""Generate NN training data (.npz) from PyCMG BSIM-CMG sweeps.

Usage (from PyCMG root):
    python scripts/generate_nn_data.py --device both --universal
    python scripts/generate_nn_data.py --device nmos --tech asap7
    python scripts/generate_nn_data.py --device both --universal --n-dense-mid 30

Output goes to --data-dir (default: ../../nn_model/data/datasets/).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import generate_dataset, generate_universal_dataset
from pycmg.sweep import save_npz


def _default_data_dir() -> Path:
    pycmg_root = Path(__file__).resolve().parents[1]
    project_root = pycmg_root.parents[1]
    return project_root / "nn_model" / "data" / "datasets"


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
    parser.add_argument("--vg-points", type=int, default=71)
    parser.add_argument("--vd-points", type=int, default=71)
    parser.add_argument("--n-dense-mid", type=int, default=0,
                        help="Extra dense points near mid-supply (default: 0)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Output directory for .npz files")
    args = parser.parse_args()

    data_dir = args.data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    devices = ["nmos", "pmos"] if args.device == "both" else [args.device]
    sweep_kw = dict(vg_points=args.vg_points, vd_points=args.vd_points,
                    n_dense_mid=args.n_dense_mid, verbose=True)

    if args.universal:
        for device_type in devices:
            data = generate_universal_dataset(device_type, **sweep_kw)
            out = data_dir / f"universal_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"])
        return

    techs = list(TECH_CONFIGS.values()) if args.tech == "all" \
        else [TECH_CONFIGS[args.tech]]
    variant_names = None if args.variants == "all" \
        else [v.strip() for v in args.variants.split(",")]

    for tech in techs:
        for device_type in devices:
            data = generate_dataset(tech, device_type,
                                    variant_names=variant_names, **sweep_kw)
            out = data_dir / f"{tech.name.lower()}_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"])


if __name__ == "__main__":
    main()
