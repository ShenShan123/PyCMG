from __future__ import annotations

import argparse
from pathlib import Path

from pycmg import ctypes_host
from tests import verify_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce ASAP7 modelcard parsing and eval.")
    parser.add_argument(
        "--modelcard",
        default="/home/shenshan/pycmg-wrapper/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm",
        help="Path to ASAP7 modelcard file.",
    )
    parser.add_argument("--model-name", default="nmos_lvt", help="Model name to target.")
    parser.add_argument("--l", type=float, default=16e-9, help="Instance L parameter.")
    parser.add_argument("--tfin", type=float, default=8e-9, help="Instance TFIN parameter.")
    parser.add_argument("--nfin", type=float, default=2.0, help="Instance NFIN parameter.")
    parser.add_argument("--vd", type=float, default=0.7, help="Drain bias for OP check.")
    parser.add_argument("--vg", type=float, default=0.7, help="Gate bias for OP check.")
    parser.add_argument("--vs", type=float, default=0.0, help="Source bias for OP check.")
    parser.add_argument("--ve", type=float, default=0.0, help="Bulk bias for OP check.")
    parser.add_argument("--temp-c", type=float, default=27.0, help="Temperature in Celsius.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modelcard = Path(args.modelcard)
    model_name = args.model_name
    parsed = ctypes_host.parse_modelcard(str(modelcard), target_model_name=model_name)
    print(f"model={parsed.name}")
    for key in ("l", "tfin"):
        if key in parsed.params:
            print(f"{key}={parsed.params[key]}")
        else:
            print(f"{key}=<missing>")

    inst_params = {"L": args.l, "TFIN": args.tfin, "NFIN": args.nfin}
    ng_modelcard = Path("build-deep-verify") / "ngspice_eval" / "asap7_nmos_lvt.osdi"
    verify_utils.make_ngspice_modelcard(modelcard, ng_modelcard, model_name, inst_params)
    ng = verify_utils.run_ngspice_op_point(
        ng_modelcard,
        model_name,
        inst_params,
        vd=args.vd,
        vg=args.vg,
        vs=args.vs,
        ve=args.ve,
        out_dir=Path("build-deep-verify") / "ngspice_eval" / "asap7_op",
        temp_c=args.temp_c,
    )
    eval_fn = verify_utils.make_pycmg_eval(modelcard, model_name, inst_params, temp_c=args.temp_c)
    py = eval_fn(args.vd, args.vg, args.vs, args.ve)
    print(f"ngspice id={ng['id']} ig={ng['ig']} is={ng['is']} ib={ng['ib']}")
    print(f"pycmg   id={py[0]} ig={py[1]} is={py[2]} ib={py[3]}")


if __name__ == "__main__":
    main()
