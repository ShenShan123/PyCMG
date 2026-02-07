from __future__ import annotations

from pathlib import Path
import re

from scripts import deep_verify
from pycmg import ctypes_host


def make_ngspice_modelcard(src: Path, dst: Path, model_name: str, overrides: dict[str, float]) -> None:
    text = src.read_text()
    # Clamp EOTACC for OSDI compatibility (matches deep_verify.ensure_modelcard).
    text = re.sub(r"EOTACC\s*=\s*([0-9eE+\-\.]+)", "EOTACC = 1.10e-10", text, flags=re.IGNORECASE)
    lines = []
    in_target = False
    found_keys: set[str] = set()
    target_lower = model_name.lower()
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped.lower().startswith(".model"):
            if in_target:
                for key, val in overrides.items():
                    key_u = key.upper()
                    if key_u not in found_keys:
                        lines.append(f"+ {key_u} = {val}")
                in_target = False
                found_keys.clear()
            parts = stripped.split()
            if len(parts) >= 3 and parts[1].lower() == target_lower:
                parts[2] = "bsimcmg"
                prefix = line[: line.lower().find(".model")]
                line = f"{prefix}{' '.join(parts)}"
                in_target = True
        elif in_target:
            for key, val in overrides.items():
                key_u = key.upper()
                pattern = rf"(?i)\b{re.escape(key)}\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)"

                def _repl(match, key_u: str = key_u, val: float = val) -> str:
                    found_keys.add(key_u)
                    return f"{key_u} = {val}"

                line, _ = re.subn(pattern, _repl, line)
        lines.append(line)
    if in_target:
        for key, val in overrides.items():
            key_u = key.upper()
            if key_u not in found_keys:
                lines.append(f"+ {key_u} = {val}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def main() -> None:
    modelcard = Path(
        "/home/shenshan/pycmg-wrapper/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm"
    )
    model_name = "nmos_lvt"
    parsed = ctypes_host.parse_modelcard(str(modelcard), target_model_name=model_name)
    print(f"model={parsed.name}")
    for key in ("l", "tfin"):
        if key in parsed.params:
            print(f"{key}={parsed.params[key]}")
        else:
            print(f"{key}=<missing>")

    inst_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0}
    ng_modelcard = Path("build-deep-verify") / "ngspice_eval" / "asap7_nmos_lvt.osdi"
    make_ngspice_modelcard(modelcard, ng_modelcard, model_name, inst_params)
    ng = deep_verify.run_ngspice_op_point(
        ng_modelcard,
        model_name,
        inst_params,
        vd=0.7,
        vg=0.7,
        vs=0.0,
        ve=0.0,
        out_dir=Path("build-deep-verify") / "ngspice_eval" / "asap7_op",
        temp_c=27.0,
    )
    eval_fn = deep_verify.make_pycmg_eval(modelcard, model_name, inst_params, temp_c=27.0)
    py = eval_fn(0.7, 0.7, 0.0, 0.0)
    print(f"ngspice id={ng['id']} ig={ng['ig']} is={ng['is']} ib={ng['ib']}")
    print(f"pycmg   id={py[0]} ig={py[1]} is={py[2]} ib={py[3]}")


if __name__ == "__main__":
    main()
