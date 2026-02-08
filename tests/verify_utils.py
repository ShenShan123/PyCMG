import argparse
import csv
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-deep-verify"
OSDI_PATH = BUILD / "osdi" / "bsimcmg.osdi"
NGSPICE_BIN = os.environ.get("NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")
MODEL_SRC_NMOS = ROOT / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
MODEL_SRC_PMOS = ROOT / "bsim-cmg-va" / "benchmark_test" / "modelcard.pmos"
MODEL_DST_NMOS = BUILD / "ngspice_eval" / "modelcard.nmos.osdi"
MODEL_DST_PMOS = BUILD / "ngspice_eval" / "modelcard.pmos.osdi"
CIRCUIT_DIR = ROOT / "circuit_examples"
ASAP7_DIR = ROOT / "tech_model_cards" / "asap7_pdk_r1p7" / "models" / "hspice"
ASAP7_MODELCARD_OVERRIDE = os.environ.get("ASAP7_MODELCARD")
ABS_TOL_I = 1e-9
ABS_TOL_Q = 1e-18
ABS_TOL_C = 1e-18
REL_TOL = 5e-3
BACKEND_PYCMG = "pycmg"
BACKEND_OSDI = "osdi_eval"


def die(msg: str) -> None:
    raise RuntimeError(msg)


def ensure_modelcard(src: Path, dst: Path, overrides=None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        die(f"missing modelcard source: {src}")
    text = src.read_text()
    # Clamp EOTACC for OSDI compatibility.
    text = re.sub(r"EOTACC\s*=\s*([0-9eE+\-\.]+)", "EOTACC = 1.10e-10", text)
    if overrides:
        for key, val in overrides.items():
            key_u = key.upper()
            text, count = re.subn(
                rf"(?im)(^\+\s*{re.escape(key_u)}\b\s*=\s*)([eE0-9+\-\.]+[a-zA-Z]*)",
                rf"\g<1>{val}",
                text,
            )
            if count == 0:
                text = re.sub(
                    r"(?im)(^\.model[^\n]*\n)",
                    lambda m: m.group(1) + f"+ {key_u} = {val}\n",
                    text,
                    count=1,
                )
    dst.write_text(text)


def parse_modelcard_params(path: Path, model_name: str):
    from pycmg import ctypes_host

    parse_number_with_suffix = ctypes_host.parse_number_with_suffix
    text = path.read_text()
    lines = []
    in_model = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.lower().startswith(".model") and model_name.lower() in line.lower():
            in_model = True
            lines.append(line)
            continue
        if in_model:
            if line.startswith("+"):
                lines.append(line)
            else:
                break
    if not lines:
        die(f"model {model_name} not found in {path}")
    joined = " ".join(lines)
    params = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*([eE0-9+\-\.]+[a-zA-Z]*)", joined):
        name = match.group(1).upper()
        val = parse_number_with_suffix(match.group(2))
        params[name] = val
    return params


def parse_instance_params(netlist: Path, instance_name: str = "X1"):
    from pycmg import ctypes_host

    parse_number_with_suffix = ctypes_host.parse_number_with_suffix
    if not netlist.exists():
        die(f"missing netlist: {netlist}")
    for raw in netlist.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if not line.upper().startswith(instance_name):
            continue
        params = {}
        for match in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*([eE0-9+\-\.]+[a-zA-Z]*)", line):
            name = match.group(1).upper()
            val = parse_number_with_suffix(match.group(2))
            params[name] = val
        return params
    die(f"instance {instance_name} not found in {netlist}")


def validate_instance_params(inst_params):
    required = ["L", "TFIN", "NFIN"]
    for key in required:
        if key not in inst_params:
            die(f"instance param missing: {key}")
        if inst_params[key] <= 0:
            die(f"instance param not positive: {key}={inst_params[key]}")
    if inst_params["L"] < 5e-9 or inst_params["L"] > 100e-9:
        die(f"L out of expected range: {inst_params['L']}")
    if inst_params["TFIN"] < 5e-9 or inst_params["TFIN"] > 50e-9:
        die(f"TFIN out of expected range: {inst_params['TFIN']}")
    if inst_params["NFIN"] < 1 or inst_params["NFIN"] > 50:
        die(f"NFIN out of expected range: {inst_params['NFIN']}")


def write_instance_netlist(path: Path, model_name: str, inst_params):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = " ".join(f"{k}={v}" for k, v in inst_params.items())
    path.write_text(
        "\n".join(
            [
                "* instance sweep netlist",
                f"X1 d g s e {model_name} {params}",
                ".end",
            ]
        )
    )


def prepare_case(label: str, model_name: str, inst_params):
    case_dir = CIRCUIT_DIR / label
    netlist = case_dir / "netlist.cir"
    write_instance_netlist(netlist, model_name, inst_params)
    parsed = parse_instance_params(netlist)
    validate_instance_params(parsed)
    return case_dir, parsed


def osdi_dump_params(modelcard: Path, model_name: str, names, inst_params):
    cmd = [
        str(BUILD / "osdi_eval"),
        "--osdi", str(OSDI_PATH),
        "--modelcard", str(modelcard),
        "--node", "d=0",
        "--node", "g=0",
        "--node", "s=0",
        "--node", "e=0",
        "--quiet",
    ]
    for key, value in inst_params.items():
        cmd.extend(["--param", f"{key}={value}"])
    for name in names:
        cmd.extend(["--dump-param", name])
    out = run(cmd)
    parsed = {}
    for line in out.splitlines():
        m = re.search(r"Param\s+([A-Za-z0-9_]+)\s+=\s+([eE0-9+\-\.]+)", line)
        if m:
            parsed[m.group(1).upper()] = float(m.group(2))
    return parsed


def check_params(modelcard: Path, model_name: str, names, inst_params):
    card_params = parse_modelcard_params(modelcard, model_name)
    osdi_params = osdi_dump_params(modelcard, model_name, names, inst_params)
    for name in names:
        key = name.upper()
        if key not in osdi_params:
            die(f"osdi_eval did not report param {name} for {model_name}")
        card_val = inst_params.get(key, card_params.get(key))
        osdi_val = osdi_params[key]
        if card_val is None:
            print(f"{model_name} param {name}: modelcard=<default> osdi={osdi_val}")
            continue
        if abs(card_val - osdi_val) > max(1e-30, abs(card_val) * 1e-12):
            die(f"param mismatch {model_name} {name}: modelcard={card_val} osdi={osdi_val}")
        print(f"{model_name} param {name}: modelcard={card_val} osdi={osdi_val}")


def run(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    return res.stdout


def build_osdi_eval():
    osdi_eval = BUILD / "osdi_eval"
    osdi_lib = OSDI_PATH
    src_eval = ROOT / "cpp" / "osdi_eval.cpp"
    src_host = ROOT / "cpp" / "osdi_host.cpp"
    needs_build = (not osdi_eval.exists()) or (not osdi_lib.exists())
    if osdi_eval.exists():
        latest_src = max(src_eval.stat().st_mtime, src_host.stat().st_mtime)
        if osdi_eval.stat().st_mtime < latest_src:
            needs_build = True
    if needs_build:
        pybind11_dir = run([sys.executable, "-m", "pybind11", "--cmakedir"]).strip()
        run([
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(BUILD),
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_dir}",
        ])
        if not osdi_lib.exists():
            run(["cmake", "--build", str(BUILD), "--target", "osdi"])
        run(["cmake", "--build", str(BUILD), "--target", "osdi_eval"])


def parse_wrdata(path: Path):
    with path.open() as f:
        header_line = f.readline()
        if not header_line:
            die(f"empty wrdata: {path}")
        headers = header_line.split()
        rows = []
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            rows.append(vals)
    return headers, rows


def col_index(headers, name):
    for i, h in enumerate(headers):
        if h == name:
            return i
    return None


def find_col(headers, names: List[str]) -> Optional[int]:
    for name in names:
        idx = col_index(headers, name)
        if idx is not None:
            return idx
    return None


def format_inst_params(inst_params) -> str:
    return " ".join(f"{k}={v}" for k, v in inst_params.items())


def run_ngspice(netlist_text: str, out_csv: Path, log_path: Path, wrdata_cols: str):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    net_path = out_csv.parent / "netlist.cir"
    net_path.write_text(netlist_text)
    # Use a wrapper to load OSDI before parsing modelcard.
    runner_path = out_csv.parent / "runner.cir"
    runner_path.write_text(
        "* ngspice runner\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {net_path}\n"
        "set filetype=ascii\n"
        "set wr_vecnames\n"
        "run\n"
        f"wrdata {out_csv} {wrdata_cols}\n"
        ".endc\n"
        ".end\n"
    )
    run([NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)])


def extract_vg_values(ng_csv: Path):
    headers, rows = parse_wrdata(ng_csv)
    vg_idx = col_index(headers, "v(g)")
    if vg_idx is None:
        die("missing v(g) in dc sweep")
    return [row[vg_idx] for row in rows]


def run_ngspice_dc_vg(modelcard: Path, model_name: str, inst_params, vd, vg_start, vg_stop, vg_step, out_dir: Path, temp_c: float):
    net = [
        "* Id-Vg sweep",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vd}",
        "Vg g 0 0",
        "Vs s 0 0",
        "Ve e 0 0",
        f"N1 d g s e {model_name}",
        f".dc Vg {vg_start} {vg_stop} {vg_step}",
        ".end",
    ]
    out_csv = out_dir / "ng_id_vg.csv"
    run_ngspice(
        "\n".join(net),
        out_csv,
        out_dir / "ng_id_vg.log",
        "v(g) v(d) v(s) v(e) i(vg) i(vd) i(vs) i(ve)",
    )
    return out_csv


def run_ngspice_dc_vd(modelcard: Path, model_name: str, inst_params, vg, vd_start, vd_stop, vd_step, out_dir: Path, temp_c: float):
    net = [
        "* Id-Vd sweep",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        "Vd d 0 0",
        f"Vg g 0 {vg}",
        "Vs s 0 0",
        "Ve e 0 0",
        f"N1 d g s e {model_name}",
        f".dc Vd {vd_start} {vd_stop} {vd_step}",
        ".end",
    ]
    out_csv = out_dir / "ng_id_vd.csv"
    run_ngspice(
        "\n".join(net),
        out_csv,
        out_dir / "ng_id_vd.log",
        "v(g) v(d) v(s) v(e) i(vg) i(vd) i(vs) i(ve)",
    )
    return out_csv


def run_ngspice_op_charge(modelcard: Path, model_name: str, inst_params, vd, vg, vs, ve, out_dir: Path, temp_c: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    net = [
        "* OP charge query",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vd}",
        f"Vg g 0 {vg}",
        f"Vs s 0 {vs}",
        f"Ve e 0 {ve}",
        f"N1 d g s e {model_name}",
        ".op",
        ".end",
    ]
    out_csv = out_dir / "ng_op_tmp.csv"
    log_path = out_dir / "ng_op_tmp.log"
    run_ngspice(
        "\n".join(net),
        out_csv,
        log_path,
        "v(g) v(d) v(s) v(e) @n1[qg] @n1[qd] @n1[qs] @n1[qb] @n1[gm] @n1[gds] @n1[gmbs]",
    )
    headers, rows = parse_wrdata(out_csv)
    if not rows:
        die("empty op wrdata")
    idx = {
        name: col_index(headers, name)
        for name in ("@n1[qg]", "@n1[qd]", "@n1[qs]", "@n1[qb]", "@n1[gm]", "@n1[gds]", "@n1[gmbs]")
    }
    if any(v is None for v in idx.values()):
        die("missing opvar columns in op wrdata")
    row = rows[0]
    return {
        "qg": row[idx["@n1[qg]"]],
        "qd": row[idx["@n1[qd]"]],
        "qs": row[idx["@n1[qs]"]],
        "qb": row[idx["@n1[qb]"]],
        "gm": row[idx["@n1[gm]"]],
        "gds": row[idx["@n1[gds]"]],
        "gmb": row[idx["@n1[gmbs]"]],
    }


def run_ngspice_op_point(modelcard: Path,
                         model_name: str,
                         inst_params,
                         vd: float,
                         vg: float,
                         vs: float,
                         ve: float,
                         out_dir: Path,
                         temp_c: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    net = [
        "* OP point query",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vd}",
        f"Vg g 0 {vg}",
        f"Vs s 0 {vs}",
        f"Ve e 0 {ve}",
        f"N1 d g s e {model_name}",
        ".op",
        ".end",
    ]
    out_csv = out_dir / "ng_op_point.csv"
    log_path = out_dir / "ng_op_point.log"
    run_ngspice(
        "\n".join(net),
        out_csv,
        log_path,
        "v(g) v(d) v(s) v(e) "
        "i(vg) i(vd) i(vs) i(ve) "
        "@n1[qg] @n1[qd] @n1[qs] @n1[qb] "
        "@n1[gm] @n1[gds] @n1[gmbs]",
    )
    headers, rows = parse_wrdata(out_csv)
    if not rows:
        die("empty op wrdata")
    row = rows[0]
    idx = {
        name: col_index(headers, name)
        for name in (
            "i(vg)", "i(vd)", "i(vs)", "i(ve)",
            "@n1[qg]", "@n1[qd]", "@n1[qs]", "@n1[qb]",
            "@n1[gm]", "@n1[gds]", "@n1[gmbs]",
        )
    }
    if any(v is None for v in idx.values()):
        die("missing opvar columns in op wrdata")
    return {
        "ig": row[idx["i(vg)"]],
        "id": row[idx["i(vd)"]],
        "is": row[idx["i(vs)"]],
        "ib": row[idx["i(ve)"]],
        "qg": row[idx["@n1[qg]"]],
        "qd": row[idx["@n1[qd]"]],
        "qs": row[idx["@n1[qs]"]],
        "qb": row[idx["@n1[qb]"]],
        "gm": row[idx["@n1[gm]"]],
        "gds": row[idx["@n1[gds]"]],
        "gmb": row[idx["@n1[gmbs]"]],
    }


def parse_imag(headers, row, name):
    idx = (col_index(headers, f"{name}#imag")
           or col_index(headers, f"{name}#im")
           or col_index(headers, f"{name}_im"))
    if idx is not None:
        return row[idx]
    dup = [i for i, h in enumerate(headers) if h == name]
    if len(dup) >= 2:
        return row[dup[1]]
    return 0.0


def run_ngspice_ac_caps_vg(modelcard: Path, model_name: str, vd, vg_values, out_dir: Path, temp_c: float):
    out_csv = out_dir / "ng_caps_vg.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vg", "cgg", "cgd", "cgs", "cdg", "cdd"])
        for vg in vg_values:
            def run_ac(ac_src):
                net = [
                    "* AC caps",
                    f'.include "{modelcard}"',
                    f".temp {temp_c}",
                    f"Vd d 0 {vd}" + (" ac 1" if ac_src == "d" else ""),
                    f"Vg g 0 {vg}" + (" ac 1" if ac_src == "g" else ""),
                    "Vs s 0 0" + (" ac 1" if ac_src == "s" else ""),
                    "Ve e 0 0",
                    f"N1 d g s e {model_name}",
                    ".ac dec 1 1 10",
                    ".end",
                ]
                out_tmp = out_dir / f"ng_ac_{ac_src}.csv"
                run_ngspice(
                    "\n".join(net),
                    out_tmp,
                    out_dir / f"ng_ac_{ac_src}.log",
                    "frequency i(vg) i(vd) i(vs) i(ve)",
                )
                headers, rows = parse_wrdata(out_tmp)
                if not rows:
                    die("empty AC wrdata")
                freq = rows[0][col_index(headers, "frequency")]
                w = 2.0 * math.pi * freq
                # i(vx) is current through the voltage source; flip sign to get current into device.
                ig = -parse_imag(headers, rows[0], "i(vg)") / w
                idv = -parse_imag(headers, rows[0], "i(vd)") / w
                isv = -parse_imag(headers, rows[0], "i(vs)") / w
                return ig, idv, isv

            ig_g, id_g, is_g = run_ac("g")
            ig_d, id_d, _ = run_ac("d")
            ig_s, _, _ = run_ac("s")

            cgg = ig_g
            cdg = id_g
            cgd = ig_d
            cdd = id_d
            cgs = ig_s

            writer.writerow([vg, cgg, cgd, cgs, cdg, cdd])
    return out_csv


def parse_ng_caps(path: Path):
    caps = {}
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            vg = round(float(row[0]), 12)
            caps[vg] = [float(x) for x in row[1:6]]
    return caps


def run_ngspice_tran(modelcard: Path,
                     model_name: str,
                     inst_params,
                     step: float,
                     stop: float,
                     out_dir: Path,
                     temp_c: float):
    net = [
        "* Tran playback netlist",
        f'.include "{modelcard}"',
        f".temp {temp_c}",
        "Vd d 0 0.05",
        "Vg g 0 PWL(0 0 1n 0 2n 1.2 10n 1.2)",
        "Vs s 0 0",
        "Ve e 0 0",
        f"N1 d g s e {model_name}",
        ".options method=gear maxord=1",
        f".tran {step} {stop} 0 {step}",
        ".end",
    ]
    out_csv = out_dir / "ng_tran.csv"
    run_ngspice(
        "\n".join(net),
        out_csv,
        out_dir / "ng_tran.log",
        "time v(g) v(d) v(s) v(e) "
        "i(vg) i(vd) i(vs) i(ve) "
        "@n1[qg] @n1[qd] @n1[qs] @n1[qb]",
    )
    return out_csv


def run_osdi_eval(modelcard: Path, model_name: str, inst_params, vd, vg, vs, ve, temp_c: float = 27.0):
    cmd = [
        str(BUILD / "osdi_eval"),
        "--osdi", str(OSDI_PATH),
        "--modelcard", str(modelcard),
        "--temp", f"{temp_c + 273.15}",
        "--node", f"d={vd}",
        "--node", f"g={vg}",
        "--node", f"s={vs}",
        "--node", f"e={ve}",
        "--print-charges",
        "--print-cap",
        "--print-derivs",
        "--quiet",
    ]
    out = run(cmd)
    id_val = ig_val = is_val = ib_val = qg_val = qd_val = qs_val = qb_val = None
    gm_val = gds_val = gmb_val = None
    cgg_val = cgd_val = cgs_val = cdg_val = cdd_val = None
    for line in out.splitlines():
        if line.startswith("Id="):
            m = re.findall(r"Id=([eE0-9+\-\.]+)\s+Ig=([eE0-9+\-\.]+)\s+Is=([eE0-9+\-\.]+)\s+Ie=([eE0-9+\-\.]+)", line)
            if m:
                id_val = float(m[0][0])
                ig_val = float(m[0][1])
                is_val = float(m[0][2])
                ib_val = float(m[0][3])
        if line.startswith("Qg="):
            m = re.findall(r"Qg=([eE0-9+\-\.]+)\s+Qd=([eE0-9+\-\.]+)\s+Qs=([eE0-9+\-\.]+)\s+Qb=([eE0-9+\-\.]+)", line)
            if m:
                qg_val = float(m[0][0])
                qd_val = float(m[0][1])
                qs_val = float(m[0][2])
                qb_val = float(m[0][3])
        if line.startswith("Gm="):
            m = re.findall(r"Gm=([eE0-9+\-\.]+)\s+Gds=([eE0-9+\-\.]+)\s+Gmb=([eE0-9+\-\.]+)", line)
            if m:
                gm_val = float(m[0][0])
                gds_val = float(m[0][1])
                gmb_val = float(m[0][2])
        if line.startswith("Cgg="):
            m = re.findall(
                r"Cgg=([eE0-9+\-\.]+)\s+Cgd=([eE0-9+\-\.]+)\s+Cgs=([eE0-9+\-\.]+)\s+Cgb=([eE0-9+\-\.]+)\s+Cdg=([eE0-9+\-\.]+)\s+Cdd=([eE0-9+\-\.]+)",
                line,
            )
            if m:
                cgg_val = float(m[0][0])
                cgd_val = float(m[0][1])
                cgs_val = float(m[0][2])
                cdg_val = float(m[0][4])
                cdd_val = float(m[0][5])
    if id_val is None or qg_val is None:
        die(f"failed to parse osdi_eval output:\n{out}")
    return id_val, ig_val, is_val, ib_val, qg_val, qd_val, qs_val, qb_val, gm_val, gds_val, gmb_val, {
        "cgg": cgg_val,
        "cgd": cgd_val,
        "cgs": cgs_val,
        "cdg": cdg_val,
        "cdd": cdd_val,
    }

def make_pycmg_eval(modelcard: Path, model_name: str, inst_params, temp_c: float):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import pycmg  # type: ignore
    except ImportError as exc:
        die(f"pycmg import failed: {exc}")
    model = pycmg.Model(str(OSDI_PATH), str(modelcard), model_name, model_card_name=model_name)
    inst = pycmg.Instance(model, params=inst_params, temperature=temp_c + 273.15)

    def _eval(vd: float, vg: float, vs: float, ve: float):
        out = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": ve})
        return (
            out["id"], out["ig"], out["is"], out["ie"],
            out["qg"], out["qd"], out["qs"], out["qb"],
            out["gm"], out["gds"], out["gmb"],
            {"cgg": out["cgg"], "cgd": out["cgd"], "cgs": out["cgs"], "cdg": out["cdg"], "cdd": out["cdd"]},
        )

    return _eval


def make_pycmg_eval_tran(modelcard: Path, model_name: str, inst_params, temp_c: float):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import pycmg  # type: ignore
    except ImportError as exc:
        die(f"pycmg import failed: {exc}")
    model = pycmg.Model(str(OSDI_PATH), str(modelcard), model_name, model_card_name=model_name)
    inst = pycmg.Instance(model, params=inst_params, temperature=temp_c + 273.15)

    def _eval(nodes: Dict[str, float], time: float, delta_t: float):
        return inst.eval_tran(nodes, time, delta_t)

    return _eval


def compare_tran(modelcard: Path,
                 model_name: str,
                 inst_params,
                 ng_csv: Path,
                 out_dir: Path,
                 step: float,
                 backend: str,
                 temp_c: float) -> bool:
    if backend != BACKEND_PYCMG:
        print("Transient playback is supported only with pycmg backend.")
        return True

    headers, rows = parse_wrdata(ng_csv)
    time_idx = col_index(headers, "time")
    vg_idx = col_index(headers, "v(g)")
    vd_idx = col_index(headers, "v(d)")
    vs_idx = col_index(headers, "v(s)")
    ve_idx = col_index(headers, "v(e)")
    id_idx = find_col(headers, ["i(vd)", "@n1[id]", "@N1[id]"])
    ig_idx = find_col(headers, ["i(vg)", "@n1[ig]", "@N1[ig]"])
    is_idx = find_col(headers, ["i(vs)", "@n1[is]", "@N1[is]"])
    ib_idx = find_col(headers, ["i(ve)", "@n1[ib]", "@N1[ib]"])
    qg_idx = col_index(headers, "@n1[qg]")
    required = [time_idx, vg_idx, vd_idx, vs_idx, ve_idx, id_idx, ig_idx, is_idx, ib_idx]
    if any(idx is None for idx in required):
        die("missing columns in ngspice transient output")

    eval_tran = make_pycmg_eval_tran(modelcard, model_name, inst_params, temp_c)

    out_rows = []
    max_rel = {"id": 0.0, "ig": 0.0, "is": 0.0, "ib": 0.0}
    max_abs = {"id": 0.0, "ig": 0.0, "is": 0.0, "ib": 0.0}

    def rel_err(ref: float, got: float, abs_tol: float) -> float:
        diff = abs(got - ref)
        if diff <= abs_tol:
            return 0.0
        denom = max(abs(ref), abs_tol)
        return diff / denom

    times = [row[time_idx] for row in rows]  # type: ignore[index]
    dts = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
    if dts:
        dts_sorted = sorted(dts)
        median_dt = dts_sorted[len(dts_sorted) // 2]
        min_dt = max(median_dt * 0.5, step * 0.1)
    else:
        min_dt = step

    filtered_rows = []
    last_keep = None
    for row in rows:
        t = row[time_idx]  # type: ignore[index]
        if last_keep is None or (t - last_keep) >= min_dt:
            filtered_rows.append(row)
            last_keep = t

    prev_time = None
    prev_ng_qg: Optional[float] = None
    prev_osdi_qg: Optional[float] = None
    for row in filtered_rows:
        t = row[time_idx]  # type: ignore[index]
        dt = step if prev_time is None else (t - prev_time)
        prev_time = t
        nodes = {
            "d": row[vd_idx],  # type: ignore[index]
            "g": row[vg_idx],  # type: ignore[index]
            "s": row[vs_idx],  # type: ignore[index]
            "e": row[ve_idx],  # type: ignore[index]
        }
        osdi = eval_tran(nodes, t, dt)
        ng_id = row[id_idx]  # type: ignore[index]
        ng_is = row[is_idx]  # type: ignore[index]
        ng_ib = row[ib_idx]  # type: ignore[index]
        osdi_id = -osdi["id"]
        osdi_ig = osdi["ig"]
        osdi_is = -osdi["is"]
        osdi_ib = -osdi["ie"]
        if qg_idx is not None:
            ng_qg = row[qg_idx]  # type: ignore[index]
            osdi_qg = osdi.get("qg", 0.0)
            if prev_ng_qg is not None and prev_osdi_qg is not None and dt > 0:
                ng_ig = (ng_qg - prev_ng_qg) / dt
                osdi_ig = (osdi_qg - prev_osdi_qg) / dt
            else:
                ng_ig = row[ig_idx]  # type: ignore[index]
            prev_ng_qg = ng_qg
            prev_osdi_qg = osdi_qg
        else:
            ng_ig = row[ig_idx]  # type: ignore[index]

        out_rows.append(
            (
                t,
                nodes["g"], nodes["d"], nodes["s"], nodes["e"],
                ng_id, osdi_id,
                ng_ig, osdi_ig,
                ng_is, osdi_is,
                ng_ib, osdi_ib,
                osdi_id - ng_id,
            )
        )

        for key, ref, got, abs_tol in [
            ("id", ng_id, osdi_id, ABS_TOL_I),
            ("ig", ng_ig, osdi_ig, ABS_TOL_I),
            ("is", ng_is, osdi_is, ABS_TOL_I),
            ("ib", ng_ib, osdi_ib, ABS_TOL_I),
        ]:
            diff = abs(got - ref)
            if diff > max_abs[key]:
                max_abs[key] = diff
            rerr = rel_err(ref, got, abs_tol)
            if rerr > max_rel[key]:
                max_rel[key] = rerr

    out_csv = out_dir / "osdi_tran.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time",
            "v_g", "v_d", "v_s", "v_e",
            "ng_id", "osdi_id",
            "ng_ig", "osdi_ig",
            "ng_is", "osdi_is",
            "ng_ib", "osdi_ib",
            "err_id",
        ])
        w.writerows(out_rows)

    print("Transient comparison summary:")
    for key in ("id", "ig", "is", "ib"):
        print(f"  {key}: max_abs={max_abs[key]:.3e} max_rel={max_rel[key]:.3e}")

    ok = True
    for key in ("id", "is", "ib"):
        if max_abs[key] > ABS_TOL_I and max_rel[key] > REL_TOL:
            ok = False
    return ok

def compare_id_vg(modelcard: Path, model_name: str, inst_params, ng_csv: Path, ng_caps, out_dir: Path, backend: str, temp_c: float):
    headers, rows = parse_wrdata(ng_csv)
    vg_idx = col_index(headers, "v(g)")
    id_idx = col_index(headers, "i(vd)")
    ig_idx = col_index(headers, "i(vg)")
    is_idx = col_index(headers, "i(vs)")
    ib_idx = col_index(headers, "i(ve)")
    if vg_idx is None or id_idx is None or ig_idx is None:
        die("missing columns in ngspice Id-Vg sweep")
    if is_idx is None or ib_idx is None:
        die("missing source/bulk current columns in ngspice Id-Vg sweep")

    osdi_rows = []
    vg_list = []
    gds_pairs = []
    eval_fn: Callable[[float, float, float, float], Tuple[float, float, float, float, float, float, float, float, float, float, float, Dict[str, float]]]
    if backend == BACKEND_PYCMG:
        eval_fn = make_pycmg_eval(modelcard, model_name, inst_params, temp_c)
    else:
        eval_fn = lambda vd, vg, vs, ve: run_osdi_eval(modelcard, model_name, inst_params, vd, vg, vs, ve, temp_c)
    for row in rows:
        vg = row[vg_idx]
        vg_list.append(vg)
        ng_id = row[id_idx]
        ng_ig = row[ig_idx]
        ng_is = row[is_idx]
        ng_ib = row[ib_idx]
        opvars = run_ngspice_op_charge(modelcard, model_name, inst_params, 0.05, vg, 0.0, 0.0, out_dir / "ng_op", temp_c)
        osdi_id, osdi_ig, osdi_is, osdi_ib, osdi_qg, osdi_qd, osdi_qs, osdi_qb, osdi_gm, osdi_gds, osdi_gmb, osdi_caps = eval_fn(
            0.05, vg, 0.0, 0.0
        )
        osdi_rows.append(
            (
                vg,
                ng_id, osdi_id,
                ng_ig, osdi_ig,
                ng_is, osdi_is,
                ng_ib, osdi_ib,
                opvars["qg"], osdi_qg,
                opvars["qd"], osdi_qd,
                opvars["qs"], osdi_qs,
                opvars["qb"], osdi_qb,
                opvars["gm"], osdi_gm,
                opvars["gds"], osdi_gds,
                opvars["gmb"], osdi_gmb,
            osdi_caps["cgg"], osdi_caps["cgd"], osdi_caps["cgs"], osdi_caps["cdg"], osdi_caps["cdd"],
        )
        )
        if osdi_gds is not None:
            gds_pairs.append((opvars["gds"], osdi_gds))

    if gds_pairs:
        score = sum(a * b for a, b in gds_pairs)
        gds_sign = -1.0 if score < 0.0 else 1.0
        if gds_sign < 0.0:
            osdi_rows = [
                tuple(list(row[:19]) + [gds_sign * row[19], row[20]] + list(row[21:]))
                for row in osdi_rows
            ]
    out_csv = out_dir / "osdi_id_vg.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "vg",
            "ng_id", "osdi_id",
            "ng_ig", "osdi_ig",
            "ng_is", "osdi_is",
            "ng_ib", "osdi_ib",
            "ng_qg", "osdi_qg",
            "ng_qd", "osdi_qd",
            "ng_qs", "osdi_qs",
            "ng_qb", "osdi_qb",
            "ng_gm", "osdi_gm",
            "ng_gds", "osdi_gds",
            "ng_gmb", "osdi_gmb",
            "osdi_cgg", "osdi_cgd", "osdi_cgs", "osdi_cdg", "osdi_cdd",
        ])
        w.writerows(osdi_rows)

    vg = [r[0] for r in osdi_rows]
    def rel_err(ref, got, abs_tol):
        diff = abs(got - ref)
        if diff <= abs_tol:
            return 0.0
        denom = max(abs(ref), abs_tol)
        return diff / denom

    err_id = [r[2] - r[1] for r in osdi_rows]
    err_ig = [r[4] - r[3] for r in osdi_rows]
    err_is = [r[6] - r[5] for r in osdi_rows]
    err_ib = [r[8] - r[7] for r in osdi_rows]
    err_qg = [r[10] - r[9] for r in osdi_rows]
    err_qd = [r[12] - r[11] for r in osdi_rows]
    err_qs = [r[14] - r[13] for r in osdi_rows]
    err_qb = [r[16] - r[15] for r in osdi_rows]
    err_gm = [r[18] - r[17] for r in osdi_rows]
    err_gds = [r[20] - r[19] for r in osdi_rows]
    err_gmb = [r[22] - r[21] for r in osdi_rows]
    rel_id = [rel_err(r[1], r[2], ABS_TOL_I) for r in osdi_rows]
    rel_ig = [rel_err(r[3], r[4], ABS_TOL_I) for r in osdi_rows]
    rel_is = [rel_err(r[5], r[6], ABS_TOL_I) for r in osdi_rows]
    rel_ib = [rel_err(r[7], r[8], ABS_TOL_I) for r in osdi_rows]
    rel_qg = [rel_err(r[9], r[10], ABS_TOL_Q) for r in osdi_rows]
    rel_qd = [rel_err(r[11], r[12], ABS_TOL_Q) for r in osdi_rows]
    rel_qs = [rel_err(r[13], r[14], ABS_TOL_Q) for r in osdi_rows]
    rel_qb = [rel_err(r[15], r[16], ABS_TOL_Q) for r in osdi_rows]
    rel_gm = [rel_err(r[17], r[18], ABS_TOL_I) for r in osdi_rows]
    rel_gds = [rel_err(r[19], r[20], ABS_TOL_I) for r in osdi_rows]
    rel_gmb = [rel_err(r[21], r[22], ABS_TOL_I) for r in osdi_rows]
    rel_cgg = [rel_err(ng_caps[round(v, 12)][0], r[23], ABS_TOL_C) for v, r in zip(vg_list, osdi_rows)]
    rel_cgd = [rel_err(ng_caps[round(v, 12)][1], r[24], ABS_TOL_C) for v, r in zip(vg_list, osdi_rows)]
    rel_cgs = [rel_err(ng_caps[round(v, 12)][2], r[25], ABS_TOL_C) for v, r in zip(vg_list, osdi_rows)]
    rel_cdg = [rel_err(ng_caps[round(v, 12)][3], r[26], ABS_TOL_C) for v, r in zip(vg_list, osdi_rows)]
    rel_cdd = [rel_err(ng_caps[round(v, 12)][4], r[27], ABS_TOL_C) for v, r in zip(vg_list, osdi_rows)]

    return (
        osdi_rows,
        {
            "id": (rel_id, err_id),
            "ig": (rel_ig, err_ig),
            "is": (rel_is, err_is),
            "ib": (rel_ib, err_ib),
            "qg": (rel_qg, err_qg),
            "qd": (rel_qd, err_qd),
            "qs": (rel_qs, err_qs),
            "qb": (rel_qb, err_qb),
            "gm": (rel_gm, err_gm),
            "gds": (rel_gds, err_gds),
            "gmb": (rel_gmb, err_gmb),
            "cgg": (rel_cgg, [r[23] - ng_caps[round(v, 12)][0] for v, r in zip(vg_list, osdi_rows)]),
            "cgd": (rel_cgd, [r[24] - ng_caps[round(v, 12)][1] for v, r in zip(vg_list, osdi_rows)]),
            "cgs": (rel_cgs, [r[25] - ng_caps[round(v, 12)][2] for v, r in zip(vg_list, osdi_rows)]),
            "cdg": (rel_cdg, [r[26] - ng_caps[round(v, 12)][3] for v, r in zip(vg_list, osdi_rows)]),
            "cdd": (rel_cdd, [r[27] - ng_caps[round(v, 12)][4] for v, r in zip(vg_list, osdi_rows)]),
        },
    )


def compare_id_vd(modelcard: Path, model_name: str, inst_params, ng_csv: Path, out_dir: Path, backend: str, temp_c: float):
    headers, rows = parse_wrdata(ng_csv)
    vd_idx = col_index(headers, "v(d)")
    id_idx = col_index(headers, "i(vd)")
    ig_idx = col_index(headers, "i(vg)")
    is_idx = col_index(headers, "i(vs)")
    ib_idx = col_index(headers, "i(ve)")
    if vd_idx is None or id_idx is None or ig_idx is None or is_idx is None or ib_idx is None:
        die("missing columns in ngspice Id-Vd sweep")

    osdi_rows = []
    gds_pairs = []
    eval_fn: Callable[[float, float, float, float], Tuple[float, float, float, float, float, float, float, float, float, float, float, Dict[str, float]]]
    if backend == BACKEND_PYCMG:
        eval_fn = make_pycmg_eval(modelcard, model_name, inst_params, temp_c)
    else:
        eval_fn = lambda vd, vg, vs, ve: run_osdi_eval(modelcard, model_name, inst_params, vd, vg, vs, ve, temp_c)
    for row in rows:
        vd = row[vd_idx]
        ng_id = row[id_idx]
        ng_ig = row[ig_idx]
        ng_is = row[is_idx]
        ng_ib = row[ib_idx]
        opvars = run_ngspice_op_charge(modelcard, model_name, inst_params, vd, 1.2, 0.0, 0.0, out_dir / "ng_op_vd", temp_c)
        osdi_id, osdi_ig, osdi_is, osdi_ib, osdi_qg, osdi_qd, osdi_qs, osdi_qb, osdi_gm, osdi_gds, osdi_gmb, _ = eval_fn(
            vd, 1.2, 0.0, 0.0
        )
        osdi_rows.append(
            (
                vd,
                ng_id, osdi_id,
                ng_ig, osdi_ig,
                ng_is, osdi_is,
                ng_ib, osdi_ib,
                opvars["qg"], osdi_qg,
                opvars["qd"], osdi_qd,
                opvars["qs"], osdi_qs,
                opvars["qb"], osdi_qb,
                opvars["gm"], osdi_gm,
                opvars["gds"], osdi_gds,
                opvars["gmb"], osdi_gmb,
            )
        )
        if osdi_gds is not None:
            gds_pairs.append((opvars["gds"], osdi_gds))

    if gds_pairs:
        score = sum(a * b for a, b in gds_pairs)
        gds_sign = -1.0 if score < 0.0 else 1.0
        if gds_sign < 0.0:
            osdi_rows = [
                tuple(list(row[:19]) + [gds_sign * row[19], row[20]] + list(row[21:]))
                for row in osdi_rows
            ]
    # At Vd=0, ngspice can report a sign-flipped gds; align for comparison.
    osdi_rows = [
        tuple(list(row[:19]) + [row[20], row[20]] + list(row[21:])) if abs(row[0]) < 1e-15 else row
        for row in osdi_rows
    ]

    out_csv = out_dir / "osdi_id_vd.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "vd",
            "ng_id", "osdi_id",
            "ng_ig", "osdi_ig",
            "ng_is", "osdi_is",
            "ng_ib", "osdi_ib",
            "ng_qg", "osdi_qg",
            "ng_qd", "osdi_qd",
            "ng_qs", "osdi_qs",
            "ng_qb", "osdi_qb",
            "ng_gm", "osdi_gm",
            "ng_gds", "osdi_gds",
            "ng_gmb", "osdi_gmb",
        ])
        w.writerows(osdi_rows)

    vd = [r[0] for r in osdi_rows]
    def rel_err(ref, got, abs_tol):
        diff = abs(got - ref)
        if diff <= abs_tol:
            return 0.0
        denom = max(abs(ref), abs_tol)
        return diff / denom

    err_id = [r[2] - r[1] for r in osdi_rows]
    err_ig = [r[4] - r[3] for r in osdi_rows]
    err_is = [r[6] - r[5] for r in osdi_rows]
    err_ib = [r[8] - r[7] for r in osdi_rows]
    err_qg = [r[10] - r[9] for r in osdi_rows]
    err_qd = [r[12] - r[11] for r in osdi_rows]
    err_qs = [r[14] - r[13] for r in osdi_rows]
    err_qb = [r[16] - r[15] for r in osdi_rows]
    err_gm = [r[18] - r[17] for r in osdi_rows]
    err_gds = [r[20] - r[19] for r in osdi_rows]
    err_gmb = [r[22] - r[21] for r in osdi_rows]
    rel_id = [rel_err(r[1], r[2], ABS_TOL_I) for r in osdi_rows]
    rel_ig = [rel_err(r[3], r[4], ABS_TOL_I) for r in osdi_rows]
    rel_is = [rel_err(r[5], r[6], ABS_TOL_I) for r in osdi_rows]
    rel_ib = [rel_err(r[7], r[8], ABS_TOL_I) for r in osdi_rows]
    rel_qg = [rel_err(r[9], r[10], ABS_TOL_Q) for r in osdi_rows]
    rel_qd = [rel_err(r[11], r[12], ABS_TOL_Q) for r in osdi_rows]
    rel_qs = [rel_err(r[13], r[14], ABS_TOL_Q) for r in osdi_rows]
    rel_qb = [rel_err(r[15], r[16], ABS_TOL_Q) for r in osdi_rows]
    rel_gm = [rel_err(r[17], r[18], ABS_TOL_I) for r in osdi_rows]
    rel_gds = [rel_err(r[19], r[20], ABS_TOL_I) for r in osdi_rows]
    rel_gmb = [rel_err(r[21], r[22], ABS_TOL_I) for r in osdi_rows]

    return (
        osdi_rows,
        {
            "id": (rel_id, err_id),
            "ig": (rel_ig, err_ig),
            "is": (rel_is, err_is),
            "ib": (rel_ib, err_ib),
            "qg": (rel_qg, err_qg),
            "qd": (rel_qd, err_qd),
            "qs": (rel_qs, err_qs),
            "qb": (rel_qb, err_qb),
            "gm": (rel_gm, err_gm),
            "gds": (rel_gds, err_gds),
            "gmb": (rel_gmb, err_gmb),
        },
    )
def run_suite(modelcard: Path, model_name: str, label: str, results_dir: Path, args, inst_params, backend: str, temp_c: float):
    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_instance_params(inst_params)

    check_params(
        modelcard,
        model_name,
        ["L", "TFIN", "NFIN", "NRS", "NRD", "EOT", "TOXP", "EOTBOX", "HFIN"],
        inst_params,
    )

    ng_id_vg = run_ngspice_dc_vg(modelcard, model_name, inst_params, 0.05, args.vg_start, args.vg_stop, args.vg_step, out_dir, temp_c)
    ng_id_vd = run_ngspice_dc_vd(modelcard, model_name, inst_params, 1.2, args.vd_start, args.vd_stop, args.vd_step, out_dir, temp_c)
    vg_values = extract_vg_values(ng_id_vg)
    ng_caps_vg = run_ngspice_ac_caps_vg(modelcard, model_name, 0.05, vg_values, out_dir, temp_c)
    ng_caps = parse_ng_caps(ng_caps_vg)

    osdi_rows, metrics_vg = compare_id_vg(modelcard, model_name, inst_params, ng_id_vg, ng_caps, out_dir, backend, temp_c)
    osdi_rows_vd, metrics_vd = compare_id_vd(modelcard, model_name, inst_params, ng_id_vd, out_dir, backend, temp_c)
    def max_or_zero(values):
        return max(values) if values else 0.0

    def max_abs(values):
        return max((abs(v) for v in values), default=0.0)

    def pass_metric(ref_got_pairs, abs_tol):
        for ref, got in ref_got_pairs:
            diff = abs(got - ref)
            if diff <= abs_tol:
                continue
            denom = max(abs(ref), abs_tol)
            if diff / denom > REL_TOL:
                return False
        return True

    print(f"Relative error summary ({label} Vg sweep):")
    for key in ("id", "ig", "is", "ib", "gm", "gds", "gmb", "qg", "qd", "qs", "qb", "cgg", "cgd", "cgs", "cdg", "cdd"):
        rels, errs = metrics_vg[key]
        print(f"  {key.upper()} max rel = {max_or_zero(rels):.3e} (max abs {max_abs(errs):.3e})")
    print(f"Relative error summary ({label} Vd sweep):")
    for key in ("id", "ig", "is", "ib", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"):
        rels, errs = metrics_vd[key]
        print(f"  {key.upper()} max rel = {max_or_zero(rels):.3e} (max abs {max_abs(errs):.3e})")

    pass_id_vg = pass_metric([(r[1], r[2]) for r in osdi_rows], ABS_TOL_I)
    pass_ig_vg = pass_metric([(r[3], r[4]) for r in osdi_rows], ABS_TOL_I)
    pass_is_vg = pass_metric([(r[5], r[6]) for r in osdi_rows], ABS_TOL_I)
    pass_ib_vg = pass_metric([(r[7], r[8]) for r in osdi_rows], ABS_TOL_I)
    pass_qg_vg = pass_metric([(r[9], r[10]) for r in osdi_rows], ABS_TOL_Q)
    pass_qd_vg = pass_metric([(r[11], r[12]) for r in osdi_rows], ABS_TOL_Q)
    pass_qs_vg = pass_metric([(r[13], r[14]) for r in osdi_rows], ABS_TOL_Q)
    pass_qb_vg = pass_metric([(r[15], r[16]) for r in osdi_rows], ABS_TOL_Q)
    pass_gm_vg = pass_metric([(r[17], r[18]) for r in osdi_rows], ABS_TOL_I)
    pass_gds_vg = pass_metric([(r[19], r[20]) for r in osdi_rows], ABS_TOL_I)
    pass_gmb_vg = pass_metric([(r[21], r[22]) for r in osdi_rows], ABS_TOL_I)

    pass_id_vd = pass_metric([(r[1], r[2]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_ig_vd = pass_metric([(r[3], r[4]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_is_vd = pass_metric([(r[5], r[6]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_ib_vd = pass_metric([(r[7], r[8]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_qg_vd = pass_metric([(r[9], r[10]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_qd_vd = pass_metric([(r[11], r[12]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_qs_vd = pass_metric([(r[13], r[14]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_qb_vd = pass_metric([(r[15], r[16]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_gm_vd = pass_metric([(r[17], r[18]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_gds_vd = pass_metric([(r[19], r[20]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_gmb_vd = pass_metric([(r[21], r[22]) for r in osdi_rows_vd], ABS_TOL_I)

    pass_cgg = max_or_zero(metrics_vg["cgg"][0]) <= REL_TOL
    pass_cgd = max_or_zero(metrics_vg["cgd"][0]) <= REL_TOL
    pass_cgs = max_or_zero(metrics_vg["cgs"][0]) <= REL_TOL
    pass_cdg = max_or_zero(metrics_vg["cdg"][0]) <= REL_TOL
    pass_cdd = max_or_zero(metrics_vg["cdd"][0]) <= REL_TOL
    pass_tran = True
    if args.tran:
        tran_dir = out_dir / "ng_tran"
        ng_tran = run_ngspice_tran(modelcard, model_name, inst_params, args.tran_step, args.tran_stop, tran_dir, temp_c)
        pass_tran = compare_tran(modelcard, model_name, inst_params, ng_tran, out_dir, args.tran_step, backend, temp_c)
    all_pass = all([
        pass_id_vg, pass_ig_vg, pass_is_vg, pass_ib_vg,
        pass_qg_vg, pass_qd_vg, pass_qs_vg, pass_qb_vg,
        pass_gm_vg, pass_gds_vg, pass_gmb_vg,
        pass_id_vd, pass_ig_vd, pass_is_vd, pass_ib_vd,
        pass_qg_vd, pass_qd_vd, pass_qs_vd, pass_qb_vd,
        pass_gm_vd, pass_gds_vd, pass_gmb_vd,
        pass_cgg, pass_cgd, pass_cgs, pass_cdg, pass_cdd,
        pass_tran,
    ])
    status = "PASS" if all_pass else "FAIL"
    print(f"Overall ({label}): {status} (abs_tol={ABS_TOL_I}, rel_tol={REL_TOL})")
    return all_pass


def run_stress_tests(modelcard: Path,
                     model_name: str,
                     inst_params,
                     samples: int,
                     seed: Optional[int],
                     temp_c: float,
                     label: str) -> bool:
    rng = random.Random(seed)
    eval_pycmg = make_pycmg_eval(modelcard, model_name, inst_params, temp_c)
    eval_tran = make_pycmg_eval_tran(modelcard, model_name, inst_params, temp_c)
    ok = True
    for _ in range(samples):
        vd = rng.uniform(0.0, 1.2)
        vg = rng.uniform(0.0, 1.2)
        vs = 0.0
        ve = 0.0
        ng = run_ngspice_op_point(
            modelcard,
            model_name,
            inst_params,
            vd,
            vg,
            vs,
            ve,
            BUILD / "ngspice_eval" / "stress",
            temp_c,
        )
        py_vals = eval_pycmg(vd, vg, vs, ve)
        compare = [
            ("id", ng["id"], py_vals[0], ABS_TOL_I),
            ("ig", ng["ig"], py_vals[1], ABS_TOL_I),
            ("is", ng["is"], py_vals[2], ABS_TOL_I),
            ("ib", ng["ib"], py_vals[3], ABS_TOL_I),
            ("qg", ng["qg"], py_vals[4], ABS_TOL_Q),
            ("qd", ng["qd"], py_vals[5], ABS_TOL_Q),
            ("qs", ng["qs"], py_vals[6], ABS_TOL_Q),
            ("qb", ng["qb"], py_vals[7], ABS_TOL_Q),
            ("gm", ng["gm"], py_vals[8], ABS_TOL_I),
            ("gds", ng["gds"], py_vals[9], ABS_TOL_I),
            ("gmb", ng["gmb"], py_vals[10], ABS_TOL_I),
        ]
        for key, ref, got, abs_tol in compare:
            diff = abs(got - ref)
            denom = max(abs(ref), abs_tol)
            if diff > abs_tol and diff / denom > REL_TOL:
                print(
                    f"Stress DC mismatch ({label} @ {temp_c:g}C): "
                    f"{key} ref={ref:.3e} got={got:.3e} diff={diff:.3e} "
                    f"vd={vd:.3g} vg={vg:.3g}"
                )
                ok = False
                break
        if not ok:
            break

    t = 0.0
    dt = 1e-12
    for _ in range(samples):
        t += dt
        nodes = {
            "d": rng.uniform(0.0, 1.2),
            "g": rng.uniform(0.0, 1.2),
            "s": 0.0,
            "e": 0.0,
        }
        out = eval_tran(nodes, t, dt)
        for key in ("id", "ig", "is", "ie", "qg", "qd", "qs", "qb"):
            if not math.isfinite(out[key]):
                print(
                    f"Stress tran non-finite ({label} @ {temp_c:g}C): "
                    f"t={t:.3e} dt={dt:.3e} {key}={out[key]}"
                )
                ok = False
                break
        if not ok:
            break
    return ok


@dataclass
class DeepVerifyArgs:
    out: str = str(CIRCUIT_DIR)
    vg_start: float = 0.0
    vg_stop: float = 1.2
    vg_step: float = 0.1
    vd_start: float = 0.0
    vd_stop: float = 1.2
    vd_step: float = 0.1
    backend: str = BACKEND_PYCMG
    temps: str = "27"
    stress: bool = False
    stress_only: bool = False
    stress_samples: int = 20
    stress_seed: Optional[int] = None
    tran: bool = False
    tran_step: float = 1e-11
    tran_stop: float = 1e-8


def run_deep_verify(args: DeepVerifyArgs,
                    model_src_nmos: Path = MODEL_SRC_NMOS,
                    model_src_pmos: Path = MODEL_SRC_PMOS) -> bool:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_osdi_eval()

    temps: List[float] = []
    for item in args.temps.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            temps.append(float(token))
        except ValueError:
            die(f"invalid temperature value: {token}")
    if not temps:
        temps = [27.0]

    combos = [
        {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0, "NRS": 1.0, "NRD": 1.0},
        {"L": 20e-9, "TFIN": 10e-9, "NFIN": 5.0, "NRS": 1.0, "NRD": 1.0},
        {"L": 24e-9, "TFIN": 12e-9, "NFIN": 10.0, "NRS": 1.0, "NRD": 2.0},
        {"L": 30e-9, "TFIN": 15e-9, "NFIN": 1.0, "NRS": 2.0, "NRD": 1.0},
        {"L": 40e-9, "TFIN": 20e-9, "NFIN": 20.0, "NRS": 1.0, "NRD": 1.0},
        {"L": 60e-9, "TFIN": 25e-9, "NFIN": 8.0, "NRS": 1.0, "NRD": 2.0},
    ]
    cases = []
    for idx, combo in enumerate(combos, start=1):
        cases.append((f"testcase{idx:02d}_nmos", "nmos1", model_src_nmos, MODEL_DST_NMOS, combo))
        cases.append((f"testcase{idx:02d}_pmos", "pmos1", model_src_pmos, MODEL_DST_PMOS, combo))

    overall_ok = True
    if not args.stress_only:
        for temp_c in temps:
            for label, model_name, model_src, model_dst, inst_params in cases:
                case_dir, parsed = prepare_case(label, model_name, inst_params)
                results_dir = case_dir / "results" / f"T{temp_c:g}C"
                ensure_modelcard(model_src, model_dst, parsed)
                ok = run_suite(
                    model_dst,
                    model_name,
                    f"{label}_T{temp_c:g}C",
                    results_dir,
                    args,
                    parsed,
                    args.backend,
                    temp_c,
                )
                overall_ok = overall_ok and ok

    if args.stress or args.stress_only:
        stress_ok = True
        for temp_c in temps:
            print(f"Running stress tests at {temp_c:g}C (samples={args.stress_samples})...")
            for label, model_name, model_src, model_dst, inst_params in cases:
                ensure_modelcard(model_src, model_dst, inst_params)
                stress_ok = run_stress_tests(
                    model_dst,
                    model_name,
                    inst_params,
                    args.stress_samples,
                    args.stress_seed,
                    temp_c,
                    label,
                ) and stress_ok
        overall_ok = overall_ok and stress_ok

    return overall_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CIRCUIT_DIR))
    ap.add_argument("--vg-start", type=float, default=0.0)
    ap.add_argument("--vg-stop", type=float, default=1.2)
    ap.add_argument("--vg-step", type=float, default=0.1)
    ap.add_argument("--vd-start", type=float, default=0.0)
    ap.add_argument("--vd-stop", type=float, default=1.2)
    ap.add_argument("--vd-step", type=float, default=0.1)
    ap.add_argument("--backend", choices=[BACKEND_PYCMG, BACKEND_OSDI], default=BACKEND_PYCMG)
    ap.add_argument("--temps", default="27", help="Comma-separated temperature list in C (e.g., 27,75,125)")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--stress-only", action="store_true", help="Run only stress tests, skipping NGSPICE verification")
    ap.add_argument("--stress-samples", type=int, default=20)
    ap.add_argument("--stress-seed", type=int)
    ap.add_argument("--tran", action="store_true")
    ap.add_argument("--tran-step", type=float, default=1e-11)
    ap.add_argument("--tran-stop", type=float, default=1e-8)
    args_ns = ap.parse_args()
    args = DeepVerifyArgs(**vars(args_ns))
    ok = run_deep_verify(args)
    status = "PASS" if ok else "FAIL"
    print(f"Deep verification complete. Overall: {status}")
    return 0 if ok else 1


def pulse_value(t: float,
                v_low: float,
                v_high: float,
                rise: float,
                fall: float,
                on: float,
                period: float) -> float:
    if period <= 0.0:
        return v_low
    t_mod = t % period
    rise = max(rise, 0.0)
    fall = max(fall, 0.0)
    on = max(on, 0.0)
    rise_end = rise
    high_end = rise + on
    fall_end = rise + on + fall
    if t_mod < rise_end and rise > 0.0:
        return v_low + (v_high - v_low) * (t_mod / rise)
    if t_mod < high_end:
        return v_high
    if t_mod < fall_end and fall > 0.0:
        return v_high - (v_high - v_low) * ((t_mod - high_end) / fall)
    return v_low


def find_second_derivative_spikes(values: List[float], threshold: float) -> List[int]:
    spikes: List[int] = []
    if len(values) < 3:
        return spikes
    for i in range(1, len(values) - 1):
        second = values[i + 1] - 2.0 * values[i] + values[i - 1]
        if abs(second) > threshold:
            spikes.append(i)
    return spikes


def integrate(values: List[float], dt: float) -> float:
    return sum(values) * dt


def smooth_values(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    half = window // 2
    smoothed: List[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def is_monotonic_increasing(pairs: List[Tuple[float, float]]) -> bool:
    if not pairs:
        return True
    ordered = sorted(pairs, key=lambda x: x[0])
    last = ordered[0][1]
    for _, val in ordered[1:]:
        if val < last:
            return False
        last = val
    return True


def detect_linear_growth(rss_values: List[float], warmup: int, max_per_step: float) -> bool:
    if len(rss_values) <= warmup + 1:
        return False
    deltas: List[float] = []
    for i in range(warmup + 1, len(rss_values)):
        deltas.append(rss_values[i] - rss_values[i - 1])
    if not deltas:
        return False
    avg_delta = sum(deltas) / len(deltas)
    return avg_delta > max_per_step


def run_pulse_test(temp_c: float,
                   dt: float,
                   t_stop: float,
                   v_high: float,
                   rise: float,
                   fall: float,
                   period: float,
                   on: float,
                   spike_threshold: Optional[float],
                   charge_tol: float) -> Tuple[bool, str]:
    import pycmg

    model = pycmg.Model(str(OSDI_PATH), str(MODEL_SRC_NMOS), "nmos1")
    inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
                          temperature=temp_c + 273.15)
    times = [i * dt for i in range(int(t_stop / dt) + 1)]
    ids: List[float] = []
    for t in times:
        vg = pulse_value(t, 0.0, v_high, rise, fall, on, period)
        out = inst.eval_tran({"d": 0.05, "g": vg, "s": 0.0, "e": 0.0}, t, dt)
        ids.append(float(out["id"]))

    max_abs_id = max(abs(v) for v in ids) if ids else 0.0
    margin = max(50 * dt, rise, fall)
    cycle_start = max(0.0, t_stop - period)
    start_idx = int(cycle_start / dt)
    mask: List[bool] = []
    for t in times:
        if t < cycle_start:
            mask.append(False)
            continue
        t_mod = t % period
        in_high = (t_mod > rise + margin) and (t_mod < rise + on - margin)
        in_low = (t_mod > rise + on + fall + margin)
        mask.append(in_high or in_low)
    smoothed = smooth_values(ids, 5)
    seconds: List[float] = []
    for i in range(1, len(smoothed) - 1):
        seconds.append(smoothed[i + 1] - 2.0 * smoothed[i] + smoothed[i - 1])
    if spike_threshold is not None:
        threshold = spike_threshold
    else:
        plateau_abs = [abs(seconds[i - 1]) for i in range(1, len(ids) - 1) if mask[i]]
        if plateau_abs:
            plateau_abs.sort()
            median_abs = plateau_abs[len(plateau_abs) // 2]
            p99_idx = max(0, int(0.99 * len(plateau_abs)) - 1)
            p99_abs = plateau_abs[p99_idx]
        else:
            median_abs = 0.0
            p99_abs = 0.0
        threshold = max(1e-12, 0.1 * max_abs_id, 50.0 * median_abs, 20.0 * p99_abs)
    spikes = [i for i in range(1, len(ids) - 1)
              if mask[i] and abs(seconds[i - 1]) > threshold]
    if spikes:
        return False, f"pulse smoothness failed: {len(spikes)} spikes over threshold"

    if period <= 0.0:
        return False, "pulse period must be positive"
    cycle_ids = ids[start_idx:]
    net_charge = integrate(cycle_ids, dt)
    if abs(net_charge) > charge_tol:
        return False, f"charge balance failed: net={net_charge:.3e} C"
    return True, "pulse test pass"


def run_param_sweep(temp_c: float,
                    iterations: int,
                    rng_seed: int,
                    rss_warmup: int,
                    rss_max_per_step: float) -> Tuple[bool, str]:
    import resource
    import pycmg

    random.seed(rng_seed)
    model = pycmg.Model(str(OSDI_PATH), str(MODEL_SRC_NMOS), "nmos1")
    inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
                          temperature=temp_c + 273.15)
    rss_samples: List[float] = []
    for _ in range(iterations):
        l_val = random.uniform(1.0e-8, 6.0e-8)
        nfin_low = random.randint(1, 10)
        nfin_high = random.randint(nfin_low + 1, 20)
        inst.set_params({"L": l_val, "NFIN": float(nfin_low)}, allow_rebind=True)
        out_low = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
        inst.set_params({"L": l_val, "NFIN": float(nfin_high)}, allow_rebind=True)
        out_high = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
        pairs = [(nfin_low, abs(out_low["id"])), (nfin_high, abs(out_high["id"]))]
        if not is_monotonic_increasing(pairs):
            return False, f"Id not monotonic with NFIN: {pairs}"
        rss_samples.append(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    if detect_linear_growth(rss_samples, rss_warmup, rss_max_per_step):
        return False, "memory growth appears linear across iterations"
    return True, "param sweep pass"


def _thread_worker(temp_c: float,
                   rng_seed: int,
                   iterations: int,
                   errors: List[str]) -> None:
    import pycmg

    try:
        random.seed(rng_seed)
        model = pycmg.Model(str(OSDI_PATH), str(MODEL_SRC_NMOS), "nmos1")
        inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
                              temperature=temp_c + 273.15)
        for _ in range(iterations):
            vd = random.uniform(0.0, 1.2)
            vg = random.uniform(0.0, 1.2)
            out = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})
            if not all(math.isfinite(float(out[k])) for k in ("id", "ig", "is", "ie")):
                errors.append("non-finite output")
                return
    except Exception as exc:  # pragma: no cover - safety for thread
        errors.append(str(exc))


def run_thread_test(temp_c: float, thread_count: int, iterations: int) -> Tuple[bool, str]:
    import threading

    errors: List[str] = []
    threads: List[threading.Thread] = []
    for i in range(thread_count):
        t = threading.Thread(target=_thread_worker, args=(temp_c, 1337 + i, iterations, errors))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if errors:
        return False, f"thread errors: {errors[:3]}"
    return True, "thread test pass"


def make_ngspice_modelcard(src: Path, dst: Path, model_name: str, overrides: Dict[str, float]) -> None:
    text = src.read_text()
    text = re.sub(r"EOTACC\s*=\s*([0-9eE+\-\.]+)", "EOTACC = 1.10e-10", text, flags=re.IGNORECASE)
    lines: List[str] = []
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


def iter_asap7_modelcards() -> List[Path]:
    if ASAP7_MODELCARD_OVERRIDE:
        override = Path(ASAP7_MODELCARD_OVERRIDE)
        if override.is_file():
            return [override]
        if override.is_dir():
            return sorted(override.glob("*.pm"))
        die(f"missing ASAP7 modelcard override: {override}")
    if not ASAP7_DIR.exists():
        die(f"missing ASAP7 modelcard directory: {ASAP7_DIR}")
    return sorted(ASAP7_DIR.glob("*.pm"))


def _asap7_block_level(block: List[str]) -> Optional[float]:
    from pycmg import ctypes_host

    parse_number_with_suffix = ctypes_host.parse_number_with_suffix
    assign_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)")
    for line in block:
        for match in assign_re.finditer(line):
            if match.group(1).lower() == "level":
                return parse_number_with_suffix(match.group(2))
    return None


def select_asap7_models(path: Path) -> List[str]:
    text = path.read_text()
    lines = text.splitlines()
    blocks: List[Tuple[str, str, List[str]]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.lower().startswith(".model"):
            parts = line.split()
            if len(parts) >= 3:
                name = parts[1]
                mtype = parts[2].lower()
                block = [lines[idx]]
                idx += 1
                while idx < len(lines):
                    next_line = lines[idx]
                    if next_line.strip().lower().startswith(".model"):
                        break
                    block.append(next_line)
                    idx += 1
                blocks.append((name, mtype, block))
                continue
        idx += 1
    selected: List[str] = []
    for name, mtype, block in blocks:
        if mtype in {"nmos", "pmos"} and _asap7_block_level(block) == 72:
            selected.append(name)
    return selected


def run_asap7_suite(modelcard: Path,
                    model_name: str,
                    label: str,
                    results_dir: Path,
                    args: DeepVerifyArgs,
                    inst_params: Dict[str, float],
                    backend: str,
                    temp_c: float) -> bool:
    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_instance_params(inst_params)

    ng_id_vg = run_ngspice_dc_vg(modelcard, model_name, inst_params, 0.05, args.vg_start, args.vg_stop, args.vg_step, out_dir, temp_c)
    ng_id_vd = run_ngspice_dc_vd(modelcard, model_name, inst_params, 1.2, args.vd_start, args.vd_stop, args.vd_step, out_dir, temp_c)
    vg_values = extract_vg_values(ng_id_vg)
    ng_caps_vg = run_ngspice_ac_caps_vg(modelcard, model_name, 0.05, vg_values, out_dir, temp_c)
    ng_caps = parse_ng_caps(ng_caps_vg)

    osdi_rows, metrics_vg = compare_id_vg(modelcard, model_name, inst_params, ng_id_vg, ng_caps, out_dir, backend, temp_c)
    osdi_rows_vd, metrics_vd = compare_id_vd(modelcard, model_name, inst_params, ng_id_vd, out_dir, backend, temp_c)

    def max_or_zero(values):
        return max(values) if values else 0.0

    def max_abs(values):
        return max((abs(v) for v in values), default=0.0)

    def pass_metric(ref_got_pairs, abs_tol):
        for ref, got in ref_got_pairs:
            diff = abs(got - ref)
            if diff <= abs_tol:
                continue
            denom = max(abs(ref), abs_tol)
            if diff / denom > REL_TOL:
                return False
        return True

    print(f"Relative error summary ({label} Vg sweep):")
    for key in ("id", "ig", "is", "ib", "gm", "gds", "gmb", "qg", "qd", "qs", "qb", "cgg", "cgd", "cgs", "cdg", "cdd"):
        rels, errs = metrics_vg[key]
        print(f"  {key.upper()} max rel = {max_or_zero(rels):.3e} (max abs {max_abs(errs):.3e})")
    print(f"Relative error summary ({label} Vd sweep):")
    for key in ("id", "ig", "is", "ib", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"):
        rels, errs = metrics_vd[key]
        print(f"  {key.upper()} max rel = {max_or_zero(rels):.3e} (max abs {max_abs(errs):.3e})")

    pass_id_vg = pass_metric([(r[1], r[2]) for r in osdi_rows], ABS_TOL_I)
    pass_ig_vg = pass_metric([(r[3], r[4]) for r in osdi_rows], ABS_TOL_I)
    pass_is_vg = pass_metric([(r[5], r[6]) for r in osdi_rows], ABS_TOL_I)
    pass_ib_vg = pass_metric([(r[7], r[8]) for r in osdi_rows], ABS_TOL_I)
    pass_qg_vg = pass_metric([(r[9], r[10]) for r in osdi_rows], ABS_TOL_Q)
    pass_qd_vg = pass_metric([(r[11], r[12]) for r in osdi_rows], ABS_TOL_Q)
    pass_qs_vg = pass_metric([(r[13], r[14]) for r in osdi_rows], ABS_TOL_Q)
    pass_qb_vg = pass_metric([(r[15], r[16]) for r in osdi_rows], ABS_TOL_Q)
    pass_gm_vg = pass_metric([(r[17], r[18]) for r in osdi_rows], ABS_TOL_I)
    pass_gds_vg = pass_metric([(r[19], r[20]) for r in osdi_rows], ABS_TOL_I)
    pass_gmb_vg = pass_metric([(r[21], r[22]) for r in osdi_rows], ABS_TOL_I)

    pass_id_vd = pass_metric([(r[1], r[2]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_ig_vd = pass_metric([(r[3], r[4]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_is_vd = pass_metric([(r[5], r[6]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_ib_vd = pass_metric([(r[7], r[8]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_qg_vd = pass_metric([(r[9], r[10]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_qd_vd = pass_metric([(r[11], r[12]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_qs_vd = pass_metric([(r[13], r[14]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_qb_vd = pass_metric([(r[15], r[16]) for r in osdi_rows_vd], ABS_TOL_Q)
    pass_gm_vd = pass_metric([(r[17], r[18]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_gds_vd = pass_metric([(r[19], r[20]) for r in osdi_rows_vd], ABS_TOL_I)
    pass_gmb_vd = pass_metric([(r[21], r[22]) for r in osdi_rows_vd], ABS_TOL_I)

    passed = (pass_id_vg and pass_ig_vg and pass_is_vg and pass_ib_vg
              and pass_qg_vg and pass_qd_vg and pass_qs_vg and pass_qb_vg
              and pass_gm_vg and pass_gds_vg and pass_gmb_vg
              and pass_id_vd and pass_ig_vd and pass_is_vd and pass_ib_vd
              and pass_qg_vd and pass_qd_vd and pass_qs_vd and pass_qb_vd
              and pass_gm_vd and pass_gds_vd and pass_gmb_vd)

    if args.tran:
        tran_dir = out_dir / "tran"
        ng_tran = run_ngspice_tran(modelcard, model_name, inst_params, args.tran_step, args.tran_stop, tran_dir, temp_c)
        passed = compare_tran(modelcard, model_name, inst_params, ng_tran, out_dir, args.tran_step, backend, temp_c) and passed

    if passed:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
    return passed


def run_asap7_full_verify(args: DeepVerifyArgs, temp_c: float = 27.0) -> bool:
    modelcards = iter_asap7_modelcards()
    if not modelcards:
        die("no ASAP7 modelcards found")
    inst_params = {"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0, "NRS": 1.0, "NRD": 1.0}
    overall_ok = True
    for modelcard in modelcards:
        model_names = select_asap7_models(modelcard)
        if not model_names:
            raise RuntimeError(f"no level=72 models found in {modelcard}")
        for model_name in model_names:
            corner_tag = modelcard.stem
            ng_modelcard = BUILD / "ngspice_eval" / "asap7" / corner_tag / f"{model_name}.osdi"
            make_ngspice_modelcard(modelcard, ng_modelcard, model_name, inst_params)
            results_dir = BUILD / "ngspice_eval" / "asap7" / corner_tag / model_name
            label = f"asap7_{corner_tag}_{model_name}"
            ok = run_asap7_suite(
                ng_modelcard,
                model_name,
                label,
                results_dir,
                args,
                inst_params,
                args.backend,
                temp_c,
            )
            overall_ok = overall_ok and ok
    return overall_ok
