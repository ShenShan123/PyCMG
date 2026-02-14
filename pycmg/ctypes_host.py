from __future__ import annotations
"""
PyCMG ctypes-based OSDI interface

TEMPERATURE UNITS:
==================
All temperature values in this module are in KELVIN (K).

- Internal OSDI model temperature: KELVIN
- Instance initialization temperature parameter: KELVIN
- User-facing temperature API: KELVIN

To convert from Celsius to Kelvin:
    temp_K = temp_C + 273.15

Example:
    # Room temperature (25°C) in Kelvin
    temp_K = 25.0 + 273.15  # = 298.15 K

    inst = Instance(model, params={"L": 16e-9}, temperature=298.15)

Common temperatures:
    -40°C  →  233.15 K  (cold start)
     0°C  →  273.15 K  (freezing point)
    25°C  →  298.15 K  (room temperature)
    27°C  →  300.15 K  (TSMC typical)
    85°C  →  358.15 K  (operating hot)
   125°C  →  398.15 K  (max junction)
"""


import ctypes
import ctypes.util
import math
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# OSDI constants (osdi_0_3.h)
PARA_TY_MASK = 3
PARA_TY_REAL = 0
PARA_TY_INT = 1
PARA_TY_STR = 2
PARA_KIND_MASK = 3 << 30
PARA_KIND_MODEL = 0 << 30
PARA_KIND_INST = 1 << 30
PARA_KIND_OPVAR = 2 << 30

ACCESS_FLAG_READ = 0
ACCESS_FLAG_SET = 1
ACCESS_FLAG_INSTANCE = 4

CALC_RESIST_RESIDUAL = 1
CALC_REACT_RESIDUAL = 2
CALC_RESIST_JACOBIAN = 4
CALC_REACT_JACOBIAN = 8
CALC_NOISE = 16
CALC_OP = 32
CALC_RESIST_LIM_RHS = 64
CALC_REACT_LIM_RHS = 128
ENABLE_LIM = 256
INIT_LIM = 512
ANALYSIS_NOISE = 1024
ANALYSIS_DC = 2048
ANALYSIS_AC = 4096
ANALYSIS_TRAN = 8192
ANALYSIS_IC = 16384
ANALYSIS_STATIC = 32768
ANALYSIS_NODESET = 65536

EVAL_RET_FLAG_LIM = 1
EVAL_RET_FLAG_FATAL = 2
EVAL_RET_FLAG_FINISH = 4
EVAL_RET_FLAG_STOP = 8

LOG_LVL_MASK = 7

INIT_ERR_OUT_OF_BOUNDS = 1

UINT32_MAX = 0xFFFFFFFF

_INSTANCE_NAME = ctypes.c_char_p(b"osdi_host")


class OsdiLimFunction(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("num_args", ctypes.c_uint32),
        ("func_ptr", ctypes.c_void_p),
    ]


class OsdiSimParas(ctypes.Structure):
    _fields_ = [
        ("names", ctypes.POINTER(ctypes.c_char_p)),
        ("vals", ctypes.POINTER(ctypes.c_double)),
        ("names_str", ctypes.POINTER(ctypes.c_char_p)),
        ("vals_str", ctypes.POINTER(ctypes.c_char_p)),
    ]


class OsdiSimInfo(ctypes.Structure):
    _fields_ = [
        ("paras", OsdiSimParas),
        ("abstime", ctypes.c_double),
        ("prev_solve", ctypes.POINTER(ctypes.c_double)),
        ("prev_state", ctypes.POINTER(ctypes.c_double)),
        ("next_state", ctypes.POINTER(ctypes.c_double)),
        ("flags", ctypes.c_uint32),
    ]


class OsdiInitErrorPayload(ctypes.Union):
    _fields_ = [("parameter_id", ctypes.c_uint32)]


class OsdiInitError(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint32),
        ("payload", OsdiInitErrorPayload),
    ]


class OsdiInitInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("num_errors", ctypes.c_uint32),
        ("errors", ctypes.POINTER(OsdiInitError)),
    ]


class OsdiNodePair(ctypes.Structure):
    _fields_ = [
        ("node_1", ctypes.c_uint32),
        ("node_2", ctypes.c_uint32),
    ]


class OsdiJacobianEntry(ctypes.Structure):
    _fields_ = [
        ("nodes", OsdiNodePair),
        ("react_ptr_off", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class OsdiNode(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("units", ctypes.c_char_p),
        ("residual_units", ctypes.c_char_p),
        ("resist_residual_off", ctypes.c_uint32),
        ("react_residual_off", ctypes.c_uint32),
        ("resist_limit_rhs_off", ctypes.c_uint32),
        ("react_limit_rhs_off", ctypes.c_uint32),
        ("is_flow", ctypes.c_bool),
    ]


class OsdiParamOpvar(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.POINTER(ctypes.c_char_p)),
        ("num_alias", ctypes.c_uint32),
        ("description", ctypes.c_char_p),
        ("units", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
    ]


class OsdiNoiseSource(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("nodes", OsdiNodePair),
    ]


ACCESS_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_uint32)
SETUP_MODEL_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.POINTER(OsdiSimParas),
                                    ctypes.POINTER(OsdiInitInfo))
SETUP_INSTANCE_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_double,
                                       ctypes.c_uint32, ctypes.POINTER(OsdiSimParas),
                                       ctypes.POINTER(OsdiInitInfo))
EVAL_FUNC = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.POINTER(OsdiSimInfo))
LOAD_NOISE_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_double, ctypes.POINTER(ctypes.c_double))
LOAD_RESIDUAL_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_double))
LOAD_SPICE_RHS_DC_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double))
LOAD_SPICE_RHS_TRAN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_double),
                                            ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_double)
LOAD_JACOBIAN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
LOAD_JACOBIAN_REACT_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_double)
LOAD_JACOBIAN_TRAN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_double)


class OsdiDescriptor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("num_nodes", ctypes.c_uint32),
        ("num_terminals", ctypes.c_uint32),
        ("nodes", ctypes.POINTER(OsdiNode)),
        ("num_jacobian_entries", ctypes.c_uint32),
        ("jacobian_entries", ctypes.POINTER(OsdiJacobianEntry)),
        ("num_collapsible", ctypes.c_uint32),
        ("collapsible", ctypes.POINTER(OsdiNodePair)),
        ("collapsed_offset", ctypes.c_uint32),
        ("noise_sources", ctypes.POINTER(OsdiNoiseSource)),
        ("num_noise_src", ctypes.c_uint32),
        ("num_params", ctypes.c_uint32),
        ("num_instance_params", ctypes.c_uint32),
        ("num_opvars", ctypes.c_uint32),
        ("param_opvar", ctypes.POINTER(OsdiParamOpvar)),
        ("node_mapping_offset", ctypes.c_uint32),
        ("jacobian_ptr_resist_offset", ctypes.c_uint32),
        ("num_states", ctypes.c_uint32),
        ("state_idx_off", ctypes.c_uint32),
        ("bound_step_offset", ctypes.c_uint32),
        ("instance_size", ctypes.c_uint32),
        ("model_size", ctypes.c_uint32),
        ("access", ACCESS_FUNC),
        ("setup_model", SETUP_MODEL_FUNC),
        ("setup_instance", SETUP_INSTANCE_FUNC),
        ("eval", EVAL_FUNC),
        ("load_noise", LOAD_NOISE_FUNC),
        ("load_residual_resist", LOAD_RESIDUAL_FUNC),
        ("load_residual_react", LOAD_RESIDUAL_FUNC),
        ("load_limit_rhs_resist", LOAD_RESIDUAL_FUNC),
        ("load_limit_rhs_react", LOAD_RESIDUAL_FUNC),
        ("load_spice_rhs_dc", LOAD_SPICE_RHS_DC_FUNC),
        ("load_spice_rhs_tran", LOAD_SPICE_RHS_TRAN_FUNC),
        ("load_jacobian_resist", LOAD_JACOBIAN_FUNC),
        ("load_jacobian_react", LOAD_JACOBIAN_REACT_FUNC),
        ("load_jacobian_tran", LOAD_JACOBIAN_TRAN_FUNC),
    ]


def _load_libc() -> Optional[ctypes.CDLL]:
    path = ctypes.util.find_library("c")
    if not path:
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_LIBC = _load_libc()


class AlignedBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.ptr = ctypes.c_void_p()
        self._buffer: Optional[ctypes.Array] = None
        self._use_libc = False
        if size <= 0:
            return
        alignment = ctypes.alignment(ctypes.c_longdouble)
        if _LIBC is not None and hasattr(_LIBC, "posix_memalign"):
            mem = ctypes.c_void_p()
            res = _LIBC.posix_memalign(ctypes.byref(mem), alignment, size)
            if res == 0 and mem.value is not None:
                ctypes.memset(mem, 0, size)
                self.ptr = mem
                self._use_libc = True
                return
        self._buffer = ctypes.create_string_buffer(size)
        self.ptr = ctypes.cast(self._buffer, ctypes.c_void_p)

    def close(self) -> None:
        if self._use_libc and self.ptr.value:
            if _LIBC is not None and hasattr(_LIBC, "free"):
                _LIBC.free(self.ptr)
            self.ptr = ctypes.c_void_p()
            self._use_libc = False
        self._buffer = None

    def __del__(self) -> None:
        self.close()


@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_bool, ctypes.POINTER(ctypes.c_bool),
                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
def _pnjlim(init: bool,
            check: ctypes.POINTER(ctypes.c_bool),
            vnew: float,
            vold: float,
            vt: float,
            vcrit: float) -> float:
    triggered = False
    if init:
        vnew = vcrit
        triggered = True
    elif vnew > vcrit and abs(vnew - vold) > 2.0 * vt:
        if vold > 0.0:
            arg = (vnew - vold) / vt
            if arg > 0.0:
                vnew = vold + vt * math.log(arg + 1.0)
            else:
                vnew = vcrit
        else:
            vnew = vt * math.log(vnew / vt)
        triggered = True
    if check:
        check[0] = triggered
    return vnew


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32)
def _osdi_log(handle: ctypes.c_void_p, msg: ctypes.c_char_p, lvl: int) -> None:
    instance = b"osdi"
    if handle:
        try:
            instance = ctypes.cast(handle, ctypes.c_char_p).value or instance
        except Exception:
            pass
    text = msg.decode("utf-8", errors="replace") if msg else ""
    level = lvl & LOG_LVL_MASK
    sys.stderr.write(f"osdi[{instance.decode('utf-8', errors='replace')}] lvl={level} {text}\n")


def parse_number_with_suffix(token: str) -> float:
    s = token.strip()
    scale = 1.0
    pos = None
    for i, ch in enumerate(s):
        if ch not in "+-0123456789.eE":
            pos = i
            break
    if pos is not None:
        suffix = s[pos:].lower()
        s = s[:pos]
        if suffix == "t":
            scale = 1e12
        elif suffix == "g":
            scale = 1e9
        elif suffix == "meg":
            scale = 1e6
        elif suffix == "k":
            scale = 1e3
        elif suffix == "m":
            scale = 1e-3
        elif suffix == "u":
            scale = 1e-6
        elif suffix == "n":
            scale = 1e-9
        elif suffix == "p":
            scale = 1e-12
        elif suffix == "f":
            scale = 1e-15
        elif suffix == "a":
            scale = 1e-18
        elif suffix == "z":
            scale = 1e-21
        elif suffix == "y":
            scale = 1e-24
    if not s or s in {"+", "-"}:
        return 0.0
    return float(s) * scale


@dataclass
class ParsedModel:
    name: str
    params: Dict[str, float]


def _to_lower(s: str) -> str:
    return s.lower()


def parse_modelcard(path: str, target_model_name: Optional[str] = None) -> ParsedModel:
    assign_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)")
    target_lower = _to_lower(target_model_name) if target_model_name else None

    def _parse_params(lines: List[str]) -> Dict[str, float]:
        parsed_params: Dict[str, float] = {}
        for line in lines:
            for match in assign_re.finditer(line):
                key = match.group(1)
                val = match.group(2)
                key_lower = _to_lower(key)
                parsed = parse_number_with_suffix(val)
                if key_lower == "eotacc" and parsed <= 1.0e-10:
                    parsed = 1.1e-10
                if key_lower == "nf":
                    parsed = 1.0  # Single-fin default
                if key_lower == "nfin":
                    parsed = 1.0  # Single-fin default
                parsed_params[key_lower] = parsed
        return parsed_params

    def _is_valid_model(model_type: str, params: Dict[str, float]) -> bool:
        mtype = _to_lower(model_type)
        if mtype == "bsimcmg":
            return True
        if mtype in {"nmos", "pmos"}:
            level = None
            for key, val in params.items():
                if _to_lower(key) == "level":
                    level = val
                    break
            return level == 72
        return False

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        trimmed = raw
        pos_comment = trimmed.find("*")
        if pos_comment != -1:
            trimmed = trimmed[:pos_comment]
        trimmed = trimmed.strip()
        if not trimmed:
            idx += 1
            continue
        if trimmed.lower().startswith(".model"):
            block_lines = [trimmed]
            idx += 1
            while idx < len(lines):
                cont_raw = lines[idx]
                cont = cont_raw
                pos_comment = cont.find("*")
                if pos_comment != -1:
                    cont = cont[:pos_comment]
                cont = cont.strip()
                if not cont:
                    idx += 1
                    continue
                if cont.startswith("+"):
                    block_lines.append(cont[1:].strip())
                    idx += 1
                    continue
                break

            parts = block_lines[0].split()
            if len(parts) >= 3:
                model_name = parts[1]
                model_type = parts[2]
                if target_lower is None or _to_lower(model_name) == target_lower:
                    params = _parse_params(block_lines)
                    if _is_valid_model(model_type, params):
                        # Inject DEVTYPE parameter for ASAP7 compatibility
                        # BSIM-CMG v107 uses DEVTYPE to distinguish NMOS (1) vs PMOS (0)
                        # ASAP7 modelcards often omit this, causing PMOS to behave incorrectly
                        model_type_lower = _to_lower(model_type)
                        if "devtype" not in params:
                            if model_type_lower == "pmos":
                                params["devtype"] = 0.0  # PMOS
                            elif model_type_lower == "nmos":
                                params["devtype"] = 1.0  # NMOS
                        return ParsedModel(name=model_name, params=params)
            continue
        idx += 1

    expected = target_model_name if target_model_name else "bsimcmg or level=72 nmos/pmos"
    raise RuntimeError(f"no {expected} model found in modelcard: {path}")


def parse_tsmc_pdk(path: str, model_type: str, device_type: str, L: float) -> ParsedModel:
    """
    Extract and merge model parameters from full TSMC PDK.

    This function works with all TSMC FinFET PDKs (TSMC5, TSMC7, TSMC12, TSMC16)
    which share the same structure:
    - .global model: base parameters for all variants
    - .1 through .N variants: length-binned models with lmin/lmax
    - Subcircuit wrappers: not needed for OSDI (we use model directly)

    Args:
        path: Path to TSMC PDK file (e.g., cln7_1d8_sp_v1d2_2p2.l)
        model_type: "nch" for NMOS, "pch" for PMOS
        device_type: Device type - "svt_mac", "lvt_mac", "ulvt_mac", "18_mac", etc.
        L: Gate length in meters (used for automatic variant selection)

    Returns:
        ParsedModel with merged global + variant parameters

    Example:
        >>> parse_tsmc_pdk("cln7_1d8_sp_v1d2_2p2.l", "nch", "svt_mac", 16e-9)
        ParsedModel(name="nch_svt_mac", params={...merged params...})
        >>> parse_tsmc_pdk("cln5_1d2_sp_v1d2_2p2.l", "pch", "lvt_mac", 20e-9)
        ParsedModel(name="pch_lvt_mac", params={...merged params...})
    """
    base_name = f"{model_type}_{device_type}"  # e.g., "nch_svt_mac"
    expected_type = "nmos" if model_type == "nch" else "pmos"

    # Extract global model parameters (base)
    try:
        global_params = _extract_model_params(path, f"{base_name}.global", expected_type)
    except RuntimeError as e:
        raise RuntimeError(
            f"TSMC7 PDK file '{path}' does not contain the expected .global model "
            f"'{base_name}.global'. This usually means:\n"
            f"  1. The file is not a valid TSMC7 PDK file\n"
            f"  2. The model_type '{model_type}' and device_type '{device_type}' combination "
            f"does not exist in this PDK\n"
            f"  3. The PDK file format has changed\n\n"
            f"Expected model: .model {base_name}.global {expected_type} (...)\n"
            f"Original error: {e}"
        ) from e

    # Find which variant matches the L value
    variant_num = _find_length_variant(path, base_name, L)

    # Extract variant model parameters
    variant_params = _extract_model_params(path, f"{base_name}.{variant_num}", expected_type)

    # Merge: variant overrides global
    merged_params = {**global_params, **variant_params}

    return ParsedModel(name=base_name, params=merged_params)


# Backward-compatible alias for parse_tsmc_pdk
def parse_tsmc7_pdk(path: str, model_type: str, device_type: str, L: float) -> ParsedModel:
    """
    Backward-compatible alias for parse_tsmc_pdk.

    See parse_tsmc_pdk for full documentation.
    """
    return parse_tsmc_pdk(path, model_type, device_type, L)


def _find_length_variant(path: str, base_name: str, L: float) -> int:
    """
    Find which length variant matches L value.

    All TSMC FinFET PDKs (TSMC5, TSMC7, TSMC12, TSMC16) use numbered bins
    with lmin/lmax ranges. The number of bins varies by technology:
    - TSMC5, TSMC12: 5 bins per corner
    - TSMC7: 30 bins
    - TSMC16: 25 bins per corner

    Supported variant suffixes:
    - Numeric (.1, .2, ...): Length-binned models with lmin/lmax
    - .global: Base parameters (handled separately in parse_tsmc_pdk)
    - Other non-numeric suffixes: Logged as warnings and skipped

    Args:
        path: Path to TSMC PDK file
        base_name: Base model name (e.g., "nch_svt_mac")
        L: Gate length in meters

    Returns:
        Variant number (integer)

    Raises:
        RuntimeError: If no variant matches the L value
    """
    assign_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)")

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        trimmed = raw.strip()

        # Skip comments and empty lines
        if not trimmed or trimmed.startswith("*"):
            idx += 1
            continue

        # Look for variant model definitions
        if trimmed.lower().startswith(".model"):
            # Check if this is a variant model for our base_name
            parts = trimmed.split()
            if len(parts) >= 3:
                model_name = parts[1]

                # Check if this is a variant of our model (e.g., nch_svt_mac.4 or nch_svt_mac.global)
                if model_name.lower().startswith(f"{base_name.lower()}."):
                    variant_suffix = model_name[len(base_name) + 1:]  # Get suffix after dot

                    # Skip .global variant (handled separately in parse_tsmc7_pdk)
                    if variant_suffix.lower() == "global":
                        idx += 1
                        continue

                    # Only process numbered variants (1-30)
                    if variant_suffix.isdigit():
                        # Parse the model block to extract lmin/lmax
                        block_lines = [trimmed]
                        idx += 1
                        while idx < len(lines):
                            cont_raw = lines[idx]
                            cont = cont_raw.strip()
                            if not cont or cont.startswith("*"):
                                idx += 1
                                continue
                            if cont.startswith("+"):
                                block_lines.append(cont[1:].strip())
                                idx += 1
                                continue
                            break

                        # Extract lmin and lmax from this variant
                        lmin = None
                        lmax = None
                        for line in block_lines:
                            for match in assign_re.finditer(line):
                                key = match.group(1).lower()
                                val = parse_number_with_suffix(match.group(2))
                                if key == "lmin":
                                    lmin = val
                                elif key == "lmax":
                                    lmax = val

                        # Check if L falls within this variant's range
                        if lmin is not None and lmax is not None:
                            if lmin <= L <= lmax:
                                return int(variant_suffix)
                    else:
                        # Log warning for unexpected non-numeric variant suffix
                        # This helps with debugging if new variant types are added
                        sys.stderr.write(
                            f"Warning: Skipping unexpected variant '{model_name}' "
                            f"(suffix '{variant_suffix}' is not numeric or 'global')\n"
                        )

        idx += 1

    raise RuntimeError(f"No length variant found for {base_name} with L={L:.3e} in file: {path}")

def _extract_model_params(path: str, model_name: str, expected_type: str) -> Dict[str, float]:
    """
    Extract parameters from a single .model block in TSMC PDK.

    Works with all TSMC FinFET PDKs (TSMC5, TSMC7, TSMC12, TSMC16).
    Reads from the model name match to the next non-continuation line.
    Parses all key=value pairs with SPICE number suffix support.

    Args:
        path: Path to TSMC PDK file
        model_name: Full model name including suffix (e.g., "nch_svt_mac.global" or "nch_svt_mac.4")
        expected_type: Expected model type ("nmos" or "pmos")

    Returns:
        Dictionary of parameter names to float values

    Raises:
        RuntimeError: If model not found
    """
    assign_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)")

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Build the exact pattern to match
    # TSMC PDKs use format: .model nch_svt_mac.global nmos (
    target_pattern = f".model {model_name} {expected_type}"

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        trimmed = raw.strip()

        # Skip comments and empty lines
        if not trimmed or trimmed.startswith("*"):
            idx += 1
            continue

        # Look for the target model
        if trimmed.lower().startswith(".model"):
            # Check if this matches our target
            # Need to be careful with case sensitivity
            if model_name in trimmed and expected_type in trimmed.lower():
                # Found it - parse the block
                block_lines = [trimmed]
                idx += 1
                while idx < len(lines):
                    cont_raw = lines[idx]
                    cont = cont_raw.strip()
                    if not cont or cont.startswith("*"):
                        idx += 1
                        continue
                    if cont.startswith("+"):
                        block_lines.append(cont[1:].strip())
                        idx += 1
                        continue
                    break

                # Parse parameters from the block
                params: Dict[str, float] = {}
                for line in block_lines:
                    for match in assign_re.finditer(line):
                        key = match.group(1)
                        val = match.group(2)
                        key_lower = _to_lower(key)
                        parsed = parse_number_with_suffix(val)

                        # Apply EOTACC clamping for OSDI compatibility
                        if key_lower == "eotacc" and parsed <= 1.0e-10:
                            parsed = 1.1e-10

                        params[key_lower] = parsed

                # Inject DEVTYPE if not present (ASAP7 compatibility)
                # TSMC7 typically has this, but provides safety net
                if "devtype" not in params:
                    expected_type_lower = _to_lower(expected_type)
                    if expected_type_lower == "pmos":
                        params["devtype"] = 0.0  # PMOS
                    elif expected_type_lower == "nmos":
                        params["devtype"] = 1.0  # NMOS

                return params

        idx += 1

    raise RuntimeError(f"Model {model_name} (type={expected_type}) not found in file: {path}")


def _check_init_result(desc: Optional[OsdiDescriptor], info: OsdiInitInfo) -> None:
    def _cleanup() -> None:
        if info.num_errors != 0 and info.errors:
            if _LIBC is not None and hasattr(_LIBC, "free"):
                _LIBC.free(info.errors)
    if info.flags & EVAL_RET_FLAG_FATAL:
        _cleanup()
        raise RuntimeError("OSDI fatal error reported during setup")
    if info.num_errors == 0:
        _cleanup()
        return
    message = []
    fatal = False
    for i in range(info.num_errors):
        err = info.errors[i]
        if err.code == INIT_ERR_OUT_OF_BOUNDS:
            param_id = err.payload.parameter_id
            name = "unknown"
            if desc is not None and desc.param_opvar and param_id < desc.num_params:
                param = desc.param_opvar[param_id]
                if param.name and param.name[0]:
                    name = param.name[0].decode("utf-8", errors="replace")
            message.append(f"parameter out of bounds: {name}")
        else:
            fatal = True
            message.append("unknown OSDI init error")
    _cleanup()
    if message:
        msg = "; ".join(message)
        if fatal:
            raise RuntimeError(msg)
        sys.stderr.write(f"OSDI init warning: {msg}\n")


class OsdiLibrary:
    def __init__(self, path: str) -> None:
        mode = getattr(ctypes, "RTLD_LOCAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        self._lib = ctypes.CDLL(path, mode=mode)
        self._check_version()
        self._count = self._load_descriptor_count()
        self._descriptors = self._load_descriptors(self._count)
        self._install_log()
        self._install_lim()

    def _check_version(self) -> None:
        try:
            major = ctypes.c_uint32.in_dll(self._lib, "OSDI_VERSION_MAJOR").value
            minor = ctypes.c_uint32.in_dll(self._lib, "OSDI_VERSION_MINOR").value
        except ValueError as exc:
            raise RuntimeError("missing OSDI version symbols") from exc
        if major != 0 or minor != 3:
            raise RuntimeError("unsupported OSDI version")

    def _load_descriptor_count(self) -> int:
        return ctypes.c_uint32.in_dll(self._lib, "OSDI_NUM_DESCRIPTORS").value

    def _load_descriptors(self, count: int) -> ctypes.Array:
        if count <= 0:
            raise RuntimeError("no OSDI descriptors")
        array_type = OsdiDescriptor * count
        try:
            return array_type.in_dll(self._lib, "OSDI_DESCRIPTORS")
        except ValueError:
            ptr = ctypes.c_void_p.in_dll(self._lib, "OSDI_DESCRIPTORS")
            if not ptr.value:
                raise RuntimeError("missing OSDI descriptors")
            return (OsdiDescriptor * count).from_address(ptr.value)

    def _install_log(self) -> None:
        try:
            log_var = ctypes.c_void_p.in_dll(self._lib, "osdi_log")
        except ValueError:
            return
        log_var.value = ctypes.cast(_osdi_log, ctypes.c_void_p).value

    def _install_lim(self) -> None:
        try:
            table_ptr = ctypes.POINTER(ctypes.POINTER(OsdiLimFunction)).in_dll(self._lib, "OSDI_LIM_TABLE")
            len_val = ctypes.c_uint32.in_dll(self._lib, "OSDI_LIM_TABLE_LEN").value
        except ValueError:
            return
        if not table_ptr or not table_ptr[0]:
            return
        table = table_ptr[0]
        for i in range(len_val):
            entry = table[i]
            if entry.name and entry.name.decode("utf-8", errors="replace") == "pnjlim":
                table[i].func_ptr = ctypes.cast(_pnjlim, ctypes.c_void_p).value

    def descriptor(self, index: int = 0) -> Optional[OsdiDescriptor]:
        if index < 0 or index >= self._count:
            return None
        return self._descriptors[index]

    def descriptor_by_name(self, name: str) -> Optional[OsdiDescriptor]:
        for i in range(self._count):
            desc = self._descriptors[i]
            if desc.name and name == desc.name.decode("utf-8", errors="replace"):
                return desc
        return None


class OsdiModel:
    def __init__(self, desc: OsdiDescriptor) -> None:
        if not desc:
            raise RuntimeError("descriptor is null")
        self._desc = desc
        self._buf = AlignedBuffer(int(desc.model_size))

    def data(self) -> ctypes.c_void_p:
        return self._buf.ptr

    @property
    def descriptor(self) -> OsdiDescriptor:
        return self._desc

    def process_params(self) -> None:
        sim_params = OsdiSimParas()
        info = OsdiInitInfo()
        self._desc.setup_model(_INSTANCE_NAME, self._buf.ptr, ctypes.byref(sim_params), ctypes.byref(info))
        _check_init_result(self._desc, info)

    def set_param(self, name: str, value: float) -> None:
        desc = self._desc
        if not desc.param_opvar:
            raise RuntimeError("descriptor has no parameter metadata")
        for i in range(desc.num_params):
            param = desc.param_opvar[i]
            param_name = param.name[0].decode("utf-8", errors="replace") if param.name else ""
            if param_name and name == param_name:
                ptr = desc.access(None, self._buf.ptr, i, ACCESS_FLAG_SET)
                if not ptr:
                    raise RuntimeError("invalid parameter access")
                ty = param.flags & PARA_TY_MASK
                if ty == PARA_TY_INT:
                    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[0] = int(value)
                elif ty == PARA_TY_REAL:
                    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0] = float(value)
                else:
                    raise RuntimeError(f"string parameter not supported: {name}")
                return
        raise RuntimeError(f"parameter not found: {name}")


class OsdiSimulation:
    def __init__(self) -> None:
        self.node_names: List[str] = ["gnd"]
        self.node_index: Dict[str, int] = {"gnd": 0}
        self.terminal_indices: List[int] = []
        self.internal_indices: List[int] = []
        self.residual_resist = self._make_array(1)
        self.residual_react = self._make_array(1)
        self.rhs_tran = self._make_array(1)
        self.solve = self._make_array(1)
        self.prev_solve = self._make_array(1)
        self.has_prev_solve = False
        self.jacobian_info: List[Tuple[int, int]] = [(0, 0)]
        self.jacobian_index: Dict[int, int] = {0: 0}
        self.jacobian_resist = self._make_array(1)
        self.jacobian_react = self._make_array(1)
        self.state_prev = self._make_array(0)
        self.state_next = self._make_array(0)
        self.noise_dense = self._make_array(0)
        self.sim_param_names: List[str] = []
        self.sim_param_vals: List[float] = []

    def _make_array(self, size: int) -> ctypes.Array:
        return (ctypes.c_double * max(size, 0))()

    def _resize_vectors(self, new_size: int) -> None:
        self.residual_resist = self._resize_array(self.residual_resist, new_size)
        self.residual_react = self._resize_array(self.residual_react, new_size)
        self.rhs_tran = self._resize_array(self.rhs_tran, new_size)
        self.solve = self._resize_array(self.solve, new_size)
        self.prev_solve = self._resize_array(self.prev_solve, new_size)

    @staticmethod
    def _resize_array(arr: ctypes.Array, new_size: int) -> ctypes.Array:
        new_arr = (ctypes.c_double * new_size)()
        for i in range(min(len(arr), new_size)):
            new_arr[i] = arr[i]
        return new_arr

    def register_node(self, name: str) -> int:
        if name in self.node_index:
            return self.node_index[name]
        idx = len(self.node_names)
        self.node_names.append(name)
        self.node_index[name] = idx
        self._resize_vectors(len(self.node_names))
        return idx

    def copy_solve_to_prev(self) -> None:
        for i in range(len(self.solve)):
            self.prev_solve[i] = self.solve[i]

    def register_jacobian_entry(self, row: int, col: int) -> None:
        if row == 0 or col == 0:
            return
        key = (row << 32) | col
        if key in self.jacobian_index:
            return
        idx = len(self.jacobian_info)
        self.jacobian_info.append((row, col))
        self.jacobian_index[key] = idx

    def get_jacobian_entry(self, row: int, col: int) -> int:
        if row == 0 or col == 0:
            return 0
        key = (row << 32) | col
        return self.jacobian_index.get(key, 0)

    def build_jacobian(self) -> None:
        size = len(self.jacobian_info)
        self.jacobian_resist = self._make_array(size)
        self.jacobian_react = self._make_array(size)

    def clear(self) -> None:
        for i in range(len(self.residual_resist)):
            self.residual_resist[i] = 0.0
            self.residual_react[i] = 0.0
            self.rhs_tran[i] = 0.0
        for i in range(len(self.jacobian_resist)):
            self.jacobian_resist[i] = 0.0
            self.jacobian_react[i] = 0.0

    def set_voltage(self, node: str, voltage: float) -> None:
        if node not in self.node_index:
            raise RuntimeError(f"unknown node: {node}")
        self.solve[self.node_index[node]] = voltage

    def set_sim_param(self, name: str, value: float) -> None:
        for i, key in enumerate(self.sim_param_names):
            if key == name:
                self.sim_param_vals[i] = value
                return
        self.sim_param_names.append(name)
        self.sim_param_vals.append(value)

    def build_sim_paras(self) -> Tuple[OsdiSimParas, ctypes.Array, ctypes.Array, ctypes.Array, ctypes.Array]:
        names = [ctypes.c_char_p(n.encode("utf-8")) for n in self.sim_param_names]
        vals = [ctypes.c_double(v) for v in self.sim_param_vals]
        names.append(ctypes.c_char_p(None))
        vals.append(ctypes.c_double(0.0))
        names_arr = (ctypes.c_char_p * len(names))(*names)
        vals_arr = (ctypes.c_double * len(vals))(*vals)
        names_str_arr = (ctypes.c_char_p * 1)(None)
        vals_str_arr = (ctypes.c_char_p * 1)(None)
        sim_params = OsdiSimParas(names=names_arr, vals=vals_arr,
                                  names_str=names_str_arr,
                                  vals_str=vals_str_arr)
        return sim_params, names_arr, vals_arr, names_str_arr, vals_str_arr


class OsdiInstance:
    def __init__(self, desc: OsdiDescriptor) -> None:
        if not desc:
            raise RuntimeError("descriptor is null")
        self._desc = desc
        self._buf = AlignedBuffer(int(desc.instance_size))

    def data(self) -> ctypes.c_void_p:
        return self._buf.ptr

    @property
    def descriptor(self) -> OsdiDescriptor:
        return self._desc

    def _ptr_at(self, offset: int, ctype: ctypes._SimpleCData) -> ctypes.POINTER:
        base = self._buf.ptr.value or 0
        addr = base + offset
        return ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctype))

    def process_params(self, model: OsdiModel, connected_terminals: int, temperature: float) -> List[int]:
        sim_params = OsdiSimParas()
        info = OsdiInitInfo()
        self._desc.setup_instance(_INSTANCE_NAME, self._buf.ptr, model.data(),
                                  temperature, connected_terminals,
                                  ctypes.byref(sim_params), ctypes.byref(info))
        _check_init_result(self._desc, info)
        return self._collapse_nodes(connected_terminals)

    def bind_simulation(self, sim: OsdiSimulation, model: OsdiModel,
                        connected_terminals: int, temperature: float) -> None:
        internal_nodes = self.process_params(model, connected_terminals, temperature)
        nodes = self._desc.nodes
        terminal_indices: List[int] = []
        for i in range(connected_terminals):
            node_name = nodes[i].name.decode("utf-8", errors="replace") if nodes[i].name else ""
            terminal_indices.append(sim.register_node(node_name))
        sim.terminal_indices = terminal_indices

        internal_indices: List[int] = []
        for idx in internal_nodes:
            node_name = nodes[idx].name.decode("utf-8", errors="replace") if nodes[idx].name else ""
            internal_indices.append(sim.register_node(node_name))
        sim.internal_indices = internal_indices

        mapping_ptr = self._ptr_at(self._desc.node_mapping_offset, ctypes.c_uint32)
        for i in range(self._desc.num_nodes):
            idx = mapping_ptr[i]
            if idx < len(terminal_indices):
                mapping_ptr[i] = terminal_indices[idx]
            elif idx == UINT32_MAX:
                mapping_ptr[i] = 0
            else:
                internal_idx = idx - len(terminal_indices)
                if internal_idx >= len(internal_indices):
                    mapping_ptr[i] = 0
                else:
                    mapping_ptr[i] = internal_indices[internal_idx]

        for i in range(self._desc.num_jacobian_entries):
            entry = self._desc.jacobian_entries[i]
            row = mapping_ptr[entry.nodes.node_1]
            col = mapping_ptr[entry.nodes.node_2]
            sim.register_jacobian_entry(int(row), int(col))
        sim.build_jacobian()

        ptr_resist = self._ptr_at(self._desc.jacobian_ptr_resist_offset,
                                  ctypes.POINTER(ctypes.c_double))
        ptr_resist = ctypes.cast(ptr_resist, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
        for i in range(self._desc.num_jacobian_entries):
            entry = self._desc.jacobian_entries[i]
            row = mapping_ptr[entry.nodes.node_1]
            col = mapping_ptr[entry.nodes.node_2]
            idx = sim.get_jacobian_entry(int(row), int(col))
            elem_ptr = ctypes.cast(ctypes.byref(sim.jacobian_resist, idx * ctypes.sizeof(ctypes.c_double)),
                                   ctypes.POINTER(ctypes.c_double))
            ptr_resist[i] = elem_ptr
            if entry.react_ptr_off != UINT32_MAX:
                react_ptr_loc = self._ptr_at(entry.react_ptr_off, ctypes.POINTER(ctypes.c_double))
                react_ptr_loc = ctypes.cast(react_ptr_loc, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
                react_ptr = ctypes.cast(ctypes.byref(sim.jacobian_react, idx * ctypes.sizeof(ctypes.c_double)),
                                        ctypes.POINTER(ctypes.c_double))
                react_ptr_loc[0] = react_ptr

        sim.state_prev = sim._make_array(self._desc.num_states)
        sim.state_next = sim._make_array(self._desc.num_states)
        sim.noise_dense = sim._make_array(self._desc.num_noise_src)

    def eval(self, model: OsdiModel, sim: OsdiSimulation, flags: int) -> int:
        return self.eval_with_time(model, sim, flags, 0.0)

    def eval_with_time(self, model: OsdiModel, sim: OsdiSimulation, flags: int, abstime: float) -> int:
        sim_params, names_arr, vals_arr, names_str_arr, vals_str_arr = sim.build_sim_paras()
        sim_info = OsdiSimInfo()
        sim_info.paras = sim_params
        sim_info.abstime = abstime
        sim_info.prev_solve = sim.prev_solve if sim.has_prev_solve else sim.solve
        sim_info.prev_state = sim.state_prev
        sim_info.next_state = sim.state_next
        sim_info.flags = flags
        # Keep arrays alive while eval runs.
        _ = (names_arr, vals_arr, names_str_arr, vals_str_arr)
        return self._desc.eval(_INSTANCE_NAME, self._buf.ptr, model.data(), ctypes.byref(sim_info))

    def load_residuals(self, model: OsdiModel, sim: OsdiSimulation) -> None:
        self._desc.load_residual_resist(self._buf.ptr, model.data(), sim.residual_resist)
        self._desc.load_limit_rhs_resist(self._buf.ptr, model.data(), sim.residual_resist)
        self._desc.load_residual_react(self._buf.ptr, model.data(), sim.residual_react)
        self._desc.load_limit_rhs_react(self._buf.ptr, model.data(), sim.residual_react)

    def load_jacobian(self, model: OsdiModel, sim: OsdiSimulation) -> None:
        self._desc.load_jacobian_resist(self._buf.ptr, model.data())
        self._desc.load_jacobian_react(self._buf.ptr, model.data(), 1.0)

    def load_spice_rhs_dc(self, model: OsdiModel, sim: OsdiSimulation) -> None:
        self._desc.load_spice_rhs_dc(self._buf.ptr, model.data(), sim.residual_resist, sim.solve)

    def load_spice_rhs_tran(self, model: OsdiModel, sim: OsdiSimulation, alpha: float) -> None:
        self._desc.load_spice_rhs_tran(self._buf.ptr, model.data(), sim.rhs_tran, sim.prev_solve, alpha)

    def load_jacobian_tran(self, model: OsdiModel, sim: OsdiSimulation, alpha: float) -> None:
        self._desc.load_jacobian_tran(self._buf.ptr, model.data(), alpha)

    def solve_internal_nodes(self, model: OsdiModel, sim: OsdiSimulation,
                             max_iter: int, tol: float) -> bool:
        if not sim.internal_indices:
            return True
        internal_pos = {idx: i for i, idx in enumerate(sim.internal_indices)}
        for _ in range(max_iter):
            flags = (ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_RESIDUAL |
                     CALC_RESIST_JACOBIAN | CALC_RESIST_LIM_RHS |
                     ENABLE_LIM | INIT_LIM)
            self.eval(model, sim, flags)
            sim.clear()
            self.load_residuals(model, sim)
            self.load_jacobian(model, sim)
            norm = max(abs(sim.residual_resist[idx]) for idx in sim.internal_indices)
            if norm < tol:
                return True
            n = len(sim.internal_indices)
            a = np.zeros((n, n), dtype=float)
            b = np.zeros(n, dtype=float)
            for i, idx in enumerate(sim.internal_indices):
                b[i] = -sim.residual_resist[idx]
            for k, (row, col) in enumerate(sim.jacobian_info):
                if row in internal_pos and col in internal_pos:
                    r = internal_pos[row]
                    c = internal_pos[col]
                    a[r, c] = sim.jacobian_resist[k]
            try:
                delta = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                return False
            for i, idx in enumerate(sim.internal_indices):
                update = float(delta[i])
                if update > 0.2:
                    update = 0.2
                elif update < -0.2:
                    update = -0.2
                sim.solve[idx] = sim.solve[idx] + update
        return False

    def solve_internal_nodes_tran(self, model: OsdiModel, sim: OsdiSimulation,
                                  abstime: float, alpha: float,
                                  max_iter: int, tol: float) -> bool:
        if not sim.internal_indices:
            return True
        internal_pos = {idx: i for i, idx in enumerate(sim.internal_indices)}
        for _ in range(max_iter):
            flags = (ANALYSIS_TRAN | CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN |
                     CALC_RESIST_LIM_RHS | CALC_REACT_RESIDUAL |
                     CALC_REACT_JACOBIAN | CALC_REACT_LIM_RHS |
                     ENABLE_LIM | INIT_LIM)
            self.eval_with_time(model, sim, flags, abstime)
            sim.clear()
            self.load_residuals(model, sim)
            self.load_jacobian_tran(model, sim, alpha)
            self.load_spice_rhs_tran(model, sim, alpha)
            total_residual = [
                sim.residual_resist[i] + alpha * sim.residual_react[i] - sim.rhs_tran[i]
                for i in range(len(sim.residual_resist))
            ]
            norm = max(abs(total_residual[idx]) for idx in sim.internal_indices)
            if norm < tol:
                return True
            n = len(sim.internal_indices)
            a = np.zeros((n, n), dtype=float)
            b = np.zeros(n, dtype=float)
            for i, idx in enumerate(sim.internal_indices):
                b[i] = -total_residual[idx]
            for k, (row, col) in enumerate(sim.jacobian_info):
                if row in internal_pos and col in internal_pos:
                    r = internal_pos[row]
                    c = internal_pos[col]
                    a[r, c] = sim.jacobian_resist[k]
            try:
                delta = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                return False
            for i, idx in enumerate(sim.internal_indices):
                update = float(delta[i])
                if update > 0.2:
                    update = 0.2
                elif update < -0.2:
                    update = -0.2
                sim.solve[idx] = sim.solve[idx] + update
        return False

    def _collapse_nodes(self, connected_terminals: int) -> List[int]:
        back_map: List[int] = list(range(connected_terminals, int(self._desc.num_nodes)))
        mapping_ptr = self._ptr_at(self._desc.node_mapping_offset, ctypes.c_uint32)
        for i in range(self._desc.num_nodes):
            mapping_ptr[i] = i
        collapsed_ptr = self._ptr_at(self._desc.collapsed_offset, ctypes.c_bool)
        for i in range(self._desc.num_collapsible):
            if not collapsed_ptr[i]:
                continue
            candidate = self._desc.collapsible[i]
            from_node = candidate.node_1
            to_node = candidate.node_2
            mapped_from = mapping_ptr[from_node]
            collapse_to_gnd = (to_node == UINT32_MAX)
            mapped_to = UINT32_MAX if collapse_to_gnd else mapping_ptr[to_node]
            if not collapse_to_gnd and mapped_to == UINT32_MAX:
                collapse_to_gnd = True
            if mapped_from < connected_terminals and (collapse_to_gnd or mapped_to < connected_terminals):
                continue
            if not collapse_to_gnd and mapped_from < mapped_to:
                mapped_from, mapped_to = mapped_to, mapped_from
            for j in range(self._desc.num_nodes):
                val = mapping_ptr[j]
                if val == mapped_from:
                    mapping_ptr[j] = mapped_to
                elif val > mapped_from and val != UINT32_MAX:
                    mapping_ptr[j] = val - 1
            if mapped_from >= connected_terminals:
                remove_idx = mapped_from - connected_terminals
                if 0 <= remove_idx < len(back_map):
                    back_map.pop(remove_idx)
        return back_map


def apply_param(desc: OsdiDescriptor,
                inst: Optional[OsdiInstance],
                model: Optional[OsdiModel],
                name: str,
                value: float,
                from_modelcard: bool) -> bool:
    applied = False
    for i in range(desc.num_params):
        param = desc.param_opvar[i]
        param_name = param.name[0].decode("utf-8", errors="replace") if param.name else ""
        if not param_name or _to_lower(name) != _to_lower(param_name):
            continue
        ptr = None
        if (param.flags & PARA_KIND_MASK) == PARA_KIND_INST:
            if inst is None:
                return False
            ptr = desc.access(inst.data(), model.data() if model else None, i,
                              ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE)
        else:
            if model is None:
                return False
            ptr = desc.access(None, model.data(), i, ACCESS_FLAG_SET)
        if not ptr:
            raise RuntimeError(f"invalid parameter access for {name}")
        ty = param.flags & PARA_TY_MASK
        if ty == PARA_TY_INT:
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[0] = int(value)
            applied = True
        elif ty == PARA_TY_REAL:
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0] = float(value)
            applied = True
        else:
            if not from_modelcard:
                raise RuntimeError(f"string parameter not supported: {name}")
            applied = True
        break
    if not applied and not from_modelcard:
        raise RuntimeError(f"parameter not found: {name}")
    return applied


class Model:
    """
    BSIM-CMG model wrapper for OSDI binary interface.

    The Model class loads an OSDI compiled model and associated modelcard parameters.
    It provides the foundation for creating device instances with specific geometry
    and operating conditions.

    Temperature handling:
        - The Model class itself does not store temperature
        - Temperature is specified when creating Instance objects
        - Temperature must be in KELVIN (see module docstring for conversion)

    Example:
        >>> model = Model(
        ...     "bsimcmg.osdi",
        ...     "asap7.pm",
        ...     "nmos_rvt"
        ... )
    """

    def __init__(self, osdi_path: str, modelcard_path: str, model_name: str,
                 model_card_name: Optional[str] = None) -> None:
        self._lib = OsdiLibrary(osdi_path)
        desc = self._lib.descriptor_by_name(model_name) if model_name else None
        if desc is None:
            desc = self._lib.descriptor(0)
        if desc is None:
            raise RuntimeError("OSDI descriptor not found")
        self._desc = desc
        self._model = OsdiModel(desc)
        self._modelcard_params: Dict[str, float] = {}
        if modelcard_path:
            # Use model_card_name if explicitly provided, otherwise fall back to
            # model_name so parse_modelcard targets the correct .model block.
            # Without this, parse_modelcard(target=None) matches the FIRST model
            # in the file — which is NMOS in multi-model files like ASAP7,
            # causing PMOS models to get DEVTYPE=1 (NMOS) instead of DEVTYPE=0.
            target = model_card_name if model_card_name else model_name
            parsed = parse_modelcard(modelcard_path, target)
            self._modelcard_params = dict(parsed.params)

    @property
    def descriptor(self) -> OsdiDescriptor:
        return self._desc

    @property
    def model(self) -> OsdiModel:
        return self._model

    @property
    def modelcard_params(self) -> Dict[str, float]:
        return dict(self._modelcard_params)


class Instance:
    """
    BSIM-CMG device instance for DC and transient evaluation.

    An Instance represents a specific device with geometry parameters and
    operating conditions (temperature, voltages). It provides methods for
    DC operating point analysis and transient simulation.

    Temperature parameter:
        - The temperature parameter MUST be in KELVIN
        - Default is 300.15 K (27°C, typical room temperature)
        - To convert from Celsius: temp_K = temp_C + 273.15

    Args:
        model: Model object containing OSDI descriptor and modelcard
        params: Instance-specific parameters (L, TFIN, NFIN, etc.)
        temperature: Operating temperature in KELVIN (default: 300.15 K = 27°C)

    Example:
        >>> # Create instance at room temperature (27°C)
        >>> inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2},
        ...                 temperature=300.15)  # 27°C in Kelvin

        >>> # Create instance at elevated temperature (85°C)
        >>> inst = Instance(model, params={"L": 16e-9},
        ...                 temperature=358.15)  # 85°C = 85 + 273.15

        >>> # Create instance at cold temperature (-40°C)
        >>> inst = Instance(model, params={"L": 16e-9},
        ...                 temperature=233.15)  # -40°C = -40 + 273.15
    """

    def __init__(self, model: Model, params: Optional[Dict[str, float]] = None,
                 temperature: float = 300.15) -> None:
        self._model = model
        self._inst = OsdiInstance(model.descriptor)
        self._temperature = temperature
        self._sim = OsdiSimulation()
        self._connected_terminals = int(model.descriptor.num_terminals)
        for key, val in model.modelcard_params.items():
            apply_param(model.descriptor, self._inst, model.model, key, val, True)
        if params:
            for key, val in params.items():
                apply_param(model.descriptor, self._inst, model.model, key, val, False)
        self._model.model.process_params()
        self._inst.bind_simulation(self._sim, model.model, self._connected_terminals, temperature)
        self._has_prev_solve = False
        self._has_prev_q = False
        self._prev_qg = 0.0
        self._prev_qd = 0.0
        self._prev_qs = 0.0
        self._prev_qb = 0.0

    def set_params(self, params: Dict[str, float], allow_rebind: bool = False) -> None:
        for key, val in params.items():
            apply_param(self._model.descriptor, self._inst, self._model.model, key, val, False)
        self._model.model.process_params()
        internal = self._inst.process_params(self._model.model, self._connected_terminals, self._temperature)
        if len(internal) != len(self._sim.internal_indices):
            if not allow_rebind:
                raise RuntimeError("topology changed; rebind required")
            self._sim = OsdiSimulation()
            self._inst.bind_simulation(self._sim, self._model.model, self._connected_terminals, self._temperature)

    def internal_node_count(self) -> int:
        return len(self._sim.internal_indices)

    def state_count(self) -> int:
        return int(self._model.descriptor.num_states)

    def _set_node_voltages(self, nodes: Dict[str, float], seed_internal: bool) -> None:
        for name in ("d", "g", "s", "e"):
            value = float(nodes.get(name, 0.0))
            self._sim.set_voltage(name, value)
        if seed_internal:
            if "di" in self._sim.node_index and "di" not in nodes:
                self._sim.set_voltage("di", self._sim.solve[self._sim.node_index["d"]])
            if "si" in self._sim.node_index and "si" not in nodes:
                self._sim.set_voltage("si", self._sim.solve[self._sim.node_index["s"]])

    def _read_current(self, name: str) -> float:
        idx = self._sim.node_index.get(name)
        if idx is None:
            return 0.0
        return -self._sim.residual_resist[idx]

    def _read_current_from(self, residuals: List[float], name: str) -> float:
        idx = self._sim.node_index.get(name)
        if idx is None or idx >= len(residuals):
            return 0.0
        return -residuals[idx]

    def _read_terminal_current(self, term: str, internal: str) -> float:
        idx_internal = self._sim.node_index.get(internal)
        if idx_internal is not None and idx_internal < len(self._sim.residual_resist):
            return float(self._sim.residual_resist[idx_internal])
        return self._read_current(term)

    def _read_opvar(self, name: str, alias: str) -> Optional[float]:
        desc = self._model.descriptor
        name_lower = _to_lower(name)
        alias_lower = _to_lower(alias)
        total = int(desc.num_params + desc.num_opvars)
        for i in range(total):
            param = desc.param_opvar[i]
            if (param.flags & PARA_KIND_MASK) != PARA_KIND_OPVAR:
                continue
            matched = False
            if param.num_alias == 0:
                if param.name and param.name[0]:
                    if _to_lower(param.name[0].decode("utf-8", errors="replace")) == name_lower:
                        matched = True
            else:
                for a in range(param.num_alias):
                    alias_name = param.name[a]
                    if not alias_name:
                        continue
                    alias_str = _to_lower(alias_name.decode("utf-8", errors="replace"))
                    if alias_str in (name_lower, alias_lower):
                        matched = True
                        break
            if not matched:
                continue
            ptr = desc.access(self._inst.data(), self._model.model.data(), i, ACCESS_FLAG_INSTANCE)
            if not ptr:
                ptr = desc.access(self._inst.data(), self._model.model.data(), i,
                                  ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE)
            if not ptr:
                return None
            ty = param.flags & PARA_TY_MASK
            if ty == PARA_TY_INT:
                return float(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[0])
            if ty == PARA_TY_REAL:
                return float(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0])
            return None
        return None

    @staticmethod
    def _build_full_jacobian(sim: OsdiSimulation, values: ctypes.Array) -> np.ndarray:
        n = len(sim.node_names)
        out = np.zeros((n, n), dtype=float)
        for k, (row, col) in enumerate(sim.jacobian_info):
            if row < n and col < n and k < len(values):
                out[row, col] = values[k]
        return out

    @staticmethod
    def _condense_capacitance(g_full: np.ndarray,
                              c_full: np.ndarray,
                              external: List[int],
                              internal: List[int]) -> np.ndarray:
        ne = len(external)
        ni = len(internal)
        c_condensed = np.zeros((ne, ne), dtype=float)
        if ne == 0:
            return c_condensed
        jw = 1j
        yee = np.zeros((ne, ne), dtype=complex)
        yei = np.zeros((ne, ni), dtype=complex)
        yie = np.zeros((ni, ne), dtype=complex)
        yii = np.zeros((ni, ni), dtype=complex)
        for r in range(ne):
            for c in range(ne):
                yee[r, c] = g_full[external[r], external[c]] + jw * c_full[external[r], external[c]]
            for c in range(ni):
                yei[r, c] = g_full[external[r], internal[c]] + jw * c_full[external[r], internal[c]]
        for r in range(ni):
            for c in range(ne):
                yie[r, c] = g_full[internal[r], external[c]] + jw * c_full[internal[r], external[c]]
            for c in range(ni):
                yii[r, c] = g_full[internal[r], internal[c]] + jw * c_full[internal[r], internal[c]]
        if ni == 0:
            c_condensed[:, :] = np.imag(yee)
            return c_condensed
        try:
            yie_sol = np.linalg.solve(yii, yie)
        except np.linalg.LinAlgError:
            return c_condensed
        for r in range(ne):
            for c in range(ne):
                accum = yee[r, c]
                accum -= np.dot(yei[r, :], yie_sol[:, c])
                c_condensed[r, c] = float(np.imag(accum))
        return c_condensed

    def _condense_caps(self) -> Dict[str, float]:
        g_full = self._build_full_jacobian(self._sim, self._sim.jacobian_resist)
        c_full = self._build_full_jacobian(self._sim, self._sim.jacobian_react)
        c_condensed = self._condense_capacitance(g_full, c_full,
                                                 self._sim.terminal_indices,
                                                 self._sim.internal_indices)
        def idx_of(name: str) -> int:
            for i, idx in enumerate(self._sim.terminal_indices):
                if self._sim.node_names[idx] == name:
                    return i
            return -1
        caps = {"cgg": 0.0, "cgd": 0.0, "cgs": 0.0, "cdg": 0.0, "cdd": 0.0}
        if c_condensed.size == 0:
            return caps
        g = idx_of("g")
        d = idx_of("d")
        s = idx_of("s")
        if g >= 0:
            caps["cgg"] = float(c_condensed[g, g])
            if d >= 0:
                caps["cgd"] = float(c_condensed[g, d])
            if s >= 0:
                caps["cgs"] = float(c_condensed[g, s])
        if d >= 0 and g >= 0:
            caps["cdg"] = float(c_condensed[d, g])
            caps["cdd"] = float(c_condensed[d, d])
        return caps

    def eval_dc(self, nodes: Dict[str, float]) -> Dict[str, float]:
        """
        Perform DC operating point analysis.

        Evaluates the device at specified terminal voltages and returns
        terminal currents, charges, derivatives, and capacitances.

        Temperature:
            Uses the temperature specified during Instance initialization (in KELVIN).
            To change temperature, create a new Instance with the desired temperature.

        Args:
            nodes: Dictionary mapping terminal names to voltages
                   Required keys: "d" (drain), "g" (gate), "s" (source), "e" (bulk)
                   Example: {"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0}

        Returns:
            Dictionary with 18 output values:
            - Currents (A): id, ig, is, ie, ids
            - Charges (C): qg, qd, qs, qb
            - Derivatives (S): gm, gds, gmb
            - Capacitances (F): cgg, cgd, cgs, cdg, cdd

        Example:
            >>> inst = Instance(model, params={"L": 16e-9}, temperature=300.15)  # 27°C
            >>> result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
            >>> print(f"Drain current: {result['id']:.6e} A")
            >>> print(f"Transconductance: {result['gm']:.6e} S")
        """
        self._set_node_voltages(nodes, True)
        self._inst.solve_internal_nodes(self._model.model, self._sim, 200, 1e-9)
        flags = (ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_JACOBIAN |
                 CALC_RESIST_RESIDUAL | CALC_RESIST_LIM_RHS |
                 CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                 CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM)
        self._inst.eval(self._model.model, self._sim, flags)
        self._sim.clear()
        self._inst.load_residuals(self._model.model, self._sim)
        self._inst.load_jacobian(self._model.model, self._sim)

        out: Dict[str, float] = {
            "id": self._read_current("d"),
            "ig": self._read_current("g"),
            "is": self._read_current("s"),
            "ie": self._read_current("e"),
        }
        # Drain-source current (Ids = Id - Is for common-source configuration)
        out["ids"] = out["id"] - out["is"]

        qg = self._read_opvar("qg", "qgate") or 0.0
        qd = self._read_opvar("qd", "qdrain") or 0.0
        qs = self._read_opvar("qs", "qsource") or 0.0
        qb = self._read_opvar("qb", "qbulk")
        if qb is None:
            qb = self._read_opvar("qe", "qe") or 0.0
        out.update({"qg": qg, "qd": qd, "qs": qs, "qb": qb})

        gm = self._read_opvar("gm", "gm") or 0.0
        gds = self._read_opvar("gds", "gds") or 0.0
        gmb = self._read_opvar("gmbs", "gmbs")
        if gmb is None:
            gmb = self._read_opvar("gmb", "gmb") or 0.0
        out.update({"gm": gm, "gds": gds, "gmb": gmb})

        out.update(self._condense_caps())
        return out

    def eval_tran(self, nodes: Dict[str, float], time: float, delta_t: float,
                  prev_state: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Perform transient analysis at a specific time point.

        Evaluates the device with time-dependent effects including charge storage
        and capacitive currents. Suitable for transient simulation and AC analysis.

        Temperature:
            Uses the temperature specified during Instance initialization (in KELVIN).
            Temperature effects on capacitances and charges are evaluated at
            the initialization temperature.

        Args:
            nodes: Dictionary mapping terminal names to voltages
                   Required keys: "d" (drain), "g" (gate), "s" (source), "e" (bulk)
                   Example: {"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0}
            time: Current simulation time in seconds
            delta_t: Time step in seconds (must be positive)
            prev_state: Optional previous state vector for multi-step simulations

        Returns:
            Dictionary with 9 output values:
            - Currents (A): id, ig, is, ie, ids (includes displacement currents)
            - Charges (C): qg, qd, qs, qb

        Example:
            >>> inst = Instance(model, params={"L": 16e-9}, temperature=358.15)  # 85°C
            >>> result = inst.eval_tran(
            ...     nodes={"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0},
            ...     time=1e-9,
            ...     delta_t=1e-12
            ... )
            >>> print(f"Drain current (with dQ/dt): {result['id']:.6e} A")
        """
        if delta_t <= 0.0:
            raise RuntimeError("delta_t must be positive")
        if prev_state is not None:
            if len(prev_state) != len(self._sim.state_prev):
                raise RuntimeError("prev_state size mismatch")
            for i, val in enumerate(prev_state):
                self._sim.state_prev[i] = val
        self._set_node_voltages(nodes, True)
        self._sim.copy_solve_to_prev()
        self._sim.has_prev_solve = True
        if not self._has_prev_solve:
            ic_flags = (ANALYSIS_IC | CALC_RESIST_RESIDUAL |
                        CALC_RESIST_LIM_RHS | CALC_REACT_RESIDUAL |
                        CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM)
            self._inst.eval_with_time(self._model.model, self._sim, ic_flags, time)
            if len(self._sim.state_prev) == len(self._sim.state_next):
                self._sim.state_prev, self._sim.state_next = self._sim.state_next, self._sim.state_prev
        num_states = self.state_count()
        alpha = 1.0 / delta_t
        if num_states > 0:
            for key, val in [
                ("dt", delta_t), ("delta_t", delta_t), ("delta", delta_t),
                ("h", delta_t), ("step", delta_t), ("alpha", alpha),
                ("t", time), ("time", time), ("abstime", time),
            ]:
                self._sim.set_sim_param(key, val)
        if num_states == 0:
            self._inst.solve_internal_nodes(self._model.model, self._sim, 200, 1e-9)
            flags = (ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_JACOBIAN |
                     CALC_RESIST_RESIDUAL | CALC_RESIST_LIM_RHS |
                     CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                     CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM)
            self._inst.eval(self._model.model, self._sim, flags)
            self._sim.clear()
            self._inst.load_residuals(self._model.model, self._sim)
        else:
            self._inst.solve_internal_nodes_tran(self._model.model, self._sim, time, alpha, 200, 1e-9)
            flags = (ANALYSIS_TRAN | CALC_RESIST_JACOBIAN | CALC_RESIST_RESIDUAL |
                     CALC_RESIST_LIM_RHS | CALC_REACT_JACOBIAN |
                     CALC_REACT_RESIDUAL | CALC_REACT_LIM_RHS |
                     CALC_OP | ENABLE_LIM | INIT_LIM)
            self._inst.eval_with_time(self._model.model, self._sim, flags, time)
            self._sim.clear()
            self._inst.load_residuals(self._model.model, self._sim)
            self._inst.load_jacobian_tran(self._model.model, self._sim, alpha)
            self._inst.load_spice_rhs_tran(self._model.model, self._sim, alpha)

        total_residual = [
            self._sim.residual_resist[i] + alpha * self._sim.residual_react[i] - self._sim.rhs_tran[i]
            for i in range(len(self._sim.residual_resist))
        ]

        qg = self._read_opvar("qg", "qgate") or 0.0
        qd = self._read_opvar("qd", "qdrain") or 0.0
        qs = self._read_opvar("qs", "qsource") or 0.0
        qb = self._read_opvar("qb", "qbulk")
        if qb is None:
            qb = self._read_opvar("qe", "qe") or 0.0

        out: Dict[str, float] = {"qg": qg, "qd": qd, "qs": qs, "qb": qb}
        if num_states == 0:
            dqg_dt = dqd_dt = dqs_dt = dqb_dt = 0.0
            if self._has_prev_q:
                dqg_dt = (qg - self._prev_qg) * alpha
                dqd_dt = (qd - self._prev_qd) * alpha
                dqs_dt = (qs - self._prev_qs) * alpha
                dqb_dt = (qb - self._prev_qb) * alpha
            out["id"] = self._read_terminal_current("d", "di") + dqd_dt
            out["ig"] = self._read_current("g") + dqg_dt
            out["is"] = self._read_terminal_current("s", "si") + dqs_dt
            out["ie"] = self._read_current("e") + dqb_dt
            self._prev_qg = qg
            self._prev_qd = qd
            self._prev_qs = qs
            self._prev_qb = qb
            self._has_prev_q = True
        else:
            out["id"] = self._read_current_from(total_residual, "d")
            out["ig"] = self._read_current_from(total_residual, "g")
            out["is"] = self._read_current_from(total_residual, "s")
            out["ie"] = self._read_current_from(total_residual, "e")

        # Drain-source current (Ids = Id - Is for common-source configuration)
        out["ids"] = out["id"] - out["is"]

        self._sim.copy_solve_to_prev()
        self._has_prev_solve = True
        self._sim.has_prev_solve = True
        if len(self._sim.state_prev) == len(self._sim.state_next):
            self._sim.state_prev, self._sim.state_next = self._sim.state_next, self._sim.state_prev
        return out


def select_tsmc7_variant(modelcard_path: str, model_type: str, L: float) -> str:
    """
    Select the appropriate TSMC7 model variant based on device length.

    TSMC7 modelcards have multiple variants (e.g., nch_svt_mac.1, .2, etc.)
    with different L ranges defined by lmin and lmax parameters.
    This function reads the modelcard and returns the full model name
    (including variant suffix) that matches the given device length.

    Args:
        modelcard_path: Path to TSMC7 .l file
        model_type: Base model name (e.g., "nch_svt_mac")
        L: Device length in meters

    Returns:
        Full model name with variant (e.g., "nch_svt_mac.5")

    Raises:
        ValueError: If no variant matches the given L, or if modelcard is invalid
    """
    # Escape special regex characters in model_type
    model_type_escaped = re.escape(model_type)
    # Match both nmos and pmos models
    variant_pattern = re.compile(
        rf'\.model ({model_type_escaped}\.\d+) (nmos|pmos)'
    )

    # Read and parse the modelcard
    variants = []
    with open(modelcard_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        # Look for model definition
        match = variant_pattern.match(line.strip())
        if match:
            variant_name = match.group(1)
            lmin = None
            lmax = None

            # First, check the initial .model line for lmin and lmax
            # (some models like pch have them on the same line)
            lmin_match = re.search(r'\blmin\s*=\s*([0-9eE+\-\.]+)', line)
            lmax_match = re.search(r'\blmax\s*=\s*([0-9eE+\-\.]+)', line)

            if lmin_match:
                lmin = parse_number_with_suffix(lmin_match.group(1))
            if lmax_match:
                lmax = parse_number_with_suffix(lmax_match.group(1))

            # If not found on the initial line, scan continuation lines
            if lmin is None or lmax is None:
                j = idx + 1
                while j < len(lines):
                    param_line = lines[j]
                    if not param_line.startswith("+"):
                        # End of model block
                        break

                    # Extract lmin and lmax
                    if lmin is None:
                        lmin_match = re.search(r'\blmin\s*=\s*([0-9eE+\-\.]+)', param_line)
                        if lmin_match:
                            lmin = parse_number_with_suffix(lmin_match.group(1))

                    if lmax is None:
                        lmax_match = re.search(r'\blmax\s*=\s*([0-9eE+\-\.]+)', param_line)
                        if lmax_match:
                            lmax = parse_number_with_suffix(lmax_match.group(1))

                    # Once we have both, we can stop scanning this model
                    if lmin is not None and lmax is not None:
                        break

                    j += 1

            # Only add variants that have both lmin and lmax
            if lmin is not None and lmax is not None:
                variants.append({
                    "name": variant_name,
                    "lmin": lmin,
                    "lmax": lmax
                })

        idx += 1

    if not variants:
        raise ValueError(
            f"No TSMC7 variants found for model '{model_type}' in {modelcard_path}"
        )

    # Find the best matching variant
    # L should satisfy: lmin <= L <= lmax
    # For overlapping ranges, prefer the variant with the smallest range
    # (i.e., the most precise match)
    matching_variants = [
        v for v in variants
        if v["lmin"] <= L <= v["lmax"]
    ]

    if not matching_variants:
        # No match found - provide helpful error message
        # Sort variants by lmin for better error display
        sorted_variants = sorted(variants, key=lambda v: v["lmin"])

        ranges_str = ", ".join(
            f"{v['name']} ({v['lmin']:.3e} to {v['lmax']:.3e})"
            for v in sorted_variants
        )

        raise ValueError(
            f"No TSMC7 variant found for L={L:.3e}. "
            f"Available ranges for {model_type}: {ranges_str}"
        )

    # Select the variant with the smallest range (most precise match)
    best_variant = min(matching_variants, key=lambda v: v["lmax"] - v["lmin"])
    return best_variant["name"]


__all__ = ["Model", "Instance", "parse_number_with_suffix", "select_tsmc7_variant"]
