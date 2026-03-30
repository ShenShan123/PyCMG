"""
Modelcard parsing and parameter extraction for SPICE/TSMC PDK files.

This module provides:
- parse_number_with_suffix: SPICE number parsing (e.g., "1n" -> 1e-9)
- parse_modelcard: General SPICE modelcard parser
- parse_tsmc_pdk: TSMC PDK parameter extraction
- ParsedModel: Dataclass for parsed model results

Dependencies: osdi_types only (no core.py imports).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from .osdi_types import _to_lower

# Module-level regex for SPICE parameter assignment: NAME = VALUE[suffix]
_ASSIGN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?[a-zA-Z]*)"
)


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


def parse_modelcard(path: str, target_model_name: Optional[str] = None) -> ParsedModel:
    assign_re = _ASSIGN_RE
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
    assign_re = _ASSIGN_RE

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
    assign_re = _ASSIGN_RE

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
