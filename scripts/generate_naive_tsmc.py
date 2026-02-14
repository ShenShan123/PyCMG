#!/usr/bin/env python3
"""
Generate naive TSMC modelcards from full TSMC PDK.

This script extracts parameters from the full TSMC PDK (which has .global + variant structure)
and generates naive single-model modelcards that can be used directly with both PyCMG and NGSPICE+OSDI.

Supports all TSMC FinFET technology nodes:
- TSMC5 (5nm)
- TSMC7 (7nm)
- TSMC12 (12nm)
- TSMC16 (16nm)

Key Design Decisions:
- Instance parameters (L, W, TFIN, NFIN, NF, MULTI) are NOT baked into modelcard
- Only process parameters (level, eot, hfin, etc.) are written to modelcard
- Instance geometry is provided in netlist (.sp/.cir file) when instantiating device

Usage:
    python scripts/generate_naive_tsmc.py \
        --tech TSMC7 \
        --pdk tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l \
        --output tech_model_cards/TSMC7/naive/ \
        --devices nch_svt_mac,nch_lvt_mac,nch_ulvt_mac,pch_svt_mac \
        --lengths 16e-9,20e-9,24e-9
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory to path to import from pycmg
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycmg.ctypes_host import _extract_model_params, _find_length_variant, parse_number_with_suffix


# Parameters that should NOT be included in naive modelcards (instance parameters)
_INSTANCE_PARAMS = {
    'l', 'lmin', 'lmax',  # Length parameters
    'w', 'wmin', 'wmax',  # Width parameters
    'tfin', 'tfinmin', 'tfinmax',  # Fin thickness
    'nfin', 'nfinmin', 'nfinmax',  # Fin count
    'nf', 'nfmulti',  # Number of fingers
    'multi',  # Multiplier
}


def generate_naive_tsmc_modelcard(
    pdk_path: str,
    model_type: str,
    device_type: str,
    L: float,
    output_path: str,
    tech: str,
) -> None:
    """
    Generate naive TSMC modelcard by merging global + variant parameters.

    Args:
        pdk_path: Path to full TSMC PDK file (e.g., cln7_1d8_sp_v1d2_2p2.l)
        model_type: "nch" for NMOS or "pch" for PMOS
        device_type: Device type (e.g., "svt_mac", "lvt_mac", "ulvt_mac")
        L: Target gate length in meters (e.g., 16e-9)
        output_path: Output file path
        tech: Technology name for header (e.g., "TSMC7")
    """
    base_name = f"{model_type}_{device_type}"  # e.g., "nch_svt_mac"

    # Extract global model parameters (base)
    expected_type = "nmos" if model_type == "nch" else "pmos"
    global_params = _extract_model_params(pdk_path, f"{base_name}.global", expected_type)

    # Find which variant matches the L value
    variant_num = _find_length_variant(pdk_path, base_name, L)

    # Extract variant model parameters
    variant_params = _extract_model_params(pdk_path, f"{base_name}.{variant_num}", expected_type)

    # Merge: variant overrides global
    merged_params = {**global_params, **variant_params}

    # Write naive modelcard (PROCESS PARAMETERS ONLY)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        # Header
        f.write(f"* Naive {tech} {device_type} modelcard for L={L*1e9:.1f}nm\n")
        f.write(f"* Generated from: {pdk_path}\n")
        f.write(f"* Device: {base_name}, Variant: .{variant_num}\n")
        f.write(f"* Process corner: TT (typical)\n")
        f.write(f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("*\n")
        f.write("* This is a NAIVE modelcard - single .model definition without subcircuits.\n")
        f.write("* Instance parameters (L, W, TFIN, NFIN, NF, MULTI) should be provided\n")
        f.write("* in the netlist when instantiating the device.\n")
        f.write("*\n")

        # Model definition
        # Use "bsimcmg" as model type for OSDI compatibility (NGSPICE doesn't recognize level=72 nmos/pmos)
        f.write(f".model {base_name} bsimcmg (\n")

        # Write all PROCESS parameters (not instance parameters!)
        param_count = 0
        skipped_sentinels = []
        for key, val in merged_params.items():
            # Skip instance parameters
            if key.lower() in _INSTANCE_PARAMS:
                continue

            # Skip sentinel values (TSMC PDKs use -999*10^n as "use default" markers)
            # These extreme values cause OSDI "out of bounds" errors during init.
            try:
                fval = float(val)
                if abs(fval) > 1e9 and str(val).lstrip('-').startswith('999'):
                    skipped_sentinels.append(f"{key}={val}")
                    continue
            except (ValueError, TypeError):
                pass

            # Format parameter line
            f.write(f"  + {key} = {val}\n")
            param_count += 1

        f.write(")\n")
        f.write(f"* Total parameters: {param_count}\n")
        if skipped_sentinels:
            f.write(f"* Skipped sentinel values: {', '.join(skipped_sentinels)}\n")

    msg = f"Generated: {output_file} ({param_count} parameters)"
    if skipped_sentinels:
        msg += f" (skipped {len(skipped_sentinels)} sentinel(s): {', '.join(skipped_sentinels)})"
    print(msg)


def batch_generate_naive_modelcards(
    pdk_path: str,
    output_dir: str,
    devices: List[str],
    lengths: List[float],
    tech: str,
) -> None:
    """
    Batch generate naive modelcards for multiple devices and lengths.

    Args:
        pdk_path: Path to full TSMC PDK file
        output_dir: Output directory for naive modelcards
        devices: List of device names (e.g., ["nch_svt_mac", "pch_svt_mac"])
        lengths: List of gate lengths in meters (e.g., [16e-9, 20e-9, 24e-9])
        tech: Technology name for header (e.g., "TSMC7")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_files = 0
    errors = []
    for device in devices:
        # Parse device name (e.g., "nch_svt_mac" -> model_type="nch", device_type="svt_mac")
        parts = device.split("_", 1)
        if len(parts) != 2:
            msg = f"Invalid device name format: {device}"
            print(f"Warning: {msg}, skipping...", file=sys.stderr)
            errors.append(msg)
            continue

        model_type, device_type = parts[0], parts[1]

        for L in lengths:
            # Generate filename: {model_name}_l{L_nm}nm.l
            L_nm = int(L * 1e9)
            filename = f"{device}_l{L_nm}nm.l"
            file_path = output_path / filename

            try:
                generate_naive_tsmc_modelcard(
                    str(pdk_path),
                    model_type,
                    device_type,
                    L,
                    str(file_path),
                    tech,
                )
                total_files += 1
            except Exception as e:
                msg = f"Error generating {filename}: {e}"
                print(msg, file=sys.stderr)
                errors.append(msg)

    print(f"\nTotal files generated: {total_files}")
    print(f"Output directory: {output_dir}")
    if errors:
        print(f"\n{len(errors)} error(s) encountered:")
        for err in errors:
            print(f"  - {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate naive TSMC modelcards from full TSMC PDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate TSMC7 modelcards
  python scripts/generate_naive_tsmc.py \
      --tech TSMC7 \
      --pdk tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l \
      --output tech_model_cards/TSMC7/naive/ \
      --devices nch_svt_mac \
      --lengths 16e-9

  # Batch generate multiple devices/lengths for TSMC5
  python scripts/generate_naive_tsmc.py \
      --tech TSMC5 \
      --pdk tech_model_cards/TSMC5/cln5_1d2_sp_v1d2_2p2.l \
      --output tech_model_cards/TSMC5/naive/ \
      --devices nch_svt_mac,nch_lvt_mac,pch_svt_mac,pch_lvt_mac \
      --lengths 16e-9,20e-9,24e-9
        """
    )

    parser.add_argument(
        "--tech",
        required=True,
        choices=["TSMC5", "TSMC7", "TSMC12", "TSMC16"],
        help="Technology node (TSMC5, TSMC7, TSMC12, or TSMC16)"
    )
    parser.add_argument(
        "--pdk",
        required=True,
        help="Path to full TSMC PDK file (e.g., cln7_1d8_sp_v1d2_2p2.l)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for naive modelcards"
    )
    parser.add_argument(
        "--devices",
        required=True,
        help="Comma-separated list of device names (e.g., nch_svt_mac,pch_svt_mac)"
    )
    parser.add_argument(
        "--lengths",
        required=True,
        help="Comma-separated list of gate lengths in meters (e.g., 16e-9,20e-9,24e-9)"
    )

    args = parser.parse_args()

    # Parse device list
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    # Parse length list (support scientific notation and suffixes like "16n" for 16e-9)
    lengths = []
    for l_str in args.lengths.split(","):
        l_str = l_str.strip()
        # First try to parse as number with suffix (e.g., "16n" -> 16e-9)
        try:
            lengths.append(parse_number_with_suffix(l_str))
        except ValueError:
            # If that fails, try direct float conversion
            try:
                lengths.append(float(l_str))
            except ValueError:
                print(f"Warning: Invalid length '{l_str}', skipping...", file=sys.stderr)

    if not devices:
        print("Error: No valid devices specified", file=sys.stderr)
        sys.exit(1)

    if not lengths:
        print("Error: No valid lengths specified", file=sys.stderr)
        sys.exit(1)

    # Check PDK file exists
    if not Path(args.pdk).exists():
        print(f"Error: PDK file not found: {args.pdk}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating naive {args.tech} modelcards...")
    print(f"  PDK: {args.pdk}")
    print(f"  Output: {args.output}")
    print(f"  Devices: {', '.join(devices)}")
    print(f"  Lengths: {', '.join(f'{L*1e9:.1f}nm' for L in lengths)}")
    print()

    batch_generate_naive_modelcards(
        args.pdk,
        args.output,
        devices,
        lengths,
        args.tech,
    )


if __name__ == "__main__":
    main()
