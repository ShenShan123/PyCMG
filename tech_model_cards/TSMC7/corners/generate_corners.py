#!/usr/bin/env python3
"""
Generate TSMC7 Process Corner Modelcards

Creates SS (Slow-Slow) and FF (Fast-Fast) corner modelcards
by applying parameter shifts to TT (typical) base.
"""

import re
import sys
from pathlib import Path

# TT (typical) parameter base values from tsmc7_simple.l
TT_PARAMS = {
    "dvt0": 0.05,
    "dvt1": 0.4,
    "u0": 0.03,
    "vsat": 150000,
    "rdsw": 15,
    "rshs": 142,
    "rshd": 142,
}

# SS (Slow-Slow) corner: Lower drive current, higher Vth
SS_SHIFTS = {
    "dvt0": 0.06,      # +0.01
    "dvt1": 0.45,      # +0.05
    "u0": 0.025,        # -17%
    "vsat": 120000,      # -20%
    "rdsw": 18,          # +20%
    "rshs": 170,         # +20%
    "rshd": 170,         # +20%
}

# FF (Fast-Fast) corner: Higher drive current, lower Vth
FF_SHIFTS = {
    "dvt0": 0.04,      # -0.01
    "dvt1": 0.35,      # -0.05
    "u0": 0.035,        # +17%
    "vsat": 180000,      # +20%
    "rdsw": 12,          # -20%
    "rshs": 114,         # -20%
    "rshd": 114,         # -20%
}


def replace_param(line, param, new_value):
    """Replace a parameter value in a line."""
    # Use non-capturing group (?:) to avoid unbalanced parenthesis
    pattern = rf"(?:{param}\\s*=\s*)([\d.eE+\-]+)"
    match = re.search(pattern, line, re.IGNORECASE)
    if match:
        old_value = match.group(2)
        return re.sub(pattern, f"+ {param} = {new_value}", line, flags=re.IGNORECASE)
    return line


def generate_corner(input_file, output_file, corner_name, shifts):
    """Generate a corner modelcard by applying parameter shifts."""
    # Read input file
    with open(input_file) as f:
        lines = f.readlines()

    # Apply shifts to all matching lines
    for i, line in enumerate(lines):
        modified = line
        for param, new_val in shifts.items():
            modified = replace_param(modified, param, new_val)
        lines[i] = modified

    # Update header comment
    header = lines[0]
    if corner_name == "SS":
        header = header.replace("Minimal", "SS (Slow-Slow)")
    elif corner_name == "FF":
        header = header.replace("Minimal", "FF (Fast-Fast)")
    lines[0] = header

    # Write output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"✓ Generated {output_file.name}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: generate_corners.py <input.l> <output.l> <corner_name>")
        print("Corner names: SS, FF")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    corner_name = sys.argv[3]

    if corner_name.upper() == "SS":
        shifts = SS_SHIFTS
    elif corner_name.upper() == "FF":
        shifts = FF_SHIFTS
    else:
        print(f"Unknown corner: {corner_name}")
        sys.exit(1)

    generate_corner(input_file, output_file, corner_name, shifts)
    print(f"✓ Created {corner_name} corner modelcard")
