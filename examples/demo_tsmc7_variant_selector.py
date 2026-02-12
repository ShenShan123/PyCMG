#!/usr/bin/env python3
"""
Demonstration of TSMC7 variant selector function.

This script shows how to use the select_tsmc7_variant() function to select
the appropriate model variant based on device length.
"""

import sys
sys.path.insert(0, '/home/shenshan/pycmg-wrapper/.worktrees/more-techs')

from pycmg.ctypes_host import select_tsmc7_variant

# Path to TSMC7 modelcard
MODELCARD_PATH = "/home/shenshan/pycmg-wrapper/.worktrees/more-techs/tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l"

print("=" * 80)
print("TSMC7 Variant Selector Demonstration")
print("=" * 80)
print()

# Test different device lengths
test_lengths = [
    (8e-9, "8 nm (minimum)"),
    (10e-9, "10 nm (short)"),
    (28e-9, "28 nm (mid-range)"),
    (50e-9, "50 nm (nominal)"),
    (72e-9, "72 nm (boundary)"),
    (100e-9, "100 nm (long)"),
    (120e-9, "120 nm (boundary)"),
    (200e-9, "200 nm (very long)"),
    (240e-9, "240 nm (maximum)"),
]

print("NCH (NMOS) Model Selection:")
print("-" * 80)
for L, description in test_lengths:
    try:
        variant = select_tsmc7_variant(MODELCARD_PATH, "nch_svt_mac", L)
        print(f"  L = {L:10.3e} m ({description:20s}) -> {variant}")
    except ValueError as e:
        print(f"  L = {L:10.3e} m ({description:20s}) -> ERROR: {e}")

print()

# Test PCH model
print("PCH (PMOS) Model Selection:")
print("-" * 80)
for L, description in test_lengths[2:6]:  # Test a subset
    try:
        variant = select_tsmc7_variant(MODELCARD_PATH, "pch_svt_mac", L)
        print(f"  L = {L:10.3e} m ({description:20s}) -> {variant}")
    except ValueError as e:
        print(f"  L = {L:10.3e} m ({description:20s}) -> ERROR: {e}")

print()

# Test error handling
print("Error Handling Examples:")
print("-" * 80)
try:
    variant = select_tsmc7_variant(MODELCARD_PATH, "nch_svt_mac", 1e-9)
except ValueError as e:
    print(f"  Out of range (L=1nm):")
    print(f"    {e}")

print()

try:
    variant = select_tsmc7_variant(MODELCARD_PATH, "nch_svt_mac", 1e-6)
except ValueError as e:
    print(f"  Out of range (L=1um):")
    # Show only first line of error
    lines = str(e).split('\n')
    print(f"    {lines[0]}")
    if len(lines) > 1:
        print(f"    {lines[1][:80]}...")

print()
print("=" * 80)
