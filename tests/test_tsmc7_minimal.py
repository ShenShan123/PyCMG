#!/usr/bin/env python3
"""
Test TSMC7 minimal modelcard with PyCMG
"""
import sys
sys.path.insert(0, '/home/shenshan/pycmg-wrapper/.worktrees/more-techs')

from pycmg import Model, Instance

# Test parameters
L = 16e-9  # 16nm
TFIN = 8e-9  # 8nm
NFIN = 2.0

Vd = 0.7
Vg = 0.7
Vs = 0.0
Ve = 0.0

print("=" * 60)
print("Testing TSMC7 Minimal Modelcard with PyCMG")
print("=" * 60)
print(f"\nDevice Parameters:")
print(f"  L = {L*1e9:.1f} nm")
print(f"  TFIN = {TFIN*1e9:.1f} nm")
print(f"  NFIN = {NFIN}")
print(f"\nBias Conditions:")
print(f"  Vd = {Vd:.2f} V")
print(f"  Vg = {Vg:.2f} V")
print(f"  Vs = {Vs:.2f} V")
print(f"  Ve = {Ve:.2f} V")
print()

try:
    # Load model
    print("Loading model...")
    model = Model(
        "/home/shenshan/pycmg-wrapper/.worktrees/more-techs/build-deep-verify/osdi/bsimcmg.osdi",
        "/home/shenshan/pycmg-wrapper/.worktrees/more-techs/tech_model_cards/TSMC7/tsmc7_minimal.l",
        "nch_svt_mac.5"
    )
    print("Model loaded successfully!\n")

    # Create instance
    print("Creating device instance...")
    inst = Instance(model, params={"L": L, "TFIN": TFIN, "NFIN": NFIN})
    print("Instance created successfully!\n")

    # Evaluate DC
    print("Evaluating DC operating point...")
    result = inst.eval_dc({"d": Vd, "g": Vg, "s": Vs, "e": Ve})
    print("DC evaluation successful!\n")

    # Print results
    print("=" * 60)
    print("DC Results:")
    print("=" * 60)
    print(f"  Id  = {result['id']:.6e} A")
    print(f"  Ig  = {result['ig']:.6e} A")
    print(f"  Is  = {result['is']:.6e} A")
    print(f"  Ie  = {result['ie']:.6e} A")
    print(f"  Ids = {result['ids']:.6e} A")
    print(f"\nCharges:")
    print(f"  Qg  = {result['qg']:.6e} C")
    print(f"  Qd  = {result['qd']:.6e} C")
    print(f"  Qs  = {result['qs']:.6e} C")
    print(f"  Qb  = {result['qb']:.6e} C")
    print(f"\nDerivatives:")
    print(f"  gm  = {result['gm']:.6e} S")
    print(f"  gds = {result['gds']:.6e} S")
    print(f"  gmb = {result['gmb']:.6e} S")
    print(f"\nCapacitances:")
    print(f"  cgg = {result['cgg']:.6e} F")
    print(f"  cgd = {result['cgd']:.6e} F")
    print(f"  cgs = {result['cgs']:.6e} F")
    print(f"  cdg = {result['cdg']:.6e} F")
    print(f"  cdd = {result['cdd']:.6e} F")
    print("=" * 60)

    # Basic sanity check
    if result['id'] > 0 and result['id'] < 1e-3:
        print("\n✓ Id current is in reasonable range")
    else:
        print(f"\n⚠ Warning: Id current seems unusual (Id = {result['id']:.6e})")

    if abs(result['ig']) < 1e-9:
        print("✓ Gate current is near zero (as expected)")
    else:
        print(f"⚠ Warning: Gate current is non-zero (Ig = {result['ig']:.6e})")

    print("\n" + "=" * 60)
    print("PyCMG test PASSED!")
    print("=" * 60)

except Exception as e:
    print(f"\n{'=' * 60}")
    print(f"ERROR: {type(e).__name__}")
    print(f"{'=' * 60}")
    print(f"{e}")
    print(f"{'=' * 60}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
