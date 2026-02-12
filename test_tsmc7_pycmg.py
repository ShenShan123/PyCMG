#!/usr/bin/env python3
"""Test PyCMG with cleaned TSMC7 modelcard"""

import sys
sys.path.insert(0, '/home/shenshan/pycmg-wrapper/.worktrees/more-techs')

from pycmg.ctypes_host import Model, parse_modelcard

# Load the cleaned modelcard
modelcard_path = '/home/shenshan/pycmg-wrapper/.worktrees/more-techs/tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2_clean.l'
osdi_path = '/home/shenshan/pycmg-wrapper/.worktrees/more-techs/build-deep-verify/osdi/bsimcmg.osdi'

print(f"Loading modelcard from: {modelcard_path}")
print(f"Loading OSDI from: {osdi_path}")

try:
    # Parse modelcard (just to show what we parsed)
    parsed = parse_modelcard(modelcard_path)
    params = parsed.params
    print(f"\nParsed {len(params)} model parameters")

    # Show some key parameters
    print("\nKey parameters:")
    for key in ['level', 'eot', 'u0', 'vsat', 'lmin', 'lmax', 'nfinmin', 'nfinmax']:
        if key in params:
            print(f"  {key} = {params[key]}")

    # Load OSDI model (pass the modelcard path directly)
    print(f"\nLoading OSDI model...")
    model = Model(osdi_path, modelcard_path, model_name=parsed.name)
    print("OSDI model loaded successfully!")

    # Create a simple instance
    print("\nCreating instance...")
    instance_params = {
        'l': 120e-9,
        'nfin': 12,
    }
    from pycmg.ctypes_host import Instance
    inst = Instance(model, instance_params)
    print("Instance created successfully!")

    # Run DC evaluation
    print("\nRunning DC evaluation...")
    result = inst.eval_dc({
        'd': 0.5,
        'g': 0.8,
        's': 0.0,
        'e': 0.0,
    })

    print("DC evaluation successful!")
    print(f"  Id = {result['id']:.6e} A")
    print(f"  Ig = {result['ig']:.6e} A")
    print(f"  Is = {result['is']:.6e} A")
    print(f"  Ids = {result['ids']:.6e} A")

    print("\n" + "="*60)
    print("SUCCESS: Cleaned TSMC7 modelcard works with PyCMG!")
    print("="*60)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
