# TSMC7 Process Corner Modelcards

This directory contains simplified SS (Slow-Slow) and FF (Fast-Fast) process corner
variations of the TSMC7 7nm BSIM-CMG model.

## Corner Descriptions

### TT (Typical-Typical)
**File**: `../tsmc7_simple.l`
**Description**: Nominal process corner with typical parameters
**Use Case**: Default for most simulations

### SS (Slow-Slow)
**File**: `tsmc7_simple_ss.l`
**Description**: Slow NMOS and slow PMOS devices
**Characteristics**:
- Lower drive current (due to lower mobility)
- Higher threshold voltage (Vth)
- Higher series resistance

**Parameter Shifts vs TT**:
- `dvt0`: +0.01 (0.05 → 0.06)
- `dvt1`: +0.05 (0.4 → 0.45)
- `u0`: -17% (0.03 → 0.025)
- `vsat`: -20% (150000 → 120000)
- `rdsw`: +20% (15 → 18)
- `rshs`/`rshd`: +20% (142 → 170)

### FF (Fast-Fast)
**File**: `tsmc7_simple_ff.l`
**Description**: Fast NMOS and fast PMOS devices
**Characteristics**:
- Higher drive current (due to higher mobility)
- Lower threshold voltage (Vth)
- Lower series resistance

**Parameter Shifts vs TT**:
- `dvt0`: -0.01 (0.05 → 0.04)
- `dvt1`: -0.05 (0.4 → 0.35)
- `u0`: +17% (0.03 → 0.035)
- `vsat`: +20% (150000 → 180000)
- `rdsw`: -20% (15 → 12)
- `rshs`/`rshd`: -20% (142 → 114)

## Generation

The corner modelcards are generated from the TT base using `generate_corners.py`:
```bash
python tech_model_cards/TSMC7/corners/generate_corners.py \
  tech_model_cards/TSMC7/tsmc7_simple.l \
  tech_model_cards/TSMC7/corners/tsmc7_simple_ss.l SS

python tech_model_cards/TSMC7/corners/generate_corners.py \
  tech_model_cards/TSMC7/tsmc7_simple.l \
  tech_model_cards/TSMC7/corners/tsmc7_simple_ff.l FF
```

## Verification

Expected electrical characteristics at nominal bias (Vd=0.6V, Vg=0.75V, T=27°C):
- **TT**: Nominal current (Id ~ 0.5 mA for NMOS)
- **SS**: ~25-30% lower Id than TT
- **FF**: ~25-30% higher Id than TT

## Notes

- Corner modelcards follow TSMC/BSIM-CMG specification (level=72)
- Only essential parameters included (no TMI extensions)
- OSDI compatible format
- Parameter shifts based on industry-standard TSMC process variation

## References

- TSMC7 7nm PDK Documentation (proprietary)
- BSIM-CMG Specification: https://bsim.berkeley.edu/
- TSMC7_PROCESS_META_PLAN.md: Detailed implementation plan
