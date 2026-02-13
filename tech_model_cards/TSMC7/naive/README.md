# Naive TSMC7 Modelcards

This directory contains **naive** TSMC7 modelcards - simplified single-model versions that work directly with both PyCMG and NGSPICE+OSDI.

## What Are Naive Modelcards?

Naive modelcards are simplified versions of the full TSMC7 PDK that:
- Use single `.model` definitions (no `.global`/`.variant` structure)
- Remove subcircuit wrappers (`.subckt`/`.ends`)
- Bake `.global` + variant parameters into one model
- Work with NGSPICE+OSDI via direct model instantiation

## Comparison: Naive vs Full PDK

| Feature | Naive Modelcards | Full TSMC7 PDK |
|---------|-----------------|------------------|
| **File size** | ~50KB | ~4MB |
| **Loading time** | <1 second | ~10 seconds |
| **Structure** | Single `.model` | `.global` + 30 variants + subcircuits |
| **Variant selection** | Pre-baked | Automatic (L-based) |
| **Subcircuits** | No | Yes (`.subckt`/`.ends`) |
| **NGSPICE+OSDI** | ✅ Direct model calls | ❌ Subcircuits don't work |
| **Use case** | Fast testing, education, debugging | Production simulations |

## File Naming Convention

Naive modelcards follow the pattern:
```
{model_name}_l{L_nm}nm.l
```

Examples:
- `nch_svt_mac_l16nm.l` - SVT NMOS for L=16nm
- `nch_lvt_mac_l20nm.l` - LVT NMOS for L=20nm
- `pch_svt_mac_l24nm.l` - SVT PMOS for L=24nm

## Available Device Types

| Device Type | Description | NMOS | PMOS |
|-------------|-------------|-------|------|
| **svt_mac** | Standard Threshold (SVT) | ✅ | ✅ |
| **lvt_mac** | Low Threshold (LVT) | ✅ | ✅ |
| **ulvt_mac** | Ultra-Low Threshold (ULVT) | ✅ | ✅ |
| **18_mac** | 1.8V I/O devices | ✅ | ✅ |

## Available Lengths

Common gate lengths supported:
- **12nm** - Shortest channel
- **16nm** - Typical short channel
- **20nm** - Medium channel
- **24nm** - Long channel
- **30nm** - Longest channel

## Usage in PyCMG

```python
from pycmg import Model, Instance, parse_modelcard

# Load naive modelcard (exactly like ASAP7)
parsed = parse_modelcard(
    "tech_model_cards/TSMC7/naive/nch_svt_mac_l16nm.l",
    "nch_svt_mac"
)

# Create model and instance
model = Model("build-deep-verify/osdi/bsimcmg.osdi", parsed.params)
inst = Instance(model, L=16e-9, TFIN=6e-9, NFIN=2.0)

# DC analysis
result = inst.eval_dc({"d": 0.75, "g": 0.75, "s": 0.0, "e": 0.0})
print(f"Id = {result.id:.6e} A")
```

## Usage in NGSPICE

**Netlist File** (`test_naive.cir`):

```spice
* NGSPICE + OSDI with Naive TSMC7 Modelcard
.osdi build-deep-verify/osdi/bsimcmg.osdi
.include tech_model_cards/TSMC7/naive/nch_svt_mac_l16nm.l

* Device (direct model, NOT subcircuit)
N1 d g s e nch_svt_mac l=16e-9 tfin=6e-9 nfin=2

* Bias
Vd d 0 0.75
Vg g 0 0.75
Vs s 0 0
Ve e 0 0

* Analysis
.temp 27
.op
.end
```

**Running NGSPICE:**

```bash
export NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice
ngspice -b test_naive.cir
```

## Generating Naive Modelcards

Naive modelcards are generated from the full TSMC7 PDK using the generation script:

```bash
# Generate single device and length
python scripts/generate_naive_tsmc7.py \
    --pdk tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l \
    --output tech_model_cards/TSMC7/naive/ \
    --devices nch_svt_mac \
    --lengths 16e-9

# Batch generate multiple devices/lengths
python scripts/generate_naive_tsmc7.py \
    --pdk tech_model_cards/TSMC7/cln7_1d8_sp_v1d2_2p2.l \
    --output tech_model_cards/TSMC7/naive/ \
    --devices nch_svt_mac,nch_lvt_mac,nch_ulvt_mac,pch_svt_mac \
    --lengths 16e-9,20e-9,24e-9
```

## Key Design Decisions

### Instance Parameters NOT Baked

Naive modelcards do **NOT** include instance parameters:
- `L` - Gate length (provided in netlist)
- `W` - Device width (provided in netlist)
- `TFIN` - Fin thickness (provided in netlist)
- `NFIN` - Fin count (provided in netlist)
- `NF` - Number of fingers (provided in netlist)
- `MULTI` - Multiplier (provided in netlist)

Only **process parameters** are baked into the modelcard:
- `level` - Model level (72 for BSIM-CMG)
- `eot` - Equivalent oxide thickness
- `hfin` - Fin height
- Other physical/process parameters

### Why This Approach?

1. **Flexibility**: Same modelcard can be used with different device geometries
2. **NGSPICE Compatibility**: OSDI requires direct model calls (no subcircuits)
3. **Binary Verification**: Naive must match full PDK exactly (same OSDI binary)
4. **Fast Testing**: Small files enable quick iteration

## Verification

Naive modelcards are verified against:
1. **Full PDK**: Must produce identical results to full TSMC7 PDK
2. **NGSPICE**: Direct comparison with NGSPICE ground truth
3. **Binary Consistency**: Both use identical `bsimcmg.osdi` file

Run verification tests:
```bash
pytest tests/test_tsmc7_naive.py -v
```

## Migration Notes

**No Breaking Changes:**
- Both naive and full PDK approaches coexist
- `parse_modelcard()` for naive modelcards (ASAP7-style)
- `parse_tsmc7_pdk()` for full PDK (variant selection)

**Recommended Usage:**
- **Development/Testing**: Use naive modelcards (fast, simple)
- **Production**: Use full TSMC7 PDK (complete, accurate)

## See Also

- **Full PDK**: `../cln7_1d8_sp_v1d2_2p2.l`
- **Generation Script**: `scripts/generate_naive_tsmc7.py`
- **Verification Tests**: `tests/test_tsmc7_naive.py`
- **Main CLAUDE.md**: `../../CLAUDE.md`
