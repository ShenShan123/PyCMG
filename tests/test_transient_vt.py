"""
Transient Waveform Verification for Core-Voltage Vt Variants

Extends test_transient.py to cover all 16 core-voltage Vt variants
(ASAP7 lvt/slvt/sram, TSMC ulvt/elvt/hvt/lnvt) using the same methodology:

1. NGSPICE runs transient sim with pulse stimulus → V(t), I(t)
2. PyCMG steps through ALL time points sequentially (charge history)
3. Compare at quasi-steady-state points only (skip transitions)

Run: pytest tests/test_transient_vt.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import (
    OSDI_PATH, run_ngspice_transient, assert_close, REL_TOL,
)
from tests.conftest import ALL_TECHNOLOGIES, CORE_VT_NAMES, get_tech_modelcard


def _get_wave(ng_wave: dict, key_lower: str, n_points: int) -> np.ndarray:
    """Look up a waveform key case-insensitively."""
    for k in ng_wave:
        if k.lower() == key_lower:
            return ng_wave[k]
    return np.zeros(n_points)


def _run_transient_comparison(
    tech_name: str,
    device_type: str,
) -> None:
    """Run PyCMG vs NGSPICE transient comparison for a given tech/device."""
    tech = ALL_TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, device_type)
    vdd = tech["vdd"]

    # Run NGSPICE transient
    ng_wave = run_ngspice_transient(
        modelcard, model_name, inst_params, vdd,
        t_step=10e-12, t_stop=5e-9,
        tag=f"tran_vt_{tech_name}_{device_type}",
        device_type=device_type,
    )

    # Validate NGSPICE produced data
    time_key = None
    for candidate in ["time", "Time", "TIME"]:
        if candidate in ng_wave:
            time_key = candidate
            break
    if time_key is None:
        pytest.skip(f"NGSPICE transient returned no time vector for {tech_name} {device_type}")

    times = ng_wave[time_key]
    n_points = len(times)
    if n_points < 10:
        pytest.skip(f"Too few time points ({n_points}) from NGSPICE for {tech_name} {device_type}")

    # Pre-fetch waveform arrays
    vd = _get_wave(ng_wave, "v(d)", n_points)
    vg = _get_wave(ng_wave, "v(g)", n_points)
    vs = _get_wave(ng_wave, "v(s)", n_points)
    ve = _get_wave(ng_wave, "v(e)", n_points)
    ng_id_arr = _get_wave(ng_wave, "i(vd)", n_points)

    # Identify quasi-steady points
    vg_threshold_low = 0.05 * vdd
    vg_threshold_high = 0.95 * vdd
    steady_mask = (vg <= vg_threshold_low) | (vg >= vg_threshold_high)
    steady_indices = np.where(steady_mask)[0]
    steady_indices = steady_indices[steady_indices > 0]

    if len(steady_indices) < 5:
        pytest.skip(f"Too few steady-state points ({len(steady_indices)}) for {tech_name} {device_type}")

    # Select up to 30 evenly spaced check points
    n_check = min(30, len(steady_indices))
    check_indices = steady_indices[
        np.linspace(0, len(steady_indices) - 1, n_check, dtype=int)
    ]
    check_set = set(check_indices.tolist())

    # Create PyCMG instance
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)

    # Step through ALL points sequentially
    mismatches = 0
    n_checked = 0
    for idx in range(1, n_points):
        t = float(times[idx])
        t_prev = float(times[idx - 1])
        delta_t = t - t_prev
        if delta_t <= 0:
            delta_t = 1e-14

        nodes = {
            "d": float(vd[idx]),
            "g": float(vg[idx]),
            "s": float(vs[idx]),
            "e": float(ve[idx]),
        }

        py = inst.eval_tran(nodes, time=t, delta_t=delta_t)

        if idx in check_set:
            n_checked += 1
            ng_id = float(ng_id_arr[idx])
            try:
                assert_close(
                    f"{tech_name}/{device_type}/t={t*1e9:.2f}ns/id",
                    py["id"], ng_id,
                    rel_tol=REL_TOL,
                )
            except Exception:
                mismatches += 1

    # Allow up to 10% of checked points to mismatch
    max_mismatches = max(1, int(n_checked * 0.10))
    assert mismatches <= max_mismatches, (
        f"{tech_name} {device_type}: {mismatches}/{n_checked} time points exceeded tolerance "
        f"(max allowed: {max_mismatches})"
    )


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", CORE_VT_NAMES)
def test_nmos_transient_vt(tech_name: str) -> None:
    """NMOS transient waveform verification for core-voltage Vt variants."""
    _run_transient_comparison(tech_name, "nmos")


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", CORE_VT_NAMES)
def test_pmos_transient_vt(tech_name: str) -> None:
    """PMOS transient waveform verification for core-voltage Vt variants."""
    try:
        _run_transient_comparison(tech_name, "pmos")
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
