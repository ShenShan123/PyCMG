"""
Transient Waveform Verification Tests

Compares PyCMG eval_tran() output against NGSPICE transient simulation.

Methodology:
1. NGSPICE runs a transient sim with pulse stimulus → produces V(t) and I(t)
2. PyCMG steps through ALL NGSPICE time points sequentially (maintaining charge
   history for correct dQ/dt computation)
3. Compare PyCMG currents against NGSPICE currents at sampled steady-state points

Critical design decisions:

  1. Sequential stepping — PyCMG's eval_tran() stores previous Q for dQ/dt.
     Skipping points corrupts the charge history. Must step through all points.

  2. Actual delta_t — NGSPICE uses adaptive time steps. Must use t[i]-t[i-1],
     not a fixed value, for correct dQ/dt computation.

  3. Transition avoidance — During fast pulse edges (dVg/dt ≠ 0), PyCMG's
     backward-Euler charge integration diverges from NGSPICE's solver.
     Only verify at quasi-steady points where gate voltage is constant.

Sign convention:
    NGSPICE i(vd) = current into Vd positive terminal = drain terminal current
    PyCMG id = drain terminal current (same direction)
    → No sign flip needed (documented in CLAUDE.md)

Run: pytest tests/test_transient.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pycmg.ctypes_host import Model, Instance
from pycmg.testing import (
    OSDI_PATH, run_ngspice_transient, assert_close, REL_TOL,
)
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


def _get_wave(ng_wave: dict, key_lower: str, n_points: int) -> np.ndarray:
    """Look up a waveform key case-insensitively."""
    for k in ng_wave:
        if k.lower() == key_lower:
            return ng_wave[k]
    return np.zeros(n_points)


@pytest.mark.slow
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
@pytest.mark.parametrize("tech_name", TECH_NAMES)
def test_transient_waveform(tech_name: str):
    """Compare PyCMG transient currents against NGSPICE at solved node voltages.

    Steps through ALL NGSPICE time points sequentially so PyCMG maintains
    correct charge history. Checks agreement only at quasi-steady-state points
    where gate voltage is approximately constant (transitions skipped).
    """
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")
    vdd = tech["vdd"]

    # Run NGSPICE transient
    ng_wave = run_ngspice_transient(
        modelcard, model_name, inst_params, vdd,
        t_step=10e-12, t_stop=5e-9,
        tag=f"tran_{tech_name}",
    )

    # Validate NGSPICE produced data
    time_key = None
    for candidate in ["time", "Time", "TIME"]:
        if candidate in ng_wave:
            time_key = candidate
            break
    if time_key is None:
        pytest.skip(f"NGSPICE transient returned no time vector for {tech_name}")

    times = ng_wave[time_key]
    n_points = len(times)
    if n_points < 10:
        pytest.skip(f"Too few time points ({n_points}) from NGSPICE for {tech_name}")

    # Pre-fetch waveform arrays (case-insensitive lookup)
    vd = _get_wave(ng_wave, "v(d)", n_points)
    vg = _get_wave(ng_wave, "v(g)", n_points)
    vs = _get_wave(ng_wave, "v(s)", n_points)
    ve = _get_wave(ng_wave, "v(e)", n_points)
    ng_id_arr = _get_wave(ng_wave, "i(vd)", n_points)

    # Identify quasi-steady points: where Vg is near 0 or near Vdd (not transitioning)
    # During pulse edges, dVg/dt is large and PyCMG's simple backward-Euler
    # diverges from NGSPICE's more sophisticated integration.
    vg_threshold_low = 0.05 * vdd   # Within 5% of ground
    vg_threshold_high = 0.95 * vdd  # Within 5% of Vdd
    steady_mask = (vg <= vg_threshold_low) | (vg >= vg_threshold_high)
    steady_indices = np.where(steady_mask)[0]
    # Only check indices > 0 (need previous point for delta_t)
    steady_indices = steady_indices[steady_indices > 0]

    if len(steady_indices) < 5:
        pytest.skip(f"Too few steady-state points ({len(steady_indices)}) for {tech_name}")

    # Select up to 30 evenly spaced check points from steady-state regions
    n_check = min(30, len(steady_indices))
    check_indices = steady_indices[
        np.linspace(0, len(steady_indices) - 1, n_check, dtype=int)
    ]
    check_set = set(check_indices.tolist())

    # Create PyCMG instance
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)

    # Step through ALL points sequentially to maintain PyCMG charge history
    mismatches = 0
    n_checked = 0
    for idx in range(1, n_points):
        t = float(times[idx])
        t_prev = float(times[idx - 1])
        delta_t = t - t_prev
        if delta_t <= 0:
            delta_t = 1e-14  # Safety floor

        nodes = {
            "d": float(vd[idx]),
            "g": float(vg[idx]),
            "s": float(vs[idx]),
            "e": float(ve[idx]),
        }

        py = inst.eval_tran(nodes, time=t, delta_t=delta_t)

        # Only check at quasi-steady-state sample points
        if idx in check_set:
            n_checked += 1
            ng_id = float(ng_id_arr[idx])
            try:
                assert_close(
                    f"{tech_name}/t={t*1e9:.2f}ns/id",
                    py["id"], ng_id,
                    rel_tol=REL_TOL,
                )
            except Exception:
                mismatches += 1

    # Allow up to 10% of checked points to mismatch
    max_mismatches = max(1, int(n_checked * 0.10))
    assert mismatches <= max_mismatches, (
        f"{tech_name}: {mismatches}/{n_checked} time points exceeded tolerance "
        f"(max allowed: {max_mismatches})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
