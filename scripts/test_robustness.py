#!/usr/bin/env python3
import argparse
import math
import random
import resource
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
MODEL_NMOS = ROOT / "bsim-cmg-va" / "benchmark_test" / "modelcard.nmos"
MODEL_PMOS = ROOT / "bsim-cmg-va" / "benchmark_test" / "modelcard.pmos"


def pulse_value(t: float,
                v_low: float,
                v_high: float,
                rise: float,
                fall: float,
                on: float,
                period: float) -> float:
    if period <= 0.0:
        return v_low
    t_mod = t % period
    rise = max(rise, 0.0)
    fall = max(fall, 0.0)
    on = max(on, 0.0)
    rise_end = rise
    high_end = rise + on
    fall_end = rise + on + fall
    if t_mod < rise_end and rise > 0.0:
        return v_low + (v_high - v_low) * (t_mod / rise)
    if t_mod < high_end:
        return v_high
    if t_mod < fall_end and fall > 0.0:
        return v_high - (v_high - v_low) * ((t_mod - high_end) / fall)
    return v_low


def find_second_derivative_spikes(values: Sequence[float], threshold: float) -> List[int]:
    spikes: List[int] = []
    if len(values) < 3:
        return spikes
    for i in range(1, len(values) - 1):
        second = values[i + 1] - 2.0 * values[i] + values[i - 1]
        if abs(second) > threshold:
            spikes.append(i)
    return spikes


def integrate(values: Sequence[float], dt: float) -> float:
    return sum(values) * dt


def smooth_values(values: Sequence[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    half = window // 2
    smoothed: List[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def is_monotonic_increasing(pairs: Sequence[Tuple[float, float]]) -> bool:
    if not pairs:
        return True
    ordered = sorted(pairs, key=lambda x: x[0])
    last = ordered[0][1]
    for _, val in ordered[1:]:
        if val < last:
            return False
        last = val
    return True


def detect_linear_growth(rss_values: Sequence[float], warmup: int, max_per_step: float) -> bool:
    if len(rss_values) <= warmup + 1:
        return False
    deltas: List[float] = []
    for i in range(warmup + 1, len(rss_values)):
        deltas.append(rss_values[i] - rss_values[i - 1])
    if not deltas:
        return False
    avg_delta = sum(deltas) / len(deltas)
    return avg_delta > max_per_step


def _require_paths() -> None:
    if not OSDI_PATH.exists():
        raise RuntimeError(f"missing OSDI: {OSDI_PATH}")
    if not MODEL_NMOS.exists() or not MODEL_PMOS.exists():
        raise RuntimeError("missing modelcard(s)")


def run_pulse_test(temp_c: float,
                   dt: float,
                   t_stop: float,
                   v_high: float,
                   rise: float,
                   fall: float,
                   period: float,
                   on: float,
                   spike_threshold: Optional[float],
                   charge_tol: float) -> Tuple[bool, str]:
    import pycmg

    model = pycmg.Model(str(OSDI_PATH), str(MODEL_NMOS), "nmos1")
    inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
                          temperature=temp_c + 273.15)
    times = [i * dt for i in range(int(t_stop / dt) + 1)]
    ids: List[float] = []
    for t in times:
        vg = pulse_value(t, 0.0, v_high, rise, fall, on, period)
        out = inst.eval_tran({"d": 0.05, "g": vg, "s": 0.0, "e": 0.0}, t, dt)
        ids.append(float(out["id"]))

    max_abs_id = max(abs(v) for v in ids) if ids else 0.0
    margin = max(50 * dt, rise, fall)
    cycle_start = max(0.0, t_stop - period)
    start_idx = int(cycle_start / dt)
    mask: List[bool] = []
    for t in times:
        if t < cycle_start:
            mask.append(False)
            continue
        t_mod = t % period
        in_high = (t_mod > rise + margin) and (t_mod < rise + on - margin)
        in_low = (t_mod > rise + on + fall + margin)
        mask.append(in_high or in_low)
    smoothed = smooth_values(ids, 5)
    seconds: List[float] = []
    for i in range(1, len(smoothed) - 1):
        seconds.append(smoothed[i + 1] - 2.0 * smoothed[i] + smoothed[i - 1])
    if spike_threshold is not None:
        threshold = spike_threshold
    else:
        plateau_abs = [abs(seconds[i - 1]) for i in range(1, len(ids) - 1) if mask[i]]
        if plateau_abs:
            plateau_abs.sort()
            median_abs = plateau_abs[len(plateau_abs) // 2]
            p99_idx = max(0, int(0.99 * len(plateau_abs)) - 1)
            p99_abs = plateau_abs[p99_idx]
        else:
            median_abs = 0.0
            p99_abs = 0.0
        threshold = max(1e-12, 0.1 * max_abs_id, 50.0 * median_abs, 20.0 * p99_abs)
    spikes = [i for i in range(1, len(ids) - 1)
              if mask[i] and abs(seconds[i - 1]) > threshold]
    if spikes:
        return False, f"pulse smoothness failed: {len(spikes)} spikes over threshold"

    if period <= 0.0:
        return False, "pulse period must be positive"
    cycle_ids = ids[start_idx:]
    net_charge = integrate(cycle_ids, dt)
    if abs(net_charge) > charge_tol:
        return False, f"charge balance failed: net={net_charge:.3e} C"
    return True, "pulse test pass"


def run_param_sweep(temp_c: float,
                    iterations: int,
                    rng_seed: int,
                    rss_warmup: int,
                    rss_max_per_step: float) -> Tuple[bool, str]:
    import pycmg

    random.seed(rng_seed)
    model = pycmg.Model(str(OSDI_PATH), str(MODEL_NMOS), "nmos1")
    inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
                          temperature=temp_c + 273.15)
    rss_samples: List[float] = []
    for _ in range(iterations):
        l_val = random.uniform(1.0e-8, 6.0e-8)
        nfin_low = random.randint(1, 10)
        nfin_high = random.randint(nfin_low + 1, 20)
        inst.set_params({"L": l_val, "NFIN": float(nfin_low)}, allow_rebind=True)
        out_low = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
        inst.set_params({"L": l_val, "NFIN": float(nfin_high)}, allow_rebind=True)
        out_high = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
        pairs = [(nfin_low, abs(out_low["id"])), (nfin_high, abs(out_high["id"]))]
        if not is_monotonic_increasing(pairs):
            return False, f"Id not monotonic with NFIN: {pairs}"
        rss_samples.append(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    if detect_linear_growth(rss_samples, rss_warmup, rss_max_per_step):
        return False, "memory growth appears linear across iterations"
    return True, "param sweep pass"


def _thread_worker(temp_c: float,
                   rng_seed: int,
                   iterations: int,
                   errors: List[str]) -> None:
    import pycmg

    try:
        random.seed(rng_seed)
        model = pycmg.Model(str(OSDI_PATH), str(MODEL_NMOS), "nmos1")
        inst = pycmg.Instance(model, params={"L": 1.6e-8, "TFIN": 8e-9, "NFIN": 2.0},
                              temperature=temp_c + 273.15)
        for _ in range(iterations):
            vd = random.uniform(0.0, 1.2)
            vg = random.uniform(0.0, 1.2)
            out = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})
            if not all(math.isfinite(float(out[k])) for k in ("id", "ig", "is", "ie")):
                errors.append("non-finite output")
                return
    except Exception as exc:  # pragma: no cover - safety for thread
        errors.append(str(exc))


def run_thread_test(temp_c: float, thread_count: int, iterations: int) -> Tuple[bool, str]:
    errors: List[str] = []
    threads: List[threading.Thread] = []
    for i in range(thread_count):
        t = threading.Thread(target=_thread_worker, args=(temp_c, 1337 + i, iterations, errors))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if errors:
        return False, f"thread errors: {errors[:3]}"
    return True, "thread test pass"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Robustness tests for pycmg backend")
    ap.add_argument("--pulse", action="store_true", help="run pulse stability test")
    ap.add_argument("--param-sweep", action="store_true", help="run parameter sensitivity test")
    ap.add_argument("--threads", action="store_true", help="run concurrency test")
    ap.add_argument("--all", action="store_true", help="run all tests")
    ap.add_argument("--temp", type=float, default=27.0, help="temperature in C")
    ap.add_argument("--dt", type=float, default=1e-12, help="time step for pulse test")
    ap.add_argument("--t-stop", type=float, default=10e-9, help="pulse test duration")
    ap.add_argument("--pulse-period", type=float, default=1e-9, help="pulse period")
    ap.add_argument("--pulse-on", type=float, default=0.5e-9, help="pulse on duration")
    ap.add_argument("--pulse-rise", type=float, default=1e-12, help="pulse rise time")
    ap.add_argument("--pulse-fall", type=float, default=1e-12, help="pulse fall time")
    ap.add_argument("--pulse-high", type=float, default=1.0, help="pulse high voltage")
    ap.add_argument("--spike-threshold", type=float, default=None, help="2nd deriv threshold")
    ap.add_argument("--charge-tol", type=float, default=1e-14, help="net charge tolerance (C)")
    ap.add_argument("--sweep-iters", type=int, default=100, help="param sweep iterations")
    ap.add_argument("--sweep-seed", type=int, default=1234, help="param sweep RNG seed")
    ap.add_argument("--rss-warmup", type=int, default=10, help="rss warmup iterations")
    ap.add_argument("--rss-max-step", type=float, default=50.0, help="rss KB growth per iter")
    ap.add_argument("--thread-count", type=int, default=4, help="thread count")
    ap.add_argument("--thread-iters", type=int, default=500, help="dc evals per thread")
    args = ap.parse_args(argv)

    if not (args.pulse or args.param_sweep or args.threads or args.all):
        ap.error("Select --pulse, --param-sweep, --threads, or --all")

    _require_paths()
    failures: List[str] = []

    if args.all or args.pulse:
        ok, msg = run_pulse_test(args.temp, args.dt, args.t_stop, args.pulse_high,
                                 args.pulse_rise, args.pulse_fall, args.pulse_period,
                                 args.pulse_on, args.spike_threshold, args.charge_tol)
        print(msg)
        if not ok:
            failures.append(msg)

    if args.all or args.param_sweep:
        ok, msg = run_param_sweep(args.temp, args.sweep_iters, args.sweep_seed,
                                  args.rss_warmup, args.rss_max_step)
        print(msg)
        if not ok:
            failures.append(msg)

    if args.all or args.threads:
        ok, msg = run_thread_test(args.temp, args.thread_count, args.thread_iters)
        print(msg)
        if not ok:
            failures.append(msg)

    if failures:
        print(f"FAIL ({len(failures)}): {failures}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
