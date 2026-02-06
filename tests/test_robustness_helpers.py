import math

from scripts import test_robustness


def test_pulse_value_rise_fall() -> None:
    rise = 1e-12
    fall = 1e-12
    on = 10e-12
    period = 20e-12
    v0 = 0.0
    v1 = 1.0
    # Rising edge midpoint.
    assert math.isclose(
        test_robustness.pulse_value(0.5e-12, v0, v1, rise, fall, on, period),
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    # High plateau.
    assert math.isclose(
        test_robustness.pulse_value(2e-12, v0, v1, rise, fall, on, period),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    # Falling edge midpoint.
    assert math.isclose(
        test_robustness.pulse_value(11.5e-12, v0, v1, rise, fall, on, period),
        0.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    # Low plateau.
    assert math.isclose(
        test_robustness.pulse_value(15e-12, v0, v1, rise, fall, on, period),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_second_derivative_spike_detects() -> None:
    values = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    spikes = test_robustness.find_second_derivative_spikes(values, 5.0)
    assert spikes


def test_second_derivative_spike_ignores_small() -> None:
    values = [0.0, 0.1, 0.2, 0.1, 0.0]
    spikes = test_robustness.find_second_derivative_spikes(values, 1.0)
    assert not spikes


def test_charge_integral_zero() -> None:
    values = [1.0, -1.0, 1.0, -1.0]
    total = test_robustness.integrate(values, 1.0)
    assert math.isclose(total, 0.0, abs_tol=1e-12)


def test_monotonic_increasing() -> None:
    pairs = [(1, 1e-6), (2, 2e-6), (3, 2.9e-6)]
    assert test_robustness.is_monotonic_increasing(pairs)


def test_monotonic_increasing_rejects_drop() -> None:
    pairs = [(1, 1e-6), (2, 2e-6), (3, 1.5e-6)]
    assert not test_robustness.is_monotonic_increasing(pairs)


def test_detect_linear_memory_growth() -> None:
    rss = [100, 101, 102, 103, 104, 105, 106]
    assert test_robustness.detect_linear_growth(rss, warmup=2, max_per_step=0.5)


def test_detect_linear_memory_growth_allows_flat() -> None:
    rss = [100, 100, 101, 100, 101, 100, 101]
    assert not test_robustness.detect_linear_growth(rss, warmup=2, max_per_step=1.0)
