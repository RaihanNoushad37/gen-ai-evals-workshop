"""Golden-value tests for metrics.py. Every number here is precomputed and exact."""

import math

import pytest

from evals_workshop import metrics


def approx(x, tol=1e-6):
    return pytest.approx(x, abs=tol)


# --- confusion / rates ---


def test_confusion_hand_built():
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    cm = metrics.confusion(y_true, y_pred)
    assert cm == {"tp": 3, "fp": 1, "fn": 2, "tn": 4}


def test_rates_hand_built():
    cm = {"tp": 3, "fp": 1, "fn": 2, "tn": 4}
    r = metrics.rates(cm)
    assert r["sensitivity"] == approx(3 / 5)
    assert r["specificity"] == approx(4 / 5)
    assert r["ppv"] == approx(3 / 4)
    assert r["npv"] == approx(4 / 6)
    assert r["accuracy"] == approx(7 / 10)


def test_rates_zero_denominator_is_nan():
    cm = {"tp": 0, "fp": 0, "fn": 5, "tn": 5}
    assert math.isnan(metrics.rates(cm)["ppv"])


def test_confusion_rejects_parse_failures():
    # -1 must not be silently counted as a correct pass
    with pytest.raises(ValueError):
        metrics.confusion([1, 0], [1, -1])


# --- intervals ---


def test_wald_interval():
    assert metrics.wald_interval(38, 40) == approx((0.882459, 1.017541))


def test_wald_interval_zero_width():
    lo, hi = metrics.wald_interval(40, 40)
    assert lo == approx(1.0)
    assert hi == approx(1.0)


def test_wilson_interval():
    assert metrics.wilson_interval(38, 40) == approx((0.834961, 0.986179))
    assert metrics.wilson_interval(40, 40) == approx((0.912378, 1.000000))
    assert metrics.wilson_interval(63, 150) == approx((0.343980, 0.500015))
    assert metrics.wilson_interval(21, 50) == approx((0.293750, 0.557666))


# --- agreement ---


def test_cohens_kappa():
    assert metrics.cohens_kappa(40, 9, 6, 45) == approx((0.850000, 0.699519))
    assert metrics.cohens_kappa(80, 9, 6, 5) == approx((0.850000, 0.315693))


def test_specific_agreement():
    assert metrics.specific_agreement(40, 9, 6, 45) == approx((0.842105, 0.857143))
    assert metrics.specific_agreement(80, 9, 6, 5) == approx((0.914286, 0.400000))


# --- bias / correction ---


def test_fixed_point():
    assert metrics.fixed_point(0.90, 0.70) == approx(0.750000)


def test_bias_at():
    assert metrics.bias_at(0.30, 0.90, 0.70) == approx(0.480000)
    assert metrics.bias_at(0.90, 0.90, 0.70) == approx(0.840000)
    assert metrics.bias_at(0.95, 0.90, 0.70) == approx(0.870000)


def test_corrected_rate():
    assert metrics.corrected_rate(66 / 1000, 63 / 150, 847 / 850) == approx(0.150000)


def test_corrected_rate_raises_on_uninformative_judge():
    with pytest.raises(ValueError):
        metrics.corrected_rate(0.5, 0.5, 0.5)


# --- variance decomposition / labels required ---


def test_variance_decomposition():
    vd = metrics.variance_decomposition(0.30, 0.90, 0.70, 200, 50, 50)
    assert vd["half_width"] == approx(0.236688)

    vd = metrics.variance_decomposition(0.30, 0.90, 0.70, 10_000, 50, 50)
    assert vd["half_width"] == approx(0.212229)
    assert vd["share_calibration"] == pytest.approx(0.995, abs=1e-3)

    vd = metrics.variance_decomposition(0.30, 0.90, 0.70, 10_000, 200, 200)
    assert vd["half_width"] == approx(0.106903)


def test_labels_required():
    assert metrics.labels_required(0.90, 0.70, 0.80, 0.10) == 290
    assert metrics.labels_required(0.95, 0.95, 0.80, 0.10) == 70


# --- McNemar / MDE ---


def test_mcnemar_n_matches_mde_inverse():
    assert metrics._mcnemar_n(0.05, 0.10) == pytest.approx(311.59, abs=0.01)
    assert metrics._mcnemar_n(0.05, 0.20) == pytest.approx(625.55, abs=0.01)


def test_minimum_detectable_effect_round_trips_mcnemar_n():
    n = metrics._mcnemar_n(0.05, 0.10)
    delta = metrics.minimum_detectable_effect(n, 0.10)
    assert delta == pytest.approx(0.05, abs=1e-3)


def test_minimum_detectable_effect_is_nan_without_discordance():
    # 0.0 would read as "any effect is detectable", the opposite of the truth
    assert math.isnan(metrics.minimum_detectable_effect(100, discordance=0.0))


def test_paired_compare_basic():
    a = [1, 1, 1, 1, 0, 0, 0, 0]
    b = [1, 1, 0, 0, 1, 1, 0, 0]
    result = metrics.paired_compare(a, b)
    assert result["n01"] == 2
    assert result["n10"] == 2
    assert result["p_value"] == pytest.approx(1.0)


# --- noise floor ---


def test_noise_floor_identical_runs():
    runs = [[1, 0, 1, 0, 1]] * 3
    nf = metrics.noise_floor(runs)
    assert nf["flip_rate"] == approx(0.0)
    assert nf["aggregate_sd"] == approx(0.0)
    assert nf["n_items"] == 5
    assert nf["k_runs"] == 3


def test_noise_floor_half_flip():
    run1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    run2 = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    nf = metrics.noise_floor([run1, run2])
    assert nf["flip_rate"] == approx(0.5)
