"""Sanity checks for the named-but-not-worked-through statistics. No golden values
required by the brief; these just confirm the implementations are self-consistent."""

import pytest

from evals_workshop import metrics


def test_clopper_pearson_wider_than_wilson():
    cp = metrics.clopper_pearson_interval(38, 40)
    wi = metrics.wilson_interval(38, 40)
    assert cp[0] <= wi[0]
    assert cp[1] >= wi[1]


def test_jeffreys_bounds():
    lo, hi = metrics.jeffreys_interval(38, 40)
    assert 0 <= lo <= 0.95 <= hi <= 1


def test_agresti_coull_reasonable():
    lo, hi = metrics.agresti_coull_interval(38, 40)
    assert 0 <= lo < 0.95 < hi <= 1.05


def test_bootstrap_ci_reproducible():
    items = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    ci1 = metrics.bootstrap_ci(items, lambda xs: sum(xs) / len(xs), n_boot=500, seed=17)
    ci2 = metrics.bootstrap_ci(items, lambda xs: sum(xs) / len(xs), n_boot=500, seed=17)
    assert ci1 == ci2


def test_fleiss_kappa_perfect_agreement():
    table = [[3, 0], [3, 0], [0, 3]]
    assert metrics.fleiss_kappa(table) == pytest.approx(1.0)


def test_krippendorff_alpha_perfect_agreement():
    data = [[1, 0, 1, 1], [1, 0, 1, 1]]
    assert metrics.krippendorff_alpha(data) == pytest.approx(1.0)


def test_pabak_matches_kappa_at_50_50():
    po, kappa = metrics.cohens_kappa(45, 5, 5, 45)
    assert metrics.pabak(45, 5, 5, 45) == pytest.approx(kappa)


def test_adaptive_allocation_sums_to_budget():
    alloc = metrics.adaptive_allocation(0.5, 0.90, 0.70, 100)
    assert alloc["m_pos"] + alloc["m_neg"] == 100
