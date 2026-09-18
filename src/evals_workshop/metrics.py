"""Pure statistics for evaluating a judge as a diagnostic test. No I/O, no globals.

Functions used in the notebooks live here. metrics_extras holds the statistics the
workshop names but does not work through; both are re-exported under
`evals_workshop.metrics`.
"""

import math
from collections.abc import Sequence

from ._stats import norm_ppf
from .metrics_extras import (
    adaptive_allocation,
    agresti_coull_interval,
    bias_index,
    bootstrap_ci,
    clopper_pearson_interval,
    fleiss_kappa,
    jeffreys_interval,
    krippendorff_alpha,
    pabak,
    prevalence_index,
)

__all__ = [
    "adaptive_allocation",
    "agresti_coull_interval",
    "bias_at",
    "bias_index",
    "bootstrap_ci",
    "clopper_pearson_interval",
    "cohens_kappa",
    "confusion",
    "corrected_rate",
    "fixed_point",
    "fleiss_kappa",
    "jeffreys_interval",
    "krippendorff_alpha",
    "labels_required",
    "minimum_detectable_effect",
    "noise_floor",
    "pabak",
    "paired_compare",
    "prevalence_index",
    "rates",
    "specific_agreement",
    "variance_decomposition",
    "wald_interval",
    "wilson_interval",
]


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Count tp/fp/fn/tn, positive class = 1 = "contains unsupported content".

    Both arguments must be strictly 0/1. Judge output of -1 (a parse failure) is
    rejected rather than counted: silently folding it into tn would score a missed
    hallucination as a correct pass. Drop or repair those rows first, and report how
    many there were.
    """
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in (0, 1) or p not in (0, 1):
            raise ValueError(f"labels must be 0 or 1, got y_true={t}, y_pred={p}")
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def rates(cm: dict[str, int]) -> dict[str, float]:
    """Sensitivity, specificity, PPV, NPV and accuracy from a confusion matrix.

    Sensitivity and specificity are conditional on true class and so do not depend on
    prevalence. PPV, NPV and accuracy do: quoting them from a balanced calibration set
    says nothing about their value on production traffic. A rate with no denominator
    is nan, not an error.
    """
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    return {
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
    }


def cohens_kappa(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Observed agreement and Cohen's kappa for two raters on a binary label.

    Counts are (both yes, r1 yes/r2 no, r1 no/r2 yes, both no).

    Kappa is prevalence-sensitive: with skewed marginals, expected chance agreement
    rises and kappa falls even when the raters plainly agree, so two tables can share
    85% observed agreement and return 0.70 and 0.32. Read it alongside
    `specific_agreement`, and do not compare kappas measured at different prevalences.
    """
    n = a + b + c + d
    po = (a + d) / n
    p1_yes = (a + b) / n
    p2_yes = (a + c) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return po, kappa


def specific_agreement(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Positive and negative specific agreement, same 2x2 counts as `cohens_kappa`.

    Agreement on each class separately, which diagnoses directly what a single kappa
    only hints at. Neither figure is chance-corrected, so report them with kappa
    rather than instead of it.
    """
    psa = _safe_div(2 * a, 2 * a + b + c)
    nsa = _safe_div(2 * d, 2 * d + b + c)
    return psa, nsa


def wald_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Textbook normal-approximation interval. Deliberately not clipped to [0, 1].

    Inappropriate almost everywhere this workshop looks: it runs past 1 near the
    boundary and collapses to zero width at k == n. Present for comparison with
    `wilson_interval`, which is what you should actually use.
    """
    p = k / n
    z = norm_ppf(1 - alpha / 2)
    se = math.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. The default choice here.

    Stays inside [0, 1] and keeps sensible width at small n and at the boundaries,
    which is the regime a 25-per-class calibration set lives in. It covers one
    proportion only: it says nothing about the corrected rate, which carries
    calibration error too (see `variance_decomposition`).
    """
    p = k / n
    z = norm_ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin


def corrected_rate(p: float, sensitivity: float, specificity: float) -> float:
    """Recover the true rate from a raw judge score: (p + Sp - 1) / (Se + Sp - 1).

    Assumes Se and Sp transport from the calibration set to the data they are applied
    to. Robust to a change in the mix of instances, but not to a change in the judge's
    error rates — if this data is harder than calibration was, no algebra fixes that.

    Not clipped to [0, 1]: Se and Sp are themselves estimates, so the result can fall
    outside legitimately. Clip at the point of display, not here.
    """
    denom = sensitivity + specificity - 1
    if denom <= 0:
        raise ValueError("sensitivity + specificity must exceed 1 for the correction to be defined")
    return (p + specificity - 1) / denom


def bias_at(theta: float, sensitivity: float, specificity: float) -> float:
    """Expected raw judge score when the true rate is `theta`.

    The raw score is pulled toward `fixed_point`, so a judge flatters a weak system
    and understates a strong one.
    """
    return (sensitivity + specificity - 1) * theta + (1 - specificity)


def fixed_point(sensitivity: float, specificity: float) -> float:
    """The one true rate at which the raw judge score is unbiased.

    Above it the raw score understates, below it the raw score overstates. It is a
    property of the judge, not of the system under test.
    """
    return (1 - specificity) / (2 - sensitivity - specificity)


def variance_decomposition(
    p: float,
    sensitivity: float,
    specificity: float,
    n: int,
    m_pos: int,
    m_neg: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Split the uncertainty in a corrected rate into test-set and calibration parts.

    `n` is items judged, `m_pos`/`m_neg` are human labels per class. Because judging
    is cheap and labelling is not, the calibration terms usually dominate: that share
    is what tells you whether to buy more judged items or more labels.

    Treats Se and Sp as independent of the test set, and ignores their covariance;
    fine for budgeting, not a substitute for a bootstrap if you need exact coverage.
    """
    theta = corrected_rate(p, sensitivity, specificity)
    z = norm_ppf(1 - alpha / 2)
    var_test = p * (1 - p) / n
    var_calibration = (theta**2) * sensitivity * (1 - sensitivity) / m_pos + (
        (1 - theta) ** 2
    ) * specificity * (1 - specificity) / m_neg
    denom = (sensitivity + specificity - 1) ** 2
    half_width = z * math.sqrt((var_test + var_calibration) / denom)
    share_calibration = var_calibration / (var_test + var_calibration)
    return {
        "half_width": half_width,
        "var_test": var_test,
        "var_calibration": var_calibration,
        "share_calibration": share_calibration,
    }


def labels_required(
    sensitivity: float,
    specificity: float,
    theta: float,
    target_width: float,
    alpha: float = 0.05,
    cap: int = 200_000,
) -> int:
    """Human labels per class needed for a CI no wider than `target_width`.

    Assumes unlimited judged items, so it answers "how many labels?" not "how big an
    eval set?". Judge quality enters quadratically through (Se + Sp - 1)^2, which is
    why improving the judge is often cheaper than buying labels. Returns `cap` when
    the target is unreachable — check for that before quoting the number.
    """
    z = norm_ppf(1 - alpha / 2)
    numerator = (theta**2) * sensitivity * (1 - sensitivity) + ((1 - theta) ** 2) * specificity * (
        1 - specificity
    )
    denom = (sensitivity + specificity - 1) ** 2
    m = 10
    while m <= cap:
        half_width = z * math.sqrt(numerator / (m * denom))
        if 2 * half_width <= target_width:
            return m
        m += 10
    return cap


def _exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    pk = pmf[k]
    return sum(pi for pi in pmf if pi <= pk * (1 + 1e-9))


def paired_compare(a: Sequence[int], b: Sequence[int]) -> dict:
    """McNemar's test on two judges' correctness over the same items.

    Inputs are per-item correctness (1 = matched the human label). Pairing cancels
    item difficulty, which is most of the variance, so compare on identical items.

    Only discordant pairs carry information: agreeing items contribute nothing however
    many there are. Run it once per true class to compare sensitivity and specificity
    separately — a single overall test mixes two different questions.
    """
    n01 = n10 = 0
    for x, y in zip(a, b, strict=True):
        if x == 0 and y == 1:
            n01 += 1
        elif x == 1 and y == 0:
            n10 += 1
    discordant = n01 + n10
    if discordant == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(n10 - n01) - 1) ** 2 / discordant
    if discordant < 25:
        p_value = _exact_binomial_two_sided(min(n01, n10), discordant)
    else:
        p_value = math.erfc(math.sqrt(statistic / 2))
    return {"n01": n01, "n10": n10, "statistic": statistic, "p_value": p_value}


def _mcnemar_n(delta: float, discordance: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Closed-form McNemar sample size to detect effect `delta` at given discordance."""
    z_a = norm_ppf(1 - alpha / 2)
    z_b = norm_ppf(power)
    d = discordance
    return (z_a * math.sqrt(d) + z_b * math.sqrt(d - delta**2)) ** 2 / delta**2


def minimum_detectable_effect(
    n: int, discordance: float, power: float = 0.80, alpha: float = 0.05
) -> float:
    """Smallest difference a paired comparison at this size could detect.

    `discordance` is the fraction of items on which the two judges disagree; it drives
    the answer, so estimate it from a pilot rather than assuming. Returns nan when
    discordance is 0, where the question is undefined — not 0, which would read as
    "any effect is detectable".
    """
    if discordance <= 0:
        return float("nan")
    z_a = norm_ppf(1 - alpha / 2)
    z_b = norm_ppf(power)
    d = discordance
    coef_a = n + z_b**2
    coef_b = -2 * z_a * math.sqrt(n * d)
    coef_c = d * (z_a**2 - z_b**2)
    discriminant = coef_b**2 - 4 * coef_a * coef_c
    return (-coef_b + math.sqrt(discriminant)) / (2 * coef_a)


def noise_floor(runs: Sequence[Sequence[int]]) -> dict[str, float]:
    """Run-to-run variation of one judge: per-item flip rate and aggregate SD.

    `runs` are k verdict vectors over the same items in the same order, from repeated
    calls with the prompt unchanged. Characterise this once per judge version and
    reuse it: a change smaller than about twice the aggregate SD is not believable.

    Measures this judge on these items only — it is a noise floor, not an error rate.
    """
    k_runs = len(runs)
    n_items = len(runs[0]) if runs else 0
    flips = sum(1 for i in range(n_items) if len({run[i] for run in runs}) > 1)
    flip_rate = flips / n_items if n_items else float("nan")
    aggregates = [sum(run) / len(run) for run in runs]
    if k_runs < 2:
        return {
            "flip_rate": flip_rate,
            "aggregate_sd": float("nan"),
            "n_items": n_items,
            "k_runs": k_runs,
        }
    mean_agg = sum(aggregates) / k_runs
    variance = sum((a - mean_agg) ** 2 for a in aggregates) / (k_runs - 1)
    return {
        "flip_rate": flip_rate,
        "aggregate_sd": math.sqrt(variance),
        "n_items": n_items,
        "k_runs": k_runs,
    }
