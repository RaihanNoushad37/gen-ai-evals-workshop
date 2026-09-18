"""Statistics named in the workshop but not worked through in the notebooks.

Implemented and documented so a curious student can follow a thread on their
own; re-exported from metrics.py so `from evals_workshop.metrics import X` works.
"""

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence

from ._stats import beta_ppf, norm_ppf


def jeffreys_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Bayesian interval with a Jeffreys Beta(0.5, 0.5) prior. Good small-sample coverage."""
    lower = 0.0 if k == 0 else beta_ppf(alpha / 2, k + 0.5, n - k + 0.5)
    upper = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
    return lower, upper


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (conservative) binomial interval by inverting the beta CDF."""
    lower = 0.0 if k == 0 else beta_ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lower, upper


def agresti_coull_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wald interval recentred on a shrunk estimate. Simpler than Wilson, similar coverage."""
    z = norm_ppf(1 - alpha / 2)
    n_tilde = n + z**2
    p_tilde = (k + z**2 / 2) / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return p_tilde - margin, p_tilde + margin


def bootstrap_ci(
    items: Sequence,
    statistic_fn: Callable[[Sequence], float],
    n_boot: int = 10_000,
    seed: int | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval. Resamples ITEMS with replacement, not cells."""
    rng = random.Random(seed)
    items = list(items)
    n = len(items)
    stats = sorted(statistic_fn([items[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(int((1 - alpha / 2) * n_boot), n_boot - 1)
    return stats[lo_idx], stats[hi_idx]


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float:
    """Agreement among a fixed number of raters per item, categories on columns.

    Inappropriate when the number of raters varies by item (use Krippendorff's alpha).
    """
    n_items = len(table)
    n_raters = sum(table[0])
    n_cats = len(table[0])
    p_j = [sum(row[j] for row in table) / (n_items * n_raters) for j in range(n_cats)]
    p_i = [(sum(x * x for x in row) - n_raters) / (n_raters * (n_raters - 1)) for row in table]
    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha(data: Sequence[Sequence]) -> float:
    """Nominal-metric alpha over a raters x items matrix; `None` marks a missing rating.

    Handles missing data and a variable number of raters per item, unlike Fleiss' kappa.
    """
    units = list(zip(*data, strict=True))
    pairs = []
    for unit in units:
        values = [v for v in unit if v is not None]
        for i in range(len(values)):
            for j in range(len(values)):
                if i != j:
                    pairs.append((values[i], values[j]))
    if not pairs:
        return float("nan")
    disagree = sum(1 for a, b in pairs if a != b)
    do = disagree / len(pairs)
    counts = Counter(v for pair in pairs for v in pair)
    total = sum(counts.values())
    de = 1 - sum((c / total) ** 2 for c in counts.values())
    return 1 - do / de if de else float("nan")


def pabak(a: int, b: int, c: int, d: int) -> float:
    """Prevalence- and bias-adjusted kappa: kappa with chance agreement fixed at 0.5."""
    n = a + b + c + d
    po = (a + d) / n
    return 2 * po - 1


def prevalence_index(a: int, b: int, c: int, d: int) -> float:
    """Byrt et al.'s prevalence index: how far the positive/negative split is from 50/50."""
    n = a + b + c + d
    return abs(a - d) / n


def bias_index(a: int, b: int, c: int, d: int) -> float:
    """Byrt et al.'s bias index: how far the two raters' marginals are from each other."""
    n = a + b + c + d
    return abs(b - c) / n


def adaptive_allocation(
    theta: float, sensitivity: float, specificity: float, total_budget: int
) -> dict[str, int]:
    """Pilot-then-allocate split of a labelling budget across the two classes.

    Neyman-style: allocates in proportion to each class's contribution to the
    variance of the corrected rate (see metrics.variance_decomposition). Only
    sensible once a pilot estimate of sensitivity/specificity exists; a fresh
    judge should still start from a balanced calibration set.
    """
    w_pos = theta * math.sqrt(sensitivity * (1 - sensitivity))
    w_neg = (1 - theta) * math.sqrt(specificity * (1 - specificity))
    total_w = w_pos + w_neg
    m_pos = round(total_budget * w_pos / total_w)
    return {"m_pos": m_pos, "m_neg": total_budget - m_pos}
