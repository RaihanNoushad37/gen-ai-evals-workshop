# %% [markdown]
# # Notebook 4 — Use the judge
#
# **Objective:** apply a characterised instrument to data whose labels you cannot see,
# predict its bias, correct it, and check whether the correction recovered the truth.
#
# **Expected runtime:** ~30 minutes. **Expected API calls:** 24 (~3 min wall clock),
# plus 24 on `calib` if you have not run Notebook 3. The `prod` run starts in the
# first code cell — treat it as a week of production traffic. Steps 6 and 7 are pure
# arithmetic on numbers already in hand: no further calls.
#
# **Stuck on a `# TODO`?** `solutions/04_use_the_judge.ipynb` has a fully worked
# version. Falling back to it is fine — nobody should be stranded mid-session.

# %%
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from evals_workshop.config import settings
from evals_workshop.data import load_split, unlock_prod_labels
from evals_workshop.judges import RUBRIC_V1, build_judge, rubric_hash, run_judge
from evals_workshop.metrics import (
    bias_at,
    confusion,
    corrected_rate,  # noqa: F401 -- for the Step 4 TODO below
    fixed_point,  # noqa: F401 -- for the Step 3 TODO below
    labels_required,
    rates,
    variance_decomposition,
)

prod = load_split("prod", with_labels=False)
judge_v1 = build_judge(RUBRIC_V1, settings.judge_model)
result_prod = run_judge(judge_v1, prod)  # 24 calls — starts now, labels still hidden

# %% [markdown]
# ## Step 1 — the raw rate
#
# Parse failures are excluded, not defaulted to "no violation".

# %%
n_parse_failures = int((result_prod["verdict"] == -1).sum())
clean_prod = result_prod[result_prod["verdict"] != -1]
p_hat = clean_prod["verdict"].mean()

print(f"raw judge hallucination rate: {p_hat:.3f}  (parse failures: {n_parse_failures})")
if n_parse_failures / len(result_prod) > 0.05:
    print("WARNING: dropping these assumes they fail at random. If the judge chokes")
    print("on a particular kind of answer, that assumption is wrong and p_hat is biased.")
print("\nWrite one sentence you would tell your manager, using only this number:")
print("> (edit this line)")

# %% [markdown]
# ## Step 2 — measure Se and Sp on `calib` (needed before you can predict anything)

# %%
calib = load_split("calib")
result_calib = run_judge(judge_v1, calib)  # 24 calls, free if Notebook 3 already ran
merged_calib = result_calib.merge(calib[["id", "label"]], on="id")
clean_calib = merged_calib[merged_calib["verdict"] != -1]
cm = confusion(clean_calib["label"].tolist(), clean_calib["verdict"].tolist())
r = rates(cm)
se, sp = r["sensitivity"], r["specificity"]
print(f"sensitivity: {se:.3f}  specificity: {sp:.3f}")

# %% [markdown]
# ## Step 3 — predict the bias direction, *before* computing the correction
#
# The raw score is pulled toward a fixed point at `(1-Sp)/(2-Se-Sp)`. Below that
# point, the raw rate overstates true hallucination (the system looks worse than it
# is); above it, the raw rate understates it (the system looks better than it is).

# %%
# TODO: predict the bias direction from Se/Sp alone, before computing the correction.
#   metrics.fixed_point(se, sp) -> fp_point
#   compare p_hat to fp_point to say whether the raw rate over- or understates truth
fp_point = None
prediction = None
if fp_point is None:
    raise NotImplementedError("Step 3: compute `fp_point` and `prediction` above.")

print(f"fixed point: {fp_point:.3f}   raw rate: {p_hat:.3f}")
print(prediction)

thetas = np.linspace(0, 1, 101)
expected_raw = [bias_at(t, se, sp) for t in thetas]
plt.figure()
plt.plot(thetas, expected_raw, label="E[raw judge score]")
plt.plot(thetas, thetas, linestyle="--", color="gray", label="unbiased (y = x)")
plt.axvline(fp_point, linestyle=":", color="black", label=f"fixed point ({fp_point:.2f})")
plt.xlabel("true hallucination rate (theta)")
plt.ylabel("expected raw judge score")
plt.title("Judge bias: raw score vs true rate")
plt.legend()
plt.show()

# %% [markdown]
# ## Step 4 — apply the correction
#
# `corrected_rate` is not clipped to [0, 1] internally — Se and Sp are themselves
# estimates, so the correction can legitimately fall outside that range. Clip only
# for display.

# %%
if se + sp <= 1:
    raise SystemExit(
        f"Se + Sp = {se + sp:.3f}, so this judge carries no information and the "
        "correction is undefined (§6.3). Improve the rubric and re-measure."
    )

# TODO: apply the correction.
#   metrics.corrected_rate(p_hat, se, sp) -> theta_hat
#   clip only for display: min(max(theta_hat, 0.0), 1.0) -> theta_hat_display
theta_hat = None
theta_hat_display = None
if theta_hat is None:
    raise NotImplementedError("Step 4: compute `theta_hat` and `theta_hat_display` above.")

clipped = "  (clipped for display)" if theta_hat != theta_hat_display else ""
print(f"raw rate:       {p_hat:.3f}")
print(f"corrected rate: {theta_hat:.3f}{clipped}")

# %% [markdown]
# ## Step 5 — reveal the true labels and check the correction
#
# Se/Sp came from 12 labels per class, so the correction itself is a noisy estimate —
# on any single run it may not beat the raw rate. That's this sample size, not a bug;
# Step 6 quantifies exactly that.

# %%
unlock_prod_labels()
prod_full = load_split("prod")
true_rate = prod_full["label"].mean()

raw_error = abs(p_hat - true_rate)
corrected_error = abs(theta_hat_display - true_rate)
print(f"true rate:       {true_rate:.3f}")
print(f"raw error:       {raw_error:.3f}")
print(f"corrected error: {corrected_error:.3f}")
print("correction helped" if corrected_error < raw_error else "correction did not help this time")

# %% [markdown]
# ## Step 6 — decompose the uncertainty: more items, or more labels?

# %%
n_items = len(clean_prod)
m_pos = cm["tp"] + cm["fn"]  # labels actually behind Se, after any parse failures
m_neg = cm["tn"] + cm["fp"]
vd_current = variance_decomposition(p_hat, se, sp, n_items, m_pos, m_neg)
vd_more_items = variance_decomposition(p_hat, se, sp, n_items * 10, m_pos, m_neg)
vd_more_labels = variance_decomposition(p_hat, se, sp, n_items, m_pos * 2, m_neg * 2)

print(f"current (n={n_items}, {m_pos}/{m_neg} labels): half-width {vd_current['half_width']:.3f}")
print(f"10x judged items (n={n_items * 10}):     half-width {vd_more_items['half_width']:.3f}")
print(
    f"2x labels ({m_pos * 2}/{m_neg * 2}):            half-width {vd_more_labels['half_width']:.3f}"
)

scenarios = ["current", "10x items", "2x labels"]
var_tests = [vd_current["var_test"], vd_more_items["var_test"], vd_more_labels["var_test"]]
var_calibs = [
    vd_current["var_calibration"],
    vd_more_items["var_calibration"],
    vd_more_labels["var_calibration"],
]
plt.figure()
plt.bar(scenarios, var_tests, label="test-set variance")
plt.bar(scenarios, var_calibs, bottom=var_tests, label="calibration variance")
plt.ylabel("variance contribution")
plt.title("Where the uncertainty comes from")
plt.legend()
plt.show()

# %% [markdown]
# Judging 10x more items barely moves the half-width; doubling the labels does much
# more. That asymmetry is §6.4's point, not a coincidence of these numbers.

# %% [markdown]
# ## Step 7 — labels needed for a 10-point interval: your judge vs a better one

# %%
target_width = 0.10
quality_grid = [(se, sp), (0.90, 0.90), (0.95, 0.95), (0.98, 0.98)]
labels_by_quality = [
    labels_required(s, p, theta_hat_display, target_width) for s, p in quality_grid
]

print(f"labels per class for a {target_width:.2f}-wide interval:")
for (s, p), m in zip(quality_grid, labels_by_quality, strict=True):
    reachable = "" if m < 200_000 else "  (unreachable — judge too weak at any budget)"
    print(f"  Se={s:.2f} Sp={p:.2f} -> {m} labels/class{reachable}")

plt.figure()
plt.bar([f"Se{s:.2f}/Sp{p:.2f}" for s, p in quality_grid], labels_by_quality)
plt.ylabel(f"labels per class for a {target_width:.2f}-wide CI")
plt.title("Improving the judge vs buying more labels")
plt.show()

print(
    "\nWould improving the judge be cheaper than labelling? Compare the labels saved"
    " above to what it would cost you to tighten the rubric or add few-shot examples."
)

# %% [markdown]
# ## Step 8 — log to MLflow
#
# Same run-identification pattern as Notebooks 2 and 3: model, rubric hash and seed
# as params, so this run can be told apart from any other rubric or model.

# %%
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
with mlflow.start_run(run_name="judge_v1_prod"):
    mlflow.log_param("model", settings.judge_model)
    mlflow.log_param("rubric_hash", rubric_hash(RUBRIC_V1))
    mlflow.log_param("rubric_version", "v1")
    mlflow.log_param("random_seed", settings.random_seed)
    mlflow.log_metric("raw_rate", p_hat)
    mlflow.log_metric("corrected_rate", theta_hat_display)
    mlflow.log_metric("true_rate", true_rate)
    mlflow.log_metric("ci_half_width", vd_current["half_width"])

# %% [markdown]
# Open this run in the MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`)
# and look at `raw_rate`,
# `corrected_rate` and `true_rate` side by side — the same view that made v1 vs v2
# comparable in Notebook 3 shows here whether the correction actually pulled the
# estimate toward the truth, not just in which direction.

# %% [markdown]
# ## What you should have concluded
#
# - You predicted the bias direction from Se/Sp alone, before seeing the truth — and
#   the reveal in Step 5 either confirmed it or taught you why it didn't.
# - The corrected rate was closer to the truth than the raw rate (usually — check your
#   own run above).
# - Judging more items stops helping almost immediately; more human labels helps a lot
#   more, and a better judge helps more than either, because it enters quadratically.
# - The question to ask on Monday is not "how big should my eval set be?" — it's "how
#   many labels can I afford, and would a better judge be cheaper?"
