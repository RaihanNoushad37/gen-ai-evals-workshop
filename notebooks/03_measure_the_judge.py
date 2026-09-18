# %% [markdown]
# # Notebook 3 — Measure the judge
#
# **Objective:** stop treating the judge as a black box. Measure it as a diagnostic
# test — sensitivity and specificity separately — against the calibration set.
#
# **Expected runtime:** ~30 minutes. **Expected API calls:** 64 (24 v1 + 24 v2 +
# 16 noise floor), plus 12 on `dev` if you have not run Notebook 2. The calibration
# run starts in the first code cell so it runs while you read.
#
# **Stuck on a `# TODO`?** `solutions/03_measure_the_judge.ipynb` has a fully worked
# version. Falling back to it is fine — nobody should be stranded mid-session.

# %%
import mlflow

from evals_workshop.config import settings
from evals_workshop.data import load_split
from evals_workshop.judges import (
    RUBRIC_V1,
    add_few_shot_examples,
    build_judge,
    rubric_hash,
    run_judge,
    select_few_shot,
)
from evals_workshop.metrics import (
    cohens_kappa,
    confusion,
    minimum_detectable_effect,  # noqa: F401 -- for the Step 5 TODO below
    noise_floor,
    paired_compare,  # noqa: F401 -- for the Step 5 TODO below
    rates,
    wilson_interval,  # noqa: F401 -- for the Step 3 TODO below
)

calib = load_split("calib")
judge_v1 = build_judge(RUBRIC_V1, settings.judge_model)
result_v1 = run_judge(judge_v1, calib)  # 24 calls — starts now, read on while it runs

# %% [markdown]
# ## While that runs: what to expect
#
# `calib` is 24 items, balanced 12/12 by construction (§6.2). Sensitivity is estimated
# only from the 12 positives and specificity only from the 12 negatives, so each is
# governed by n=12, not n=24. Expect wide intervals — that is the subject of §6.4,
# not a defect in the measurement.

# %%
merged_v1 = result_v1.merge(calib[["id", "label"]], on="id")
merged_v1[["id", "label", "verdict"]].head(5)

# %% [markdown]
# ## Step 1 — confusion matrix, Se and Sp separately, then accuracy
#
# Parse failures are excluded from the confusion matrix, not defaulted to "no
# violation" — that would quietly inflate specificity.

# %%
n_parse_failures_v1 = int((merged_v1["verdict"] == -1).sum())
clean_v1 = merged_v1[merged_v1["verdict"] != -1]

# TODO: confusion matrix and Se/Sp for v1 on calib.
#   metrics.confusion(clean_v1["label"].tolist(), clean_v1["verdict"].tolist()) -> cm_v1
#   metrics.rates(cm_v1) -> r_v1
cm_v1 = None
r_v1 = None
if r_v1 is None:
    raise NotImplementedError("Step 1: compute `cm_v1` and `r_v1` above.")

print(f"parse failures: {n_parse_failures_v1}/{len(merged_v1)}")
print(f"confusion: {cm_v1}")
print(f"sensitivity: {r_v1['sensitivity']:.3f}")
print(f"specificity: {r_v1['specificity']:.3f}")
print(f"accuracy:    {r_v1['accuracy']:.3f}  <- would look fine even if Se were poor")

# %% [markdown]
# If sensitivity is much lower than specificity (or vice versa), accuracy on this
# balanced set still looks respectable — it averages the two. At natural prevalence
# (Notebook 4), that average is weighted toward whichever class dominates, which is
# exactly how accuracy hides a judge that misses most of the failures.

# %% [markdown]
# ## Step 2 — kappa against the human labels: are you better or worse than the judge?
#
# Paste the kappa Notebook 1 printed for you, then compare.

# %%
my_nb1_kappa = None  # TODO: paste the "cohen's kappa (yours)" value from Notebook 1

raw_agreement_v1, judge_kappa_v1 = cohens_kappa(cm_v1["tp"], cm_v1["fp"], cm_v1["fn"], cm_v1["tn"])
print(f"judge kappa vs human labels: {judge_kappa_v1:.3f}")
if my_nb1_kappa is not None:
    verdict = "beats" if judge_kappa_v1 > my_nb1_kappa else "loses to"
    print(f"your kappa was {my_nb1_kappa:.3f} — the judge {verdict} you")

# %% [markdown]
# Both kappas measure agreement with the same annotators, but at different
# prevalences — yours on `explore` at natural prevalence, the judge's on a balanced
# `calib`. Kappa is prevalence-sensitive (§6.5), so read the comparison as indicative,
# not exact.

# %% [markdown]
# ## Step 3 — Wilson interval on Se and on Sp
#
# Se is estimated from 12 positives, Sp from 12 negatives — expect this to be
# uncomfortably wide. That discomfort is the reason §6.4 exists. Notebook 4's bias
# correction is built from these same Se/Sp, so it inherits the noise too.

# %%
# TODO: Wilson 95% interval on sensitivity and on specificity.
#   metrics.wilson_interval(tp, tp + fn) -> se_ci
#   metrics.wilson_interval(tn, tn + fp) -> sp_ci
se_ci = None
sp_ci = None
if se_ci is None:
    raise NotImplementedError("Step 3: compute `se_ci` and `sp_ci` above.")

print(f"sensitivity 95% CI: ({se_ci[0]:.3f}, {se_ci[1]:.3f})  width={se_ci[1] - se_ci[0]:.3f}")
print(f"specificity 95% CI: ({sp_ci[0]:.3f}, {sp_ci[1]:.3f})  width={sp_ci[1] - sp_ci[0]:.3f}")

# %% [markdown]
# ## Step 4 — build v2 from `dev` disagreements, run on the same `calib` items
#
# The discipline point: v2's few-shot examples come from `dev` (mined identically to
# Notebook 2), never from `calib` disagreements. Improving the judge using `calib`
# and then re-measuring on `calib` is contamination — this cell does it the clean way.

# %%
dev = load_split("dev")
dev_sample = dev.head(12)
result_v1_dev = run_judge(judge_v1, dev_sample)  # 12 calls, free if Notebook 2 already ran
merged_v1_dev = result_v1_dev.merge(dev_sample, on="id")
disagreements = merged_v1_dev[merged_v1_dev["verdict"] != merged_v1_dev["label"]]
RUBRIC_V2 = add_few_shot_examples(RUBRIC_V1, select_few_shot(disagreements, n=2))

judge_v2 = build_judge(RUBRIC_V2, settings.judge_model)
result_v2 = run_judge(judge_v2, calib)  # 24 calls

merged_v2 = result_v2.merge(calib[["id", "label"]], on="id")
n_parse_failures_v2 = int((merged_v2["verdict"] == -1).sum())
clean_v2 = merged_v2[merged_v2["verdict"] != -1]
cm_v2 = confusion(clean_v2["label"].tolist(), clean_v2["verdict"].tolist())
r_v2 = rates(cm_v2)
_, judge_kappa_v2 = cohens_kappa(cm_v2["tp"], cm_v2["fp"], cm_v2["fn"], cm_v2["tn"])
print(f"v1 — sensitivity: {r_v1['sensitivity']:.3f}  specificity: {r_v1['specificity']:.3f}")
print(f"v2 — sensitivity: {r_v2['sensitivity']:.3f}  specificity: {r_v2['specificity']:.3f}")

# %% [markdown]
# ## Step 5 — paired comparison of v1 vs v2, on Se and on Sp separately
#
# Paired on identical items: most variance is item-specific, so pairing cancels it —
# the largest single lever in §6.6, and free.
#
# Run once per true class. A single test over all items mixes two different
# questions: whether v2 catches more hallucinations, and whether it passes more clean
# answers. A rubric change usually trades one against the other.

# %%
aligned = merged_v1[["id", "verdict", "label"]].merge(
    merged_v2[["id", "verdict"]], on="id", suffixes=("_v1", "_v2")
)
# Same exclusion as the confusion matrices above: an unparseable verdict is missing
# data, not a wrong answer.
aligned = aligned[(aligned["verdict_v1"] != -1) & (aligned["verdict_v2"] != -1)]

for class_label, metric_name in [(1, "sensitivity"), (0, "specificity")]:
    subset = aligned[aligned["label"] == class_label]
    correct_v1 = (subset["verdict_v1"] == subset["label"]).astype(int).tolist()
    correct_v2 = (subset["verdict_v2"] == subset["label"]).astype(int).tolist()

    # TODO: paired McNemar comparison and the minimum detectable effect.
    #   metrics.paired_compare(correct_v1, correct_v2) -> comparison (has n01, n10)
    #   discordance = (comparison["n01"] + comparison["n10"]) / len(subset)
    #   metrics.minimum_detectable_effect(len(subset), discordance) -> mde
    comparison = None
    discordance = None
    mde = None
    if comparison is None:
        raise NotImplementedError("Step 5: compute `comparison`, `discordance`, `mde` above.")

    print(f"{metric_name}: n={len(subset)} {comparison}")
    print(f"  minimum detectable effect: {mde:.3f}")

print("\nA large minimum detectable effect means this comparison could not have")
print("found a real improvement even if there were one. nan means the two judges")
print("never disagreed, so there was nothing to test.")

# %% [markdown]
# ## Step 6 — measure the noise floor
#
# Two more passes of judge v1 over the same 8 `calib` items, with `fresh=True` so the
# judge is sampled again rather than replayed from cache. Re-running this cell spends
# those calls again, by design: the measurement is of run-to-run variation.

# %%
noise_items = calib.head(8)
run1 = merged_v1.set_index("id").loc[noise_items["id"], "verdict"].tolist()
run2 = run_judge(judge_v1, noise_items, fresh=True)
run2 = run2.set_index("id").loc[noise_items["id"], "verdict"].tolist()
run3 = run_judge(judge_v1, noise_items, fresh=True)
run3 = run3.set_index("id").loc[noise_items["id"], "verdict"].tolist()

nf = noise_floor([run1, run2, run3])
print(nf)
print(f"about twice the noise floor's SD ({2 * nf['aggregate_sd']:.3f}) is the smallest")
print("change in the aggregate score you should treat as believable.")

# %% [markdown]
# ## Step 7 — log to MLflow, one run per rubric version
#
# v1 and v2 each get their own run, logged under the *same* metric names —
# `sensitivity`, `specificity`, `kappa`, `parse_failures`. MLflow's compare view
# only lines metrics up across runs when the names match; one run per version is
# what makes that view worth opening. Noise-floor figures are run-to-run variation
# of v1 specifically, so they stay on the v1 run.

# %%
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

with mlflow.start_run(run_name="judge_v1_calib"):
    mlflow.log_param("model", settings.judge_model)
    mlflow.log_param("rubric_hash", rubric_hash(RUBRIC_V1))
    mlflow.log_param("rubric_version", "v1")
    mlflow.log_param("random_seed", settings.random_seed)
    mlflow.log_metric("sensitivity", r_v1["sensitivity"])
    mlflow.log_metric("specificity", r_v1["specificity"])
    mlflow.log_metric("kappa", judge_kappa_v1)
    mlflow.log_metric("parse_failures", n_parse_failures_v1)
    mlflow.log_metric("noise_flip_rate", nf["flip_rate"])
    mlflow.log_metric("noise_aggregate_sd", nf["aggregate_sd"])

with mlflow.start_run(run_name="judge_v2_calib"):
    mlflow.log_param("model", settings.judge_model)
    mlflow.log_param("rubric_hash", rubric_hash(RUBRIC_V2))
    mlflow.log_param("rubric_version", "v2")
    mlflow.log_param("random_seed", settings.random_seed)
    mlflow.log_metric("sensitivity", r_v2["sensitivity"])
    mlflow.log_metric("specificity", r_v2["specificity"])
    mlflow.log_metric("kappa", judge_kappa_v2)
    mlflow.log_metric("parse_failures", n_parse_failures_v2)

# %% [markdown]
# ## See it, don't just read it
#
# Launch `mlflow ui` from the repo root (default <http://localhost:5000>), select
# `judge_v1_calib` and `judge_v2_calib` in the run list, and hit Compare. Look at
# whether sensitivity and specificity actually moved between the two runs, and
# whether that movement is bigger than the noise floor logged alongside v1 — if
# it isn't, Step 5's minimum detectable effect already told you why.

# %% [markdown]
# ## What you should have concluded
#
# - Se and Sp came from 12 items each — the Wilson intervals above are wide, and that
#   is a real property of a 24-item calibration set, not a bug in this notebook.
# - The judge vs. your Notebook 1 kappa is a genuine comparison, with the prevalence
#   caveat noted above — whichever way it went, it's informative.
# - The paired v1-vs-v2 comparison likely could not detect a small real improvement at
#   this n; the minimum detectable effect tells you how small "small" is.
# - The noise floor sets a threshold: don't trust any future prompt-change result
#   smaller than about 2x this run's aggregate SD.
