# %% [markdown]
# # Notebook 2 — Build a judge
#
# **Objective:** turn Notebook 1's failure modes into a rubric, construct a judge with
# `make_judge()`, and iterate against `dev` only.
#
# **Expected runtime:** ~20 minutes. **Expected API calls:** 28 (12 v1 + 16 v2, ~3 min
# wall clock).
#
# **Closing point:** the prompt and its few-shot examples are fitted parameters. By
# the end of this notebook you will have performed a fit without a training loop —
# and you are about to evaluate it on held-out data in Notebook 3.
#
# **Stuck on a `# TODO`?** `solutions/02_build_a_judge.ipynb` has a fully worked
# version. Falling back to it is fine — nobody should be stranded mid-session.

# %%
import mlflow

from evals_workshop.config import settings
from evals_workshop.data import load_split
from evals_workshop.judges import (
    RUBRIC_V1,
    add_few_shot_examples,  # noqa: F401 -- for the Step 4 TODO below
    build_judge,
    rubric_hash,
    run_judge,
    select_few_shot,  # noqa: F401 -- for the Step 4 TODO below
)
from evals_workshop.metrics import confusion, rates  # noqa: F401 -- for the Step 5 TODO below

# %% [markdown]
# ## Discipline check, up front
#
# Few-shot examples are mined from `dev`. This assertion is what actually prevents a
# `dev` id leaking into `calib` or `prod` — not just a rule stated in a comment.

# %%
dev_ids = set(load_split("dev")["id"])
calib_ids = set(load_split("calib")["id"])
prod_ids = set(load_split("prod", with_labels=False)["id"])
assert not (dev_ids & calib_ids), "dev/calib id leakage"
assert not (dev_ids & prod_ids), "dev/prod id leakage"
print("OK: no dev id appears in calib or prod")

# %% [markdown]
# ## Step 1 — the rubric, from Notebook 1's failure modes
#
# `RUBRIC_V1` (in `judges.py`) already encodes the categories most annotators land on:
# fabricated facts, contradictions, overgeneralization, unsupported inference. If your
# own Notebook 1 categories differ, this is the string to edit.

# %%
print(RUBRIC_V1)

# %% [markdown]
# ## Step 2 — construct the judge
#
# The model always comes from `settings.judge_model` — never hardcode a model string.
# Outside Databricks, a judge built without an explicit `model=` silently falls back
# to `openai:/gpt-4o-mini` and fails on a missing OpenAI key; `build_judge` never lets
# that happen.

# %%
judge_v1 = build_judge(RUBRIC_V1, settings.judge_model)
print(judge_v1.name, "| rubric hash:", rubric_hash(RUBRIC_V1))

# %% [markdown]
# ## Step 3 — run on 12 dev rows, read the reasons

# %%
dev = load_split("dev")
dev_sample = dev.head(12)
result_v1_sample = run_judge(judge_v1, dev_sample)
result_v1_sample.merge(dev_sample[["id", "label"]], on="id")[["id", "label", "verdict", "reason"]]

# %% [markdown]
# Read a few reasons above, especially where `verdict != label`. What is the judge
# getting wrong?

# %% [markdown]
# ## Step 4 — iterate: add few-shot examples mined from `dev` disagreements
#
# Pick a few `dev` rows the v1 judge got wrong and fold them into the rubric as worked
# examples. These can never be reused as calibration or production items — the
# assertion cell above guarantees that structurally, not just by convention.

# %%
merged_v1 = result_v1_sample.merge(dev_sample, on="id")
disagreements = merged_v1[merged_v1["verdict"] != merged_v1["label"]]

# TODO: mine up to 2 worked examples from `disagreements` and build RUBRIC_V2.
#   select_few_shot(disagreements, n=2) -> the rows to fold in
#   add_few_shot_examples(RUBRIC_V1, examples) -> the new rubric
#   (if `examples` comes back empty, v1 got every dev row right — keep RUBRIC_V2 = RUBRIC_V1)
examples = None
RUBRIC_V2 = None
if RUBRIC_V2 is None:
    raise NotImplementedError("Step 4: fill in `examples` and `RUBRIC_V2` above.")

print(f"{len(examples)} worked example(s) added. Rubric hash: {rubric_hash(RUBRIC_V2)}")

# %% [markdown]
# ## Step 5 — run v2 on the full `dev` split, log to MLflow

# %%
judge_v2 = build_judge(RUBRIC_V2, settings.judge_model)
result_v2 = run_judge(judge_v2, dev)
merged_v2 = result_v2.merge(dev[["id", "label"]], on="id")

n_parse_failures = int((merged_v2["verdict"] == -1).sum())
clean = merged_v2[merged_v2["verdict"] != -1]

# TODO: confusion matrix and Se/Sp for v2 on the full dev split.
#   metrics.confusion(clean["label"].tolist(), clean["verdict"].tolist()) -> cm
#   metrics.rates(cm) -> r
cm = None
r = None
if r is None:
    raise NotImplementedError("Step 5: compute `cm` and `r` above.")

print(f"parse failures: {n_parse_failures}/{len(merged_v2)}")
print(f"dev sensitivity: {r['sensitivity']:.3f}  dev specificity: {r['specificity']:.3f}")

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
with mlflow.start_run(run_name="judge_v2_dev"):
    mlflow.log_param("model", settings.judge_model)
    mlflow.log_param("rubric_hash", rubric_hash(RUBRIC_V2))
    mlflow.log_param("rubric_version", "v2")
    mlflow.log_param("random_seed", settings.random_seed)
    mlflow.log_metric("dev_sensitivity", r["sensitivity"])
    mlflow.log_metric("dev_specificity", r["specificity"])
    mlflow.log_metric("dev_accuracy", r["accuracy"])
    mlflow.log_metric("dev_parse_failures", n_parse_failures)

# %% [markdown]
# ## What you should have concluded
#
# - You just fitted a prompt and its few-shot examples to `dev` — a fit without a
#   training/optimization loop, but a fit nonetheless.
# - `dev` sensitivity/specificity above are optimistic: this is the data the judge was
#   tuned on. Notebook 3 measures the real thing on `calib`, which the judge never saw.
# - Parse failures are counted, not silently defaulted to "no violation" — that would
#   quietly bias sensitivity downward.
