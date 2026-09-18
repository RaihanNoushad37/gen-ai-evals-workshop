# %% [markdown]
# # Notebook 1 — Annotate
#
# **Objective:** be the second annotator before you build anything that automates the
# job. Label 12 responses by hand, then compare your judgements to the trained
# annotators who produced this dataset's ground truth.
#
# **Expected runtime:** a few minutes. **Expected API calls:** 0.
#
# This notebook needs no API key at all — labelling is free, and the highest-value
# few minutes of the workshop do not depend on your Groq key working.

# %%
from collections import Counter

from evals_workshop.annotate import Labeller
from evals_workshop.data import load_split
from evals_workshop.metrics import cohens_kappa, confusion, specific_agreement

# %% [markdown]
# ## Step 1 — load 12 examples, labels hidden
#
# Each row has a `query`, `context` (the retrieved passage) and `response` (the
# model's answer). Decide whether `response` contains a claim that is **not**
# supported by `context`. Step 2 shows them one at a time; there's nothing to read
# here.

# %%
explore = load_split("explore")
explore_visible = explore.drop(columns=["label", "spans"])
print(f"{len(explore_visible)} examples loaded, labels hidden until Step 4")

# %% [markdown]
# ## Step 2 — label each example
#
# `Labeller` shows one example at a time — query, context, and the response you're
# judging, laid out so the response is visually distinct from the context it's
# supposed to rely on. Three buttons: the response contains unsupported content (1),
# it's clean (0), or you're not sure (unsure). Use unsure honestly — it's excluded
# from scoring in Step 4, not penalised, so there's no reason to force a guess. A note
# box next to the buttons captures anything about the current row worth remembering;
# it carries into Step 3 automatically. Click through all 12; you can go back and
# change an earlier answer at any point, which is normal once you've calibrated on the
# first few.
#
# Work is saved after every click, keyed by row id, and reloaded automatically if you
# restart the kernel — you'll be told how many judgements were restored. To start a
# clean run instead, call `labeller.reset()`.
#
# If ipywidgets can't render here (for example, this notebook running as a plain
# script instead of a live kernel), `Labeller` prints every example instead, with
# instructions for recording labels by hand — you're always told which path you're on.

# %%
labeller = Labeller(explore_visible)

# %% [markdown]
# ## Step 3 — review your ambiguous notes
#
# Whatever you typed in the note box while labelling shows up here — nothing to enter
# by hand. These notes become Step 5's material.

# %%
ambiguous_notes = labeller.notes()
for row_id, note in ambiguous_notes:
    print(f"{row_id}: {note}")

# %% [markdown]
# ## Step 4 — reveal the RAGTruth annotations
#
# Compare your judgements to the trained annotators, once labelling is finished. Rows
# you marked unsure are excluded from the numbers below and reported separately,
# rather than guessed — exactly how Notebook 3 treats the judge's own unparseable
# verdicts. Whether those rows are missing at random, or systematically the hardest
# ones, is a question worth sitting with before Notebook 3 asks it about the judge.
#
# Keep the four numbers this cell prints — Notebook 3 asks whether the judge you build
# beats them.

# %%
if labeller.is_complete:
    true_labels = explore["label"].tolist()
    true_confident, student_confident = labeller.ordered_labels(true_labels)
    cm = confusion(true_confident, student_confident)

    raw_agreement, my_kappa = cohens_kappa(cm["tp"], cm["fp"], cm["fn"], cm["tn"])
    pos_agreement, neg_agreement = specific_agreement(cm["tp"], cm["fp"], cm["fn"], cm["tn"])

    print(f"unsure:                      {labeller.unsure_count}/{len(explore)} (excluded below)")
    print(f"raw agreement:               {raw_agreement:.3f}")
    print(f"cohen's kappa (yours):       {my_kappa:.3f}")
    print(f"positive specific agreement: {pos_agreement:.3f}")
    print(f"negative specific agreement: {neg_agreement:.3f}")
else:
    print(
        f"{labeller.remaining} example(s) still unlabelled — finish clicking through "
        "the labeller in Step 2 (or call labeller.record(row_id, verdict) for each "
        "remaining id), then re-run this cell."
    )

# %% [markdown]
# ## Step 5 — cluster your ambiguous notes into failure modes
#
# Assign each note to one of a small set of categories. These categories become the
# rubric criteria in Notebook 2 — the rubric is derived from what you actually saw,
# not invented from scratch.

# %%
FAILURE_MODE_CATEGORIES = [
    "fabricated fact or number",
    "contradicts the context",
    "overgeneralizes beyond the context",
    "unsupported inference",
    "other",
]

# Map each entry in ambiguous_notes (same order) to one of the categories above.
note_categories: list[str] = ["other" for _ in ambiguous_notes]

print(Counter(note_categories))

# %% [markdown]
# ## What you should have concluded
#
# - Your agreement with trained annotators (kappa, above) is a real number you now
#   own — and it is probably well short of 1.0. Labelling is hard even for humans.
#   (If Step 4 told you to finish labelling first, go back and do that before you
#   trust this bullet.)
# - Raw agreement and kappa can diverge once the positive class is a minority; if
#   yours did, that's the point of §6.5, not a mistake.
# - The failure-mode categories above are the seed of Notebook 2's rubric — the next
#   notebook asks you to write a judge that catches what you just spent a few minutes
#   catching by hand.
