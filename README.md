# Evaluating GenAI Systems — workshop repo

You build an LLM-as-a-judge, measure it as a diagnostic instrument, discover its raw
output is biased in a predictable direction, correct for that bias, and work out where
to spend your next pound — on more human labels, on a better judge, or on neither.

The claim this workshop rests on: **the interesting problems in GenAI evaluation are
measurement problems, not engineering problems.** You already own diagnostic accuracy,
inter-rater reliability and interval estimation. Almost nobody in this industry applies
them here.

## Setup

Required: Python 3.11+, VS Code with the Python and Jupyter extensions, Git,
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), and a free Groq API key.

```bash
git clone https://github.com/RaihanNoushad37/gen-ai-evals-workshop.git
cd gen-ai-evals-workshop
uv sync
cp .env.example .env        # paste your Groq key into GROQ_API_KEY
uv run python -m scripts.smoke          # four checks, all should PASS
uv run jupytext --to ipynb notebooks/*.py solutions/*.py
```

Then open `notebooks/01_annotate.ipynb`. The data splits are committed, so there's no
build step — run `uv run python -m scripts.prepare_data` only if you want to rebuild
them from source.

**Getting a Groq key:** sign up at [console.groq.com](https://console.groq.com), go to
API Keys, create one, paste it into `.env`. No card required.

**Google AI Studio instead:** get a key at
[aistudio.google.com](https://aistudio.google.com/apikey) and change one line in `.env`:

```
JUDGE_MODEL=gemini:/gemini-2.0-flash
GEMINI_API_KEY=...
```

Nothing else changes — every model reference reads from `config.py`.

## Commands

`make` is not installed on Windows by default. Every target is a one-line command, so
use whichever column applies to you:

| | with `make` | without |
|---|---|---|
| install dependencies | `make setup` | `uv sync` |
| build the four splits (once) | `make data` | `uv run python -m scripts.prepare_data` |
| pre-flight checks | `make smoke` | `uv run python -m scripts.smoke` |
| generate the notebooks | `make notebooks` | `uv run jupytext --to ipynb notebooks/*.py solutions/*.py` |
| MLflow UI | `make ui` | `uv run mlflow ui` |
| tests | `make test` | `uv run pytest` |

**If `uv` is not recognised** but your virtual environment is active (your prompt starts
`(evals-workshop)`), drop the `uv run` prefix — the tools are already on your PATH:
`mlflow ui`, `pytest`, `jupytext --to ipynb notebooks/*.py`.

### Seeing your results in MLflow

Every judge run logs itself. Start the UI from the repo root in a second terminal —
`make ui`, then open http://localhost:5000. Each run carries the model, rubric hash and
seed that produced it, so a number is never separated from the configuration behind it.

| run | logged in | contains |
|---|---|---|
| `judge_v2_dev` | Notebook 2 | dev sensitivity, specificity, accuracy, parse failures |
| `judge_v1_calib` | Notebook 3 | calib Se, Sp, kappa, parse failures, noise floor |
| `judge_v2_calib` | Notebook 3 | the same metrics, for the rubric with few-shot examples |
| `judge_v1_prod` | Notebook 4 | raw rate, corrected rate, true rate, CI half-width |

The comparison worth making is **`judge_v1_calib` against `judge_v2_calib`**: tick both
in the run list and press *Compare*. They log the same metric names, so the UI lines
them up. Ask whether v2's sensitivity or specificity actually moved, and whether the
move is larger than the noise floor logged beside it. Usually it isn't — which is the
point.

## The notebooks

| | Notebook | Time | API calls |
|---|---|---|---|
| 1 | Annotate — you are the second annotator | 10 min | **0** |
| 2 | Build a judge — write a rubric, iterate on `dev` | 20 min | 28 |
| 3 | Measure the judge — Se and Sp, kappa, noise floor | 30 min | 64 |
| 4 | Use the judge — predict the bias, correct it, check it | 30 min | 24 |

Notebook 1 needs no API key at all. If your key is broken on the day, you still get the
exercise the rest of the workshop is measured against.

Notebooks 2–4 assume you run them in order: each reuses the previous one's cached judge
calls. Run one standalone and it re-issues those calls, which costs more of your budget.

## API budget

The binding constraint on a free tier is **tokens**, not requests — and there are two
limits, per minute (TPM) and per day (TPD). Only the per-minute one appears in the
response headers; the daily one shows up in the error body when you hit it.

Measured on a free Groq account, September 2026:

| | value |
|---|---|
| tokens per minute | 8,000 (same on every chat model on this tier) |
| tokens per day | 200,000 **per model** |
| pinned model | `openai/gpt-oss-20b` |

The workshop is sized to run **twice** inside that, so a mistake or a re-run does not
strand you:

| notebook | calls | tokens |
|---|---|---|
| 2 | 28 | ~21,000 |
| 3 | 64 | ~57,000 |
| 4 | 24 | ~14,000 |
| **total** | **116** | **~92,000** (46% of the daily cap; two runs fit) |

About 12 minutes of API time across the whole session, most of it overlapped with
reading. The sizing — a 400-token cap on retrieved context, 24-item calibration and
production sets — exists to fit this budget, and costs precision in a way the
notebooks are explicit about. On a higher tier, raise `JUDGE_TPM` in `.env` and enlarge
`SPLIT_SIZES` in `scripts/prepare_data.py`: the statistics get tighter, nothing else
changes.

**If you exhaust the daily budget**, the judge calls raise `DailyBudgetExhausted`.
Re-run with `USE_CACHED_RESULTS=1` to continue from shipped results.

**What the offline fallback covers, honestly.** Notebook 4 replays end to end from the
shipped cache. Notebooks 2 and 3 replay their judge-v1 steps but not judge v2, because
the v2 rubric is built from *your* judge's disagreements on `dev` — and the judge is
non-deterministic, so your v2 rubric is not the one the cache was recorded against.
That is a genuine property of the exercise rather than an oversight: anything derived
from live model output is not reproducible, which is the same lesson Notebook 3's noise
floor teaches. If you need the v2 steps offline, pin `RUBRIC_V2` to a fixed string
instead of mining it.

## What this is, and what it is not

This is a worked illustration, not a production evaluation. It is sized to run twice
inside a free API tier in a single afternoon, and several things are deliberately
simpler than you would make them on a real system. They are listed here because
knowing *which* corner was cut, and what it costs you, is part of the skill:

- **The sets are far too small.** 12 human labels per class. Sensitivity comes out with
  a confidence interval roughly ±25 points wide, and the bias correction is itself a
  noisy estimate — on any single run it may not beat the uncorrected rate. Notebook 4
  measures exactly this, and computes how many labels you would actually need. A real
  calibration set is in the hundreds.
- **One judge model, one rubric family.** No ensembling, no comparison across model
  families, no attempt to see whether a different judge has different blind spots.
- **Hallucination reduced to one binary label.** RAGTruth annotates spans and
  distinguishes evident conflicts from baseless additions; we collapse all of that to
  "does this response contain unsupported content?". Span-level scoring and per-type
  error rates are where you would go next.
- **QA task only, short contexts only.** See below — this one is load-bearing and
  becomes a discussion point rather than a footnote.
- **The judge is run once per item.** Run-to-run variation is characterised separately
  in Notebook 3 rather than averaged away, and the noise floor is measured on 8 items,
  which is enough to see the effect and not enough to pin it down.
- **No agent traces, no multi-turn.** One prompt, one response, one judgement.
- **`align()` not used.** MLflow can automate some of the calibration this workshop
  does by hand. Doing it by hand first is the point; the automation is easier to trust
  afterwards.

None of these change the *method*. They change the precision, and the notebooks are
explicit about where.

## About the data

[RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed): human span-level
hallucination annotations over LLM responses grounded in retrieved context. A binary
label — *does this response contain unsupported content?* — falls straight out of the
spans, and the labels come from trained annotators under a published protocol rather
than from an afternoon of our own guessing.

**Two filters, both of which bias the sample, and both disclosed because one of them
becomes a discussion point:**

1. **QA task only** (5,034 of 15,090 rows).
2. **`context + response` ≤ 400 tokens**, which retains 2,079 of those 5,034.

The second filter is not innocent. The retained pool hallucinates at **0.235**; the
discarded pool at **0.364**. So the judge is calibrated on systematically shorter, and
apparently easier, examples than the ones it was filtered away from. Sensitivity and
specificity measured on `calib` transport to data with a different *mix* of instances,
but not to data where the judge's *error rates* differ — and long contexts are exactly
where you'd expect them to differ. **A sampling decision taken for budget reasons
propagates into a production estimate**, and no amount of correct algebra downstream
fixes it. The exact numbers are regenerated into `data/report.md` every time you run
`make data`.

| split | n | composition | labels | purpose |
|---|---|---|---|---|
| `explore` | 12 | natural prevalence | revealed at NB1 step 4 | manual annotation |
| `dev` | 16 | balanced 8/8 | visible | judge iteration, few-shot mining |
| `calib` | 24 | balanced 12/12 | visible | measuring Se and Sp |
| `prod` | 24 | natural prevalence | **hidden until NB4 step 5** | stands in for production |

`calib` is balanced and `prod` is not because they answer different questions. `calib`
estimates Se and Sp, for which sampling by true label is the efficient design. `prod`
stands in for real traffic, where prevalence is whatever it is — and the whole point of
Notebook 4 is that you do not know it in advance.

The `prod` lock is enforced in `data.py`, not left to your restraint.

## Troubleshooting

**`RuntimeError: Missing required environment variable GROQ_API_KEY`** — you have no
`.env`, or it has no key in it. `cp .env.example .env` and paste one in.

**`DailyBudgetExhausted`** — you have spent that model's daily token allowance. It does
not come back within the session. Re-run with `USE_CACHED_RESULTS=1`, or switch
`JUDGE_MODEL` to another model, which has its own separate daily allowance.

**Judge calls are slow.** Expected. At 8,000 TPM with ~600-token prompts you get roughly
12 calls a minute, and the pacing in `llm.py` deliberately waits rather than getting
itself rate-limited. Progress prints every 25 calls. Start the cell and read on.

**`verdict == -1` on some rows.** The judge's reply did not parse. These are counted and
reported separately, never silently scored as "no violation" — dropping them assumes
they fail at random, which is worth a moment's thought if there are many.

**A notebook re-runs calls you thought were cached.** The cache key covers the model and
the exact prompt, so changing `JUDGE_MODEL` or editing a rubric invalidates it. That is
the intent: those are different measurements.

**`mlflow` complains about the file store.** Handled — `config.py` opts back into it.

## What this repo is also demonstrating

Stated explicitly, because it is a module objective:

| practice | where |
|---|---|
| Reproducible environments | `uv` with a committed lockfile |
| Secrets hygiene | `.env.example` committed, `.env` git-ignored, key never in a notebook |
| Thin notebooks | every reusable function in `src/`, imported; no `def` in any notebook |
| Unit-tested analysis code | every statistic checked against independently computed values |
| Data integrity as a test | split disjointness and document-level leakage asserted |
| Configuration over hardcoding | provider swap is one line in `.env` |
| Graceful degradation | token-aware pacing, capped backoff, cached fallback |
| Recording the experiment | model, rubric hash, seed logged to MLflow on every run |
| Held-out evaluation | four disjoint splits, the discipline enforced not just stated |

A score without its configuration is not a measurement. You will have been recording
configuration all session without being told that is what you were doing.

Notebooks 2, 3 and 4 each log a run — Notebook 3 logs two, one per rubric version.
Run `make ui` (or `mlflow ui`) from the repo root and open <http://localhost:5000>
to browse what got logged. Select `judge_v1_calib` and `judge_v2_calib` in the run
list and hit Compare: v1 and v2 share metric names (`sensitivity`, `specificity`,
`kappa`, `parse_failures`), so the compare view lines them up and shows whether the
rubric change moved anything, and by how much.
