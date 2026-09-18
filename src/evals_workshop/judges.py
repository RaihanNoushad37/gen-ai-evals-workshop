"""Judge construction and execution.

`build_judge` passes `model=` to mlflow's `make_judge` explicitly: outside Databricks
the default is `openai:/gpt-4o-mini`, which fails on a missing OpenAI key.

`run_judge` renders the judge's rubric into one prompt per row and sends the batch
through `llm.judge_batch`, rather than calling the mlflow Judge per row, so the run is
paced against the token budget and cached. The Judge object still carries the
versioned rubric.
"""

import hashlib
import re

import pandas as pd
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import Judge

from .config import DATA_DIR, settings
from .llm import judge_batch

RUBRIC_V1 = """You are evaluating an AI assistant's answer to a question, given the \
retrieved context it was supposed to rely on.

Decide whether the answer contains any claim that is NOT supported by the context —
including fabricated facts, invented numbers or names, or claims that contradict the
context. Reasonable summarization, paraphrasing, or drawing an inference that is
directly implied by the context is NOT a violation.

## Question and context
{{ inputs }}

## Answer to evaluate
{{ outputs }}

## Response format
Respond in exactly this format, two lines:
Reason: <one or two sentences explaining your judgement>
Verdict: <yes if the answer contains unsupported content, otherwise no>
"""

_VERDICT_RE = re.compile(r"verdict\s*:\s*(yes|no)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+?)(?:\n\s*verdict|$)", re.IGNORECASE | re.DOTALL)


def rubric_hash(rubric: str) -> str:
    return hashlib.sha256(rubric.encode()).hexdigest()[:8]


def build_judge(rubric: str, model: str) -> Judge:
    return make_judge(
        name=f"faithfulness_judge_{rubric_hash(rubric)}",
        instructions=rubric,
        model=model,
        description="Binary faithfulness judge for QA responses grounded in retrieved context.",
        feedback_value_type=bool,
        generate_rationale_first=True,
    )


def _render_prompt(rubric: str, row: pd.Series) -> str:
    inputs_text = f"Question: {row['query']}\nContext: {row['context']}"
    return rubric.replace("{{ inputs }}", inputs_text).replace(
        "{{ outputs }}", str(row["response"])
    )


def _parse_response(text: str | None) -> tuple[int, str]:
    if text is None:
        return -1, ""
    verdict_match = _VERDICT_RE.search(text)
    if not verdict_match:
        return -1, text.strip()
    verdict = 1 if verdict_match.group(1).lower() == "yes" else 0
    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return verdict, reason


def run_judge(judge: Judge, rows: pd.DataFrame, fresh: bool = False) -> pd.DataFrame:
    """Run `judge` over `rows`, returning id/verdict/reason. Verdict -1 means unparseable.

    `fresh=True` bypasses the cache, so repeating a run samples the judge again rather
    than replaying it — what the noise-floor measurement needs.
    """
    prompts = [_render_prompt(judge.instructions, row) for _, row in rows.iterrows()]
    cache_path = None if fresh else DATA_DIR / ".cache" / "judge_cache.json"
    results = judge_batch(
        prompts,
        settings.judge_model,
        settings.judge_tpm,
        settings.max_concurrency,
        cache_path=cache_path,
    )
    parsed = [_parse_response(r["text"]) for r in results]
    verdicts, reasons = zip(*parsed, strict=True) if parsed else ((), ())
    return pd.DataFrame({"id": rows["id"].values, "verdict": verdicts, "reason": reasons})


def _escape_template_braces(text: str) -> str:
    """Neutralise literal `{{ }}` in source text, which mlflow would read as a variable."""
    return text.replace("{{", "{ {").replace("}}", "} }")


def select_few_shot(candidates: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Pick the `n` shortest candidate rows to use as worked examples.

    Shortest rather than first: every example is prepended to every later prompt, and
    a long QA context can crowd out the instructions. Examples go in whole — trimming
    one can remove the very span it turns on and teach the judge the wrong lesson.
    """
    return candidates.nsmallest(n, "n_tokens") if "n_tokens" in candidates else candidates.head(n)


def add_few_shot_examples(rubric: str, examples: pd.DataFrame) -> str:
    """Fold worked examples into `rubric`, above the response-format section.

    Mine these from `dev` only: examples drawn from `calib` or `prod` contaminate the
    sets that measure the judge.
    """
    block = "\n\n## Worked examples\n"
    for _, row in examples.iterrows():
        correct = "yes" if row["label"] == 1 else "no"
        query = _escape_template_braces(str(row["query"]))
        context = _escape_template_braces(str(row["context"]))
        response = _escape_template_braces(str(row["response"]))
        block += (
            f"\nQuestion: {query}\nContext: {context}\nAnswer: {response}\nVerdict: {correct}\n"
        )
    return rubric.replace("## Response format", block + "\n## Response format")
