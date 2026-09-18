"""Runs each solution notebook end to end with the judge stubbed out. No network calls.

The stub looks up real labels (read straight from the parquet, bypassing the prod
lock) so it agrees with truth often enough for sensitivity + specificity > 1, while
still emitting some wrong and some unparseable replies — a judge that only says "no"
or only ever parses cleanly would leave half the statistics untested.

`solutions/` holds the complete, reference notebooks and is what gets executed here.
`notebooks/` holds the student versions with TODOs in place of the statistical steps;
those are checked for conversion and for the presence of a TODO, not executed.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from evals_workshop import annotate, data, judges

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SOLUTIONS_DIR = REPO_ROOT / "solutions"
SPLITS_PATH = Path("data/splits.parquet")

# NB1 is left as-is -- it's already hands-on, nothing to exercise out of it.
EXERCISED_NOTEBOOKS = [
    "02_build_a_judge.py",
    "03_measure_the_judge.py",
    "04_use_the_judge.py",
]

# Only the notebook-execution tests need built data; the jupytext/TODO checks below
# do not, so the skip is applied per-test rather than to the whole module.
needs_data = pytest.mark.skipif(
    not SPLITS_PATH.exists(), reason="data/splits.parquet not built; run `make data`"
)

# Matches the row rendered into a prompt by judges._render_prompt, to recover
# which row a stub call is about.
_ROW_RE = re.compile(
    r"Question: (.*?)\nContext: (.*)\n\n## Answer to evaluate\n(.*)\n\n## Response format",
    re.DOTALL,
)


@pytest.fixture(scope="module")
def truth() -> dict[tuple[str, str, str], int]:
    df = pd.read_parquet(SPLITS_PATH)
    return {(str(r.query), str(r.context), str(r.response)): int(r.label) for r in df.itertuples()}


def _make_stub(truth: dict[tuple[str, str, str], int]):
    """~9% unparseable, ~10% of the rest wrong, everything else matches truth."""

    def _stub(prompts, model, tpm, max_concurrency, cache_path=None):
        replies = []
        for prompt in prompts:
            digest = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
            if digest % 11 == 0:
                garbled = "the model trailed off with no clear verdict"
                replies.append({"text": garbled, "error": None})
                continue
            match = _ROW_RE.search(prompt)
            label = truth.get(match.groups()) if match else None
            if label is None:
                verdict = "yes" if digest % 2 == 0 else "no"
            else:
                correct = "yes" if label == 1 else "no"
                wrong = "no" if correct == "yes" else "yes"
                verdict = wrong if digest % 10 == 0 else correct
            replies.append({"text": f"Reason: stub judgement.\nVerdict: {verdict}", "error": None})
        return replies

    return _stub


def _exec_notebook(path: Path) -> None:
    source = path.read_text()
    exec(compile(source, str(path), "exec"), {"__name__": "__main__"})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@needs_data
def test_notebook_01_annotate(monkeypatch, truth, tmp_path):
    # No judge calls at all, so no stub is wired up -- but the notebook's kappa /
    # specific-agreement step only runs once the Labeller reports complete, so patch
    # Labeller itself to auto-record the ground-truth label for every row. That
    # exercises the reveal step instead of the "finish labelling" skip path.
    # The default storage path is also redirected under tmp_path, so this never
    # touches the real data/my_labels.json.
    monkeypatch.setattr(annotate, "_DEFAULT_LABELS_PATH", tmp_path / "my_labels.json")
    real_labeller = annotate.Labeller

    def _pre_labelled(df: pd.DataFrame) -> annotate.Labeller:
        labeller = real_labeller(df)
        for row in df.itertuples():
            key = (str(row.query), str(row.context), str(row.response))
            labeller.record(row.id, truth[key])
        return labeller

    monkeypatch.setattr(annotate, "Labeller", _pre_labelled)
    _exec_notebook(SOLUTIONS_DIR / "01_annotate.py")


@needs_data
def test_notebook_02_build_a_judge(monkeypatch, truth):
    monkeypatch.setattr(judges, "judge_batch", _make_stub(truth))
    _exec_notebook(SOLUTIONS_DIR / "02_build_a_judge.py")


@needs_data
def test_notebook_03_measure_the_judge(monkeypatch, truth):
    monkeypatch.setattr(judges, "judge_batch", _make_stub(truth))
    _exec_notebook(SOLUTIONS_DIR / "03_measure_the_judge.py")


@needs_data
def test_notebook_04_use_the_judge(monkeypatch, truth, tmp_path):
    monkeypatch.setattr(judges, "judge_batch", _make_stub(truth))
    # Gate the reveal sentinel to a tmp path: notebook 4 calls unlock_prod_labels(),
    # which must never touch the real data/.prod_unlocked.
    monkeypatch.setattr(data, "_UNLOCK_SENTINEL", tmp_path / ".prod_unlocked")
    _exec_notebook(SOLUTIONS_DIR / "04_use_the_judge.py")


@pytest.mark.parametrize("name", sorted(p.name for p in NOTEBOOKS_DIR.glob("*.py")))
def test_student_notebook_converts_with_jupytext(name, tmp_path):
    # The student notebooks are never executed (they stop at the first TODO by
    # design) -- this only confirms jupytext still accepts them as valid percent-format
    # source, which is all `make notebooks` needs from them.
    out = tmp_path / name.replace(".py", ".ipynb")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "ipynb",
            "--output",
            str(out),
            str(NOTEBOOKS_DIR / name),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert out.exists()


@pytest.mark.parametrize("name", EXERCISED_NOTEBOOKS)
def test_student_notebook_has_todo(name):
    # Every exercised notebook must actually contain an exercise -- catches a TODO
    # being accidentally filled in or removed while editing the student version.
    source = (NOTEBOOKS_DIR / name).read_text()
    assert "# TODO" in source
