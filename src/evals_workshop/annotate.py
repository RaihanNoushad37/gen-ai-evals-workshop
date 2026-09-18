"""Click-through labeller for Notebook 1.

`Labeller` shows one `explore` row at a time and records a verdict per click --
0 clean, 1 unsupported, -1 unsure -- keyed by the row's `id`, never by position, so
going back and relabelling an earlier row can't shift what a later answer means.
Unsure mirrors the judge's own -1 for an unparseable reply: missing data, excluded
from the confusion matrix rather than guessed. Every recorded verdict (and its note)
is persisted to disk immediately and reloaded on construction, so a kernel restart
resumes rather than starting over.

It degrades instead of failing: where ipywidgets has no live frontend to render
into (a plain-script run, as `tests/test_notebooks.py` does, or any environment
without a notebook kernel), it prints every example plus instructions for
recording labels by hand instead of building widgets that would never be
clickable.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR

_REQUIRED_COLUMNS = ("id", "query", "context", "response")
_VALID_VERDICTS = (0, 1, -1)
_VERDICT_NAMES = {0: "clean (0)", 1: "unsupported (1)", -1: "unsure"}

# Overridable by tests via monkeypatch; the notebook always constructs Labeller
# with no storage_path, so this is the only hook that changes where it writes.
_DEFAULT_LABELS_PATH = DATA_DIR / "my_labels.json"


def _widget_backend() -> tuple[Any, Any] | None:
    """Return (ipywidgets, display) if there's a live kernel to render into, else None.

    `get_ipython() is None` is the plain-script case: no frontend is attached, so a
    widget would only ever print its repr, never take a click. Any other failure
    (package missing, frontend broken) is caught too rather than left to propagate.
    """
    try:
        import ipywidgets as widgets
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is None:
            return None
        return widgets, display
    except Exception:
        return None


def _load_labels(path: Path) -> dict[str, Any]:
    """Read a persisted labels file, tolerating absent, empty, or corrupt content.

    A broken file must never block the labeller from opening -- worst case is
    losing prior progress, not a crash.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _panel(title: str, text: str, background: str, border: str) -> str:
    """One HTML panel: a title bar over pre-wrapped body text, so long context reads
    as a paragraph rather than collapsing into a single scrolled line."""
    safe = html.escape(str(text))
    return (
        f'<div style="border-left: 4px solid {border}; background: {background}; '
        f'padding: 8px 12px; margin: 6px 0; border-radius: 4px;">'
        f'<div style="font-weight: 600; margin-bottom: 4px;">{title}</div>'
        f'<div style="white-space: pre-wrap;">{safe}</div>'
        f"</div>"
    )


class Labeller:
    """One-example-at-a-time labeller over an `explore`-shaped dataframe.

    Three buttons record a verdict: the response contains unsupported content (1),
    it's clean (0), or you're not sure (-1); a back button revisits and changes an
    earlier answer. A note box captures free text against whichever row is on
    screen. Labels and notes are stored by row `id` and written to disk after every
    click, so `ordered_labels()` can always reconstruct row order regardless of
    click order, and progress survives a kernel restart.
    """

    def __init__(self, df: pd.DataFrame, storage_path: Path | None = None) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Labeller needs columns {_REQUIRED_COLUMNS}, missing {missing}")
        self._df = df.reset_index(drop=True)
        self._ids: list[Any] = self._df["id"].tolist()
        self._id_set = set(self._ids)
        self._storage_path = storage_path if storage_path is not None else _DEFAULT_LABELS_PATH
        self._labels: dict[Any, dict[str, Any]] = {}
        self._restore_persisted()
        self._pos = 0
        self._backend: tuple[Any, Any] | None = None
        try:
            backend = _widget_backend()
            if backend is not None:
                self._backend = backend
                self._build_widgets()
        except Exception:
            self._backend = None
        if self._backend is not None:
            self._render()
        else:
            self._print_fallback()

    # -- persistence ----------------------------------------------------------

    def _restore_persisted(self) -> None:
        """Repopulate `self._labels` from disk, keyed by id as a string so it
        survives a round trip through JSON regardless of the id column's dtype."""
        raw = _load_labels(self._storage_path)
        restored = 0
        for rid in self._ids:
            entry = raw.get(str(rid))
            if not isinstance(entry, dict) or entry.get("verdict") not in _VALID_VERDICTS:
                continue
            note = entry.get("note", "")
            self._labels[rid] = {
                "verdict": entry["verdict"],
                "note": note if isinstance(note, str) else "",
            }
            restored += 1
        if restored:
            print(
                f"Restored {restored} of {len(self._ids)} previously recorded judgement(s) "
                f"from {self._storage_path}. Call labeller.reset() to start over."
            )

    def _persist(self) -> None:
        payload = {str(rid): entry for rid, entry in self._labels.items()}
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- widget construction (only reached when a live frontend is present) -----

    def _build_widgets(self) -> None:
        widgets, display = self._backend  # type: ignore[misc]
        self._progress = widgets.HTML()
        self._query_html = widgets.HTML()
        self._context_html = widgets.HTML()
        self._response_html = widgets.HTML()
        self._note_box = widgets.Textarea(
            placeholder="optional: what makes this one hard to call?",
            layout=widgets.Layout(width="100%"),
        )
        self._back_button = widgets.Button(description="< back")
        self._clean_button = widgets.Button(description="clean (0)", button_style="success")
        self._bad_button = widgets.Button(description="unsupported (1)", button_style="danger")
        self._unsure_button = widgets.Button(description="unsure", button_style="warning")
        self._back_button.on_click(self._handle_back)
        self._clean_button.on_click(lambda _button: self._handle_label(0))
        self._bad_button.on_click(lambda _button: self._handle_label(1))
        self._unsure_button.on_click(lambda _button: self._handle_label(-1))
        buttons = widgets.HBox(
            [self._back_button, self._clean_button, self._bad_button, self._unsure_button]
        )
        container = widgets.VBox(
            [
                self._progress,
                self._query_html,
                self._context_html,
                self._response_html,
                self._note_box,
                buttons,
            ]
        )
        display(container)

    def _handle_label(self, verdict: int) -> None:
        self.record(self._ids[self._pos], verdict, note=self._note_box.value)
        if self._pos < len(self._ids) - 1:
            self._pos += 1
        self._render()

    def _handle_back(self, _button: Any) -> None:
        self._pos = max(0, self._pos - 1)
        self._render()

    def _render(self) -> None:
        row = self._df.iloc[self._pos]
        entry = self._labels.get(row["id"])
        status = f"marked: {_VERDICT_NAMES[entry['verdict']]}" if entry else "not yet labelled"
        self._progress.value = (
            f"<b>{self._pos + 1} / {len(self._df)}</b> &middot; "
            f"{len(self._labels)} of {len(self._df)} labelled &middot; {status}"
        )
        self._query_html.value = _panel("QUERY", row["query"], "#f5f5f5", "#999999")
        self._context_html.value = _panel(
            "CONTEXT (retrieved)", row["context"], "#eef3fb", "#6a8fc2"
        )
        self._response_html.value = _panel(
            "RESPONSE — judge this one", row["response"], "#fff4e5", "#d98a1f"
        )
        self._note_box.value = entry["note"] if entry else ""
        self._back_button.disabled = self._pos == 0

    def _print_fallback(self) -> None:
        print(
            "ipywidgets has no live frontend to render into here (for example, this "
            "notebook is running as a plain script), so every example is printed "
            "below instead.\n"
            'Record each judgement by calling labeller.record(row_id, verdict, note="..."): '
            "1 if the response contains a claim the context doesn't support, 0 if it's "
            "clean, -1 if you're not sure. note is optional free text.\n"
        )
        for _, row in self._df.iterrows():
            print(f"id={row['id']}")
            print(f"  query:    {row['query']}")
            print(f"  context:  {row['context']}")
            print(f"  response: {row['response']}")
            print()

    # -- public API ---------------------------------------------------------

    def record(self, row_id: Any, verdict: int, note: str = "") -> None:
        """Record `verdict` for the row with this id: 1 unsupported, 0 clean, -1
        unsure -- the same convention the judge uses for its own unparseable
        replies. Overwrites any earlier answer and note for the same id (this is
        how a student revises a judgement, by clicking back or by calling this
        again), then persists to disk immediately.
        """
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {_VALID_VERDICTS}, got {verdict!r}")
        if row_id not in self._id_set:
            raise KeyError(f"{row_id!r} is not a row id in this labeller's dataframe")
        self._labels[row_id] = {"verdict": verdict, "note": note}
        self._persist()

    def reset(self) -> None:
        """Clear every recorded verdict and note, in memory and on disk."""
        self._labels.clear()
        self._storage_path.unlink(missing_ok=True)
        if self._backend is not None:
            self._pos = 0
            self._render()

    @property
    def is_complete(self) -> bool:
        """Whether every row has a recorded verdict -- unsure counts as recorded."""
        return len(self._labels) == len(self._ids)

    @property
    def remaining(self) -> int:
        """How many rows still have no recorded verdict."""
        return len(self._id_set - self._labels.keys())

    @property
    def unsure_count(self) -> int:
        """How many recorded verdicts are unsure, and so excluded from confusion."""
        return sum(1 for entry in self._labels.values() if entry["verdict"] == -1)

    def notes(self) -> list[tuple[Any, str]]:
        """(row id, note) pairs for every non-empty note, in dataframe row order."""
        return [
            (rid, self._labels[rid]["note"])
            for rid in self._ids
            if rid in self._labels and self._labels[rid]["note"].strip()
        ]

    def ordered_labels(self, true_labels: Sequence[int]) -> tuple[list[int], list[int]]:
        """Confident verdicts paired with the matching ground truth, in row order,
        ready for `metrics.confusion`.

        Unsure rows are dropped from both lists -- the same treatment Notebook 3
        gives the judge's own -1 parse failures -- rather than defaulted to a
        class, which would misrepresent both distributions. Raises if labelling
        isn't finished; unsure counts as finished, just not confident.
        """
        if not self.is_complete:
            missing = [rid for rid in self._ids if rid not in self._labels]
            raise RuntimeError(
                f"{len(missing)} of {len(self._ids)} examples still have no verdict "
                f"(ids: {missing}), unsure included. Finish clicking through the "
                "labeller above -- or call labeller.record(row_id, verdict) for each "
                "remaining id -- before calling ordered_labels()."
            )
        if len(true_labels) != len(self._ids):
            raise ValueError(
                f"true_labels has {len(true_labels)} entries, expected {len(self._ids)}"
            )
        true_confident: list[int] = []
        pred_confident: list[int] = []
        for rid, true in zip(self._ids, true_labels, strict=True):
            verdict = self._labels[rid]["verdict"]
            if verdict == -1:
                continue
            true_confident.append(true)
            pred_confident.append(verdict)
        return true_confident, pred_confident
