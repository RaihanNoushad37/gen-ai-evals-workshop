"""Tests for the Labeller widget wrapper. No Jupyter kernel runs under pytest, so
these also double as proof that construction degrades to the print fallback instead
of raising -- see `test_construction_is_headless_safe`.

Every test monkeypatches the default storage path under `tmp_path`, so the real
`data/my_labels.json` is never read or written by the suite.
"""

import pandas as pd
import pytest

from evals_workshop import annotate
from evals_workshop.annotate import Labeller


def _df() -> pd.DataFrame:
    # Deliberately not in id order, so a bug that reads labels back by position
    # rather than by id would be caught.
    return pd.DataFrame(
        {
            "id": [30, 10, 20],
            "query": ["q30", "q10", "q20"],
            "context": ["c30", "c10", "c20"],
            "response": ["r30", "r10", "r20"],
        }
    )


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(annotate, "_DEFAULT_LABELS_PATH", tmp_path / "my_labels.json")
    return tmp_path / "my_labels.json"


def test_construction_is_headless_safe(capsys):
    # No widget frontend is attached under pytest, so this must fall back to
    # printing rather than blocking or raising.
    labeller = Labeller(_df())
    assert labeller.remaining == 3
    assert not labeller.is_complete
    captured = capsys.readouterr()
    assert "record(row_id, verdict" in captured.out


def test_ordered_labels_matches_dataframe_row_order():
    labeller = Labeller(_df())
    labeller.record(10, 1)
    labeller.record(20, 0)
    labeller.record(30, 1)
    assert labeller.is_complete
    assert labeller.remaining == 0
    # Row order is id 30, 10, 20 -- not label-call order or id order.
    true_labels = [0, 0, 0]
    true_confident, pred_confident = labeller.ordered_labels(true_labels)
    assert pred_confident == [1, 1, 0]
    assert true_confident == true_labels


def test_incomplete_access_raises_informative_error():
    labeller = Labeller(_df())
    labeller.record(10, 1)
    with pytest.raises(RuntimeError, match="2 of 3"):
        labeller.ordered_labels([0, 0, 0])


def test_relabel_overwrites_previous_value():
    labeller = Labeller(_df())
    labeller.record(10, 1)
    labeller.record(10, 0)
    labeller.record(20, 0)
    labeller.record(30, 0)
    _, pred_confident = labeller.ordered_labels([0, 0, 0])
    assert pred_confident == [0, 0, 0]


def test_record_rejects_unknown_id():
    labeller = Labeller(_df())
    with pytest.raises(KeyError):
        labeller.record(999, 1)


def test_record_rejects_invalid_verdict():
    labeller = Labeller(_df())
    with pytest.raises(ValueError):
        labeller.record(10, 2)


# -- unsure verdict -----------------------------------------------------------


def test_unsure_counts_as_labelled_but_excluded_from_confusion():
    labeller = Labeller(_df())
    labeller.record(10, 1)
    labeller.record(20, -1)
    labeller.record(30, 0)
    # is_complete only needs *some* verdict per row -- unsure included.
    assert labeller.is_complete
    assert labeller.remaining == 0
    assert labeller.unsure_count == 1

    true_confident, pred_confident = labeller.ordered_labels([1, 1, 0])
    # Row order is 30, 10, 20 -- id 20 (unsure) is dropped from both lists,
    # leaving row 30 (true=1, marked clean=0) and row 10 (true=1, marked 1).
    assert true_confident == [1, 1]
    assert pred_confident == [0, 1]


# -- notes --------------------------------------------------------------------


def test_notes_recorded_and_retrieved_by_id():
    labeller = Labeller(_df())
    labeller.record(10, 1, note="fabricated date")
    labeller.record(20, 0)
    labeller.record(30, -1, note="genuinely unclear")
    assert labeller.notes() == [
        (30, "genuinely unclear"),
        (10, "fabricated date"),
    ]


# -- persistence ----------------------------------------------------------


def test_labels_persist_across_new_instance(_isolated_storage):
    first = Labeller(_df())
    first.record(10, 1, note="a note")
    first.record(20, -1)

    second = Labeller(_df())
    assert second.remaining == 1
    assert second.unsure_count == 1
    assert second.notes() == [(10, "a note")]


def test_construction_reports_restored_progress(_isolated_storage, capsys):
    first = Labeller(_df())
    first.record(10, 1)
    capsys.readouterr()

    Labeller(_df())
    captured = capsys.readouterr()
    assert "Restored 1 of 3" in captured.out


def test_corrupt_storage_file_does_not_break_construction(_isolated_storage):
    _isolated_storage.parent.mkdir(parents=True, exist_ok=True)
    _isolated_storage.write_text("{not valid json", encoding="utf-8")
    labeller = Labeller(_df())
    assert labeller.remaining == 3


def test_empty_storage_file_does_not_break_construction(_isolated_storage):
    _isolated_storage.parent.mkdir(parents=True, exist_ok=True)
    _isolated_storage.write_text("", encoding="utf-8")
    labeller = Labeller(_df())
    assert labeller.remaining == 3


def test_missing_storage_file_does_not_break_construction(_isolated_storage):
    assert not _isolated_storage.exists()
    labeller = Labeller(_df())
    assert labeller.remaining == 3


def test_reset_clears_memory_and_file(_isolated_storage):
    labeller = Labeller(_df())
    labeller.record(10, 1)
    assert _isolated_storage.exists()

    labeller.reset()
    assert labeller.remaining == 3
    assert not labeller.is_complete
    assert not _isolated_storage.exists()

    # A fresh instance sees no restored progress either.
    reloaded = Labeller(_df())
    assert reloaded.remaining == 3
