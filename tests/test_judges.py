"""Judge contract tests. Model calls are mocked; nothing here touches the network."""

import pandas as pd

from evals_workshop import judges


def test_rubric_hash_deterministic():
    h1 = judges.rubric_hash(judges.RUBRIC_V1)
    h2 = judges.rubric_hash(judges.RUBRIC_V1)
    assert h1 == h2
    assert len(h1) == 8


def test_rubric_hash_changes_with_content():
    assert judges.rubric_hash("a") != judges.rubric_hash("b")


def test_build_judge_receives_explicit_model():
    judge = judges.build_judge(judges.RUBRIC_V1, "groq:/dummy-model")
    stored_model = judge.model_dump()["instructions_judge_pydantic_data"]["model"]
    assert stored_model == "groq:/dummy-model"


def test_render_prompt_substitutes_placeholders():
    row = pd.Series({"query": "What year?", "context": "It happened in 1990.", "response": "1990"})
    prompt = judges._render_prompt(judges.RUBRIC_V1, row)
    assert "{{ inputs }}" not in prompt
    assert "{{ outputs }}" not in prompt
    assert "What year?" in prompt
    assert "1990" in prompt


def test_parse_response_extracts_verdict_and_reason():
    text = "Reason: the answer invents a date not in the context.\nVerdict: yes"
    verdict, reason = judges._parse_response(text)
    assert verdict == 1
    assert "invents a date" in reason


def test_parse_response_no_case():
    verdict, _ = judges._parse_response("Reason: fully supported.\nVerdict: no")
    assert verdict == 0


def test_parse_response_malformed_is_minus_one():
    verdict, reason = judges._parse_response("I think this is fine.")
    assert verdict == -1


def test_parse_response_missing_text_is_minus_one():
    verdict, reason = judges._parse_response(None)
    assert verdict == -1
    assert reason == ""


def test_run_judge_builds_expected_dataframe(monkeypatch):
    rows = pd.DataFrame(
        {
            "id": ["a", "b"],
            "query": ["q1", "q2"],
            "context": ["c1", "c2"],
            "response": ["r1", "r2"],
        }
    )

    def fake_judge_batch(prompts, model, tpm, max_concurrency, cache_path=None):
        assert len(prompts) == 2
        return [
            {"text": "Reason: fine.\nVerdict: no", "error": None},
            {"text": "Reason: made something up.\nVerdict: yes", "error": None},
        ]

    monkeypatch.setattr(judges, "judge_batch", fake_judge_batch)

    judge = judges.build_judge(judges.RUBRIC_V1, "groq:/dummy-model")
    result = judges.run_judge(judge, rows)

    assert list(result["id"]) == ["a", "b"]
    assert list(result["verdict"]) == [0, 1]
    assert "made something up" in result["reason"].iloc[1]


def test_add_few_shot_examples_inserts_before_response_format():
    examples = pd.DataFrame({"query": ["q1"], "context": ["c1"], "response": ["r1"], "label": [1]})
    rubric_v2 = judges.add_few_shot_examples(judges.RUBRIC_V1, examples)
    assert "Worked examples" in rubric_v2
    assert rubric_v2.index("Worked examples") < rubric_v2.index("Response format")
    assert "Verdict: yes" in rubric_v2


def test_select_few_shot_picks_shortest():
    candidates = pd.DataFrame(
        {
            "query": ["long", "short", "mid"],
            "context": ["c", "c", "c"],
            "response": ["r", "r", "r"],
            "label": [1, 1, 0],
            "n_tokens": [400, 100, 250],
        }
    )
    picked = judges.select_few_shot(candidates, n=2)
    assert sorted(picked["n_tokens"]) == [100, 250]


def test_add_few_shot_examples_escapes_literal_double_braces():
    examples = pd.DataFrame(
        {
            "query": ["q1"],
            "context": ["Bake at 425. {{model.addEditText}} Print. 1 Prep."],
            "response": ["r1"],
            "label": [1],
        }
    )
    rubric_v2 = judges.add_few_shot_examples(judges.RUBRIC_V1, examples)
    assert "{{model.addEditText}}" not in rubric_v2
    # mlflow's make_judge must accept the result: no unescaped "{{ }}" beyond ours.
    judges.build_judge(rubric_v2, "groq:/dummy-model")


def test_run_judge_fresh_bypasses_cache(monkeypatch):
    rows = pd.DataFrame({"id": ["a"], "query": ["q"], "context": ["c"], "response": ["r"]})
    seen_paths = []

    def fake_judge_batch(prompts, model, tpm, max_concurrency, cache_path=None):
        seen_paths.append(cache_path)
        return [{"text": "Reason: ok.\nVerdict: no", "error": None}]

    monkeypatch.setattr(judges, "judge_batch", fake_judge_batch)
    judge = judges.build_judge(judges.RUBRIC_V1, "groq:/dummy-model")
    judges.run_judge(judge, rows, fresh=True)
    judges.run_judge(judge, rows)
    assert seen_paths[0] is None
    assert seen_paths[1] is not None


def test_run_judge_counts_parse_failures_as_minus_one(monkeypatch):
    rows = pd.DataFrame({"id": ["a"], "query": ["q"], "context": ["c"], "response": ["r"]})

    def fake_judge_batch(prompts, model, tpm, max_concurrency, cache_path=None):
        return [{"text": "garbage output", "error": None}]

    monkeypatch.setattr(judges, "judge_batch", fake_judge_batch)

    judge = judges.build_judge(judges.RUBRIC_V1, "groq:/dummy-model")
    result = judges.run_judge(judge, rows)
    assert result["verdict"].iloc[0] == -1
