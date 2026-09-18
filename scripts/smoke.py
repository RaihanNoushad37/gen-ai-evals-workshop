"""Four checks students run before the workshop. Each prints PASS or FAIL plus a remedy."""

import sys


def check_config() -> bool:
    try:
        from evals_workshop.config import settings

        assert settings.groq_api_key
        print("PASS: config loads and GROQ_API_KEY is present")
        return True
    except Exception as exc:
        print(f"FAIL: config check — {exc}")
        print("  Remedy: copy .env.example to .env and paste your Groq API key.")
        return False


def check_model_call() -> bool:
    import time

    import litellm

    from evals_workshop.config import settings

    try:
        model = settings.judge_model.replace(":/", "/")
        start = time.monotonic()
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            timeout=30,
        )
        elapsed = time.monotonic() - start
        print(f"PASS: one model call returned in {elapsed:.1f}s")
        return True
    except Exception as exc:
        print(f"FAIL: model call — {exc}")
        print("  Remedy: check GROQ_API_KEY and JUDGE_MODEL in .env, and your network connection.")
        return False


def check_data() -> bool:
    try:
        from evals_workshop.data import check_integrity, load_split

        load_split("dev")
        check_integrity()
        print("PASS: data/splits.parquet loads and check_integrity() passes")
        return True
    except Exception as exc:
        print(f"FAIL: data check — {exc}")
        print("  Remedy: run `make data` to build data/splits.parquet.")
        return False


def check_mlflow() -> bool:
    try:
        import mlflow

        from evals_workshop.config import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_name="smoke_test"):
            pass
        print("PASS: an MLflow run opens and closes against the local store")
        print("  View your runs with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        return True
    except Exception as exc:
        print(f"FAIL: mlflow check — {exc}")
        print("  Remedy: check MLFLOW_TRACKING_URI in .env and that ./mlruns is writable.")
        return False


def main() -> int:
    checks = [check_config, check_model_call, check_data, check_mlflow]
    results = [check() for check in checks]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
