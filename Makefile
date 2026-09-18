setup:     ; uv sync
test:      ; uv run pytest
lint:      ; uv run ruff check src tests scripts notebooks solutions && uv run ruff format --check src tests scripts notebooks solutions
notebooks: ; uv run jupytext --to ipynb notebooks/*.py solutions/*.py
smoke:     ; uv run python -m scripts.smoke
data:      ; uv run python -m scripts.prepare_data
cache:     ; uv run python -m scripts.build_cache
ui:        ; uv run mlflow ui
clean:     ; rm -rf .pytest_cache **/__pycache__ mlruns
