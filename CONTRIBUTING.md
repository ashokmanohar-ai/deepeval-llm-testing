# Contributing

1. Create a feature branch.
2. Add or update an evaluation case and explain the expected quality behavior.
3. Run `python scripts/validate_repository.py`, `ruff check .`, and
   `python -m pytest -m "not azure"`.
4. With Azure credentials configured, run `deepeval test run tests`.
5. Never commit `.env`, real prompts containing private data, or captured credentials.

When changing a threshold, include evidence from a labeled calibration set rather than
choosing a value from a single example.

