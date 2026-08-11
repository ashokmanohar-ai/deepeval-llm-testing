.PHONY: install check test eval expected-failures

install:
	python -m pip install -r requirements-dev.txt

check:
	python scripts/validate_repository.py
	ruff check .
	python -m pytest -m "not azure"

test:
	deepeval test run tests

eval:
	deepeval test run tests -c

expected-failures:
	python -m pytest tests/expected_failures -o addopts="" -v

