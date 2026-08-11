"""Fast repository checks that require no third-party packages or network access."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    ".env.example",
    ".github/workflows/llm-tests.yml",
    "README.md",
    "data/evaluation_cases.json",
    "src/deepeval_llm_testing/azure_openai_client.py",
    "src/deepeval_llm_testing/evaluation/agent_evaluation.py",
    "tests/test_llm_unit.py",
    "tests/test_hallucination.py",
    "tests/test_relevancy.py",
    "tests/test_faithfulness.py",
    "tests/test_geval.py",
    "tests/test_agent_evaluation.py",
]
REQUIRED_ENVIRONMENT_NAMES = {
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
}
AZURE_KEY_PATTERN = re.compile(r"(?i)(?:azure[_-]openai[_-]api[_-]key).{0,20}[=:]\s*[0-9a-f]{32,}")


def validate_structure() -> list[str]:
    return [
        f"Missing required path: {path}" for path in REQUIRED_PATHS if not (ROOT / path).exists()
    ]


def validate_environment_template() -> list[str]:
    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    declared = {
        line.split("=", maxsplit=1)[0].strip()
        for line in template.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    missing = REQUIRED_ENVIRONMENT_NAMES - declared
    errors = [f".env.example is missing {name}" for name in sorted(missing)]
    if (ROOT / ".env").exists():
        errors.append("A real .env file must not be included in the repository.")
    if ".env" not in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        errors.append(".gitignore must contain an exact .env entry.")
    return errors


def validate_python_syntax() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Syntax error in {path.relative_to(ROOT)}: {exc}")
    return errors


def validate_dataset() -> list[str]:
    payload = json.loads((ROOT / "data/evaluation_cases.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(payload, list) or len(payload) < 10:
        errors.append("The sample dataset must contain at least 10 cases.")
        return errors
    identifiers = [item.get("id") for item in payload]
    if len(identifiers) != len(set(identifiers)):
        errors.append("Dataset IDs must be unique.")
    if not any(item.get("should_pass") is False for item in payload):
        errors.append("The dataset must include negative/failing examples.")
    return errors


def scan_for_probable_azure_keys() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if AZURE_KEY_PATTERN.search(content):
            errors.append(f"Probable Azure API key found in {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors = [
        *validate_structure(),
        *validate_environment_template(),
        *validate_python_syntax(),
        *validate_dataset(),
        *scan_for_probable_azure_keys(),
    ]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Repository validation passed: structure, syntax, dataset, and secret checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
