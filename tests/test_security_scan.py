import re
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "path,excluded",
    [
        (".venv/Lib/example.py", True),
        (r".venv\Lib\example.py", True),
        (r"src\foco_llm\core\evaluation.py", False),
        ("src/foco_llm/config.py", False),
        (".env.example", False),
    ],
)
def test_secret_scan_excludes_dependencies_on_both_platforms(path, excluded):
    settings = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "check_secrets.py"))
    assert bool(re.search(settings["EXCLUDED_PATHS"], path)) is excluded


@pytest.mark.parametrize(
    "key,size,suffix,excluded",
    [
        ("prompt_sha256", 64, ",", True),
        ("dataset_sha256", 64, "", True),
        ("evaluation_dataset_sha256", 64, ",", True),
        ("training_dataset_sha256", 64, ",", True),
        ("script_sha256", 64, ",", True),
        ("script_sha256", 32, "", False),
        ("adapter_sha256", 64, ",", True),
        ("adapter_config_sha256", 64, ",", True),
        ("training_manifest_sha256", 64, ",", True),
        ("sha256", 64, "", True),
        ("adapter_sha256", 32, "", False),
        ("sha256", 64, ', "extra": "value"', False),
        ("model_revision", 40, ",", True),
        ("api_key", 64, "", False),
        ("revision", 32, "", False),
        # Synthetic extra field verifies that the exclusion cannot hide credentials.
        ("prompt_sha256", 64, ', "api_key": "value"', False),  # pragma: allowlist secret
    ],
)
def test_only_complete_provenance_fields_are_excluded(key, size, suffix, excluded):
    settings = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "check_secrets.py"))
    line = f'  "{key}": "{"a" * size}"{suffix}'
    assert bool(re.search(settings["PROVENANCE_LINES"], line)) is excluded


@pytest.mark.parametrize(
    "value,suffix,excluded",
    [
        ("d3382a884de2c421", ",", True),  # pragma: allowlist secret -- public run ID
        ("6836fbda70e99de1", "", True),  # pragma: allowlist secret -- public run ID
        ("a" * 16, "", False),
        ("d3382a884de2c421", ', "other": "value"', False),  # pragma: allowlist secret
    ],
)
def test_only_audited_standalone_run_ids_are_excluded(value, suffix, excluded):
    settings = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "check_secrets.py"))
    assert bool(re.search(settings["PROVENANCE_LINES"], f'  "{value}"{suffix}')) is excluded
