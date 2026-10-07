"""Fail the quality gate when detect-secrets reports potential credentials."""

from __future__ import annotations

import json
import subprocess
import sys

from loguru import logger

EXCLUDED_PATHS = (
    r"(^|[\\/])(\.git|\.venv|data|logs|\.mypy_cache|\.pytest_cache|"
    r"\.ruff_cache|\.hypothesis|__pycache__)[\\/]"
)
# Audited provenance fields are public digests, not credentials. The whole line
# must be one JSON field; other keys and additional content remain scanned.
PROVENANCE_LINES = (
    r'^\s*"(?:prompt|dataset|source|script|selection_scores)_sha256": "[a-f0-9]{64}",?\s*$'
    r'|^\s*"(?:evaluation_dataset|training_dataset|source_dataset)_sha256": "[a-f0-9]{64}",?\s*$'
    # Verified hashes of local adapter files and comparison input artifacts.
    r'|^\s*"(?:adapter_sha256|adapter_config_sha256|training_manifest_sha256|sha256)"'
    r': "[a-f0-9]{64}",?\s*$'
    r'|^\s*"(?:model_revision|revision)": "[a-f0-9]{40}",?\s*$'
    # Audited public run IDs in the BF16 comparison; never exclude arbitrary IDs.
    r'|^\s*"(?:d3382a884de2c421|6836fbda70e99de1)",?\s*$'
)


def main() -> int:
    """Scan project files and fail without displaying suspected secret values."""
    command = [
        sys.executable,
        "-m",
        "detect_secrets",
        "scan",
        "--all-files",
        "--exclude-files",
        EXCLUDED_PATHS,
        "--exclude-lines",
        PROVENANCE_LINES,
    ]
    result = subprocess.run(  # noqa: S603 -- fixed module and arguments, no shell.
        command, capture_output=True, text=True, encoding="utf-8", timeout=120, check=True
    )
    findings = json.loads(result.stdout)["results"]
    if findings:
        logger.error(
            "Potential secrets detected in {} files; review before publishing", len(findings)
        )
        return 1
    logger.info("Secret scan passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
