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
