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
