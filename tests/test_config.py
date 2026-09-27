"""Regression coverage for persistent diagnostic logging."""

from pathlib import Path

from loguru import logger

from foco_llm.utils.logger import setup_logging


def test_setup_logging_writes_debug_records_to_disk(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, level="INFO")
    logger.debug("marker record {}", 7)
    logger.remove()
    written = list(tmp_path.glob("*.log"))
    assert len(written) == 1
    assert "marker record 7" in written[0].read_text(encoding="utf-8")
