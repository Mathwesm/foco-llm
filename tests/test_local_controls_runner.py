import runpy
from pathlib import Path


def test_runner_does_not_force_offline_metadata_failure(tmp_path, monkeypatch):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_local_controls.py"))
    captured = {}

    def record_command(command, **kwargs):
        captured.update(
            offline="HF_HUB_OFFLINE" in kwargs["env"],
            utf8=kwargs["env"]["PYTHONUTF8"],
            timeout=kwargs["timeout"],
            check=kwargs["check"],
        )

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(module["subprocess"], "run", record_command)
    module["execute"](["-m", "foco_llm.train_pilot"], tmp_path / "stage.log")
    assert captured["offline"] is False
    assert captured["utf8"] == "1"
    assert captured["timeout"] == 1800
    assert captured["check"] is True
