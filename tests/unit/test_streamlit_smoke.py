from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_exception(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("STREAMLIT_CLOUD", "true")
    monkeypatch.setenv("MAX_WORKERS", "2")
    monkeypatch.setenv("MAX_WORKER_LIMIT", "8")

    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=30)

    assert not app.exception
    worker_input = next(item for item in app.number_input if item.label == "Maximum workers")
    assert worker_input.value == 2
    assert worker_input.max == 8
