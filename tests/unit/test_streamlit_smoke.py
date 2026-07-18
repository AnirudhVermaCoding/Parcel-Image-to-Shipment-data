from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_streamlit_app_renders_in_fresh_process(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "RUNTIME_DIR": str(tmp_path / "runtime"),
            "STREAMLIT_CLOUD": "true",
            "MAX_WORKERS": "2",
            "MAX_WORKER_LIMIT": "8",
        }
    )
    script = """
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app.py").run(timeout=30)
assert not app.exception, [item.message for item in app.exception]
worker_input = next(item for item in app.number_input if item.label == "Maximum workers")
assert worker_input.value == 2
assert worker_input.max == 8
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
