from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


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


def test_streamlit_server_starts_in_fresh_process(tmp_path: Path) -> None:
    """Exercise Streamlit's real server startup, not only its test runner."""
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "RUNTIME_DIR": str(tmp_path / "runtime-server"),
            "STREAMLIT_CLOUD": "true",
            "MAX_WORKERS": "2",
            "MAX_WORKER_LIMIT": "8",
        }
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + 35
    startup_error: str | None = None
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                startup_error = "Streamlit exited before becoming healthy."
                break
            try:
                with urlopen(health_url, timeout=1) as response:
                    if response.status == 200 and response.read().strip() == b"ok":
                        return
            except (OSError, URLError):
                time.sleep(0.25)
        startup_error = startup_error or "Streamlit did not become healthy within 35 seconds."
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        output = process.stdout.read() if process.stdout else ""

    raise AssertionError(f"{startup_error}\n{output}")
