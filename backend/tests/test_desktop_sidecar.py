from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _request(url: str, *, token: str | None = None, shutdown_token: str | None = None):
    headers = {}
    if token:
        headers["X-Frame-Session"] = token
    if shutdown_token:
        headers["X-Frame-Shutdown-Token"] = shutdown_token
    method = "POST" if shutdown_token else "GET"
    request = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, json.load(response)


def test_packaged_sidecar_startup_authenticated_health_and_graceful_shutdown():
    launcher = Path(__file__).resolve().parents[1] / "launcher.py"
    port = _free_loopback_port()
    token = secrets.token_hex(32)
    environment = os.environ.copy()
    environment["FRAME_DESKTOP_SESSION_TOKEN"] = token
    environment["FRAME_DESKTOP_SHUTDOWN_TOKEN"] = token
    process = subprocess.Popen(
        [sys.executable, str(launcher), "--port", str(port)],
        cwd=launcher.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"The sidecar exited before becoming healthy.\n{output}")
            try:
                status, health = _request(f"{base_url}/health", token=token)
                if status == 200 and health.get("status") == "ok":
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        else:
            raise AssertionError("The sidecar did not become healthy within 20 seconds.")

        for invalid_token in (None, "wrong-token"):
            try:
                _request(f"{base_url}/health", token=invalid_token)
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError("Health must require the matching per-launch session token.")

        status, result = _request(
            f"{base_url}/internal/shutdown",
            token=token,
            shutdown_token=token,
        )
        assert status == 200
        assert result == {"ok": True}
        assert process.wait(timeout=8) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if process.stdout:
            process.stdout.close()
