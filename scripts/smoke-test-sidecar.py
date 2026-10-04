#!/usr/bin/env python3
"""Launch a PyInstaller sidecar and verify authenticated readiness/shutdown."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def request(url: str, *, session_token: str | None = None, shutdown: bool = False):
    headers = {}
    if session_token:
        headers["X-Frame-Session"] = session_token
    if shutdown and session_token:
        headers["X-Frame-Shutdown-Token"] = session_token
    req = urllib.request.Request(url, headers=headers, method="POST" if shutdown else "GET")
    with urllib.request.urlopen(req, timeout=2) as response:
        return response.status, json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=25)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Packaged sidecar was not found: {executable}")

    port = free_loopback_port()
    token = secrets.token_hex(32)
    environment = os.environ.copy()
    environment["FRAME_DESKTOP_SESSION_TOKEN"] = token
    environment["FRAME_DESKTOP_SHUTDOWN_TOKEN"] = token
    process = subprocess.Popen(
        [str(executable), "--port", str(port)],
        cwd=executable.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Sidecar exited during startup with code {process.returncode}.")
            try:
                status, health = request(f"{base_url}/health", session_token=token)
                if status == 200 and health.get("status") == "ok":
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError(f"Sidecar did not become healthy within {args.timeout:g} seconds.")

        for invalid_token in (None, "wrong-token"):
            try:
                request(f"{base_url}/health", session_token=invalid_token)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise RuntimeError(f"Expected unauthenticated health to return 404, got {error.code}.")
            else:
                raise RuntimeError("The sidecar health endpoint accepted a missing or incorrect token.")

        status, result = request(f"{base_url}/internal/shutdown", session_token=token, shutdown=True)
        if status != 200 or result != {"ok": True}:
            raise RuntimeError("The authenticated sidecar shutdown request was rejected.")
        process.wait(timeout=8)
        if process.returncode != 0:
            raise RuntimeError(f"Sidecar shutdown returned exit code {process.returncode}.")
        print(f"Packaged sidecar smoke test passed (port {port}, authenticated health, graceful exit).")
        return 0
    except Exception as error:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        output = process.communicate(timeout=2)[0] if process.stdout else ""
        raise RuntimeError(f"{error}\nSidecar output:\n{output}") from error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if process.stdout:
            process.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
