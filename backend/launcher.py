"""Desktop sidecar entry point; deliberately binds only to loopback."""
from __future__ import annotations

import argparse

import uvicorn

from app.main import app


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAME local computer-vision service")
    parser.add_argument("--port", type=int, required=True, help="Electron-selected localhost port")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
            access_log=False,
        )
    )
    # Electron asks the sidecar to exit over an authenticated loopback endpoint so
    # Uvicorn can run its normal lifespan/shutdown hooks before the parent falls back
    # to terminating the child process.
    app.state.request_desktop_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()
