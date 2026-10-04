# Windows desktop application

FRAME now has an Electron shell around the existing React/Vite UI and Python/FastAPI vision service. The renderer is sandboxed; Electron starts the Python sidecar on loopback, waits for its health check, then gives the UI a narrow preload bridge containing the runtime endpoint and a per-launch session token.

```text
Electron main (Windows)
  ├─ starts/stops Python sidecar
  ├─ chooses an ephemeral port on 127.0.0.1
  ├─ owns a random per-launch session/shutdown token
  └─ creates a hardened BrowserWindow (context isolation; no renderer Node API)
       └─ React/Vite renderer
            ├─ asks for camera authorization only after Start Camera is clicked
            ├─ sends webcam JPEGs over authenticated local WebSocket
            └─ calls local FastAPI REST routes for upload/settings/status
                  └─ OpenCV → alignment → optional transformer → blend → processed JPEG
```

Camera frames and the source photo (only after the user explicitly selects it) go to the Python process over loopback. The backend keeps source pixels and derived identity features in memory only; it does not write or log them. The desktop backend binds to `127.0.0.1` only. There is no cloud relay or telemetry in the desktop path. The virtual-camera output remains a future phase; `canvas.captureStream()` is still only a browser stream.

## Requirements

- Windows 10/11 x64 for the packaged installer.
- Python 3.11+ for development and building the sidecar.
- Node.js 22.12+ and npm.
- A supported camera/browser permission from Windows. Camera access is requested only by the Start Camera button.

## Development: regular web workflow

The existing Vite/FastAPI workflow remains available:

```powershell
# From the repository root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements-dev.txt

# Terminal 1: local backend, bound to loopback
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

# Terminal 2
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The Vite proxy continues to send `/api/*` and `/ws/stream` to the development backend.

## Development: Electron desktop window

Install dependencies as above, then from `frontend` set the Python interpreter if it is not already on PATH and start Electron:

```powershell
cd frontend
$env:FRAME_PYTHON = (Resolve-Path ..\.venv\Scripts\python.exe).Path
npm ci
npm run dev:desktop
```

`dev:desktop` builds the Electron main/preload TypeScript, starts Vite on `127.0.0.1:5173`, waits for it, then opens the desktop window. Electron launches `backend/launcher.py` itself; that launcher binds only to loopback on a dynamically allocated port. The desktop renderer uses the preload-provided runtime URL rather than assuming port 8000. Vite-only browser mode still uses its `/api` and WebSocket proxy.

If the backend cannot start, Electron shows a native startup error dialog with the reason and recent backend output. If the sidecar exits after startup, the renderer displays a backend-stopped message. The backend health route is polled before the window loads.

## How the Python process is launched

- **Development:** Electron spawns `FRAME_PYTHON backend/launcher.py --port <ephemeral-port>` with `backend/` as the working directory. `FRAME_PYTHON` is optional if `python`/`python3` resolves to the environment containing `backend/requirements.txt`.
- **Packaged app:** Electron spawns `process.resourcesPath\backend\FrameBackend.exe --port <ephemeral-port>`. No system Python installation is needed by the end user.
- Electron tests port availability, retries startup if a port is claimed between allocation and bind, and polls `GET /health` with the per-launch session header.
- The preload exposes only `getRuntimeConfig`, `authorizeCamera`, and a backend-error subscription. Renderer Node integration is off and context isolation is on.
- When quitting, Electron calls the authenticated `/internal/shutdown` hook so Uvicorn can exit cleanly; it terminates the child if graceful shutdown times out.

## Per-user installation and application data

The NSIS installer is one-click and per-user (`perMachine: false`), with elevation and custom install paths disabled. It installs under the current Windows user's local Programs directory; no administrator privileges are required. NSIS creates an uninstaller and shortcuts. Uninstall removes the app but intentionally retains user settings so reinstalling does not lose preferences.

Electron stores only the inference-provider choice, transform intensity, and processing resolution in `%LOCALAPPDATA%\FRAME\settings.json` for packaged Windows installs. Development uses `%LOCALAPPDATA%\FRAME Development\settings.json`. These folders are created with the current user's permissions. Source photos, face features, camera frames, session tokens, and model weights are not written there. To remove preferences as well, delete the `FRAME` folder under `%LOCALAPPDATA%` after uninstalling.

## Build the Windows NSIS installer

Build on Windows (or a Windows CI runner); PyInstaller does not cross-compile a Windows Python executable from the Linux development container.

```powershell
# From the repository root; activate an environment with Python 3.11+
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements-build.txt
cd frontend
npm ci
npm run dist:win
```

The PowerShell build script:

1. Runs PyInstaller in `--onedir` mode for the FastAPI service and collects OpenCV/ONNX Runtime binaries.
2. Builds the Vite renderer and compiles the Electron main/preload processes.
3. Launches the newly built sidecar and verifies authenticated health, rejects missing/wrong session tokens, and exercises authenticated graceful shutdown.
4. Runs electron-builder with the **NSIS x64** target.

The installer is written to `frontend/release/`. Build outputs are ignored by Git. Electron’s Chromium and the renderer assets are packaged by electron-builder; the Python sidecar is copied as an extra resource. The Windows CI workflow at `.github/workflows/windows-desktop.yml` also silently installs the generated installer as the current user, checks the production app and sidecar files, runs the uninstaller, and uploads the installer artifact.

## Packaged backend location

The installer places the sidecar under Electron’s resources directory:

```text
<install directory>\resources\backend\FrameBackend.exe
<install directory>\resources\backend\_internal\...   # PyInstaller runtime libraries/data
```

The main process resolves the executable from `process.resourcesPath`, not the current working directory. Do not move the executable out of the PyInstaller folder without its `_internal` files.

## Errors and diagnostics

- **Startup dialog:** executable/Python missing, dependency import failure, sidecar exits, port bind failure, or health timeout. Recent stdout/stderr is included when available.
- **Runtime banner:** if the child exits unexpectedly, Electron informs the renderer, which releases the camera and shows the failure.
- **Camera errors:** remain visible in the existing inline camera panel; Electron authorizes only the camera request initiated by the Start Camera UI action.
- **Backend:** development logs appear in the Electron process terminal. The packaged sidecar suppresses access logs to avoid logging uploaded image/video requests.

## Windows desktop validation checklist

The automated Windows workflow covers sidecar startup/authentication/shutdown plus a silent per-user installer/uninstaller round trip. On a Windows machine with a camera, also verify the UI/device behavior manually:

1. Install as a standard user and launch FRAME. Confirm the sidecar becomes healthy and the UI loads from the installed production bundle; no Vite server or system Python should be needed.
2. Leave the camera idle and confirm no camera permission prompt or active camera indicator appears. Click **Start camera** and confirm that is the first camera permission request; deny and retry once, then allow and verify live preview.
3. Confirm the app sends frames over the local WebSocket, receives processed JPEGs and telemetry, and releases the camera after **Stop camera** and app exit.
4. Change provider/intensity/resolution, restart the app, and confirm those preferences return while the camera remains stopped. Confirm source photos/identity data are not retained after removing them or exiting.
5. Uninstall from Windows as a standard user; verify the installed files and shortcuts are removed. The preferences folder is intentionally retained unless manually deleted.

A physical camera is not required for the automated CI job and is not exercised by the sandbox tests.

## Model and GPU notes

No face-transformation weights are bundled. The configured local model bundle must have terms that permit the intended use. The transformer interface and existing GPU-provider detection/fallback remain unchanged. The default Python requirements install CPU ONNX Runtime; CUDA execution requires a compatible GPU-enabled ONNX Runtime build and matching Windows CUDA/cuDNN dependencies. Validate that configuration on the target Windows machine before distributing a GPU-enabled build.

A system virtual camera is not included in this phase. A future implementation will add a platform-isolated output adapter and document the required Windows virtual-camera provider/driver separately.
