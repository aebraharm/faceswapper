# FRAME — Real-time Face AI

A local-first face-transformation application with a React/Vite development UI and an Electron Windows desktop shell. Choose a source portrait, start the camera explicitly, and inspect a live processed preview with target tracking, performance telemetry, and model status. Electron launches the Python/FastAPI computer-vision backend as a loopback-only sidecar.

> **Model-weight status:** no identity-swapping weights are included. The app ships with face detection, reusable alignment, camera streaming, masking/blending, a replaceable `FaceTransformer` interface, and an ONNX Runtime adapter. MediaPipe landmarks are optional; OpenCV detection/alignment fallback runs without them. A compatible, properly licensed model bundle must be installed locally before identity transformation can be enabled. Without it, the live preview and face analysis still work, and the transform switch fails closed rather than pretending a pasted overlay is an identity transformation.

## What it does

- Captures camera video only after the user clicks **Start camera**. Browser permission and camera-device selection remain browser-controlled; audio is never requested.
- Accepts JPG/JPEG, PNG, and WEBP portraits (8 MB maximum), validates actual file bytes, checks dimensions, detects faces, supports explicit selection when there are multiple, and aligns the chosen face.
- Keeps source pixels and prepared identity features in memory only. Uploads are not written to disk or served through a public URL.
- Streams downsized camera frames over a same-origin WebSocket to FastAPI. The backend detects/tracks faces, highlights candidates, returns processed frames and reports FPS/latency/resolution.
- Provides a replaceable `FaceTransformer` abstraction and an ONNX adapter with separate source-identity encoding and target-conditioned transformation. Source features are prepared once per selection/model load, not once per frame.
- Includes a feathered inverse-warp compositing stage. Transformation is enabled only when a model and selected source representation are ready.
- Exposes the processed browser canvas as a `MediaStream` for integrations inside the browser. It does not install a desktop virtual camera.

## Architecture

```text
Electron main (desktop only)
  starts Python sidecar on an ephemeral 127.0.0.1 port; waits for /health
  └─ hardened BrowserWindow + context-isolated preload bridge
       └─ frontend (React + TypeScript + Vite)
            getUserMedia (explicit click) -> original preview
            JPEG frames ---------------------> local WebSocket /ws/stream
            processed JPEG + telemetry <----- local FastAPI sidecar
            processed canvas -> browser integration only (not a system camera)

backend (FastAPI)
  upload -> validate -> detect faces -> select -> align -> cached source representation
  frame -> detect/landmark -> target selection/tracking -> align -> transformer (optional)
       -> soft mask + inverse warp -> JPEG output
```

The regular web-development workflow is retained. In Electron, HTTP and WebSocket calls use the runtime URL supplied by preload; the backend binds only to loopback and is protected by a per-launch session token.

The source upload uses a bounded raw-image request body (not multipart temp-file uploads) and the backend stores no upload on disk. The backend is organized by responsibility under `backend/app/`: `detection`, `landmarks`, `alignment`, `transformation`, `blending`, `camera`, and `output`. Read [docs/architecture.md](docs/architecture.md) for the request and frame flows.

## Requirements

- Python 3.11+
- Node.js 22.12+ for the current npm toolchain (web and Electron workflows).
- A modern browser with WebRTC-era media APIs (`getUserMedia`, WebSocket, Canvas); camera use requires HTTPS or localhost.
- Optional: a MediaPipe face-landmarker asset, ONNX Runtime-compatible model bundle, and compatible accelerator/runtime for GPU inference.

## Install and run locally

### 1. Backend

```bash
python3.11 -m venv .venv
source .venv/bin/activate             # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# Start from the repository root:
PYTHONPATH=backend uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

API docs: <http://localhost:8000/docs>. Health check: <http://localhost:8000/health>.

### 2. Frontend

In another terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Open <http://localhost:5173>. Vite proxies `/api/*` and `/ws/stream` to the local FastAPI backend. The browser never calls a hard-coded localhost URL. In Arena's hosted preview, the browser sends selected source imagery and camera frames to the workspace backend; run both processes on your own machine for entirely on-device processing. In the preview, both processes must be running; the Vite host allowlist is opened for the proxied preview domain.

For a frontend production build:

```bash
cd frontend && npm run build
```

### Windows desktop app

Electron development, local Python-sidecar startup, secure camera permission handling, Windows PyInstaller packaging, and the NSIS installer workflow are documented in [docs/desktop-windows.md](docs/desktop-windows.md). In short, run `npm run dev:desktop` from `frontend` for the desktop window or `npm run dist:win` on Windows to build the installer.

### 3. Use the app

1. Choose a JPG, PNG, or WEBP portrait. The upload is validated and face-detected immediately.
2. If several faces are detected, select one in the source preview or the face buttons. A single detected face is selected automatically.
3. Press **Start camera** and grant browser permission. Pick a different camera from the selector before starting if needed.
4. The live target feed is analyzed and shown beside the processed output. With several target faces, explicitly choose a target; the app never transforms all faces.
5. Install and load a compatible local model bundle (see [models/README.md](models/README.md)). The switch only enables once both a selected source face and model-generated identity representation exist.
6. Adjust intensity and processing resolution. Turning transformation off immediately returns the untransformed camera frame.
7. Stop the camera to release the browser's media track. Remove the source portrait to clear its in-memory pixels/representation.

## Model installation and licensing

No weights are distributed. Identity-swap model weights frequently carry terms different from their source code, restrict commercial use, or have training-data conditions. FRAME avoids bundling or endorsing weights without a clearly verifiable grant.

The included ONNX adapter expects **two local graphs**. Obtain/convert these only from a provider whose model-weight, training-data, face-image, and intended-use terms you have reviewed. Configure paths in your shell or copy `.env.example` to a local `.env` (not committed):

```bash
export FACE_SOURCE_ENCODER_MODEL="$PWD/models/source_encoder.onnx"
export FACE_TRANSFORMER_MODEL="$PWD/models/face_transformer.onnx"
export FACE_ONNX_PROVIDER=auto       # auto, CPUExecutionProvider, CUDAExecutionProvider
```

Then restart FastAPI and press **Load configured model**. See [models/README.md](models/README.md) for the exact tensor ABI, smoke checks, privacy/consent guidance, and provider setup. A model with a different ABI should be wrapped behind `FaceTransformer` or exported/adapted; the camera pipeline does not need to change.

### Optional MediaPipe landmarks

The core installation stays headless and uses OpenCV face detection/alignment fallback. To enable MediaPipe landmarks, install the optional dependencies without allowing them to replace headless OpenCV, then configure `FACE_LANDMARKER_TASK` with a compatible local Face Landmarker task asset when using the Tasks API:

```bash
pip install --no-deps -r backend/requirements-mediapipe.txt
```

Newer MediaPipe distributions removed the legacy `solutions` API, so the backend reports landmark availability rather than assuming it exists. Review the task asset license/provenance and do not commit it. Some older MediaPipe releases expose `solutions.face_mesh` without a task asset; the adapter supports that API too.

### GPU

The default requirements install CPU `onnxruntime`. For a CUDA-compatible machine, follow ONNX Runtime's current installation matrix for your CUDA/cuDNN version, replace `onnxruntime` with the matching `onnxruntime-gpu` package, and check `onnxruntime.get_available_providers()`. Do not install both packages into the same environment. The status panel reports the provider selected by the ONNX session. CPU is the stable fallback; webcam frames are resized, and the UI offers a processing-resolution control. Hardware, model design, and camera throughput determine attainable FPS.

## API overview

- `GET /health`
- `GET /camera/devices` (documents browser-owned device enumeration)
- `POST /camera/start`, `POST /camera/stop`, `GET /camera/status`
- `POST /camera/target`, `POST /camera/settings`
- `POST /source-face/upload`, `GET /source-face/status`, `POST /source-face/select`, `DELETE /source-face`
- `POST /transformer/load`, `POST /transformer/unload`, `GET /transformer/status`
- `GET /performance`
- `WS /ws/stream` (binary JPEG in; JSON telemetry then binary JPEG out)

Uploads are capped, MIME and actual file format must agree, decoded dimensions/pixels are bounded, and unsupported files are rejected. Camera frames are size-limited. Error responses are actionable and invalid/missing model state does not activate transformation.

## Video-call output

The frontend exports the processed canvas using `getProcessedVideoStream(canvas, fps)` in `frontend/src/services/output.ts`; when enabled, the current stream is also available as `window.faceTransformOutput` for browser-side integrations to attach to an `RTCPeerConnection`. This is not automatically visible as a camera in desktop meeting software. A virtual-camera bridge is platform-specific and intentionally isolated/not included. It is not required to test or use the local processing preview.

## Troubleshooting

- **Engine offline:** start FastAPI on port 8000 and reload the Vite page.
- **Camera unavailable:** use localhost/HTTPS, grant permission, close other apps holding the camera, then restart it. The application never opens a device before the button click.
- **No source face found:** use a well-lit image with a clear, unobstructed face; oversized or mismatched MIME files are rejected.
- **Several faces:** pick the source face and live target explicitly. No target is transformed while a multi-face scene has no selected target.
- **Model cannot load:** verify both configured paths, ONNX Runtime provider availability, input/output names and tensor shapes. See `models/README.md`; no built-in model is silently substituted.
- **Slow preview:** lower processing resolution, close GPU-heavy apps, prefer a CUDA provider if installed correctly, and use a smaller camera capture mode. Detection/preview remains available without identity model weights.
- **Camera starts but preview freezes:** inspect the FastAPI terminal and browser console; stop/restart the session. The WebSocket applies backpressure and drops frames if its send queue grows.

## Privacy, consent, and safe use

This is an opt-in local creative tool. Use portraits and live camera feeds only with the depicted person's permission, do not misrepresent generated footage as authentic, and follow applicable laws and model terms. A visible active-state indicator is shown when transformation is enabled. Source images are not persisted by this app; stopping/unloading/removing the source clears the corresponding session data. Do not expose the development API to an untrusted network: it is designed for a trusted local workstation and has no user-authentication layer.

## Tests

```bash
pip install -r backend/requirements-dev.txt
pytest
```

The tests use synthetic images/frames and fake detectors/models; physical camera hardware and model weights are not required.

## Project map

```text
backend/app/          FastAPI service and modular vision pipeline
backend/launcher.py   loopback-only Electron sidecar entry point
backend/tests/        validation, alignment, model, frame and API tests
frontend/src/         React/TypeScript UI, API client, browser output adapter
frontend/electron/    Electron main process and secure preload bridge
models/README.md      model acquisition notes and ONNX ABI
scripts/              local setup/run and Windows packaging helpers
docs/                 architecture, pipeline and desktop notes
```
