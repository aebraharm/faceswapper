# Architecture

## Design goals

- Keep source **identity** and live **target** as distinct values throughout the pipeline.
- Make every camera activation an explicit browser action; the API does not enumerate or open a physical camera.
- Keep the camera/data path stable when changing inference models.
- Bound upload/frame sizes, avoid writing portraits to disk, and fail closed if transformation prerequisites are missing.
- Support one selected target now while surfacing all detections for a later multi-face workflow.

## Components

```text
React UI (frontend/src/App.tsx)
  ├─ browser getUserMedia, camera picker, original preview
  ├─ source photo upload, face selection, model/target/settings controls
  ├─ WebSocket JPEG sender + processed canvas + live telemetry
  └─ canvas.captureStream() browser integration helper

FastAPI (backend/app/main.py)
  ├─ API routes (api/routes.py) and bounded WebSocket (camera/websocket.py)
  ├─ FaceAnalyzer (MediaPipe Face Mesh; OpenCV Haar fallback)
  ├─ FaceAligner + SourceFaceStore (in-memory selection/features)
  ├─ CameraSession + TargetFaceTracker + FrameProcessor
  ├─ FaceTransformer contract + ONNX Runtime adapter/manager
  └─ FaceCompositor (elliptical feathered mask + inverse affine warp)
```

### Source path

`POST /source-face/upload` checks declared MIME against decoded file type, byte size, dimensions and pixel count. It analyzes the image, rejects zero detections, and keeps the decoded pixels only in the in-memory `SourceFaceStore`. A single face is selected automatically; with multiple faces the aligned source is withheld until `POST /source-face/select` chooses an index. The selected face is aligned to the canonical crop. If a model is already loaded, `prepare_source()` computes an identity representation once. If the model loads later, the aligned crop is encoded once at load time. Only geometry/status metadata is returned to the browser; the browser's preview is a local object URL.

### Target-frame path

The browser requests camera permission only when Start is clicked. It sends bounded JPEG frames over a same-origin WebSocket. `FrameProcessor` decodes and resizes each frame, detects/landmarks faces, smooths the selected target box, and records telemetry. With multiple detections and no explicit target choice, it highlights candidates but does not transform any of them. When transformation is enabled and both a model and selected source representation exist, it aligns only the chosen target, runs the model, warps the output and soft mask back to the frame, and blends. If detection/model/source is unavailable, it returns the original frame with analysis overlays instead of crashing or making a false transformation claim.

### Model boundary

`FaceTransformer` separates `load_model`, `prepare_source`, `transform`, and `unload_model`. `TransformerManager` owns the model lifecycle. The shipped `OnnxIdentityTransformer` requires a source encoder graph and a target-conditioned transformation graph with the ABI in `models/README.md`. No weights are included. Models with another input contract should be wrapped behind the same interface, leaving camera, alignment and UI services intact.

### Output

The processed frame is displayed on a canvas. The frontend's `getProcessedVideoStream()` helper returns `canvas.captureStream()` for a browser-side WebRTC peer connection. Signaling and OS virtual-camera integration are not part of this MVP; a desktop virtual camera is platform-specific and not needed to test the core pipeline.

## Concurrency and lifecycle

- CPU frame inference is dispatched via `asyncio.to_thread` so the FastAPI event loop can continue handling control requests.
- The browser bounds its WebSocket send queue and drops stale frames rather than accumulating latency.
- Camera session, transformer lifecycle and source-face state use locks where state can cross thread/request boundaries.
- Stop camera releases browser tracks and disables transformation. Remove source clears image/crop/embedding arrays. Unload model clears the identity embedding and ONNX sessions.
- Current v1 supports one active browser stream per shared process. A multi-user deployment would require per-session state, quotas and authentication before exposure.

## Trust boundary

The UI uses a relative same-origin API/WebSocket URL; Vite proxies these in development. Uploads are never executed, stored on disk, or publicly served. Run the development server on a trusted machine/network only: this MVP does not add authentication, CSRF protection, tenant isolation, or hostile-network hardening.
