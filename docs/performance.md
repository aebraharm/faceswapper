# Performance and stability

## Current controls

- Camera frames are resized on the server to a configurable maximum processing dimension (320, 480, 640 or 720 px); capture is capped at 960 px maximum side in the browser.
- The frontend sends a JPEG at a bounded rate and checks WebSocket `bufferedAmount`; it drops stale frames rather than queueing them.
- The backend offloads CPU frame processing with `asyncio.to_thread`, reports rolling end-to-end FPS and average latency, and returns the original frame if there is no valid target/model/source.
- ONNX Runtime selects CUDA only when explicitly requested and available or when `auto` sees `CUDAExecutionProvider`; otherwise CPU is used. Model load includes a warm-up pass. `FACE_ONNX_INTRA_THREADS` tunes the intra-op thread count for the catalog LivePortrait session; the warping/generator graph input order is resolved from the actual ONNX graph at load time rather than hardcoded.
- Source embedding/representation computation is done once on source selection/model load; only compact numeric arrays are cached (never source pixels). Target inference and alignment/blend run per frame.
- `POST /camera/settings` accepts `performance_mode`: `auto` (default), `quality`, or `performance`. In `auto`, once the stream has sustained a low rolling FPS for several frames, inference is skipped on alternating frames and the last composited frame is reused instead of falling further behind; `quality` always runs the model every frame; `performance` always skips every other frame. This keeps the UI responsive on a slow/4 GB machine rather than freezing or crashing, and the camera never shows a stale frame once transformation is disabled.
- A model failure mid-stream (e.g. the OS refuses a memory allocation while running the warping network) is caught and converted into a recoverable per-frame error; the stream keeps running on the next frame instead of crashing.

## Tuning

1. Start at 320 or 480 px; increase only after measuring latency and output quality.
2. Reduce camera capture resolution in browser settings if the device/driver allows it. Current UI reports actual frame dimensions.
3. Close other applications using the GPU and choose an ONNX Runtime build matching the installed CUDA/cuDNN versions.
3a. On a 4 GB RAM Windows laptop with the catalog LivePortrait model loaded (~1.5–2.5 GB resident), close other memory-heavy apps and switch to "Performance" mode and/or 320 px processing resolution before increasing quality settings.
4. Benchmark source encoder, transformer, detection and blending separately for a new model. Model graph architecture is often the largest factor.
5. Keep the browser tab foregrounded; browsers throttle animation callbacks in background tabs.

## Interpretation

The UI reports processed frames per second and processing latency from the local stream, not a guarantee of 20–30 FPS. Actual rate depends on model, CPU/GPU, camera and browser. Face detection falls back to CPU Haar cascades when MediaPipe is absent. Stability and bounded latency are prioritized over a fixed FPS target.

## Future optimization seams

The frame processor and transformer interfaces allow detector cadence/tracking backends, GPU-specific aligned crop buffers, dynamic frame skipping and an asynchronous inference queue to be added independently. A production multi-user service needs isolated per-user sessions, authentication, upload quotas, backpressure and model-resource scheduling.
