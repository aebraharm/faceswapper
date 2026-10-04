# Performance and stability

## Current controls

- Camera frames are resized on the server to a configurable maximum processing dimension (320, 480, 640 or 720 px); capture is capped at 960 px maximum side in the browser.
- The frontend sends a JPEG at a bounded rate and checks WebSocket `bufferedAmount`; it drops stale frames rather than queueing them.
- The backend offloads CPU frame processing with `asyncio.to_thread`, reports rolling end-to-end FPS and average latency, and returns the original frame if there is no valid target/model/source.
- ONNX Runtime selects CUDA only when explicitly requested and available or when `auto` sees `CUDAExecutionProvider`; otherwise CPU is used. Model load includes a warm-up pass.
- Source embedding computation is done once on source selection/model load; target inference is per frame.

## Tuning

1. Start at 320 or 480 px; increase only after measuring latency and output quality.
2. Reduce camera capture resolution in browser settings if the device/driver allows it. Current UI reports actual frame dimensions.
3. Close other applications using the GPU and choose an ONNX Runtime build matching the installed CUDA/cuDNN versions.
4. Benchmark source encoder, transformer, detection and blending separately for a new model. Model graph architecture is often the largest factor.
5. Keep the browser tab foregrounded; browsers throttle animation callbacks in background tabs.

## Interpretation

The UI reports processed frames per second and processing latency from the local stream, not a guarantee of 20–30 FPS. Actual rate depends on model, CPU/GPU, camera and browser. Face detection falls back to CPU Haar cascades when MediaPipe is absent. Stability and bounded latency are prioritized over a fixed FPS target.

## Future optimization seams

The frame processor and transformer interfaces allow detector cadence/tracking backends, GPU-specific aligned crop buffers, dynamic frame skipping and an asynchronous inference queue to be added independently. A production multi-user service needs isolated per-user sessions, authentication, upload quotas, backpressure and model-resource scheduling.
