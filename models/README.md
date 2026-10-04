# Model files and licensing

FRAME's source code (this repository) does **not** include any face-transformation weights, and nothing is downloaded automatically. Model files are either installed on request from the in-app **Model Status** panel, or supplied by you as your own ONNX files. Before relying on any model, independently verify that the exact checkpoint, training data, dependencies and your intended use are allowed by their license and terms — code licenses do not automatically grant rights to pretrained weights, training data, face identities, or every use case. In particular confirm whether commercial use, redistribution, biometric processing and use of the pictured identity are permitted. The user is responsible for consent and applicable-law compliance.

## Option A (recommended): install the built-in LivePortrait catalog model

FRAME ships a small catalog (`backend/app/transformation/model_catalog.py`) of models it knows how to install and run. Today it contains one entry:

| Model | License | Size (download) | Approx. runtime RAM | Source |
| --- | --- | --- | --- | --- |
| LivePortrait (identity-preserving reenactment) | **MIT** ([LICENSE](https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE)) | ~537 MB (3 ONNX graphs) | ~1.5–2.5 GB while the model is loaded | ONNX export republished at [warmshao/FasterLivePortrait](https://huggingface.co/warmshao/FasterLivePortrait/tree/main/liveportrait_onnx) |

Notes on this choice:

- **Licensing:** the three reenactment networks FRAME downloads (`appearance_feature_extractor.onnx`, `motion_extractor.onnx`, `warping_spade.onnx`) are the original MIT-licensed LivePortrait (KwaiVGI) checkpoints. The official LivePortrait LICENSE separately notes that its *reference pipeline* also bundles InsightFace's face detector/landmarker, which is non-commercial-only, and recommends replacing it for MIT compliance. **FRAME never uses the InsightFace assets** — it reuses its own OpenCV + MediaPipe face detector/landmarker/aligner, so only the MIT-licensed networks are ever downloaded or run. Rejected alternatives and why: `inswapper_128` (weights are non-commercial-only despite MIT code), SimSwap (CC-BY-NC 4.0), GHOST/sber-swap (heavier multi-framework runtime, poor fit for a 4 GB machine).
- **Redistribution:** nothing is bundled in this Git repository or in the Windows installer. The app fetches the files at install time directly from the hosting above with a published SHA-256 checksum, verified before the file is considered installed.
- **Size vs. the 100–500 MB target:** at ~537 MB on disk and ~1.5–2.5 GB resident while loaded, this model modestly exceeds the original 100–500 MB target. No smaller, credibly and permissively licensed alternative was found with comparable identity preservation and pose/expression tracking; this is a documented tradeoff, not an oversight. On a 4 GB Windows laptop this leaves a few hundred MB to 1 GB free for the OS, browser/Electron shell and camera pipeline — tight but workable at reduced processing resolution (320–480 px) and with "Performance" mode (see below). Users on very constrained machines can instead supply their own smaller BYO ONNX bundle (Option B).
- **Where it's stored:** downloaded files live under a per-user app-data directory, never inside the installer or `Program Files`/the Git repo:
  - Packaged desktop app: `%LOCALAPPDATA%\FRAME\models\<model-id>\` (Electron sets `FRAME_MODELS_DIR` when it launches the backend).
  - Running the backend directly (dev/CLI): `~/.frame/models/<model-id>/`, or wherever `FRAME_MODELS_DIR` points.
- **How to install it:** open the app, go to the Model Status panel, and click **Install** next to "LivePortrait". This calls `POST /models/liveportrait-v1/install`, which downloads and verifies the files in a background thread — the UI shows progress and you can keep using the rest of the app. Nothing is downloaded until you click Install, and it is never re-downloaded automatically on a later launch. The same model can be removed from disk with the **Remove** button (`DELETE /models/liveportrait-v1`) once it is unloaded.
- **API:** `GET /models/catalog` lists every catalog entry with its license, size and install state; `GET /models/{id}/status`, `POST /models/{id}/install`, `DELETE /models/{id}` manage one entry.

Adding a new catalog entry later must independently confirm the exact checkpoint (not just the surrounding code) is permissively licensed and that the hosting terms allow programmatic download, following the same pattern as the LivePortrait entry above.

## Option B: bring your own ONNX model files

If you have your own licensed two-graph ONNX bundle (source-identity encoder + target-conditioned transformer), FRAME's original BYO adapter still works and takes priority over the catalog model whenever both are configured:

```text
models/
├── source_encoder.onnx     # aligned source face -> identity feature tensor
└── face_transformer.onnx   # cached identity + aligned target -> transformed target crop
```

The `.onnx` pattern is ignored by Git. Keep model files on local storage or another private artifact store; do not commit weights unless their license explicitly allows redistribution and you intentionally choose to do so. An optional MediaPipe Tasks face-landmarker asset can be configured with `FACE_LANDMARKER_TASK`; `.task` files are ignored as well. Verify that asset's terms separately.

Set paths before starting FastAPI:

```bash
export FACE_SOURCE_ENCODER_MODEL="$PWD/models/source_encoder.onnx"
export FACE_TRANSFORMER_MODEL="$PWD/models/face_transformer.onnx"
export FACE_ONNX_PROVIDER=auto
```

`FACE_ONNX_PROVIDER` accepts `auto`, `CPUExecutionProvider`, or `CUDAExecutionProvider`. `auto` selects CUDA only when the installed ONNX Runtime reports it; otherwise CPU is used. Install the ONNX Runtime package matching your system's CUDA/cuDNN versions rather than adding GPU packages blindly.

### BYO tensor ABI

This adapter intentionally supports a documented ABI rather than guessing at unrelated face-swap graphs:

- Both input faces are aligned, RGB, square `256 × 256` crops.
- Image tensors are `float32`, NCHW, batch one, normalized to `[-1, 1]`.
- The encoder takes one image input and returns a finite `float32` identity tensor, usually `[1, D]`. Its output is computed once after source selection (or once when the model is loaded if the source was already selected).
- The transformer graph must have `float32` inputs named exactly `source_identity` and `target_face`. `source_identity` must accept the encoder output shape; `target_face` accepts `[1, 3, 256, 256]`.
- The transformer returns one RGB face crop, either `[1, 3, 256, 256]` or `[256, 256, 3]`. The current adapter interprets output values in `[-1, 1]` or `[0, 1]` and converts to 8-bit RGB.
- The transformation graph must condition output on the target image so pose and expression are driven by the live target. Check this behavior in representative tests before using it; a model that only returns/pastes the source portrait is not a compatible implementation.

If model tensors/names differ, either export a compatible graph or implement `FaceTransformer` in `backend/app/transformation/` and register it with `TransformerManager`. Keep source encoding in `prepare_source()` and per-frame work in `transform()`.

> **Adapter compatibility note:** graph signatures and identity dimensions differ. The adapter validates names and execution but cannot prove model quality, consent, legal status, or semantic pose/expression preservation. Run a consented local validation set before relying on output. No universal ONNX face-swap ABI exists.

## Performance modes and graceful degradation

`POST /camera/settings` accepts `performance_mode`: `auto` (default — degrades by reusing the last composited frame on alternating frames only once the stream has sustained a low rolling FPS), `quality` (always run the model every frame), or `performance` (always skip every other frame's inference). A frame that genuinely fails during inference (e.g. the OS refuses an allocation) is reported as a recoverable error and processing continues with the next frame rather than crashing the stream.

## Installation smoke test

1. Install a model (Option A, in-app) or point the BYO environment variables at approved local files (Option B) and install `backend/requirements.txt`.
2. Start the backend.
3. Query `GET /transformer/status`, then call `POST /transformer/load` with `{"provider":"auto"}`.
4. Upload a source portrait, select the intended face, and confirm `/source-face/status` has `model_ready: true`.
5. Test a live feed at 320–480 px first. Verify output pose/expression follows the target, source identity is not re-encoded per frame, FPS/latency are reasonable, and the blend does not spill outside the face mask.
6. Unload the model or stop the backend to release runtime sessions and identity features.

The application remains useful for camera preview, face detection, source alignment, and API development when no model files are installed; the transformation switch stays disabled in that state.
