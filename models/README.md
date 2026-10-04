# Model files and licensing

FRAME does **not** include, download, or redistribute face-transformation weights. The source code is an adapter and an inference pipeline, not itself a trained identity-swap model. This is deliberate: code licenses do not automatically grant rights to pretrained weights, training data, face identities, or every intended use.

Before enabling a model, independently verify that the exact checkpoint, encoder, training data, dependencies, and use case are allowed by their respective licenses and terms. In particular, confirm whether commercial use, redistribution, biometric processing, and use of the pictured identity are permitted. Prefer a model with a clear permissive grant and documented provenance. The user is responsible for consent and applicable-law compliance.

## Expected local files

The built-in ONNX adapter consumes a two-graph bundle:

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

## Tensor ABI

This adapter intentionally supports a documented ABI rather than guessing at unrelated face-swap graphs:

- Both input faces are aligned, RGB, square `256 × 256` crops.
- Image tensors are `float32`, NCHW, batch one, normalized to `[-1, 1]`.
- The encoder takes one image input and returns a finite `float32` identity tensor, usually `[1, D]`. Its output is computed once after source selection (or once when the model is loaded if the source was already selected).
- The transformer graph must have `float32` inputs named exactly `source_identity` and `target_face`. `source_identity` must accept the encoder output shape; `target_face` accepts `[1, 3, 256, 256]`.
- The transformer returns one RGB face crop, either `[1, 3, 256, 256]` or `[256, 256, 3]`. The current adapter interprets output values in `[-1, 1]` or `[0, 1]` and converts to 8-bit RGB.
- The transformation graph must condition output on the target image so pose and expression are driven by the live target. Check this behavior in representative tests before using it; a model that only returns/pastes the source portrait is not a compatible implementation.

If model tensors/names differ, either export a compatible graph or implement `FaceTransformer` in `backend/app/transformation/` and register it with `TransformerManager`. Keep source encoding in `prepare_source()` and per-frame work in `transform()`.

> **Adapter compatibility note:** graph signatures and identity dimensions differ. The adapter validates names and execution but cannot prove model quality, consent, legal status, or semantic pose/expression preservation. Run a consented local validation set before relying on output. No universal ONNX face-swap ABI exists.

## Installation smoke test

1. Put approved local files at the paths above and install `backend/requirements.txt`.
2. Start the backend with those environment variables.
3. Query `GET /transformer/status`, then call `POST /transformer/load` with `{"provider":"auto"}`.
4. Upload a source portrait, select the intended face, and confirm `/source-face/status` has `model_ready: true`.
5. Test a live feed at 320–480 px first. Verify output pose/expression follows the target, source identity is not re-encoded per frame, FPS/latency are reasonable, and the blend does not spill outside the face mask.
6. Unload the model or stop the backend to release runtime sessions and identity features.

The application remains useful for camera preview, face detection, source alignment, and API development when no model files are installed; the transformation switch stays disabled in that state.
