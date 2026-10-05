# Model integration

## Adapter lifecycle

Implement `app.transformation.base.FaceTransformer`:

```python
class FaceTransformer(ABC):
    def load_model(self) -> None: ...
    def prepare_source(self, source_face_rgb: np.ndarray) -> Any: ...
    def transform(self, source_representation: Any, target_face_rgb: np.ndarray) -> np.ndarray: ...
    def unload_model(self) -> None: ...
```

- `load_model()` validates configuration, initializes provider/session state and warms the graph.
- `prepare_source()` runs once for a selected/aligned source. Return a compact, owned feature/identity representation, not a reference to a temporary tensor.
- `transform()` runs per selected target crop. The target crop contains current pose and expression. Return an RGB crop in the target's canonical coordinates; do not simply return the source portrait. The LivePortrait adapter preserves the fused `warping_spade` graph's native 512×512 RGB render and the compositor performs its explicit 512→256 `INTER_AREA` conversion immediately before inverse alignment.
- `unload_model()` releases sessions, GPU state and caches.

Register an implementation in `TransformerManager` without altering the source upload or frame API. The current `OnnxIdentityTransformer` is a concrete ONNX Runtime adapter requiring separate source-encoder and target-conditioned transformer graphs. Its names/shapes are documented in `models/README.md`.

## Identity and target are not interchangeable

`SOURCE FACE` means the user-selected identity image. `TARGET FACE` means the current live camera crop. Source features are cached; each frame supplies the target's current pose/expression. The transformer should generate the source identity under target appearance conditions. A 2D portrait crop/mask overlay is not an identity-transform model and is intentionally not used as a fallback.

## Weights and licenses

The repository contains no pretrained weights and provides no automatic downloader. Verify weight, code, training-data and use terms for the exact files you install. A code repository marked permissive does not prove pretrained weights or identity data are similarly licensed. Do not put non-redistributable model files in version control. See `models/README.md`.

## Errors

Missing paths, unavailable providers, unsupported graph names and inference errors surface as explicit model/API errors. Transformation remains off unless a model and selected source representation are ready. CPU execution is the fallback; CUDA is requested only when ONNX Runtime reports it. API and tests run without model files using synthetic/mock frames.
