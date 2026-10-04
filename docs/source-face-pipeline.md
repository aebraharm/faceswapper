# Source-face pipeline

## Upload and validation

1. The user selects or drops a source portrait. Nothing is read from the filesystem until that selection.
2. The browser sends the selected file bytes directly as a raw image request body (no multipart spool). The backend accepts JPG/JPEG, PNG, and WEBP only, checks the declared MIME against Pillow's decoded format, streams into a bounded in-memory buffer capped at 8 MiB, requires at least 64 pixels per dimension, and limits dimensions to 6000 px and total pixels to 24 million.
3. The image is EXIF-oriented and decoded into RGB memory. No temporary file is created, and the original upload is not logged or returned through an image URL.
4. `FaceAnalyzer` uses MediaPipe Face Mesh when installed; OpenCV's bundled Haar frontal-face detector remains the fallback. A no-face result returns a helpful 422 and does not replace the previous valid source.
5. A single face is chosen automatically. Multiple faces are shown with numbered boxes; the user selects the desired source. No representation is prepared from an arbitrary face in a multi-face image.
6. `FaceAligner` uses five facial landmarks and a similarity transform when available. Its reusable box-based crop/scale/translation fallback works without landmarks.
7. The aligned 256×256 RGB face is kept in memory. A loaded transformer's `prepare_source()` creates the identity representation once. When a model loads after the upload, it consumes the cached aligned crop once.

## Replacement/removal

A valid replacement supersedes and zeroes the previous decoded/cropped arrays. Invalid/no-face uploads do not destroy the previous valid source. `DELETE /source-face` zeros image/crop/array representation data and disables transformation. The server process does not persist uploads, and restarting it naturally discards in-memory session data.

## API response

`POST /source-face/upload` and `/source-face/status` return detected boxes, dimensions, selected index and readiness flags. They do not return portrait bytes, model embeddings, landmarks or a public URL. `POST /source-face/select` requires an integer face index from the current detected set.

## Operational notes

Use a clear, reasonably frontal, well-lit photo with one unobstructed face when possible. Detection is not guaranteed under severe occlusion, profile angles, small faces, or unusual lighting. Do not use a model/source identity without appropriate permission. The model's output and source-face rights remain the operator's responsibility.
