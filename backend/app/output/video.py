"""Output capability metadata; camera capture stays in the user's browser."""


def output_capabilities() -> dict[str, object]:
    return {
        "browser_canvas_stream": True,
        "webrtc_signaling": False,
        "virtual_camera": False,
        "note": "The processed canvas can be exposed with HTMLCanvasElement.captureStream(); desktop virtual cameras are intentionally optional and platform-specific.",
    }
