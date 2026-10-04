/**
 * A processed canvas can be handed to a WebRTC RTCPeerConnection or another
 * browser component. Desktop meeting apps need an optional OS virtual-camera
 * bridge; this helper does not silently install or start one.
 */
export function getProcessedVideoStream(canvas: HTMLCanvasElement, frameRate = 30): MediaStream {
  if (typeof canvas.captureStream !== 'function') {
    throw new Error('Canvas video output is not supported in this browser.');
  }
  return canvas.captureStream(frameRate);
}
