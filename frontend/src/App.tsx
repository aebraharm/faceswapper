import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import {
  Aperture,
  AlertCircle,
  ArrowDownToLine,
  Camera,
  Check,
  ChevronDown,
  CircleHelp,
  Cpu,
  ImagePlus,
  LoaderCircle,
  LockKeyhole,
  Play,
  ScanFace,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Video,
  X,
  Zap,
} from 'lucide-react';
import { MetricCard } from './components/MetricCard';
import { api } from './services/api';
import { getSourceImageContentType } from './services/sourceImage';
import { getProcessedVideoStream } from './services/output';
import type { DesktopPreferences } from './types/desktop';
import type {
  CameraStatus,
  FaceBox,
  FrameStats,
  ModelCatalogEntry,
  SourceFaceStatus,
  TransformerStatus,
} from './types/api';

const emptySource: SourceFaceStatus = {
  uploaded: false,
  face_count: 0,
  faces: [],
  selected_face_index: null,
  ready: false,
  model_ready: false,
  width: 0,
  height: 0,
  storage: 'memory only',
};

const MAX_SOURCE_BYTES = 8 * 1024 * 1024;

declare global {
  interface Window {
    /** Optional integration hook for browser WebRTC callers in this page. */
    faceTransformOutput?: MediaStream;
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof TypeError && /fetch|network/i.test(error.message)) {
    return 'Could not reach the local AI backend. Check that the local engine is connected, then try again.';
  }
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}

function hasUsableVideoFrame(video: HTMLVideoElement): boolean {
  return (
    video.srcObject !== null
    && !video.paused
    && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
    && video.videoWidth > 0
    && video.videoHeight > 0
  );
}

async function waitForUsableVideoFrame(video: HTMLVideoElement, timeoutMs = 8_000): Promise<void> {
  if (hasUsableVideoFrame(video)) return;
  await new Promise<void>((resolve, reject) => {
    const events = ['loadedmetadata', 'loadeddata', 'canplay', 'playing'];
    const cleanUp = () => {
      window.clearTimeout(timeout);
      events.forEach((event) => video.removeEventListener(event, onVideoStateChange));
    };
    const onVideoStateChange = () => {
      if (!hasUsableVideoFrame(video)) return;
      cleanUp();
      resolve();
    };
    const timeout = window.setTimeout(() => {
      cleanUp();
      reject(new Error('The camera stream started but did not provide a usable video frame within 8 seconds.'));
    }, timeoutMs);
    events.forEach((event) => video.addEventListener(event, onVideoStateChange));
    // Covers state changes that happened between the first check and listener setup.
    onVideoStateChange();
  });
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const processedCanvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCaptureRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const sendAnimationRef = useRef<number | null>(null);
  const outputStreamRef = useRef<MediaStream | null>(null);
  const frameInFlightRef = useRef(false);
  const intentionallyStoppingRef = useRef(false);
  const previewUrlRef = useRef<string | null>(null);
  const pendingPreviewUrlRef = useRef<string | null>(null);
  // This is set only by the backend's opt-in sampled diagnostic stats frame;
  // normal production preview rendering does not write per-frame console logs.
  const pendingFrameDiagnosticsRef = useRef<string | null>(null);
  // The backend advertises this only for opt-in investigations. It lets the
  // renderer emit sparse capture/send checkpoints before a binary frame could
  // block in local inference; normal camera streaming sends none of them.
  const backendDiagnosticsEnabledRef = useRef(false);
  const transformEnabledRef = useRef(false);

  const [backendReady, setBackendReady] = useState(false);
  const [landmarksAvailable, setLandmarksAvailable] = useState(false);
  const [source, setSource] = useState<SourceFaceStatus>(emptySource);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [model, setModel] = useState<TransformerStatus>({
    loaded: false,
    name: 'No model loaded',
    device: 'Checking…',
  });
  const [cameraLive, setCameraLive] = useState(false);
  const [cameraBusy, setCameraBusy] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [sourceUploadError, setSourceUploadError] = useState('');
  const [notice, setNotice] = useState('');
  const [noticeTone, setNoticeTone] = useState<'success' | 'error'>('success');
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [transformEnabled, setTransformEnabled] = useState(false);
  const [provider, setProvider] = useState<DesktopPreferences['provider']>('auto');
  const [transformationRunning, setTransformationRunning] = useState(false);
  const [intensity, setIntensity] = useState(0.85);
  const [resolution, setResolution] = useState<DesktopPreferences['resolution']>(640);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [cameraResolution, setCameraResolution] = useState('—');
  const [targetFaces, setTargetFaces] = useState<FaceBox[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<number | null>(null);
  const [outputEnabled, setOutputEnabled] = useState(false);
  const [catalog, setCatalog] = useState<ModelCatalogEntry[]>([]);
  const [installingId, setInstallingId] = useState<string | null>(null);
  const [performanceMode, setPerformanceMode] = useState<'auto' | 'quality' | 'performance'>('auto');

  const showNotice = (message: string, tone: 'success' | 'error' = 'success') => {
    setNoticeTone(tone);
    setNotice(message);
  };

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    const cameras = allDevices.filter((device) => device.kind === 'videoinput');
    setDevices(cameras);
    setSelectedDevice((current) => current || cameras[0]?.deviceId || '');
  }, []);

  useEffect(() => {
    transformEnabledRef.current = transformEnabled;
  }, [transformEnabled]);

  useEffect(() => {
    let mounted = true;
    if (window.frameDesktop) {
      void window.frameDesktop.loadSettings()
        .then((preferences) => {
          if (!mounted) return;
          setProvider(preferences.provider);
          setIntensity(preferences.intensity);
          setResolution(preferences.resolution);
        })
        .catch(() => undefined)
        .finally(() => {
          if (mounted) setPreferencesLoaded(true);
        });
    } else {
      setPreferencesLoaded(true);
    }
    const removeBackendErrorListener = window.frameDesktop?.onBackendError((message) => {
      setCameraError(`Local AI backend stopped: ${message}`);
      setBackendReady(false);
      intentionallyStoppingRef.current = true;
      if (sendAnimationRef.current !== null) cancelAnimationFrame(sendAnimationRef.current);
      sendAnimationRef.current = null;
      socketRef.current?.close();
      socketRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      outputStreamRef.current?.getTracks().forEach((track) => track.stop());
      outputStreamRef.current = null;
      delete window.faceTransformOutput;
      if (videoRef.current) videoRef.current.srcObject = null;
      setOutputEnabled(false);
      setCameraLive(false);
      setCameraBusy(false);
      setTransformEnabled(false);
      setTransformationRunning(false);
      setTargetFaces([]);
      setSelectedTarget(null);
    });
    Promise.all([api.health(), api.sourceStatus(), api.transformerStatus()])
      .then(([health, sourceStatus, modelStatus]) => {
        if (!mounted) return;
        setBackendReady(true);
        backendDiagnosticsEnabledRef.current = Boolean(health.frame_diagnostics_enabled);
        setLandmarksAvailable(health.landmarks_available);
        setSource(sourceStatus);
        setModel(modelStatus);
      })
      .catch(() => {
        if (mounted) setBackendReady(false);
      });
    void refreshDevices();
    navigator.mediaDevices?.addEventListener?.('devicechange', refreshDevices);
    return () => {
      mounted = false;
      navigator.mediaDevices?.removeEventListener?.('devicechange', refreshDevices);
      removeBackendErrorListener?.();
      intentionallyStoppingRef.current = true;
      if (sendAnimationRef.current !== null) cancelAnimationFrame(sendAnimationRef.current);
      socketRef.current?.close();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      outputStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (window.faceTransformOutput) delete window.faceTransformOutput;
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (pendingPreviewUrlRef.current) URL.revokeObjectURL(pendingPreviewUrlRef.current);
    };
  }, [refreshDevices]);

  // Installable-model catalog is fetched separately from the core health/source/model
  // bootstrap above so a slow or unmocked catalog call never blocks (or is required by)
  // the rest of the UI.
  useEffect(() => {
    let mounted = true;
    let pollTimer: number | null = null;

    const loadCatalog = () => {
      api
        .modelCatalog()
        .then((response) => {
          if (!mounted) return;
          setCatalog(response.models);
          const stillBusy = response.models.some((entry) => entry.downloading);
          pollTimer = window.setTimeout(loadCatalog, stillBusy ? 800 : 15000);
        })
        .catch(() => {
          // The catalog is informational only; the rest of the app keeps working
          // (e.g. a bring-your-own ONNX bundle) if this call fails.
        });
    };
    loadCatalog();

    return () => {
      mounted = false;
      if (pollTimer !== null) window.clearTimeout(pollTimer);
    };
  }, []);

  useEffect(() => {
    if (!preferencesLoaded || !window.frameDesktop) return;
    const timer = window.setTimeout(() => {
      void window.frameDesktop?.saveSettings({ provider, intensity, resolution }).catch(() => undefined);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [preferencesLoaded, provider, intensity, resolution]);

  useEffect(() => {
    if (!cameraLive) return;
    const timer = window.setTimeout(() => {
      void api.cameraSettings({ intensity, processing_resolution: resolution, performance_mode: performanceMode }).catch((error) => {
        showNotice(errorMessage(error), 'error');
      });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [intensity, resolution, performanceMode, cameraLive]);

  const handleFrameMessage = useCallback(async (event: MessageEvent) => {
    if (typeof event.data === 'string') {
      try {
        const stats = JSON.parse(event.data) as FrameStats;
        if (stats.type === 'error') {
          frameInFlightRef.current = false;
          const message = stats.message ?? 'The camera frame could not be processed.';
          if (backendDiagnosticsEnabledRef.current && socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({
              type: 'camera_backend_error',
              transport_id: `backend-error-${Date.now()}`,
              message,
              transformation_enabled: transformEnabledRef.current,
            }));
          }
          setCameraError(message);
          return;
        }
        setFps(stats.fps ?? 0);
        setLatency(stats.latency_ms ?? 0);
        setCameraResolution(stats.camera_resolution ?? '—');
        setTargetFaces(stats.faces ?? []);
        setSelectedTarget(stats.selected_target ?? null);
        setTransformationRunning(Boolean(stats.transformation_active));
        if (stats.model_status) setModel(stats.model_status);
        if (stats.frame_diagnostics) {
          pendingFrameDiagnosticsRef.current = stats.frame_diagnostics_id ?? null;
          // The following binary WebSocket message is the JPEG described by this
          // trace. The paired bitmap log confirms Electron/browser decode and
          // canvas dimensions without adding a visual overlay or production spam.
          console.debug('[FRAME] backend frame diagnostics', stats.frame_diagnostics);
        }
      } catch {
        // Ignore malformed telemetry and keep the last good preview on screen.
      }
      return;
    }
    const traceId = pendingFrameDiagnosticsRef.current;
    try {
      const bitmap = await createImageBitmap(event.data as Blob);
      frameInFlightRef.current = false;
      const canvas = processedCanvasRef.current;
      const canvasBeforeDraw = canvas ? { width: canvas.width, height: canvas.height } : null;
      const context = canvas?.getContext('2d');
      let drawn = false;
      if (canvas && context) {
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
        }
        context.drawImage(bitmap, 0, 0);
        drawn = true;
      }
      if (traceId) {
        const diagnostic = {
          type: 'frame_diagnostics_displayed',
          trace_id: traceId,
          bitmap_width: bitmap.width,
          bitmap_height: bitmap.height,
          canvas_before_draw: canvasBeforeDraw,
          canvas_after_draw: canvas ? { width: canvas.width, height: canvas.height } : null,
          canvas_context_available: Boolean(context),
          drawn,
        };
        console.debug('[FRAME] frontend JPEG decode', diagnostic);
        if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.send(JSON.stringify(diagnostic));
        pendingFrameDiagnosticsRef.current = null;
      }
      bitmap.close();
    } catch (error) {
      frameInFlightRef.current = false;
      if (traceId) {
        const diagnostic = {
          type: 'frame_diagnostics_displayed',
          trace_id: traceId,
          bitmap_decode_failed: true,
          error: error instanceof Error ? error.message : String(error),
        };
        console.debug('[FRAME] frontend JPEG decode failed', diagnostic);
        if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.send(JSON.stringify(diagnostic));
        pendingFrameDiagnosticsRef.current = null;
      }
      setCameraError('The processed video frame could not be displayed.');
    }
  }, []);

  const stopCamera = useCallback(async (showMessage = false) => {
    intentionallyStoppingRef.current = true;
    if (sendAnimationRef.current !== null) cancelAnimationFrame(sendAnimationRef.current);
    sendAnimationRef.current = null;
    frameInFlightRef.current = false;
    socketRef.current?.close();
    socketRef.current = null;
    outputStreamRef.current?.getTracks().forEach((track) => track.stop());
    outputStreamRef.current = null;
    delete window.faceTransformOutput;
    setOutputEnabled(false);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = processedCanvasRef.current;
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    try {
      await api.stopCamera();
    } catch {
      // Local media tracks are always stopped even if the API has gone away.
    }
    setCameraLive(false);
    setCameraBusy(false);
    setTransformEnabled(false);
    setTransformationRunning(false);
    setTargetFaces([]);
    setSelectedTarget(null);
    setFps(0);
    setLatency(0);
    setCameraResolution('—');
    if (showMessage) showNotice('Camera stopped. The local video track has been released.');
  }, []);

  const startCamera = async () => {
    setCameraError('');
    setNotice('');
    if (!backendReady) {
      setCameraError('Start the FastAPI backend first, then refresh this page.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is not available. Use a modern browser over HTTPS or localhost.');
      return;
    }
    setCameraBusy(true);
    intentionallyStoppingRef.current = false;
    frameInFlightRef.current = false;
    let localStream: MediaStream | null = null;
    let connectingSocket: WebSocket | null = null;
    try {
      const desktopCameraAuthorized = await window.frameDesktop?.authorizeCamera();
      if (window.frameDesktop && desktopCameraAuthorized === false) {
        throw new Error('The desktop camera permission could not be authorized. Try again from the Start camera button.');
      }
      const videoConstraints: MediaTrackConstraints = selectedDevice
        ? { deviceId: { exact: selectedDevice }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' };
      localStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
      const initialTrack = localStream.getVideoTracks()[0];
      if (!initialTrack || initialTrack.readyState !== 'live') {
        throw new Error('The camera permission request succeeded, but no live video track was returned.');
      }
      streamRef.current = localStream;
      const previewVideo = videoRef.current;
      if (!previewVideo) {
        throw new Error('The camera preview element is not mounted.');
      }
      previewVideo.srcObject = localStream;
      await previewVideo.play();
      // Electron can resolve play() before dimensions/current data are available.
      // Do not begin the canvas loop until the attached stream has a drawable frame.
      await waitForUsableVideoFrame(previewVideo);
      await refreshDevices();
      await api.cameraSettings({ transform_enabled: false, intensity, processing_resolution: resolution, performance_mode: performanceMode });
      await api.startCamera(selectedDevice || null);

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      let runtime = null;
      try {
        runtime = (await window.frameDesktop?.getRuntimeConfig()) ?? null;
      } catch {
        // Browser/Vite mode uses the same-origin WebSocket proxy below.
      }
      const socketUrl = runtime
        ? `${runtime.websocketUrl}/ws/stream`
        : `${protocol}//${window.location.host}/ws/stream`;
      const protocols = runtime?.sessionToken ? ["frame-v1", runtime.sessionToken] : undefined;
      const socket = protocols ? new WebSocket(socketUrl, protocols) : new WebSocket(socketUrl);
      connectingSocket = socket;
      socket.binaryType = 'blob';
      socket.onmessage = (event) => void handleFrameMessage(event);
      socket.onerror = () => {
        if (backendDiagnosticsEnabledRef.current) console.debug('[FRAME] WebSocket error before/while camera streaming');
        setCameraError('The local processing stream could not connect to the backend.');
      };
      socket.onclose = (event) => {
        if (backendDiagnosticsEnabledRef.current) {
          console.debug('[FRAME] WebSocket closed', { code: event.code, reason: event.reason, wasClean: event.wasClean });
        }
        if (!intentionallyStoppingRef.current && streamRef.current) {
          setCameraError('The processing connection closed. Stop and restart the camera to reconnect.');
          setCameraLive(false);
          setTransformEnabled(false);
          setTransformationRunning(false);
          outputStreamRef.current?.getTracks().forEach((track) => track.stop());
          outputStreamRef.current = null;
          delete window.faceTransformOutput;
          setOutputEnabled(false);
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
          void api.stopCamera().catch(() => undefined);
        }
      };
      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(() => reject(new Error('The processing backend did not respond.')), 8000);
        socket.addEventListener('open', () => {
          window.clearTimeout(timer);
          resolve();
        }, { once: true });
        socket.addEventListener('error', () => {
          window.clearTimeout(timer);
          reject(new Error('Could not connect to the local frame processor.'));
        }, { once: true });
      });
      socketRef.current = socket;
      connectingSocket = null;
      setCameraLive(true);
      setCameraBusy(false);
      showNotice('Camera is live. Frames are sent to the local processor; no audio is captured.');

      const video = videoRef.current;
      const sourceCanvas = hiddenCaptureRef.current;
      const sourceContext = sourceCanvas?.getContext('2d', { alpha: false });
      const captureState = (phase: string) => {
        const track = localStream?.getVideoTracks()[0] ?? null;
        return {
          phase,
          stream_present: Boolean(localStream),
          video_element_present: Boolean(video),
          video_src_object_present: Boolean(video?.srcObject),
          video_ready_state: video?.readyState ?? null,
          video_width: video?.videoWidth ?? null,
          video_height: video?.videoHeight ?? null,
          video_paused: video?.paused ?? null,
          capture_canvas_present: Boolean(sourceCanvas),
          capture_context_present: Boolean(sourceContext),
          video_track_present: Boolean(track),
          video_track_ready_state: track?.readyState ?? null,
          video_track_enabled: track?.enabled ?? null,
          video_track_muted: track?.muted ?? null,
          transformation_enabled: transformEnabledRef.current,
        };
      };
      let lastSent = 0;
      let nextTransportDiagnosticAt = 0;
      let transportSequence = 0;
      let jpegEncodingInFlight = false;
      const emitTransportDiagnostic = (type: string, transportId: string, details: Record<string, unknown> = {}) => {
        if (!backendDiagnosticsEnabledRef.current || socket.readyState !== WebSocket.OPEN) return;
        try {
          socket.send(JSON.stringify({ type, transport_id: transportId, ...details }));
        } catch (error) {
          // A failed WebSocket cannot report its own diagnostic record. Keep an
          // opt-in renderer-console fact instead of changing camera behavior.
          console.debug('[FRAME] diagnostic WebSocket send failed', { type, error });
        }
      };
      const streamTransportId = `stream-${Date.now()}-${transportSequence++}`;
      emitTransportDiagnostic('camera_stream_started', streamTransportId, {
        websocket_open: true,
        ...captureState('stream_started'),
      });
      const captureStateId = `capture-state-${Date.now()}-${transportSequence++}`;
      emitTransportDiagnostic('camera_capture_state', captureStateId, captureState('capture_source_checked'));
      const captureReady = Boolean(
        video
        && sourceCanvas
        && sourceContext
        && hasUsableVideoFrame(video)
        && localStream?.getVideoTracks()[0]?.readyState === 'live'
      );
      if (captureReady) {
        emitTransportDiagnostic('camera_capture_ready', captureStateId, captureState('capture_ready'));
      }
      const send = (timestamp: number) => {
        if (socket.readyState !== WebSocket.OPEN) return;
        if (!video || !sourceCanvas || !sourceContext) {
          if (backendDiagnosticsEnabledRef.current && timestamp >= nextTransportDiagnosticAt) {
            nextTransportDiagnosticAt = timestamp + 1_000;
            emitTransportDiagnostic(
              'camera_capture_unavailable',
              `unavailable-${Math.round(timestamp)}-${transportSequence++}`,
              captureState('capture_unavailable'),
            );
          }
          return;
        }
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth <= 0) {
          if (backendDiagnosticsEnabledRef.current && timestamp >= nextTransportDiagnosticAt) {
            nextTransportDiagnosticAt = timestamp + 1_000;
            emitTransportDiagnostic('camera_frame_loop_waiting', `video-${Math.round(timestamp)}-${transportSequence++}`, {
              blocked_by: 'video_not_ready',
              video_ready_state: video.readyState,
              video_width: video.videoWidth,
              video_height: video.videoHeight,
            });
          }
        } else if (timestamp - lastSent >= 33) {
          lastSent = timestamp;
          const sourceWidth = video.videoWidth;
          const sourceHeight = video.videoHeight;
          const scale = Math.min(1, 960 / Math.max(sourceWidth, sourceHeight));
          const frameWidth = Math.max(2, Math.round(sourceWidth * scale));
          const frameHeight = Math.max(2, Math.round(sourceHeight * scale));
          if (sourceCanvas.width !== frameWidth || sourceCanvas.height !== frameHeight) {
            sourceCanvas.width = frameWidth;
            sourceCanvas.height = frameHeight;
          }
          if (
            backendDiagnosticsEnabledRef.current
            && (frameInFlightRef.current || socket.bufferedAmount >= 500_000)
            && timestamp >= nextTransportDiagnosticAt
          ) {
            const transportId = `wait-${Math.round(timestamp)}-${transportSequence++}`;
            nextTransportDiagnosticAt = timestamp + 1_000;
            emitTransportDiagnostic('camera_frame_loop_waiting', transportId, {
              blocked_by: jpegEncodingInFlight
                ? 'jpeg_encoding'
                : (frameInFlightRef.current ? 'awaiting_backend_preview' : 'websocket_backpressure'),
              buffered_amount: socket.bufferedAmount,
              transformation_enabled: transformEnabledRef.current,
            });
          }
          if (!frameInFlightRef.current && socket.bufferedAmount < 500_000) {
            const transportId = backendDiagnosticsEnabledRef.current && timestamp >= nextTransportDiagnosticAt
              ? `frame-${Math.round(timestamp)}-${transportSequence++}`
              : null;
            if (transportId) {
              nextTransportDiagnosticAt = timestamp + 1_000;
              emitTransportDiagnostic('camera_frame_capture', transportId, {
                video_width: sourceWidth,
                video_height: sourceHeight,
                capture_width: frameWidth,
                capture_height: frameHeight,
                transformation_enabled: transformEnabledRef.current,
              });
            }
            frameInFlightRef.current = true;
            try {
              sourceContext.drawImage(video, 0, 0, frameWidth, frameHeight);
            } catch (error) {
              if (transportId) {
                emitTransportDiagnostic('camera_frame_capture_failed', transportId, {
                  error: error instanceof Error ? error.message : String(error),
                });
              }
              return;
            }
            jpegEncodingInFlight = true;
            try {
              sourceCanvas.toBlob((blob) => {
                jpegEncodingInFlight = false;
                if (blob && socket.readyState === WebSocket.OPEN && socket.bufferedAmount < 500_000) {
                  if (transportId) {
                    emitTransportDiagnostic('camera_frame_encoded', transportId, {
                      jpeg_bytes: blob.size,
                      capture_width: frameWidth,
                      capture_height: frameHeight,
                    });
                  }
                  try {
                    socket.send(blob);
                  } catch (error) {
                    if (transportId) {
                      emitTransportDiagnostic('camera_frame_send_failed', transportId, {
                        error: error instanceof Error ? error.message : String(error),
                      });
                    }
                  }
                } else {
                  if (transportId) {
                    emitTransportDiagnostic(blob ? 'camera_frame_send_skipped' : 'camera_frame_encode_failed', transportId, {
                      reason: blob ? (socket.readyState !== WebSocket.OPEN ? 'websocket_not_open' : 'websocket_backpressure') : 'canvas_to_blob_returned_null',
                    });
                  }
                  frameInFlightRef.current = false;
                }
              }, 'image/jpeg', 0.78);
            } catch (error) {
              jpegEncodingInFlight = false;
              if (transportId) {
                emitTransportDiagnostic('camera_frame_encode_failed', transportId, {
                  error: error instanceof Error ? error.message : String(error),
                });
              }
              return;
            }
          }
        }
        sendAnimationRef.current = requestAnimationFrame(send);
      };
      sendAnimationRef.current = requestAnimationFrame(send);
    } catch (error) {
      intentionallyStoppingRef.current = true;
      connectingSocket?.close();
      localStream?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      socketRef.current?.close();
      socketRef.current = null;
      await api.stopCamera().catch(() => undefined);
      setCameraLive(false);
      setCameraBusy(false);
      setCameraError(errorMessage(error));
    }
  };

  const handleImageFile = async (file?: File) => {
    if (!file || uploadBusy) return;
    setSourceUploadError('');
    setNotice('');

    const contentType = getSourceImageContentType(file);
    if (!contentType) {
      setSourceUploadError('Choose a JPG, PNG, or WEBP image.');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }
    if (file.size > MAX_SOURCE_BYTES) {
      setSourceUploadError('This image exceeds the 8 MB upload limit. Choose a smaller file.');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    // Preview the user's local file immediately. Do not make rendering depend on
    // the backend response: a no-face result or runtime/network failure must not
    // make a valid first selection disappear.
    let previewUrl: string;
    try {
      previewUrl = URL.createObjectURL(file);
    } catch {
      setSourceUploadError('The selected photo could not be opened for preview. Choose another JPG, PNG, or WEBP image.');
      return;
    }
    const previousPreviewUrl = previewUrlRef.current;
    const previousSource = source;
    pendingPreviewUrlRef.current = previewUrl;
    setSourcePreview(previewUrl);
    setSource(emptySource);

    setUploadBusy(true);
    try {
      const result = await api.uploadSource(file);
      if (previousPreviewUrl) URL.revokeObjectURL(previousPreviewUrl);
      previewUrlRef.current = previewUrl;
      pendingPreviewUrlRef.current = null;
      setSource(result);
      setSourceUploadError('');
      showNotice(result.face_count > 1
        ? `Found ${result.face_count} faces. Choose the source identity below.`
        : 'Source face detected and aligned in memory.');
    } catch (error) {
      const message = errorMessage(error);
      pendingPreviewUrlRef.current = null;
      if (previousSource.uploaded) {
        // The backend keeps the previous valid source if replacement fails. Roll
        // the preview/status back too, so face boxes and the active identity can
        // never refer to a different photo than the one shown.
        URL.revokeObjectURL(previewUrl);
        previewUrlRef.current = previousPreviewUrl;
        setSourcePreview(previousPreviewUrl);
        setSource(previousSource);
        setSourceUploadError(`${message} The previous source photo remains active.`);
      } else {
        if (previousPreviewUrl) URL.revokeObjectURL(previousPreviewUrl);
        previewUrlRef.current = previewUrl;
        setSource(emptySource);
        setSourceUploadError(message);
      }
    } finally {
      setUploadBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => void handleImageFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    void handleImageFile(event.dataTransfer.files[0]);
  };

  const selectSourceFace = async (index: number) => {
    try {
      const selected = await api.selectSource(index);
      setSource(selected);
      showNotice(`Source face ${index + 1} selected and aligned.`);
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const removeSource = async () => {
    try {
      const result = await api.removeSource();
      setSource(result);
      setTransformEnabled(false);
      setTransformationRunning(false);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (pendingPreviewUrlRef.current) URL.revokeObjectURL(pendingPreviewUrlRef.current);
      previewUrlRef.current = null;
      pendingPreviewUrlRef.current = null;
      setSourcePreview(null);
      setSourceUploadError('');
      showNotice('Source photo removed from the application session.');
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const loadModel = async () => {
    setNotice('');
    try {
      const loaded = await api.loadTransformer(provider);
      setModel(loaded);
      setSource(await api.sourceStatus());
      showNotice(`Model loaded on ${loaded.device}. Source identity features are now prepared once.`);
    } catch (error) {
      showNotice(errorMessage(error), 'error');
      setModel(await api.transformerStatus().catch(() => model));
    }
  };

  const installCatalogModel = async (modelId: string) => {
    setInstallingId(modelId);
    try {
      const status = await api.installModel(modelId);
      setCatalog((current) => current.map((entry) => (entry.id === modelId ? status : entry)));
      showNotice(status.error ? errorMessage(new Error(status.error)) : `Downloading ${status.display_name}…`, status.error ? 'error' : 'success');
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    } finally {
      setInstallingId(null);
    }
  };

  const removeCatalogModel = async (modelId: string) => {
    try {
      const status = await api.removeModel(modelId);
      setCatalog((current) => current.map((entry) => (entry.id === modelId ? status : entry)));
      showNotice(`${status.display_name} removed from disk.`);
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const changePerformanceMode = async (mode: 'auto' | 'quality' | 'performance') => {
    setPerformanceMode(mode);
    if (!cameraLive) return;
    try {
      await api.cameraSettings({ performance_mode: mode });
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const unloadModel = async () => {
    try {
      const unloaded = await api.unloadTransformer();
      setModel(unloaded);
      setSource(await api.sourceStatus());
      setTransformEnabled(false);
      setTransformationRunning(false);
      showNotice('Model unloaded and cached identity features cleared.');
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const toggleTransformation = async () => {
    const next = !transformEnabled;
    try {
      await api.cameraSettings({ transform_enabled: next, intensity, processing_resolution: resolution });
      setTransformEnabled(next);
      showNotice(next ? 'AI face transformation enabled.' : 'Transformation disabled. Original camera frames are shown.');
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const selectTargetFace = async (index: number | null) => {
    try {
      const status: CameraStatus = await api.selectTarget(index);
      setSelectedTarget(status.selected_target_index);
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const toggleOutput = () => {
    const canvas = processedCanvasRef.current;
    if (!canvas) return;
    if (outputStreamRef.current) {
      outputStreamRef.current.getTracks().forEach((track) => track.stop());
      outputStreamRef.current = null;
      delete window.faceTransformOutput;
      setOutputEnabled(false);
      return;
    }
    try {
      const stream = getProcessedVideoStream(canvas, 30);
      outputStreamRef.current = stream;
      window.faceTransformOutput = stream;
      setOutputEnabled(true);
    } catch (error) {
      showNotice(errorMessage(error), 'error');
    }
  };

  const canTransform = Boolean(cameraLive && source.model_ready && model.loaded);
  const aiConfigured = transformEnabled && canTransform;
  const targetNeedsSelection = targetFaces.length > 1 && selectedTarget === null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="FRAME home">
          <span className="brand-mark"><Aperture size={21} strokeWidth={2.2} /></span>
          <span className="brand-name">FRAME<span className="brand-dot">.</span></span>
          <span className="brand-divider" />
          <span className="brand-product">REAL-TIME FACE AI</span>
        </a>
        <div className="topbar-right">
          <span className={`backend-chip ${backendReady ? 'is-connected' : 'is-disconnected'}`}>
            <span className="tiny-dot" />
            {backendReady ? 'LOCAL ENGINE CONNECTED' : 'ENGINE OFFLINE'}
          </span>
          <a className="topbar-help" href="#privacy"><CircleHelp size={16} /><span>Privacy & help</span></a>
        </div>
      </header>

      <main id="top" className="page-content">
        <section className="intro-row">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> YOUR CAMERA, REIMAGINED</div>
            <h1>Face transformation,<br /><span>in real time.</span></h1>
            <p className="intro-copy">Choose a source identity, start your camera, and preview a live, pose-aware transformation — all processed locally.</p>
          </div>
          <div className="intro-side-note">
            <div className="privacy-seal"><LockKeyhole size={16} /></div>
            <div><strong>Private by design</strong><span>Session-only processing · no disk storage</span></div>
          </div>
        </section>

        <div className="workspace-grid">
          <section className="panel source-panel" aria-labelledby="source-heading">
            <div className="panel-heading">
              <div className="step-label"><span>01</span><span className="step-rule" /> SOURCE IDENTITY</div>
              {source.ready ? <span className="mini-status ready"><Check size={13} /> FACE READY</span> : <span className="mini-status">OPTIONAL</span>}
            </div>
            <div className="section-title-row">
              <div><h2 id="source-heading">Source face</h2><p>The identity you choose to apply</p></div>
              {sourcePreview && <button className="icon-button" onClick={() => void removeSource()} disabled={uploadBusy} aria-label="Remove source face" title="Remove source face"><Trash2 size={16} /></button>}
            </div>

            <div
              className={`source-dropzone${isDragging ? ' is-dragging' : ''}${sourcePreview ? ' has-preview' : ''}`}
              onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              style={sourcePreview && source.width && source.height ? { aspectRatio: `${source.width} / ${source.height}` } : undefined}
              aria-busy={uploadBusy}
            >
              {sourcePreview ? (
                <>
                  <img src={sourcePreview} alt="Selected source photo" className="source-image" />
                  {uploadBusy && <div className="source-processing-indicator"><LoaderCircle className="spin" size={13} /> CHECKING PHOTO</div>}
                  {source.uploaded && <div className="source-scanline" />}
                  {source.faces.map((face, index) => (
                    <button
                      key={index}
                      type="button"
                      className={`face-outline${source.selected_face_index === index ? ' selected' : ''}`}
                      style={{ left: `${(face.x / source.width) * 100}%`, top: `${(face.y / source.height) * 100}%`, width: `${(face.width / source.width) * 100}%`, height: `${(face.height / source.height) * 100}%` }}
                      onClick={() => void selectSourceFace(index)}
                      aria-label={`Select source face ${index + 1}`}
                      title={`Select face ${index + 1}`}
                    ><span>{index + 1}</span></button>
                  ))}
                  {source.uploaded && <div className="image-caption"><ScanFace size={14} /> {source.face_count} {source.face_count === 1 ? 'FACE' : 'FACES'} DETECTED</div>}
                </>
              ) : (
                <div className="drop-placeholder">
                  <div className="upload-glyph"><ImagePlus size={23} strokeWidth={1.6} /></div>
                  <strong>{uploadBusy ? 'Analyzing source…' : 'Drop a portrait here'}</strong>
                  <span>JPG, PNG or WEBP · up to 8 MB</span>
                  {uploadBusy && <LoaderCircle className="spin upload-spinner" size={18} />}
                </div>
              )}
            </div>
            <input ref={fileInputRef} type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" hidden onChange={onFileChange} />
            <div className="source-actions">
              <button className="button button--outline source-upload-button" onClick={() => fileInputRef.current?.click()} disabled={!backendReady || uploadBusy}>
                {sourcePreview ? <Upload size={16} /> : <ImagePlus size={16} />}
                {sourcePreview ? 'Replace image' : 'Choose image'}
              </button>
              <span className="memory-note"><LockKeyhole size={12} /> NOT STORED</span>
            </div>
            {source.face_count > 1 && (
              <div className="face-selection" aria-live="polite">
                <div className="face-selection-title">SELECT SOURCE FACE <span>({source.face_count} found)</span></div>
                <div className="face-pills">
                  {source.faces.map((_, index) => (
                    <button key={index} className={`face-pill${source.selected_face_index === index ? ' selected' : ''}`} onClick={() => void selectSourceFace(index)}>
                      <ScanFace size={14} /> Face {index + 1} {source.selected_face_index === index && <Check size={13} />}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div
              className={`source-detection-note${source.ready ? ' note-ready' : ''}${sourceUploadError ? ' note-error' : ''}`}
              role={sourceUploadError ? 'alert' : 'status'}
              aria-live={sourceUploadError ? 'assertive' : 'polite'}
            >
              <span className="note-icon">{sourceUploadError ? <AlertCircle size={14} /> : source.ready ? <Check size={14} /> : uploadBusy ? <LoaderCircle className="spin" size={14} /> : <ScanFace size={14} />}</span>
              <span>{sourceUploadError || (uploadBusy
                ? 'Checking the selected photo for a face…'
                : source.ready
                  ? 'Face detected and aligned. Identity representation will be cached when a model is loaded.'
                  : source.face_count > 1
                    ? 'Choose one detected source face above before continuing.'
                    : 'Upload a clear portrait to detect and align its face.')}</span>
            </div>
          </section>

          <section className="panel camera-panel" aria-labelledby="camera-heading">
            <div className="panel-heading">
              <div className="step-label"><span>02</span><span className="step-rule" /> LIVE CAMERA</div>
              <span className={`mini-status ${cameraLive ? 'ready' : ''}`}><span className={`mini-dot${cameraLive ? ' live' : ''}`} />{cameraLive ? 'CAMERA LIVE' : 'CAMERA OFF'}</span>
            </div>
            <div className="camera-topline">
              <div><h2 id="camera-heading">Your live studio</h2><p>Camera capture starts only when you press Start</p></div>
              <div className="camera-picker">
                <Camera size={15} />
                <select value={selectedDevice} onChange={(event) => setSelectedDevice(event.target.value)} disabled={cameraLive} aria-label="Select camera">
                  {devices.length === 0 ? <option value="">Default camera</option> : devices.map((device, index) => (
                    <option key={device.deviceId || index} value={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>
                  ))}
                </select>
                <ChevronDown size={13} className="select-chevron" />
              </div>
            </div>

            <div className="video-grid">
              <div className="video-card">
                <div className="video-card-head"><span><span className="video-index">A</span> ORIGINAL</span><span className="video-live-indicator">{cameraLive ? 'LIVE' : 'STANDBY'}</span></div>
                <div className={`video-stage${cameraLive ? ' is-live' : ''}`}>
                  <video ref={videoRef} className="camera-video mirrored" muted playsInline autoPlay />
                  {/* Permanently mounted offscreen source for JPEG encoding; it never displays pixels. */}
                  <canvas ref={hiddenCaptureRef} className="hidden-capture-canvas" aria-hidden="true" />
                  {!cameraLive && <div className="video-placeholder"><div className="placeholder-icon"><Video size={21} /></div><strong>Camera preview</strong><span>Start your camera to see the original feed</span></div>}
                  <span className="stage-corner stage-corner--tl" /><span className="stage-corner stage-corner--br" />
                </div>
              </div>
              <div className="video-card">
                <div className="video-card-head"><span><span className="video-index video-index--green">B</span> PROCESSED OUTPUT</span>
                  {transformationRunning
                    ? <span className="active-indicator"><span className="tiny-dot" /> AI TRANSFORMATION ACTIVE</span>
                    : <span className="video-live-indicator">{cameraLive ? (aiConfigured ? 'AI READY · WAITING' : 'ANALYSIS ON') : 'STANDBY'}</span>}
                </div>
                <div className={`video-stage processed-stage${cameraLive ? ' is-live' : ''}`}>
                  <canvas ref={processedCanvasRef} className="processed-canvas mirrored" />
                  {!cameraLive && <div className="video-placeholder"><div className="placeholder-icon placeholder-icon--green"><Sparkles size={21} /></div><strong>Processed preview</strong><span>Live detection and model output appear here</span></div>}
                  {cameraLive && transformationRunning && <div className="active-banner"><span className="active-banner-dot" /> AI FACE TRANSFORMATION ACTIVE</div>}
                  {cameraLive && targetNeedsSelection && <div className="selection-overlay">Select a target below</div>}
                  <span className="stage-corner stage-corner--tl" /><span className="stage-corner stage-corner--br" />
                </div>
              </div>
            </div>

            {cameraLive && targetFaces.length > 1 && (
              <div className="target-selector">
                <div><strong>Choose target face</strong><span>Only the selected face is eligible for transformation.</span></div>
                <div className="target-buttons">
                  {targetFaces.map((_, index) => (
                    <button key={index} className={`target-button${selectedTarget === index ? ' active' : ''}`} onClick={() => void selectTargetFace(index)}>
                      <ScanFace size={14} /> Target {index + 1}
                    </button>
                  ))}
                  <button className={`target-button clear-target${selectedTarget === null ? ' active' : ''}`} onClick={() => void selectTargetFace(null)}><X size={13} /> None</button>
                </div>
              </div>
            )}
            {cameraLive && aiConfigured && !transformationRunning && (
              <div className="waiting-target"><AlertCircle size={14} />{targetNeedsSelection ? 'Select one target face to continue.' : 'Waiting for a clearly detected target face.'}</div>
            )}

            <div className="camera-actions">
              {!cameraLive ? (
                <button className="button button--primary start-button" onClick={() => void startCamera()} disabled={!backendReady || cameraBusy}>
                  {cameraBusy ? <LoaderCircle className="spin" size={17} /> : <Play size={16} fill="currentColor" />}
                  {cameraBusy ? 'Connecting camera…' : 'Start camera'}
                </button>
              ) : (
                <button className="button button--danger start-button" onClick={() => void stopCamera(true)}><Square size={15} fill="currentColor" /> Stop camera</button>
              )}
              <span className="camera-privacy"><ShieldCheck size={14} /> Camera access is browser-controlled · no audio</span>
            </div>
            {cameraError && <div className="inline-error"><AlertCircle size={15} />{cameraError}<button onClick={() => setCameraError('')} aria-label="Dismiss error"><X size={14} /></button></div>}
          </section>
        </div>

        <section className="control-grid">
          <div className="panel control-panel">
            <div className="panel-heading"><div className="step-label"><span>03</span><span className="step-rule" /> TRANSFORM CONTROLS</div><SlidersHorizontal size={17} className="quiet-icon" /></div>
            <div className="control-main-row">
              <div><h2>Transformation</h2><p>Model output follows the target pose and expression</p></div>
              <button
                type="button"
                role="switch"
                aria-checked={transformEnabled}
                aria-label="Enable AI face transformation"
                className={`switch${transformEnabled ? ' is-on' : ''}`}
                onClick={() => void toggleTransformation()}
                disabled={!canTransform}
              ><span /></button>
            </div>
            {!canTransform && <div className="control-hint"><AlertCircle size={14} /> Start the camera, load a compatible model, and select a source face to enable.</div>}
            <div className="range-control">
              <div className="range-label"><span>Transformation intensity</span><strong>{Math.round(intensity * 100)}<small>%</small></strong></div>
              <input type="range" min="0" max="100" value={Math.round(intensity * 100)} onChange={(event) => setIntensity(Number(event.target.value) / 100)} style={{ '--range-progress': `${intensity * 100}%` } as React.CSSProperties} />
              <div className="range-ends"><span>Subtle</span><span>Full blend</span></div>
            </div>
            <div className="resolution-control">
              <label htmlFor="resolution-select">Processing resolution</label>
              <div className="resolution-select-wrap"><select id="resolution-select" value={resolution} onChange={(event) => setResolution(Number(event.target.value) as DesktopPreferences['resolution'])}>
                <option value={320}>320 px · fastest</option><option value={480}>480 px · balanced</option><option value={640}>640 px · high quality</option><option value={720}>720 px · max detail</option>
              </select><ChevronDown size={14} /></div>
            </div>
          </div>

          <div className="panel model-panel">
            <div className="panel-heading"><div className="step-label"><span>04</span><span className="step-rule" /> MODEL STATUS</div><span className={`model-pill${model.loaded ? ' model-pill--ready' : ''}`}><span className="tiny-dot" />{model.loaded ? 'READY' : 'NEEDS SETUP'}</span></div>
            <div className="model-content">
              <div className="model-icon"><Cpu size={20} /></div>
              <div className="model-copy"><h2>{model.loaded ? 'Identity model loaded' : 'Bring a licensed model'}</h2><p>{model.loaded ? model.name : 'The processing pipeline is ready for a compatible ONNX model bundle.'}</p></div>
            </div>
            <div className="model-detail-row"><span><span className="detail-key">DEVICE</span><strong>{model.device}</strong></span><span><span className="detail-key">LANDMARKS</span><strong>{landmarksAvailable ? 'MediaPipe' : 'OpenCV fallback'}</strong></span></div>
            <div className="provider-row"><label htmlFor="provider-select">INFERENCE PROVIDER</label><div className="provider-select-wrap"><select id="provider-select" value={provider} onChange={(event) => setProvider(event.target.value as DesktopPreferences['provider'])} disabled={model.loaded}><option value="auto">Auto · recommended</option><option value="CPUExecutionProvider">CPU</option><option value="CUDAExecutionProvider">CUDA GPU</option></select><ChevronDown size={12} /></div></div>
            <button className={`button ${model.loaded ? 'button--outline' : 'button--dark'} model-action`} onClick={() => void (model.loaded ? unloadModel() : loadModel())} disabled={!backendReady}>
              {model.loaded ? <><ArrowDownToLine size={15} className="rotate-180" /> Unload model</> : <><Zap size={15} /> Load configured model</>}
            </button>
            {!model.loaded && <a className="model-doc-link" href="/models/README.md" onClick={(event) => { event.preventDefault(); showNotice('See models/README.md in the repository for the model ABI, installation steps and licensing checklist.'); }}><CircleHelp size={13} /> Model setup & licensing <span>↗</span></a>}

            {catalog.length > 0 && (
              <div className="model-catalog">
                {catalog.map((entry) => {
                  const mb = (bytes: number) => `${Math.max(1, Math.round(bytes / 1024 / 1024))} MB`;
                  const busy = entry.downloading || installingId === entry.id;
                  return (
                    <div key={entry.id} className="model-catalog-entry">
                      <div className="model-catalog-row">
                        <div>
                          <strong>{entry.display_name}</strong>
                          <p className="model-catalog-summary">{entry.summary}</p>
                          <span className="model-catalog-meta">
                            {entry.license_name} license · {mb(entry.approx_total_bytes)}
                          </span>
                        </div>
                        {entry.installed ? (
                          <button
                            type="button"
                            className="button button--outline model-catalog-action"
                            onClick={() => void removeCatalogModel(entry.id)}
                            disabled={model.loaded && model.model_id === entry.id}
                            title={model.loaded && model.model_id === entry.id ? 'Unload the model before removing it.' : undefined}
                          >
                            <Trash2 size={13} /> Remove
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="button button--dark model-catalog-action"
                            onClick={() => void installCatalogModel(entry.id)}
                            disabled={busy}
                          >
                            {busy ? <LoaderCircle size={13} className="spin" /> : <ArrowDownToLine size={13} />}
                            {entry.downloading ? `${Math.round(entry.progress * 100)}%` : 'Install'}
                          </button>
                        )}
                      </div>
                      {entry.downloading && (
                        <div className="model-catalog-progress" role="progressbar" aria-valuenow={Math.round(entry.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                          <div className="model-catalog-progress-fill" style={{ width: `${Math.round(entry.progress * 100)}%` }} />
                        </div>
                      )}
                      {entry.error && <p className="model-catalog-error">{entry.error}</p>}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="panel output-panel">
            <div className="panel-heading"><div className="step-label"><span>05</span><span className="step-rule" /> VIDEO-CALL OUTPUT</div><span className={`output-dot${outputEnabled ? ' output-dot--on' : ''}`} /></div>
            <div className="output-content"><div className="output-icon"><Video size={18} /></div><div><h2>Browser video stream</h2><p>Capture the processed canvas for a browser WebRTC integration.</p></div></div>
            <button className={`button ${outputEnabled ? 'button--outline' : 'button--outline'} output-button`} onClick={toggleOutput} disabled={!cameraLive}>
              {outputEnabled ? <><Square size={14} fill="currentColor" /> Stop stream</> : <><Video size={15} /> {cameraLive ? 'Enable stream output' : 'Start camera first'}</>}
            </button>
            <div className="output-footnote">Desktop meeting apps need a separate OS virtual-camera bridge.</div>
          </div>
        </section>

        <section className="performance-section" aria-label="Live performance">
          <div className="performance-header">
            <div><span className="eyebrow-line" /><span className="step-label-text">LIVE PERFORMANCE</span></div>
            <span className="performance-subtitle">Updates while camera is streaming</span>
            <div className="performance-mode-control">
              <label htmlFor="performance-mode-select">Mode</label>
              <div className="provider-select-wrap">
                <select
                  id="performance-mode-select"
                  value={performanceMode}
                  onChange={(event) => void changePerformanceMode(event.target.value as 'auto' | 'quality' | 'performance')}
                >
                  <option value="auto">Auto · degrade if slow</option>
                  <option value="quality">Quality · every frame</option>
                  <option value="performance">Performance · skip frames</option>
                </select>
                <ChevronDown size={12} />
              </div>
            </div>
          </div>
          <div className="metrics-grid">
            <MetricCard label="PROCESSING FPS" value={cameraLive ? fps.toFixed(1) : '—'} note={cameraLive ? 'End-to-end stream' : 'Camera is off'} icon={Zap} accent />
            <MetricCard label="FRAME LATENCY" value={cameraLive ? `${latency.toFixed(0)} ms` : '—'} note="Detection + processing" icon={Aperture} />
            <MetricCard label="CAMERA RESOLUTION" value={cameraLive ? cameraResolution : '—'} note={`Processing max ${resolution} px`} icon={Camera} />
            <MetricCard label="COMPUTE DEVICE" value={model.device === 'not loaded' ? 'CPU' : model.device.replace(' (ONNX Runtime not installed)', '')} note={model.loaded ? 'Model inference' : 'Face detection pipeline'} icon={Cpu} />
          </div>
        </section>

        {notice && <div className={`notice-bar${noticeTone === 'error' ? ' is-error' : ''}`} role={noticeTone === 'error' ? 'alert' : 'status'}><span>{noticeTone === 'error' ? <AlertCircle size={14} /> : <Check size={14} />}</span><p>{notice}</p><button onClick={() => setNotice('')} aria-label="Dismiss message"><X size={15} /></button></div>}

        <footer id="privacy" className="footer">
          <div className="footer-brand"><span className="footer-mark"><Aperture size={16} /></span><strong>FRAME</strong><span>Local-first face AI</span></div>
          <div className="footer-privacy"><LockKeyhole size={13} /><span>Source photos are processed in memory and not written to disk. In a hosted preview, frames reach this workspace backend; run locally for on-device processing.</span></div>
          <span className="footer-version">MODEL WEIGHTS ARE NOT INCLUDED</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
