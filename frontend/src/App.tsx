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
import { getProcessedVideoStream } from './services/output';
import type { DesktopPreferences } from './types/desktop';
import type { CameraStatus, FaceBox, FrameStats, SourceFaceStatus, TransformerStatus } from './types/api';

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
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
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
  const [notice, setNotice] = useState('');
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

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    const cameras = allDevices.filter((device) => device.kind === 'videoinput');
    setDevices(cameras);
    setSelectedDevice((current) => current || cameras[0]?.deviceId || '');
  }, []);

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
    };
  }, [refreshDevices]);

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
      void api.cameraSettings({ intensity, processing_resolution: resolution }).catch((error) => {
        setNotice(errorMessage(error));
      });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [intensity, resolution, cameraLive]);

  const handleFrameMessage = useCallback(async (event: MessageEvent) => {
    if (typeof event.data === 'string') {
      try {
        const stats = JSON.parse(event.data) as FrameStats;
        if (stats.type === 'error') {
          frameInFlightRef.current = false;
          setCameraError(stats.message ?? 'The camera frame could not be processed.');
          return;
        }
        setFps(stats.fps ?? 0);
        setLatency(stats.latency_ms ?? 0);
        setCameraResolution(stats.camera_resolution ?? '—');
        setTargetFaces(stats.faces ?? []);
        setSelectedTarget(stats.selected_target ?? null);
        setTransformationRunning(Boolean(stats.transformation_active));
        if (stats.model_status) setModel(stats.model_status);
      } catch {
        // Ignore malformed telemetry and keep the last good preview on screen.
      }
      return;
    }
    try {
      const bitmap = await createImageBitmap(event.data as Blob);
      frameInFlightRef.current = false;
      const canvas = processedCanvasRef.current;
      const context = canvas?.getContext('2d');
      if (canvas && context) {
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
        }
        context.drawImage(bitmap, 0, 0);
      }
      bitmap.close();
    } catch {
      frameInFlightRef.current = false;
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
    if (showMessage) setNotice('Camera stopped. The local video track has been released.');
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
      streamRef.current = localStream;
      if (videoRef.current) {
        videoRef.current.srcObject = localStream;
        await videoRef.current.play();
      }
      await refreshDevices();
      await api.cameraSettings({ transform_enabled: false, intensity, processing_resolution: resolution });
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
      socket.onerror = () => setCameraError('The local processing stream could not connect to the backend.');
      socket.onclose = () => {
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
      setNotice('Camera is live. Frames are sent to the local processor; no audio is captured.');

      const video = videoRef.current;
      const sourceCanvas = hiddenCaptureRef.current;
      const sourceContext = sourceCanvas?.getContext('2d', { alpha: false });
      let lastSent = 0;
      const send = (timestamp: number) => {
        if (socket.readyState !== WebSocket.OPEN || !video || !sourceCanvas || !sourceContext) return;
        if (timestamp - lastSent >= 33 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
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
          if (!frameInFlightRef.current && socket.bufferedAmount < 500_000) {
            frameInFlightRef.current = true;
            sourceContext.drawImage(video, 0, 0, frameWidth, frameHeight);
            sourceCanvas.toBlob((blob) => {
              if (blob && socket.readyState === WebSocket.OPEN && socket.bufferedAmount < 500_000) socket.send(blob);
              else frameInFlightRef.current = false;
            }, 'image/jpeg', 0.78);
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
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setNotice('Choose a JPG, PNG, or WEBP image.');
      return;
    }
    if (file.size > MAX_SOURCE_BYTES) {
      setNotice('This image exceeds the 8 MB upload limit.');
      return;
    }
    setUploadBusy(true);
    setNotice('');
    try {
      const result = await api.uploadSource(file);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      const url = URL.createObjectURL(file);
      previewUrlRef.current = url;
      setSourcePreview(url);
      setSource(result);
      setNotice(result.face_count > 1
        ? `Found ${result.face_count} faces. Choose the source identity below.`
        : 'Source face detected and aligned in memory.');
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (error) {
      setNotice(errorMessage(error));
    } finally {
      setUploadBusy(false);
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
      setNotice(`Source face ${index + 1} selected and aligned.`);
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  const removeSource = async () => {
    try {
      const result = await api.removeSource();
      setSource(result);
      setTransformEnabled(false);
      setTransformationRunning(false);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
      setSourcePreview(null);
      setNotice('Source photo removed from the application session.');
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  const loadModel = async () => {
    setNotice('');
    try {
      const loaded = await api.loadTransformer(provider);
      setModel(loaded);
      setSource(await api.sourceStatus());
      setNotice(`Model loaded on ${loaded.device}. Source identity features are now prepared once.`);
    } catch (error) {
      setNotice(errorMessage(error));
      setModel(await api.transformerStatus().catch(() => model));
    }
  };

  const unloadModel = async () => {
    try {
      const unloaded = await api.unloadTransformer();
      setModel(unloaded);
      setSource(await api.sourceStatus());
      setTransformEnabled(false);
      setTransformationRunning(false);
      setNotice('Model unloaded and cached identity features cleared.');
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  const toggleTransformation = async () => {
    const next = !transformEnabled;
    try {
      await api.cameraSettings({ transform_enabled: next, intensity, processing_resolution: resolution });
      setTransformEnabled(next);
      setNotice(next ? 'AI face transformation enabled.' : 'Transformation disabled. Original camera frames are shown.');
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  const selectTargetFace = async (index: number | null) => {
    try {
      const status: CameraStatus = await api.selectTarget(index);
      setSelectedTarget(status.selected_target_index);
    } catch (error) {
      setNotice(errorMessage(error));
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
      setNotice(errorMessage(error));
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
              {sourcePreview && <button className="icon-button" onClick={() => void removeSource()} aria-label="Remove source face" title="Remove source face"><Trash2 size={16} /></button>}
            </div>

            <div
              className={`source-dropzone${isDragging ? ' is-dragging' : ''}${sourcePreview ? ' has-preview' : ''}`}
              onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              style={sourcePreview && source.width && source.height ? { aspectRatio: `${source.width} / ${source.height}` } : undefined}
            >
              {sourcePreview ? (
                <>
                  <img src={sourcePreview} alt="Selected source face" className="source-image" />
                  <div className="source-scanline" />
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
                  <div className="image-caption"><ScanFace size={14} /> {source.face_count} {source.face_count === 1 ? 'FACE' : 'FACES'} DETECTED</div>
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
            <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={onFileChange} />
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
            <div className={`source-detection-note${source.ready ? ' note-ready' : ''}`}>
              <span className="note-icon">{source.ready ? <Check size={14} /> : <ScanFace size={14} />}</span>
              <span>{source.ready ? 'Face detected and aligned. Identity representation will be cached when a model is loaded.' : 'Upload a clear portrait to detect and align its face.'}</span>
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
            {!model.loaded && <a className="model-doc-link" href="/models/README.md" onClick={(event) => { event.preventDefault(); setNotice('See models/README.md in the repository for the model ABI, installation steps and licensing checklist.'); }}><CircleHelp size={13} /> Model setup & licensing <span>↗</span></a>}
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
          <div className="performance-header"><div><span className="eyebrow-line" /><span className="step-label-text">LIVE PERFORMANCE</span></div><span className="performance-subtitle">Updates while camera is streaming</span></div>
          <div className="metrics-grid">
            <MetricCard label="PROCESSING FPS" value={cameraLive ? fps.toFixed(1) : '—'} note={cameraLive ? 'End-to-end stream' : 'Camera is off'} icon={Zap} accent />
            <MetricCard label="FRAME LATENCY" value={cameraLive ? `${latency.toFixed(0)} ms` : '—'} note="Detection + processing" icon={Aperture} />
            <MetricCard label="CAMERA RESOLUTION" value={cameraLive ? cameraResolution : '—'} note={`Processing max ${resolution} px`} icon={Camera} />
            <MetricCard label="COMPUTE DEVICE" value={model.device === 'not loaded' ? 'CPU' : model.device.replace(' (ONNX Runtime not installed)', '')} note={model.loaded ? 'Model inference' : 'Face detection pipeline'} icon={Cpu} />
          </div>
        </section>

        {notice && <div className="notice-bar" role="status"><span><Check size={14} /></span><p>{notice}</p><button onClick={() => setNotice('')} aria-label="Dismiss message"><X size={15} /></button></div>}

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
