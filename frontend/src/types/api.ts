export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence?: number | null;
}

export interface SourceFaceStatus {
  uploaded: boolean;
  face_count: number;
  faces: FaceBox[];
  selected_face_index: number | null;
  ready: boolean;
  model_ready: boolean;
  width: number;
  height: number;
  storage: string;
  format?: string;
}

export interface TransformerStatus {
  loaded: boolean;
  name: string;
  device: string;
  provider?: string | null;
  error?: string | null;
  requires_model_files?: boolean;
  model_id?: string | null;
}

export interface ModelCatalogEntry {
  id: string;
  display_name: string;
  license_name: string;
  license_url: string;
  source_url: string;
  summary: string;
  approx_total_bytes: number;
  installed: boolean;
  downloading: boolean;
  bytes_downloaded: number;
  total_bytes: number;
  progress: number;
  error?: string | null;
}

export interface ModelCatalogResponse {
  models: ModelCatalogEntry[];
  selected_model_id: string;
}

export interface CameraStatus {
  active: boolean;
  connected: boolean;
  device_id?: string | null;
  face_count: number;
  faces: FaceBox[];
  selected_target: number | null;
  selected_target_index: number | null;
  transform_enabled: boolean;
  frames_processed: number;
  fps: number;
  latency_ms: number;
  camera_resolution: string;
  processing_resolution: number;
}

export interface FrameDiagnostics {
  first_invalid: { stage: string; reason: string } | null;
  stages: Array<Record<string, unknown>>;
}

export interface FrameStats {
  type: 'stats' | 'error';
  message?: string;
  fps?: number;
  latency_ms?: number;
  camera_resolution?: string;
  processing_resolution?: number;
  face_count?: number;
  faces?: FaceBox[];
  selected_target?: number | null;
  transformation_active?: boolean;
  model_status?: TransformerStatus;
  device?: string;
  /** Present only when FRAME_FRAME_DIAGNOSTICS=1 on the local backend. */
  frame_diagnostics?: FrameDiagnostics;
  frame_diagnostics_id?: string;
}
