import type { CameraStatus, ModelCatalogEntry, ModelCatalogResponse, SourceFaceStatus, TransformerStatus } from '../types/api';
import { getSourceImageContentType } from './sourceImage';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let runtime = null;
  try {
    runtime = (await window.frameDesktop?.getRuntimeConfig()) ?? null;
  } catch {
    // If the desktop bridge is unavailable, use the regular Vite development proxy.
  }
  const url = runtime ? `${runtime.backendUrl}${path}` : `/api${path}`;
  const requestInit = { ...init, headers: { ...(init?.headers as Record<string, string> | undefined) } };
  if (runtime?.sessionToken) requestInit.headers["X-Frame-Session"] = runtime.sessionToken;
  const response = await fetch(url, requestInit);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (runtime && response.status === 404) {
      throw new Error('The local AI backend rejected this desktop session. Close and reopen FRAME, then try again.');
    }
    const detail = typeof body.detail === 'string' ? body.detail : `The local AI backend request failed (${response.status}).`;
    throw new Error(detail);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: string; landmarks_available: boolean }>('/health'),
  sourceStatus: () => request<SourceFaceStatus>('/source-face/status'),
  uploadSource: async (file: File): Promise<SourceFaceStatus> => {
    const contentType = getSourceImageContentType(file);
    if (!contentType) throw new Error('Choose a JPG, PNG, or WEBP image.');
    return request<SourceFaceStatus>('/source-face/upload', {
      method: 'POST',
      headers: { 'Content-Type': contentType },
      // This API uses a raw image body, not multipart/form-data. The FastAPI
      // endpoint streams and decodes these exact bytes in memory.
      body: file,
    });
  },
  selectSource: (face_index: number) => request<SourceFaceStatus>('/source-face/select', json({ face_index })),
  removeSource: () => request<SourceFaceStatus & { ok: boolean }>('/source-face', { method: 'DELETE' }),
  transformerStatus: () => request<TransformerStatus>('/transformer/status'),
  loadTransformer: (provider: string) => request<TransformerStatus>('/transformer/load', json({ provider })),
  unloadTransformer: () => request<TransformerStatus>('/transformer/unload', { method: 'POST' }),
  startCamera: (device_id: string | null) => request<CameraStatus>('/camera/start', json({ device_id })),
  stopCamera: () => request<CameraStatus>('/camera/stop', { method: 'POST' }),
  cameraStatus: () => request<CameraStatus>('/camera/status'),
  selectTarget: (face_index: number | null) => request<CameraStatus>('/camera/target', json({ face_index })),
  cameraSettings: (settings: {
    transform_enabled?: boolean;
    intensity?: number;
    processing_resolution?: number;
    performance_mode?: 'auto' | 'quality' | 'performance';
  }) => request<Record<string, unknown>>('/camera/settings', json(settings)),
  modelCatalog: () => request<ModelCatalogResponse>('/models/catalog'),
  modelStatus: (modelId: string) => request<ModelCatalogEntry>(`/models/${modelId}/status`),
  installModel: (modelId: string) => request<ModelCatalogEntry>(`/models/${modelId}/install`, { method: 'POST' }),
  removeModel: (modelId: string) => request<ModelCatalogEntry>(`/models/${modelId}`, { method: 'DELETE' }),
};
