import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';
import type { DesktopRuntimeConfig, FrameDesktopBridge } from '../types/desktop';

function makeResponse(body: unknown, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 503,
    json: async () => body,
  } as Response;
}

describe('runtime API routing', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(makeResponse({ status: 'ok', landmarks_available: false }));
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('uses the Vite /api proxy when the Electron bridge is absent', async () => {
    vi.stubGlobal('window', {});
    await api.health();
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/health');
  });

  it('uses the runtime loopback backend URL and session header in desktop mode', async () => {
    const config: DesktopRuntimeConfig = {
      backendUrl: 'http://127.0.0.1:43210',
      websocketUrl: 'ws://127.0.0.1:43210',
      sessionToken: 'test-session-token',
    };
    const bridge: FrameDesktopBridge = {
      getRuntimeConfig: vi.fn().mockResolvedValue(config),
      loadSettings: vi.fn().mockResolvedValue({ provider: 'auto', intensity: 0.85, resolution: 640 }),
      saveSettings: vi.fn().mockResolvedValue({ provider: 'auto', intensity: 0.85, resolution: 640 }),
      authorizeCamera: vi.fn().mockResolvedValue(true),
      onBackendError: vi.fn().mockReturnValue(() => undefined),
    };
    vi.stubGlobal('window', { frameDesktop: bridge });

    await api.health();

    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://127.0.0.1:43210/health');
    expect(fetchMock.mock.calls[0]?.[1]?.headers).toMatchObject({ 'X-Frame-Session': 'test-session-token' });
  });

  it('sends a raw source image to the packaged runtime with its session token', async () => {
    const config: DesktopRuntimeConfig = {
      backendUrl: 'http://127.0.0.1:43210',
      websocketUrl: 'ws://127.0.0.1:43210',
      sessionToken: 'upload-session-token',
    };
    const bridge: FrameDesktopBridge = {
      getRuntimeConfig: vi.fn().mockResolvedValue(config),
      loadSettings: vi.fn().mockResolvedValue({ provider: 'auto', intensity: 0.85, resolution: 640 }),
      saveSettings: vi.fn().mockResolvedValue({ provider: 'auto', intensity: 0.85, resolution: 640 }),
      authorizeCamera: vi.fn().mockResolvedValue(true),
      onBackendError: vi.fn().mockReturnValue(() => undefined),
    };
    vi.stubGlobal('window', { frameDesktop: bridge });
    const file = new File(['fake image bytes'], 'portrait.jpeg', { type: '' });

    await api.uploadSource(file);

    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://127.0.0.1:43210/source-face/upload');
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: 'POST',
      headers: { 'Content-Type': 'image/jpeg', 'X-Frame-Session': 'upload-session-token' },
      body: file,
    });
  });

  it('rejects unsupported local source types before making a request', async () => {
    vi.stubGlobal('window', {});
    const file = new File(['not an image'], 'portrait.heic', { type: 'image/heic' });

    await expect(api.uploadSource(file)).rejects.toThrow('Choose a JPG, PNG, or WEBP image.');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('explains an authenticated desktop-session rejection instead of showing a generic 404', async () => {
    const bridge: FrameDesktopBridge = {
      getRuntimeConfig: vi.fn().mockResolvedValue({
        backendUrl: 'http://127.0.0.1:43210',
        websocketUrl: 'ws://127.0.0.1:43210',
        sessionToken: 'expired-session-token',
      }),
      loadSettings: vi.fn().mockResolvedValue({ provider: 'auto', intensity: 0.85, resolution: 640 }),
      saveSettings: vi.fn().mockResolvedValue({ provider: 'auto', intensity: 0.85, resolution: 640 }),
      authorizeCamera: vi.fn().mockResolvedValue(true),
      onBackendError: vi.fn().mockReturnValue(() => undefined),
    };
    vi.stubGlobal('window', { frameDesktop: bridge });
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: 'Not found.' }),
    } as Response);

    await expect(api.health()).rejects.toThrow('rejected this desktop session');
  });

  it('surfaces backend API error messages', async () => {
    vi.stubGlobal('window', {});
    fetchMock.mockResolvedValueOnce(makeResponse({ detail: 'backend unavailable' }, false));
    await expect(api.health()).rejects.toThrow('backend unavailable');
  });

  it('fetches the installable model catalog', async () => {
    vi.stubGlobal('window', {});
    fetchMock.mockResolvedValueOnce(makeResponse({ models: [], selected_model_id: 'liveportrait-v1' }));
    const response = await api.modelCatalog();
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/models/catalog');
    expect(response.selected_model_id).toBe('liveportrait-v1');
  });

  it('starts an install and can remove an installed model by id', async () => {
    vi.stubGlobal('window', {});
    fetchMock.mockResolvedValueOnce(makeResponse({ id: 'liveportrait-v1', installed: false, downloading: true }));
    await api.installModel('liveportrait-v1');
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/models/liveportrait-v1/install');
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ method: 'POST' });

    fetchMock.mockResolvedValueOnce(makeResponse({ id: 'liveportrait-v1', installed: false, downloading: false }));
    await api.removeModel('liveportrait-v1');
    expect(fetchMock.mock.calls[1]?.[0]).toBe('/api/models/liveportrait-v1');
    expect(fetchMock.mock.calls[1]?.[1]).toMatchObject({ method: 'DELETE' });
  });
});
