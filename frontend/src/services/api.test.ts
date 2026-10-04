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
      authorizeCamera: vi.fn().mockResolvedValue(true),
      onBackendError: vi.fn().mockReturnValue(() => undefined),
    };
    vi.stubGlobal('window', { frameDesktop: bridge });

    await api.health();

    expect(fetchMock.mock.calls[0]?.[0]).toBe('http://127.0.0.1:43210/health');
    expect(fetchMock.mock.calls[0]?.[1]?.headers).toMatchObject({ 'X-Frame-Session': 'test-session-token' });
  });

  it('surfaces backend API error messages', async () => {
    vi.stubGlobal('window', {});
    fetchMock.mockResolvedValueOnce(makeResponse({ detail: 'backend unavailable' }, false));
    await expect(api.health()).rejects.toThrow('backend unavailable');
  });
});
