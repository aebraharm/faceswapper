import { afterEach, describe, expect, it, vi } from 'vitest';
import packageConfig from '../../package.json';
import {
  createAuthenticatedHealthRequest,
  createAuthenticatedShutdownRequest,
  createRuntimeConfig,
  waitForAuthenticatedBackend,
} from '../../electron/backend-lifecycle';
import { CameraPermissionGate } from '../../electron/camera-permission';
import {
  createBackendCommand,
  resolveDevelopmentBackendPaths,
  resolvePackagedBackendPath,
  resolveRendererEntry,
  resolveUserDataPath,
} from '../../electron/desktop-paths';
import {
  DEFAULT_DESKTOP_PREFERENCES,
  loadDesktopPreferences,
  saveDesktopPreferences,
} from '../../electron/user-settings';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  const { rm } = await import('node:fs/promises');
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe('packaged Electron paths and launch configuration', () => {
  it('defines a non-elevated per-user NSIS installer with app, React bundle and sidecar resources', () => {
    expect(packageConfig.build.win.target).toEqual([{ target: 'nsis', arch: ['x64'] }]);
    expect(packageConfig.build.nsis).toMatchObject({
      oneClick: true,
      perMachine: false,
      allowElevation: false,
      deleteAppDataOnUninstall: false,
    });
    expect(packageConfig.build.files).toContain('dist/**/*');
    expect(packageConfig.build.files).toContain('dist-electron/**/*');
    expect(packageConfig.build.extraResources[0]?.from).toBe('../backend/dist/FrameBackend');
  });

  it('uses a per-user local application-data directory for packaged settings', () => {
    expect(resolveUserDataPath('C:\\Users\\Ada\\AppData\\Local', true, 'win32'))
      .toBe('C:\\Users\\Ada\\AppData\\Local\\FRAME');
    expect(resolveUserDataPath('C:\\Users\\Ada\\AppData\\Local', false, 'win32'))
      .toBe('C:\\Users\\Ada\\AppData\\Local\\FRAME Development');
  });

  it('loads the production React build from the packaged app instead of Vite', () => {
    const entry = resolveRendererEntry(
      true,
      'C:\\Program Files\\FRAME\\resources\\app.asar',
      'http://127.0.0.1:5173/',
      'win32',
    );
    expect(entry).toEqual({
      kind: 'file',
      path: 'C:\\Program Files\\FRAME\\resources\\app.asar\\dist\\index.html',
    });
  });

  it('keeps Vite as the development renderer entry', () => {
    expect(resolveRendererEntry(false, '/repo/frontend', 'http://127.0.0.1:5173/', 'linux'))
      .toEqual({ kind: 'url', url: 'http://127.0.0.1:5173/' });
  });

  it('discovers the packaged PyInstaller executable and never falls back to system Python', () => {
    const resourcesPath = 'C:\\Users\\Ada\\AppData\\Local\\Programs\\FRAME\\resources';
    expect(resolvePackagedBackendPath(resourcesPath, 'win32'))
      .toBe('C:\\Users\\Ada\\AppData\\Local\\Programs\\FRAME\\resources\\backend\\FrameBackend.exe');
    const command = createBackendCommand({
      isPackaged: true,
      resourcesPath,
      compiledMainDirectory: 'C:\\repo\\frontend\\dist-electron\\electron',
      platform: 'win32',
      port: 43_217,
      sessionToken: 'launch-secret',
      baseEnvironment: { PATH: 'C:\\Windows\\System32' },
      pythonExecutable: 'C:\\Python311\\python.exe',
    });
    expect(command.command).toBe(resolvePackagedBackendPath(resourcesPath, 'win32'));
    expect(command.args).toEqual(['--port', '43217']);
    expect(command.cwd).toBe('C:\\Users\\Ada\\AppData\\Local\\Programs\\FRAME\\resources\\backend');
    expect(command.env.FRAME_DESKTOP_SESSION_TOKEN).toBe('launch-secret');
    expect(command.env.FRAME_DESKTOP_SHUTDOWN_TOKEN).toBe('launch-secret');
  });

  it('resolves the development Python launcher relative to the compiled Electron main process', () => {
    expect(resolveDevelopmentBackendPaths('C:\\repo\\frontend\\dist-electron\\electron', 'win32'))
      .toEqual({
        repositoryRoot: 'C:\\repo',
        backendDirectory: 'C:\\repo\\backend',
        launcherPath: 'C:\\repo\\backend\\launcher.py',
      });
  });
});

describe('sidecar lifecycle requests', () => {
  it('builds loopback runtime URLs for the selected dynamic port and launch token', () => {
    expect(createRuntimeConfig(43_217, 'session-secret')).toEqual({
      backendUrl: 'http://127.0.0.1:43217',
      websocketUrl: 'ws://127.0.0.1:43217',
      sessionToken: 'session-secret',
    });
    expect(() => createRuntimeConfig(0, 'session-secret')).toThrow('port is invalid');
  });

  it('sends the session token while polling the authenticated health endpoint', async () => {
    const unauthorized = { ok: false, json: async () => ({ detail: 'Not found.' }) } as Response;
    const healthy = { ok: true, json: async () => ({ status: 'ok' }) } as Response;
    const fetcher = vi.fn().mockResolvedValueOnce(unauthorized).mockResolvedValueOnce(healthy);
    await waitForAuthenticatedBackend({
      port: 43_217,
      sessionToken: 'health-secret',
      getExitCode: () => null,
      getSpawnError: () => null,
      fetcher,
      sleep: async () => undefined,
      timeoutMs: 2_000,
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(fetcher.mock.calls[0]?.[0]).toBe('http://127.0.0.1:43217/health');
    expect(fetcher.mock.calls[0]?.[1]?.headers).toEqual({ 'X-Frame-Session': 'health-secret' });
  });

  it('sends both authentication headers for graceful shutdown', () => {
    expect(createAuthenticatedShutdownRequest('shutdown-secret')).toMatchObject({
      method: 'POST',
      headers: {
        'X-Frame-Session': 'shutdown-secret',
        'X-Frame-Shutdown-Token': 'shutdown-secret',
      },
    });
    expect(createAuthenticatedHealthRequest('health-secret').headers)
      .toEqual({ 'X-Frame-Session': 'health-secret' });
  });
});

describe('explicit camera permission gate', () => {
  it('denies camera requests until authorized by the Start Camera action, then permits only one trusted media request', () => {
    const gate = new CameraPermissionGate();
    expect(gate.consume('media', true, 1_000)).toBe(false);
    gate.authorize(1_000);
    expect(gate.consume('media', false, 1_001)).toBe(false);
    expect(gate.consume('media', true, 1_002)).toBe(false);
    gate.authorize(2_000);
    expect(gate.consume('media', true, 2_001)).toBe(true);
    expect(gate.consume('media', true, 2_002)).toBe(false);
  });

  it('expires camera authorization if getUserMedia is not requested promptly', () => {
    const gate = new CameraPermissionGate();
    gate.authorize(1_000, 500);
    expect(gate.consume('media', true, 1_500)).toBe(false);
  });
});

describe('per-user desktop preferences', () => {
  it('stores only validated settings in the user-data folder and restores them', async () => {
    const { mkdtemp } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const directory = await mkdtemp(join(tmpdir(), 'frame-settings-'));
    temporaryDirectories.push(directory);
    const saved = await saveDesktopPreferences(directory, {
      provider: 'CUDAExecutionProvider',
      intensity: 0.64,
      resolution: 480,
      sourcePhoto: 'must-not-persist',
      sessionToken: 'must-not-persist',
    });
    expect(saved).toEqual({ provider: 'CUDAExecutionProvider', intensity: 0.64, resolution: 480 });
    expect(await loadDesktopPreferences(directory)).toEqual(saved);
    const stored = JSON.parse(await (await import('node:fs/promises')).readFile(join(directory, 'settings.json'), 'utf8'));
    expect(Object.keys(stored).sort()).toEqual(['intensity', 'provider', 'resolution']);
  });

  it('uses defaults for missing settings and rejects invalid preference writes', async () => {
    const { mkdtemp } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const directory = await mkdtemp(join(tmpdir(), 'frame-settings-'));
    temporaryDirectories.push(directory);
    expect(await loadDesktopPreferences(directory)).toEqual(DEFAULT_DESKTOP_PREFERENCES);
    await expect(saveDesktopPreferences(directory, { provider: 'unknown', intensity: 2, resolution: 123 }))
      .rejects.toThrow('inference provider');
  });
});
