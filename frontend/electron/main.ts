import { app, BrowserWindow, dialog, ipcMain, session } from 'electron';
import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:net';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

interface RuntimeConfig {
  backendUrl: string;
  websocketUrl: string;
  sessionToken: string;
}

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let runtimeConfig: RuntimeConfig | null = null;
let backendStopping = false;
let isQuitting = false;
let nextCameraPermissionAllowed = false;
let backendLogTail = '';
let backendSpawnError: Error | null = null;

// tsc emits this file under frontend/dist-electron/electron in development.
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const isPackaged = app.isPackaged;
const BACKEND_READY_TIMEOUT_MS = 20_000;
const BACKEND_SHUTDOWN_TIMEOUT_MS = 4_000;
const DEV_RENDERER_ORIGIN = 'http://127.0.0.1:5173';
const PACKAGED_RENDERER_URL = pathToFileURL(path.join(app.getAppPath(), 'dist', 'index.html')).href;

function isTrustedRendererUrl(url: string): boolean {
  if (isPackaged) return url === PACKAGED_RENDERER_URL || url.startsWith(`${PACKAGED_RENDERER_URL}#`);
  try {
    return new URL(url).origin === DEV_RENDERER_ORIGIN;
  } catch {
    return false;
  }
}

function isTrustedIpcSender(event: Electron.IpcMainInvokeEvent): boolean {
  const frame = event.senderFrame;
  return Boolean(
    mainWindow
    && !mainWindow.isDestroyed()
    && event.sender.id === mainWindow.webContents.id
    && frame?.parent === null
    && isTrustedRendererUrl(frame.url)
  );
}

function appendBackendLog(chunk: Buffer | string): void {
  backendLogTail = `${backendLogTail}${chunk.toString()}`.slice(-8_000);
}

function allocateLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address();
      if (!address || typeof address === 'string') {
        probe.close();
        reject(new Error('Could not allocate a loopback port for the AI backend.'));
        return;
      }
      const { port } = address;
      probe.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function spawnBackend(port: number, sessionToken: string): ChildProcess {
  const backendDirectory = path.join(repoRoot, 'backend');
  const environment = {
    ...process.env,
    FRAME_DESKTOP_SESSION_TOKEN: sessionToken,
    FRAME_DESKTOP_SHUTDOWN_TOKEN: sessionToken,
  };

  if (isPackaged) {
    const executable = path.join(process.resourcesPath, 'backend', 'FrameBackend.exe');
    const child = spawn(executable, ['--port', String(port)], {
      cwd: path.dirname(executable),
      env: environment,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
    child.stdout?.on('data', appendBackendLog);
    child.stderr?.on('data', appendBackendLog);
    child.on('error', (error) => { backendSpawnError = error; });
    return child;
  }

  const python = process.env.FRAME_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  const launcher = path.join(backendDirectory, 'launcher.py');
  const child = spawn(python, [launcher, '--port', String(port)], {
    cwd: backendDirectory,
    env: environment,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  child.stdout?.on('data', appendBackendLog);
  child.stderr?.on('data', appendBackendLog);
  child.on('error', (error) => { backendSpawnError = error; });
  return child;
}

async function waitForBackend(port: number, token: string, child: ChildProcess): Promise<void> {
  const startedAt = Date.now();
  const healthUrl = `http://127.0.0.1:${port}/health`;
  while (Date.now() - startedAt < BACKEND_READY_TIMEOUT_MS) {
    if (backendSpawnError) throw backendSpawnError;
    if (child.exitCode !== null) {
      throw new Error(`The Python backend exited during startup (exit code ${child.exitCode}).`);
    }
    try {
      const response = await fetch(healthUrl, {
        headers: { 'X-Frame-Session': token },
        signal: AbortSignal.timeout(1_200),
      });
      if (response.ok) {
        const health = await response.json() as { status?: string };
        if (health.status === 'ok') return;
      }
    } catch {
      // The backend may still be importing OpenCV/ONNX Runtime; retry until timeout.
    }
    await delay(180);
  }
  throw new Error(`The Python backend did not become healthy within ${BACKEND_READY_TIMEOUT_MS / 1_000} seconds.`);
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill();
      resolve();
    }, 1_500);
    child.once('exit', () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function launchBackend(): Promise<RuntimeConfig> {
  const errors: string[] = [];
  backendLogTail = '';
  backendSpawnError = null;

  for (let attempt = 0; attempt < 5; attempt += 1) {
    const port = await allocateLoopbackPort();
    const token = randomBytes(32).toString('hex');
    const child = spawnBackend(port, token);
    backendProcess = child;
    try {
      await waitForBackend(port, token, child);
      child.on('exit', (code, signal) => {
        if (backendStopping || isQuitting || !mainWindow || mainWindow.isDestroyed()) return;
        const message = `Local AI backend stopped unexpectedly (code ${code ?? 'unknown'}, signal ${signal ?? 'none'}).`;
        mainWindow.webContents.send('desktop:backend-error', message);
      });
      child.on('error', (error) => {
        if (backendStopping || isQuitting || !mainWindow || mainWindow.isDestroyed()) return;
        mainWindow.webContents.send('desktop:backend-error', `Could not run the local AI backend: ${error.message}`);
      });
      return {
        backendUrl: `http://127.0.0.1:${port}`,
        websocketUrl: `ws://127.0.0.1:${port}`,
        sessionToken: token,
      };
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
      await stopChild(child);
      backendProcess = null;
      const spawnFailure = backendSpawnError as Error | null;
      if (spawnFailure?.message.includes('ENOENT') || spawnFailure?.message.includes('EACCES')) break;
      backendSpawnError = null;
    }
  }

  const detail = [
    'FRAME could not start its local Python computer-vision backend.',
    ...errors,
    backendLogTail.trim() ? `Backend output:\n${backendLogTail.trim()}` : '',
  ].filter(Boolean).join('\n\n');
  throw new Error(detail);
}

async function stopBackend(): Promise<void> {
  if (backendStopping) return;
  backendStopping = true;
  const child = backendProcess;
  if (!child || child.exitCode !== null) return;

  if (runtimeConfig) {
    try {
      await fetch(`${runtimeConfig.backendUrl}/internal/shutdown`, {
        method: 'POST',
        headers: {
          'X-Frame-Session': runtimeConfig.sessionToken,
          'X-Frame-Shutdown-Token': runtimeConfig.sessionToken,
        },
        signal: AbortSignal.timeout(1_500),
      });
    } catch {
      // If the service is already gone, the child-process fallback below still runs.
    }
  }

  await Promise.race([
    new Promise<void>((resolve) => child.once('exit', () => resolve())),
    delay(BACKEND_SHUTDOWN_TIMEOUT_MS),
  ]);
  if (child.exitCode === null) await stopChild(child);
  backendProcess = null;
}

function configureCameraPermissions(): void {
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const trustedWindow = mainWindow
      && !mainWindow.isDestroyed()
      && webContents.id === mainWindow.webContents.id
      && isTrustedRendererUrl(webContents.getURL());
    const allow = permission === 'media' && trustedWindow && nextCameraPermissionAllowed;
    nextCameraPermissionAllowed = false;
    callback(Boolean(allow));
  });

  ipcMain.handle('desktop:authorize-camera', (event) => {
    if (!isTrustedIpcSender(event) || !mainWindow?.isVisible()) return false;
    nextCameraPermissionAllowed = true;
    setTimeout(() => { nextCameraPermissionAllowed = false; }, 5_000);
    return true;
  });
}

async function createMainWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 980,
    minWidth: 920,
    minHeight: 700,
    show: false,
    backgroundColor: '#f1f2ee',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  mainWindow.once('ready-to-show', () => mainWindow?.show());
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!isTrustedRendererUrl(url)) event.preventDefault();
  });
  mainWindow.webContents.on('will-redirect', (event, url) => {
    if (!isTrustedRendererUrl(url)) event.preventDefault();
  });
  mainWindow.on('closed', () => { mainWindow = null; });

  if (isPackaged) {
    await mainWindow.loadFile(path.join(app.getAppPath(), 'dist', 'index.html'));
  } else {
    await mainWindow.loadURL(`${DEV_RENDERER_ORIGIN}/`);
  }
}

ipcMain.handle('desktop:get-runtime-config', (event) => {
  if (!isTrustedIpcSender(event)) throw new Error('Untrusted renderer requested desktop configuration.');
  if (!runtimeConfig) throw new Error('The local AI backend is not ready.');
  return runtimeConfig;
});

app.whenReady().then(async () => {
  configureCameraPermissions();
  try {
    runtimeConfig = await launchBackend();
    await createMainWindow();
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    dialog.showErrorBox('FRAME could not start', `${detail}\n\nCheck the installation and Python backend files, then try again.`);
    await stopBackend();
    app.quit();
  }
});

app.on('window-all-closed', () => app.quit());
app.on('before-quit', (event) => {
  if (isQuitting) return;
  event.preventDefault();
  isQuitting = true;
  void stopBackend().finally(() => app.quit());
});
