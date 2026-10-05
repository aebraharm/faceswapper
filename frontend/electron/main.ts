import { app, BrowserWindow, dialog, ipcMain, session } from 'electron';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'node:net';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  BACKEND_READY_TIMEOUT_MS,
  BACKEND_SHUTDOWN_TIMEOUT_MS,
  createAuthenticatedShutdownRequest,
  createRuntimeConfig,
  waitForAuthenticatedBackend,
  type DesktopRuntimeConfig,
} from './backend-lifecycle';
import { CameraPermissionGate } from './camera-permission';
import {
  createBackendCommand,
  resolvePackagedRendererPath,
  resolveRendererEntry,
  resolveUserDataPath,
} from './desktop-paths';
import { loadDesktopPreferences, saveDesktopPreferences } from './user-settings';

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let runtimeConfig: DesktopRuntimeConfig | null = null;
let backendStopping = false;
let isQuitting = false;
let backendLogTail = '';
let backendSpawnError: Error | null = null;

// tsc emits this file under frontend/dist-electron/electron in development.
const isPackaged = app.isPackaged;
const DEV_RENDERER_ORIGIN = 'http://127.0.0.1:5173';
app.setName('FRAME');
const appDataRoot = process.platform === 'win32'
  ? process.env.LOCALAPPDATA || app.getPath('appData')
  : app.getPath('appData');
const userDataDirectory = resolveUserDataPath(appDataRoot, isPackaged, process.platform);
app.setPath('userData', userDataDirectory);
app.setPath('sessionData', userDataDirectory);
const packagedRendererPath = resolvePackagedRendererPath(app.getAppPath(), process.platform);
const PACKAGED_RENDERER_URL = pathToFileURL(packagedRendererPath).href;
const cameraPermissionGate = new CameraPermissionGate();

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
  const command = createBackendCommand({
    isPackaged,
    resourcesPath: process.resourcesPath,
    compiledMainDirectory: __dirname,
    platform: process.platform,
    port,
    sessionToken,
    baseEnvironment: process.env,
    pythonExecutable: process.env.FRAME_PYTHON,
    modelsDirectory: path.join(userDataDirectory, 'models'),
  });
  const child = spawn(command.command, command.args, {
    cwd: command.cwd,
    env: command.env,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  child.stdout?.on('data', appendBackendLog);
  child.stderr?.on('data', appendBackendLog);
  child.on('error', (error) => { backendSpawnError = error; });
  return child;
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

async function launchBackend(): Promise<DesktopRuntimeConfig> {
  const errors: string[] = [];
  backendLogTail = '';
  backendSpawnError = null;

  for (let attempt = 0; attempt < 5; attempt += 1) {
    const port = await allocateLoopbackPort();
    const token = randomBytes(32).toString('hex');
    const child = spawnBackend(port, token);
    backendProcess = child;
    try {
      await waitForAuthenticatedBackend({
        port,
        sessionToken: token,
        getExitCode: () => child.exitCode,
        getSpawnError: () => backendSpawnError,
        timeoutMs: BACKEND_READY_TIMEOUT_MS,
      });
      child.on('exit', (code, signal) => {
        if (backendStopping || isQuitting || !mainWindow || mainWindow.isDestroyed()) return;
        const message = `Local AI backend stopped unexpectedly (code ${code ?? 'unknown'}, signal ${signal ?? 'none'}).`;
        mainWindow.webContents.send('desktop:backend-error', message);
      });
      child.on('error', (error) => {
        if (backendStopping || isQuitting || !mainWindow || mainWindow.isDestroyed()) return;
        mainWindow.webContents.send('desktop:backend-error', `Could not run the local AI backend: ${error.message}`);
      });
      return createRuntimeConfig(port, token);
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
      await fetch(
        `${runtimeConfig.backendUrl}/internal/shutdown`,
        createAuthenticatedShutdownRequest(runtimeConfig.sessionToken),
      );
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
    callback(cameraPermissionGate.consume(permission, Boolean(trustedWindow)));
  });

  ipcMain.handle('desktop:authorize-camera', (event) => {
    if (!isTrustedIpcSender(event) || !mainWindow?.isVisible()) return false;
    cameraPermissionGate.authorize();
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

  const renderer = resolveRendererEntry(
    isPackaged,
    app.getAppPath(),
    `${DEV_RENDERER_ORIGIN}/`,
    process.platform,
  );
  if (renderer.kind === 'file') {
    await mainWindow.loadFile(renderer.path);
  } else {
    await mainWindow.loadURL(renderer.url);
  }
}

ipcMain.handle('desktop:get-runtime-config', (event) => {
  if (!isTrustedIpcSender(event)) throw new Error('Untrusted renderer requested desktop configuration.');
  if (!runtimeConfig) throw new Error('The local AI backend is not ready.');
  return runtimeConfig;
});

ipcMain.handle('desktop:load-settings', (event) => {
  if (!isTrustedIpcSender(event)) throw new Error('Untrusted renderer requested desktop settings.');
  return loadDesktopPreferences(app.getPath('userData'));
});

ipcMain.handle('desktop:save-settings', (event, preferences: unknown) => {
  if (!isTrustedIpcSender(event)) throw new Error('Untrusted renderer requested desktop settings.');
  return saveDesktopPreferences(app.getPath('userData'), preferences);
});

app.whenReady().then(async () => {
  configureCameraPermissions();
  try {
    await mkdir(app.getPath('userData'), { recursive: true });
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
