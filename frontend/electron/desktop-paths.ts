import path from 'node:path';

export interface DevelopmentBackendPaths {
  repositoryRoot: string;
  backendDirectory: string;
  launcherPath: string;
}

export interface BackendCommandOptions {
  isPackaged: boolean;
  resourcesPath: string;
  compiledMainDirectory: string;
  platform: NodeJS.Platform;
  port: number;
  sessionToken: string;
  baseEnvironment: NodeJS.ProcessEnv;
  pythonExecutable?: string;
}

export interface BackendCommand {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
}

export type RendererEntry =
  | { kind: 'file'; path: string }
  | { kind: 'url'; url: string };

function pathApi(platform: string): typeof path {
  return platform === 'win32' ? path.win32 : path;
}

export function resolveUserDataPath(appDataPath: string, isPackaged: boolean, platform = process.platform): string {
  const directoryName = isPackaged ? 'FRAME' : 'FRAME Development';
  return pathApi(platform).join(appDataPath, directoryName);
}

export function resolvePackagedRendererPath(appPath: string, platform = process.platform): string {
  return pathApi(platform).join(appPath, 'dist', 'index.html');
}

export function resolvePackagedBackendPath(resourcesPath: string, platform = process.platform): string {
  const pathForPlatform = pathApi(platform);
  const executable = platform === 'win32' ? 'FrameBackend.exe' : 'FrameBackend';
  return pathForPlatform.join(resourcesPath, 'backend', executable);
}

export function resolveDevelopmentBackendPaths(
  compiledMainDirectory: string,
  platform = process.platform,
): DevelopmentBackendPaths {
  const pathForPlatform = pathApi(platform);
  const repositoryRoot = pathForPlatform.resolve(compiledMainDirectory, '..', '..', '..');
  const backendDirectory = pathForPlatform.join(repositoryRoot, 'backend');
  return {
    repositoryRoot,
    backendDirectory,
    launcherPath: pathForPlatform.join(backendDirectory, 'launcher.py'),
  };
}

export function resolveRendererEntry(
  isPackaged: boolean,
  appPath: string,
  developmentUrl: string,
  platform = process.platform,
): RendererEntry {
  return isPackaged
    ? { kind: 'file', path: resolvePackagedRendererPath(appPath, platform) }
    : { kind: 'url', url: developmentUrl };
}

export function createBackendCommand(options: BackendCommandOptions): BackendCommand {
  const pathForPlatform = pathApi(options.platform);
  const environment: NodeJS.ProcessEnv = {
    ...options.baseEnvironment,
    FRAME_DESKTOP_SESSION_TOKEN: options.sessionToken,
    FRAME_DESKTOP_SHUTDOWN_TOKEN: options.sessionToken,
  };
  const portArgument = ['--port', String(options.port)];

  if (options.isPackaged) {
    const executable = resolvePackagedBackendPath(options.resourcesPath, options.platform);
    return {
      command: executable,
      args: portArgument,
      cwd: pathForPlatform.dirname(executable),
      env: environment,
    };
  }

  const { backendDirectory, launcherPath } = resolveDevelopmentBackendPaths(
    options.compiledMainDirectory,
    options.platform,
  );
  return {
    command: options.pythonExecutable || (options.platform === 'win32' ? 'python' : 'python3'),
    args: [launcherPath, ...portArgument],
    cwd: backendDirectory,
    env: environment,
  };
}
