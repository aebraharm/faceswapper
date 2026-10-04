export interface DesktopRuntimeConfig {
  backendUrl: string;
  websocketUrl: string;
  sessionToken: string;
}

export interface BackendReadinessOptions {
  port: number;
  sessionToken: string;
  getExitCode: () => number | null;
  getSpawnError: () => Error | null;
  fetcher?: (url: string, init?: RequestInit) => Promise<Response>;
  sleep?: (milliseconds: number) => Promise<void>;
  now?: () => number;
  timeoutMs?: number;
  pollIntervalMs?: number;
  requestTimeoutMs?: number;
}

export const BACKEND_READY_TIMEOUT_MS = 20_000;
export const BACKEND_SHUTDOWN_TIMEOUT_MS = 4_000;

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

export function createRuntimeConfig(port: number, sessionToken: string): DesktopRuntimeConfig {
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error('The local AI backend port is invalid.');
  }
  if (!sessionToken) throw new Error('The local AI backend session token is missing.');
  return {
    backendUrl: `http://127.0.0.1:${port}`,
    websocketUrl: `ws://127.0.0.1:${port}`,
    sessionToken,
  };
}

export function createAuthenticatedHealthRequest(sessionToken: string, timeoutMs = 1_200): RequestInit {
  return {
    headers: { 'X-Frame-Session': sessionToken },
    signal: AbortSignal.timeout(timeoutMs),
  };
}

export function createAuthenticatedShutdownRequest(sessionToken: string, timeoutMs = 1_500): RequestInit {
  return {
    method: 'POST',
    headers: {
      'X-Frame-Session': sessionToken,
      'X-Frame-Shutdown-Token': sessionToken,
    },
    signal: AbortSignal.timeout(timeoutMs),
  };
}

export async function waitForAuthenticatedBackend(options: BackendReadinessOptions): Promise<void> {
  const timeoutMs = options.timeoutMs ?? BACKEND_READY_TIMEOUT_MS;
  const pollIntervalMs = options.pollIntervalMs ?? 180;
  const fetcher = options.fetcher ?? fetch;
  const sleep = options.sleep ?? delay;
  const now = options.now ?? Date.now;
  const startedAt = now();
  const healthUrl = `http://127.0.0.1:${options.port}/health`;

  while (now() - startedAt < timeoutMs) {
    const spawnError = options.getSpawnError();
    if (spawnError) throw spawnError;
    const exitCode = options.getExitCode();
    if (exitCode !== null) {
      throw new Error(`The Python backend exited during startup (exit code ${exitCode}).`);
    }

    try {
      const response = await fetcher(
        healthUrl,
        createAuthenticatedHealthRequest(options.sessionToken, options.requestTimeoutMs),
      );
      if (response.ok) {
        const health = await response.json() as { status?: string };
        if (health.status === 'ok') return;
      }
    } catch {
      // The process may still be importing OpenCV/ONNX Runtime; retry until timeout.
    }
    await sleep(pollIntervalMs);
  }
  throw new Error(`The Python backend did not become healthy within ${timeoutMs / 1_000} seconds.`);
}
