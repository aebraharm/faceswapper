import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { DesktopPreferences } from '../src/types/desktop';

export const DEFAULT_DESKTOP_PREFERENCES: DesktopPreferences = {
  provider: 'auto',
  intensity: 0.85,
  resolution: 640,
};

const VALID_PROVIDERS = new Set(['auto', 'CPUExecutionProvider', 'CUDAExecutionProvider']);
const VALID_RESOLUTIONS = new Set([320, 480, 640, 720]);

export function validateDesktopPreferences(value: unknown): DesktopPreferences {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Desktop preferences must be an object.');
  }
  const candidate = value as Record<string, unknown>;
  if (typeof candidate.provider !== 'string' || !VALID_PROVIDERS.has(candidate.provider)) {
    throw new Error('The inference provider preference is invalid.');
  }
  if (typeof candidate.intensity !== 'number' || !Number.isFinite(candidate.intensity)
    || candidate.intensity < 0 || candidate.intensity > 1) {
    throw new Error('The transformation intensity preference must be between 0 and 1.');
  }
  if (typeof candidate.resolution !== 'number' || !VALID_RESOLUTIONS.has(candidate.resolution)) {
    throw new Error('The processing resolution preference is invalid.');
  }

  // Whitelist only display preferences; source images, model paths and session tokens
  // are deliberately never accepted by the persistence layer.
  return {
    provider: candidate.provider as DesktopPreferences['provider'],
    intensity: candidate.intensity,
    resolution: candidate.resolution as DesktopPreferences['resolution'],
  };
}

export async function loadDesktopPreferences(userDataDirectory: string): Promise<DesktopPreferences> {
  let contents: string;
  try {
    contents = await readFile(path.join(userDataDirectory, 'settings.json'), 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { ...DEFAULT_DESKTOP_PREFERENCES };
    throw error;
  }

  try {
    return validateDesktopPreferences(JSON.parse(contents));
  } catch {
    return { ...DEFAULT_DESKTOP_PREFERENCES };
  }
}

export async function saveDesktopPreferences(
  userDataDirectory: string,
  value: unknown,
): Promise<DesktopPreferences> {
  const preferences = validateDesktopPreferences(value);
  await mkdir(userDataDirectory, { recursive: true });
  const destination = path.join(userDataDirectory, 'settings.json');
  const temporary = path.join(userDataDirectory, `.settings-${randomUUID()}.tmp`);
  try {
    await writeFile(temporary, `${JSON.stringify(preferences, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
    await rename(temporary, destination);
  } catch (error) {
    await unlink(temporary).catch(() => undefined);
    throw error;
  }
  return preferences;
}
