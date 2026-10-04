const MIME_BY_EXTENSION: Record<string, string> = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
};

const MIME_ALIASES: Record<string, string> = {
  'image/jpg': 'image/jpeg',
  'image/pjpeg': 'image/jpeg',
  'image/x-png': 'image/png',
  'image/x-webp': 'image/webp',
};

const SUPPORTED_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

/**
 * Windows/Electron can provide an empty or legacy File.type for a supported
 * extension. Normalize common aliases and fall back to the filename extension;
 * the backend still verifies the actual bytes against this MIME type.
 */
export function getSourceImageContentType(file: Pick<File, 'name' | 'type'>): string | null {
  const declaredType = file.type.trim().toLowerCase().split(';', 1)[0] ?? '';
  const normalizedType = MIME_ALIASES[declaredType] ?? declaredType;
  if (SUPPORTED_MIME_TYPES.has(normalizedType)) return normalizedType;

  const extension = file.name.toLowerCase().split('.').pop() ?? '';
  return MIME_BY_EXTENSION[extension] ?? null;
}
