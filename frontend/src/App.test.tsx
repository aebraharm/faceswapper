// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { api } from './services/api';
import type { SourceFaceStatus, TransformerStatus } from './types/api';

const emptySource: SourceFaceStatus = {
  uploaded: false,
  face_count: 0,
  faces: [],
  selected_face_index: null,
  ready: false,
  model_ready: false,
  width: 0,
  height: 0,
  storage: 'memory only',
};

const oneFaceSource: SourceFaceStatus = {
  uploaded: true,
  face_count: 1,
  faces: [{ x: 40, y: 30, width: 70, height: 80 }],
  selected_face_index: 0,
  ready: true,
  model_ready: false,
  width: 160,
  height: 160,
  storage: 'memory only',
  format: 'PNG',
};

const twoFaceSource: SourceFaceStatus = {
  ...oneFaceSource,
  face_count: 2,
  faces: [
    { x: 20, y: 30, width: 45, height: 50 },
    { x: 90, y: 32, width: 46, height: 52 },
  ],
  selected_face_index: null,
  ready: false,
};

const noModel: TransformerStatus = {
  loaded: false,
  name: 'No model loaded',
  device: 'CPU',
  requires_model_files: true,
};

describe('source photo selection and preview', () => {
  let host: HTMLDivElement;
  let root: Root;
  let previewUrlNumber = 0;

  beforeEach(() => {
    Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', {
      configurable: true,
      writable: true,
      value: true,
    });
    vi.restoreAllMocks();
    previewUrlNumber = 0;
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => `blob:source-preview-${++previewUrlNumber}`),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    });
    vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', landmarks_available: false });
    vi.spyOn(api, 'sourceStatus').mockResolvedValue(emptySource);
    vi.spyOn(api, 'transformerStatus').mockResolvedValue(noModel);
    vi.spyOn(api, 'uploadSource').mockResolvedValue(oneFaceSource);
    vi.spyOn(api, 'selectSource').mockResolvedValue({ ...twoFaceSource, selected_face_index: 1, ready: true });
    vi.spyOn(api, 'removeSource').mockResolvedValue({ ...emptySource, ok: true });
    host = document.createElement('div');
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.restoreAllMocks();
  });

  async function renderApp(): Promise<void> {
    await act(async () => {
      root.render(<App />);
      await Promise.resolve();
    });
  }

  async function choose(file: File): Promise<void> {
    const input = host.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error('Source image input was not rendered.');
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }

  it('previews the local selection immediately, then clearly confirms successful face detection', async () => {
    let resolveUpload!: (status: SourceFaceStatus) => void;
    const pendingUpload = new Promise<SourceFaceStatus>((resolve) => { resolveUpload = resolve; });
    vi.mocked(api.uploadSource).mockReturnValue(pendingUpload);
    await renderApp();

    await choose(new File(['jpeg bytes'], 'portrait.jpg', { type: '' }));

    expect(host.querySelector<HTMLImageElement>('.source-image')?.src).toBe('blob:source-preview-1');
    expect(host.querySelector('.source-dropzone')?.getAttribute('aria-busy')).toBe('true');
    expect(host.textContent).toContain('Checking the selected photo for a face');
    expect(host.textContent).not.toContain('FACE READY');

    await act(async () => {
      resolveUpload(oneFaceSource);
      await pendingUpload;
    });

    expect(host.querySelector<HTMLImageElement>('.source-image')?.src).toBe('blob:source-preview-1');
    expect(host.textContent).toContain('FACE READY');
    expect(host.textContent).toContain('1 FACE DETECTED');
    expect(host.textContent).toContain('Source face detected and aligned in memory.');
  });

  it('keeps the selected preview and announces the backend no-face error', async () => {
    vi.mocked(api.uploadSource).mockRejectedValue(new Error('No face was found. Choose a clear, front-facing photo and try again.'));
    await renderApp();

    await choose(new File(['jpeg bytes'], 'portrait.jpg', { type: 'image/jpeg' }));
    await vi.waitFor(() => expect(host.querySelector('[role="alert"]')?.textContent).toContain('No face was found'));

    expect(host.querySelector<HTMLImageElement>('.source-image')?.src).toBe('blob:source-preview-1');
    expect(host.textContent).not.toContain('FACE READY');
    expect(host.querySelector('.image-caption')).toBeNull();
  });

  it('restores the previous source if a replacement upload fails', async () => {
    await renderApp();
    await choose(new File(['first photo'], 'first.jpg', { type: 'image/jpeg' }));
    await vi.waitFor(() => expect(host.textContent).toContain('Source face detected and aligned in memory.'));

    let rejectUpload!: (error: Error) => void;
    const pendingUpload = new Promise<SourceFaceStatus>((_, reject) => { rejectUpload = reject; });
    vi.mocked(api.uploadSource).mockReturnValueOnce(pendingUpload);
    await choose(new File(['replacement photo'], 'replacement.jpg', { type: 'image/jpeg' }));
    expect(host.querySelector<HTMLImageElement>('.source-image')?.src).toBe('blob:source-preview-2');

    await act(async () => {
      rejectUpload(new Error('No face was found. Choose a clear, front-facing photo and try again.'));
      await pendingUpload.catch(() => undefined);
    });

    expect(host.querySelector<HTMLImageElement>('.source-image')?.src).toBe('blob:source-preview-1');
    expect(host.textContent).toContain('The previous source photo remains active.');
    expect(host.textContent).toContain('FACE READY');
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:source-preview-2');
  });

  it('shows an actionable runtime error without dropping the local preview', async () => {
    vi.mocked(api.uploadSource).mockRejectedValue(new TypeError('Failed to fetch'));
    await renderApp();

    await choose(new File(['jpeg bytes'], 'portrait.jpg', { type: 'image/jpeg' }));
    await vi.waitFor(() => expect(host.querySelector('[role="alert"]')?.textContent).toContain('local AI backend'));

    expect(host.querySelector<HTMLImageElement>('.source-image')?.src).toBe('blob:source-preview-1');
  });

  it('announces unsupported selections and does not send them to the backend', async () => {
    await renderApp();

    await choose(new File(['unsupported'], 'portrait.heic', { type: 'image/heic' }));

    expect(host.querySelector('.source-image')).toBeNull();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('Choose a JPG, PNG, or WEBP image');
    expect(api.uploadSource).not.toHaveBeenCalled();
  });

  it('shows a local preview error if the renderer cannot create an object URL', async () => {
    vi.mocked(URL.createObjectURL).mockImplementation(() => { throw new Error('object URLs unavailable'); });
    await renderApp();

    await choose(new File(['jpeg bytes'], 'portrait.jpg', { type: 'image/jpeg' }));

    expect(host.querySelector('.source-image')).toBeNull();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('could not be opened for preview');
    expect(api.uploadSource).not.toHaveBeenCalled();
  });

  it('requires an explicit source selection when the upload contains multiple faces', async () => {
    vi.mocked(api.uploadSource).mockResolvedValue(twoFaceSource);
    await renderApp();

    await choose(new File(['jpeg bytes'], 'group.jpg', { type: 'image/jpeg' }));
    await vi.waitFor(() => expect(host.textContent).toContain('Found 2 faces. Choose the source identity below.'));

    expect(host.textContent).toContain('Face 1');
    expect(host.textContent).toContain('Face 2');
    expect(host.textContent).toContain('Choose one detected source face above before continuing.');
    expect(host.textContent).not.toContain('FACE READY');

    const faceTwo = Array.from(host.querySelectorAll<HTMLButtonElement>('.face-pill'))
      .find((button) => button.textContent?.includes('Face 2'));
    expect(faceTwo).toBeDefined();
    await act(async () => {
      faceTwo?.click();
      await Promise.resolve();
    });
    expect(api.selectSource).toHaveBeenCalledWith(1);
    expect(host.textContent).toContain('FACE READY');
  });
});
