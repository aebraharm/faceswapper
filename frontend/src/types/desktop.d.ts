export interface DesktopRuntimeConfig {
  backendUrl: string;
  websocketUrl: string;
  sessionToken: string;
}

export interface DesktopPreferences {
  provider: 'auto' | 'CPUExecutionProvider' | 'CUDAExecutionProvider';
  intensity: number;
  resolution: 320 | 480 | 640 | 720;
}

export interface FrameDesktopBridge {
  getRuntimeConfig(): Promise<DesktopRuntimeConfig>;
  loadSettings(): Promise<DesktopPreferences>;
  saveSettings(settings: DesktopPreferences): Promise<DesktopPreferences>;
  authorizeCamera(): Promise<boolean>;
  onBackendError(callback: (message: string) => void): () => void;
}

declare global {
  interface Window {
    frameDesktop?: FrameDesktopBridge;
  }
}
