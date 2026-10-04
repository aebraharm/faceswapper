export interface DesktopRuntimeConfig {
  backendUrl: string;
  websocketUrl: string;
  sessionToken: string;
}

export interface FrameDesktopBridge {
  getRuntimeConfig(): Promise<DesktopRuntimeConfig>;
  authorizeCamera(): Promise<boolean>;
  onBackendError(callback: (message: string) => void): () => void;
}

declare global {
  interface Window {
    frameDesktop?: FrameDesktopBridge;
  }
}
