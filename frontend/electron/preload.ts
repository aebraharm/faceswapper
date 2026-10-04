import { contextBridge, ipcRenderer } from 'electron';
import type { DesktopRuntimeConfig, FrameDesktopBridge } from '../src/types/desktop';

const frameDesktop: FrameDesktopBridge = {
  getRuntimeConfig: () => ipcRenderer.invoke('desktop:get-runtime-config') as Promise<DesktopRuntimeConfig>,
  authorizeCamera: () => ipcRenderer.invoke('desktop:authorize-camera') as Promise<boolean>,
  onBackendError: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, message: string) => callback(message);
    ipcRenderer.on('desktop:backend-error', listener);
    return () => ipcRenderer.removeListener('desktop:backend-error', listener);
  },
};

contextBridge.exposeInMainWorld('frameDesktop', frameDesktop);
