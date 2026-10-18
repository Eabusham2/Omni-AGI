import { contextBridge, ipcRenderer } from "electron";

// Kept literal so this sandbox preload is one separately bundled entry. No
// Studio preload, Node object, arbitrary channel, event object or secret leaks.
const requestChannel = "omni:plugin-runtime:request";
const deliveryChannel = "omni:plugin-runtime:delivery";
const limitArgument = process.argv.find(value => value.startsWith("--omni-plugin-runtime-max-frame="));
const maximum = Number(limitArgument?.split("=", 2)[1]);
contextBridge.exposeInMainWorld("__omniRuntimeTransport", Object.freeze({
  request: (message: unknown): Promise<unknown> => {
    if (!Number.isSafeInteger(maximum) || maximum < 4096) return Promise.reject(new Error("Runtime transfer admission was not configured."));
    let serialized: string; try { serialized = JSON.stringify(message); } catch { return Promise.reject(new Error("Runtime message is not structured JSON.")); }
    if (typeof serialized !== "string" || Buffer.byteLength(serialized, "utf8") > maximum) return Promise.reject(new Error("Runtime message exceeds its admitted transfer; use pages."));
    return ipcRenderer.invoke(requestChannel, JSON.parse(serialized));
  },
  subscribe: (listener: (message: unknown) => void): (() => void) => {
    if (typeof listener !== "function") throw new Error("Runtime listener must be a function.");
    const receive = (_event: Electron.IpcRendererEvent, message: unknown): void => { listener(message); };
    ipcRenderer.on(deliveryChannel, receive); return () => { ipcRenderer.removeListener(deliveryChannel, receive); };
  }
}));
