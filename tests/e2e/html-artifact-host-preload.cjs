const { contextBridge, ipcRenderer } = require("electron");
if (process.isMainFrame) {
  contextBridge.exposeInMainWorld("omni", { artifacts: {
    open: request => ipcRenderer.invoke("fixture:artifact-open", request),
    activate: request => ipcRenderer.invoke("fixture:artifact-activate", request),
    request: request => ipcRenderer.invoke("fixture:artifact-request", request),
    review: request => ipcRenderer.invoke("fixture:artifact-review", request),
    close: (brainId, viewId, nonce) => ipcRenderer.invoke("fixture:artifact-close", brainId, viewId, nonce)
  } });
  contextBridge.exposeInMainWorld("htmlArtifactFixture", {
    descriptors: () => ipcRenderer.invoke("fixture:artifact-descriptors"),
    trace: () => ipcRenderer.invoke("fixture:artifact-trace")
  });
}
