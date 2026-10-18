const { contextBridge, ipcRenderer } = require("electron");
if (process.isMainFrame) {
  contextBridge.exposeInMainWorld("omni", { mcp: { apps: {
    describe: () => Promise.resolve(undefined),
    open: (brainId, viewId) => ipcRenderer.invoke("fixture:app-open", brainId, viewId),
    request: request => ipcRenderer.invoke("fixture:app-request", request),
    close: (brainId, viewId, token) => ipcRenderer.invoke("fixture:app-close", brainId, viewId, token),
    onDisplayMode: listener => {
      const wrapped = (_event, value) => listener(value);
      ipcRenderer.on("fixture:app-display-mode", wrapped);
      return () => ipcRenderer.removeListener("fixture:app-display-mode", wrapped);
    },
    acknowledgeDisplayMode: request => ipcRenderer.invoke("fixture:app-display-ack", request)
  } } });
  contextBridge.exposeInMainWorld("mcpAppFixture", {
    descriptor: () => ipcRenderer.invoke("fixture:descriptor"),
    pluginPanel: () => ipcRenderer.invoke("fixture:plugin-panel"),
    trace: () => ipcRenderer.invoke("fixture:trace"),
    setAppAccess: enabled => ipcRenderer.invoke("fixture:app-access", enabled),
    onAppAccess: listener => {
      const wrapped = (_event, enabled) => listener(enabled);
      ipcRenderer.on("fixture:app-access-changed", wrapped);
      return () => ipcRenderer.removeListener("fixture:app-access-changed", wrapped);
    },
    acknowledgement: mode => ipcRenderer.invoke("fixture:display-ack-policy", mode),
    releaseAcknowledgements: () => ipcRenderer.invoke("fixture:display-ack-release")
  });
}
