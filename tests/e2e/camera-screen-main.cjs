const { app, BrowserWindow, session } = require("electron");
const { mkdtempSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
app.setPath("userData", mkdtempSync(join(tmpdir(), "omni-call-preview-ui-")));
app.whenReady().then(async () => {
  // The fixture substitutes canvas streams; actual device requests remain denied.
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, reply) => reply(false));
  const window = new BrowserWindow({ width: 1100, height: 900, title: "Call camera/screen — synthetic UI check",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5192/tests/e2e/camera-screen-fixture.html");
});
app.on("window-all-closed", () => app.quit());
