const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_CACHED_UI_DATA) throw new Error("An isolated synthetic UI profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_CACHED_UI_DATA));
app.setName("Omni cached computation UI check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1100, height: 900, minWidth: 360, minHeight: 480,
    title: "Cached computation · synthetic no-engine UI",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/cached-decoder-fixture.html");
});
app.on("window-all-closed", () => app.quit());
