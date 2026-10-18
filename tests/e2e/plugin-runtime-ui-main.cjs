const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_PLUGIN_UI_DATA) throw new Error("An isolated UI profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_PLUGIN_UI_DATA));
app.setName("Omni Plugin UI Check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1100, height: 1000, title: "Omni plugins — synthetic UI only",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/plugin-runtime-ui-fixture.html");
});
app.on("window-all-closed", () => app.quit());
