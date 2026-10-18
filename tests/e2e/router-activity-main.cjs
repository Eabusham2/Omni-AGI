const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_ROUTER_UI_DATA) throw new Error("An isolated synthetic UI profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_ROUTER_UI_DATA));
app.setName("Omni router activity UI check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1160, height: 960, minWidth: 360, minHeight: 480,
    title: "Router connections · synthetic no-engine UI",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/router-spike-fixture.html");
});
app.on("window-all-closed", () => app.quit());
