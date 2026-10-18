const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_DESKTOP_UI_DATA) throw new Error("An isolated interface audit profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_DESKTOP_UI_DATA));
app.setName("Omni current interface audit");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1180, height: 850, minWidth: 360, minHeight: 480,
    title: "Omni current interface · no engine", webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/desktop-current-fixture.html");
});
app.on("window-all-closed", () => app.quit());
