const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_OPERATION_UI_DATA) throw new Error("An isolated synthetic UI profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_OPERATION_UI_DATA));
app.setName("Omni native operation UI check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1050, height: 900, minWidth: 360, minHeight: 480,
    title: "Native operation controls · synthetic no-engine UI", webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/native-operation-ui-fixture.html");
});
app.on("window-all-closed", () => app.quit());
