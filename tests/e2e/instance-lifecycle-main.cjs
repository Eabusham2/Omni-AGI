const { app, BrowserWindow } = require("electron");
const path = require("node:path");
app.setPath("userData", process.env.OMNI_LIFECYCLE_UI_DATA);
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1100, height: 850, show: false, webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadFile(path.join(process.env.OMNI_LIFECYCLE_UI_REPO, ".runtime/instance-lifecycle-ui/tests/e2e/instance-lifecycle-fixture.html"));
  window.show();
});
app.on("window-all-closed", () => app.quit());
