const { app, BrowserWindow } = require("electron");
const path = require("node:path");
app.setPath("userData", process.env.OMNI_MERGE_UI_DATA);
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1200, height: 960, show: false, webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadFile(path.join(process.env.OMNI_MERGE_UI_REPO, ".runtime/full-merge-ui/tests/e2e/full-merge-fixture.html"));
  window.show();
});
app.on("window-all-closed", () => app.quit());
