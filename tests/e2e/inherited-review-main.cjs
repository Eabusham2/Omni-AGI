const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_REVIEW_UI_DATA) throw new Error("An isolated synthetic review profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_REVIEW_UI_DATA));
app.setName("Omni saved file review UI check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1000, height: 850, minWidth: 360, minHeight: 480,
    title: "Saved file review · synthetic no-file UI", webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/inherited-review-fixture.html");
});
app.on("window-all-closed", () => app.quit());
