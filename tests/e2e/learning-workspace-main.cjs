const { app, BrowserWindow } = require("electron");
const path = require("node:path");
app.setPath("userData", path.resolve(__dirname, "../../.runtime/learning-ui-profile"));
app.setName("Omni Learning UI Check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1100, height: 900, title: "Omni Learning — synthetic UI check",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  const fixture = process.env.OMNI_UI_FIXTURE === "mcp-review" ? "mcp-review-fixture.html" :
    process.env.OMNI_UI_FIXTURE === "automation-plugin" ? "automation-plugin-fixture.html" :
    process.env.OMNI_UI_FIXTURE === "clock" ? "clock-workspace-fixture.html" : "learning-workspace-fixture.html";
  await window.loadURL(`http://127.0.0.1:5192/tests/e2e/${fixture}`);
});
app.on("window-all-closed", () => app.quit());
