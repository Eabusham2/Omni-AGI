const { app, BrowserWindow } = require("electron");
const path = require("node:path");

const profile = process.env.OMNI_REGIONAL_UI_DATA;
if (!profile) throw new Error("OMNI_REGIONAL_UI_DATA must name an isolated fixture profile.");
app.setPath("userData", path.resolve(profile));
app.setName("Omni regional execution UI check");

app.whenReady().then(async () => {
  const url = process.env.OMNI_REGIONAL_UI_URL ||
    "http://127.0.0.1:5189/tests/e2e/regional-execution-fixture.html";
  const window = new BrowserWindow({ width: 1160, height: 960, minWidth: 360, minHeight: 480,
    title: "Regional execution · synthetic no-engine UI",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL(url);
});

app.on("window-all-closed", () => app.quit());
