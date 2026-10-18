const { app, BrowserWindow } = require("electron");
const path = require("node:path");

const profile = process.env.OMNI_CHAT_COMPOSER_UI_DATA;
if (!profile) throw new Error("OMNI_CHAT_COMPOSER_UI_DATA must name an isolated fixture profile.");
app.setPath("userData", path.resolve(profile));
app.setName("Omni composer and queue UI check");

app.whenReady().then(async () => {
  const url = process.env.OMNI_CHAT_COMPOSER_UI_URL ||
    "http://127.0.0.1:5189/tests/e2e/chat-composer-fixture.html";
  const window = new BrowserWindow({
    width: 1120,
    height: 920,
    minWidth: 360,
    minHeight: 480,
    title: "Composer and queue · synthetic callbacks only",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false }
  });
  await window.loadURL(url);
});

app.on("window-all-closed", () => app.quit());
