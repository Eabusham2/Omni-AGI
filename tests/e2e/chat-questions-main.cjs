const { app, BrowserWindow } = require("electron");
const path = require("node:path");
if (!process.env.OMNI_QUESTIONS_UI_DATA) throw new Error("An isolated synthetic questions profile is required.");
app.setPath("userData", path.resolve(process.env.OMNI_QUESTIONS_UI_DATA));
app.setName("Omni conversation question UI check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1060, height: 900, minWidth: 360, minHeight: 480,
    title: "Conversation questions · synthetic no-engine UI",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  await window.loadURL("http://127.0.0.1:5189/tests/e2e/chat-questions-fixture.html");
});
app.on("window-all-closed", () => app.quit());
