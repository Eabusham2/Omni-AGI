const { app, BrowserWindow } = require("electron");
const fs = require("node:fs/promises");
const path = require("node:path");
const profile = process.env.OMNI_MCP_TASK_UI_DATA;
if (!profile) throw new Error("OMNI_MCP_TASK_UI_DATA must name an isolated inert fixture profile.");
app.setPath("userData", path.resolve(profile)); app.setName("Omni MCP task UI check");
app.whenReady().then(async () => {
  const window = new BrowserWindow({ width: 1100, height: 860, minWidth: 360, minHeight: 480, title: "MCP tasks · inert no-network UI",
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false } });
  let capture = 0;
  window.webContents.on("before-input-event", (event, input) => {
    const screenshots = process.env.OMNI_MCP_TASK_UI_SCREENSHOTS;
    if (!screenshots || input.type !== "keyDown" || !input.meta || !input.shift || input.key.toLowerCase() !== "s") return;
    event.preventDefault(); void (async () => { await fs.mkdir(screenshots, { recursive: true }); const file = path.join(screenshots, `${String(++capture).padStart(2, "0")}-capture.png`);
      await fs.writeFile(file, (await window.webContents.capturePage()).toPNG()); console.log(`MCP_TASK_UI_SCREENSHOT=${file}`); })().catch(error => console.error("Inert task fixture screenshot failed", error));
  });
  await window.loadURL(process.env.OMNI_MCP_TASK_UI_URL || "http://127.0.0.1:5189/tests/e2e/mcp-tasks-fixture.html");
});
app.on("window-all-closed", () => app.quit());
