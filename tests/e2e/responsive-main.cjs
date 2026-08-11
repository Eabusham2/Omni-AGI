const { app, BrowserWindow } = require("electron");
const path = require("node:path");

app.whenReady().then(async () => {
  const repository = process.env.OMNI_RESPONSIVE_REPOSITORY;
  if (!repository) throw new Error("OMNI_RESPONSIVE_REPOSITORY is required.");
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 360,
    minHeight: 480,
    show: false,
    webPreferences: {
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false
    }
  });
  await window.loadFile(path.join(repository, "out", "renderer", "index.html"));
  window.show();
});

app.on("window-all-closed", () => app.quit());
