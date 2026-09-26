const { app, BrowserWindow } = require("electron");
const path = require("node:path");

const isolatedUserDataDirectory = process.env.OMNI_RESPONSIVE_USER_DATA_DIR;
if (isolatedUserDataDirectory) {
  app.setPath("userData", path.resolve(isolatedUserDataDirectory));
}

app.whenReady().then(async () => {
  const repository = process.env.OMNI_RESPONSIVE_REPOSITORY;
  if (!repository) throw new Error("OMNI_RESPONSIVE_REPOSITORY is required.");
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 360,
    minHeight: 480,
    show: false,
    ...(process.platform === "darwin"
      ? {
          titleBarStyle: "hiddenInset",
          trafficLightPosition: { x: 14, y: 15 }
        }
      : {}),
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
