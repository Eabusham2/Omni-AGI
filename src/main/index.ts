import { dirname, extname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, session } from "electron";
import { IPC } from "../shared/ipc";
import { BrainRepository, resolveBrainDataRoot } from "./brainRepository";
import { BrainService, RuntimeJobManager } from "./brainService";
import { EngineSupervisor } from "./engineSupervisor";
import { registerIpcHandlers } from "./ipc";
import { ToolExecutor } from "./toolExecutor";

const moduleDirectory = dirname(fileURLToPath(import.meta.url));
const developmentRendererUrl = process.env.ELECTRON_RENDERER_URL;
let mainWindow: BrowserWindow | undefined;
let disposeIpc: (() => void) | undefined;
let engine: EngineSupervisor | undefined;
let brainRepository: BrainRepository | undefined;
let quitAfterCleanup = false;
const pendingImports: string[] = [];

function queuedOmniPaths(argv: string[]): string[] {
  return argv
    .filter(
      (value) =>
        typeof value === "string" &&
        value.length <= 32_000 &&
        isAbsolute(value) &&
        extname(value).toLocaleLowerCase() === ".omni"
    )
    .map((path) => resolve(path))
    .slice(0, 16);
}

async function importQueuedBundles(paths: string[]): Promise<void> {
  if (!brainRepository || !mainWindow) {
    pendingImports.push(...paths);
    return;
  }
  for (const path of [...new Set(paths)]) {
    try {
      const brain = await brainRepository.importBundle(path);
      await engine?.tryRequest("unload", { brainId: brain.id }, 30_000);
      await engine?.tryRequest(
        "load",
        {
          brainId: brain.id,
          config: brain.config,
          storagePath: brainRepository.brainDirectory(brain.id)
        },
        300_000
      );
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send(IPC.brain.imported, brain);
        }
      }, 100);
    } catch (error) {
      console.error(`Failed to import ${path}:`, error);
    }
  }
}

function rendererOrigin(): string | undefined {
  if (!developmentRendererUrl) return undefined;
  try {
    return new URL(developmentRendererUrl).origin;
  } catch {
    return undefined;
  }
}

function installSecurityPolicy(): void {
  const allowedDevelopmentOrigin = rendererOrigin();
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false);
  });
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const developmentConnect = allowedDevelopmentOrigin
      ? ` ${allowedDevelopmentOrigin} ws://${new URL(allowedDevelopmentOrigin).host}`
      : "";
    const policy = [
      "default-src 'self'",
      "base-uri 'none'",
      "object-src 'none'",
      "frame-src 'none'",
      "form-action 'none'",
      allowedDevelopmentOrigin
        ? "script-src 'self' 'unsafe-inline'"
        : "script-src 'self'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "media-src 'self' data: blob:",
      "font-src 'self' data:",
      allowedDevelopmentOrigin
        ? "worker-src 'self' blob:"
        : "worker-src 'self'",
      `connect-src 'self'${developmentConnect}`
    ].join("; ");
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        "Content-Security-Policy": [policy],
        "X-Content-Type-Options": ["nosniff"],
        "Referrer-Policy": ["no-referrer"],
        "Cross-Origin-Opener-Policy": ["same-origin"]
      }
    });
  });
}

async function createWindow(): Promise<BrowserWindow> {
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1040,
    minHeight: 700,
    show: false,
    title: "Omni AGI Studio",
    backgroundColor: "#0b1018",
    autoHideMenuBar: true,
    ...(process.platform === "win32"
      ? {
          backgroundMaterial: "mica" as const,
          titleBarStyle: "hidden" as const,
          titleBarOverlay: {
            color: "#0b1018",
            symbolColor: "#e8edf7",
            height: 46
          }
        }
      : {}),
    webPreferences: {
      preload: join(moduleDirectory, "../preload/index.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      nodeIntegrationInWorker: false,
      nodeIntegrationInSubFrames: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      spellcheck: true,
      backgroundThrottling: false,
      safeDialogs: true,
      navigateOnDragDrop: false
    }
  });

  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event, target) => {
    let allowed = false;
    try {
      allowed = developmentRendererUrl
        ? new URL(target).origin === new URL(developmentRendererUrl).origin
        : new URL(target).protocol === "file:";
    } catch {
      allowed = false;
    }
    if (!allowed) event.preventDefault();
  });
  window.webContents.on("will-attach-webview", (event) => event.preventDefault());
  window.once("ready-to-show", () => window.show());
  window.on("closed", () => {
    if (mainWindow === window) mainWindow = undefined;
  });

  if (developmentRendererUrl) await window.loadURL(developmentRendererUrl);
  else await window.loadFile(join(moduleDirectory, "../renderer/index.html"));
  return window;
}

async function bootstrap(): Promise<void> {
  if (process.platform === "win32") app.setAppUserModelId("ai.omniagi.studio");
  installSecurityPolicy();
  const appPath = app.getAppPath();
  const repository = new BrainRepository(resolveBrainDataRoot(app.getPath("userData")));
  brainRepository = repository;
  await repository.initialize();
  engine = new EngineSupervisor({
    appPath,
    resourcesPath: process.resourcesPath
  });
  const service = new BrainService(repository, engine);
  const jobs = new RuntimeJobManager(service, engine);
  const tools = new ToolExecutor(service, jobs);
  disposeIpc = registerIpcHandlers({
    repository,
    service,
    jobs,
    engine,
    tools,
    appPath
  });
  mainWindow = await createWindow();
  void engine.start();
  const startupImports = [...pendingImports.splice(0), ...queuedOmniPaths(process.argv)];
  if (startupImports.length > 0) await importQueuedBundles(startupImports);
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", (_event, argv) => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
    void importQueuedBundles(queuedOmniPaths(argv));
  });

  app.on("open-file", (event, path) => {
    event.preventDefault();
    void importQueuedBundles(queuedOmniPaths([path]));
  });

  app.whenReady().then(bootstrap).catch((error) => {
    console.error("Failed to start Omni AGI Studio:", error);
    app.quit();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow().then((window) => {
        mainWindow = window;
      });
    }
  });

  app.on("window-all-closed", () => app.quit());
  app.on("before-quit", (event) => {
    if (quitAfterCleanup) return;
    event.preventDefault();
    quitAfterCleanup = true;
    disposeIpc?.();
    void (engine?.stop() ?? Promise.resolve()).finally(() => app.quit());
  });
}
