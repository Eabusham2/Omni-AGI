import { basename, dirname, extname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  app,
  BrowserWindow,
  desktopCapturer,
  dialog,
  nativeTheme,
  net,
  protocol,
  safeStorage,
  session
} from "electron";
import { IPC } from "../shared/ipc";
import { BrainRepository, resolveBrainDataRoot } from "./brainRepository";
import { BrainService, RuntimeJobManager } from "./brainService";
import { EngineSupervisor } from "./engineSupervisor";
import { VideoRuntimeProvisioner } from "./videoRuntimeProvisioner";
import {
  initialLearningBuildProgressEvent,
  registerIpcHandlers
} from "./ipc";
import { ToolExecutor } from "./toolExecutor";
import { ChatActionController } from "./chatActionController";
import { EvolutionController } from "./evolutionController";
import { IdleCognitionScheduler } from "./idleCognitionScheduler";
import {
  BackgroundRuntimeController,
  BackgroundRuntimePreferenceStore
} from "./backgroundRuntimeController";
import { ElectronBackgroundTray } from "./electronBackgroundTray";
import { ElectronSourceRuntimeLifecycle } from "./sourceRuntimeLifecycle";
import {
  allowTrustedDisplayMedia,
  allowTrustedStudioMedia
} from "./mediaPermissionPolicy";
import { ResourcePlanner } from "./resourcePlanner";
import { SharedResourceRegistry } from "./sharedResourceRegistry";
import { createNativeComputeMeasurementCollector } from "./nativeComputeMeasurement";
import { BuildResourceSelectionStore } from "./buildResourceSelections";
import { BuildInitializationCoordinator } from "./buildInitializationCoordinator";
import { MobileGateway } from "./mobileGateway";
import { SecureSecretStore } from "./secureSecretStore";
import { ApiTeacherTrainingService } from "./teacherTraining";
import { McpClientService } from "./mcpClient";
import { ToolPreferencesStore } from "./toolPreferences";
import { MediaArtifactRegistry } from "./mediaArtifactRegistry";
import {
  authorizedNativeMediaResponse,
  OMNI_MEDIA_SCHEME,
  OMNI_MEDIA_SCHEME_PRIVILEGES
} from "./mediaProtocol";

// A detached development/packaged window may outlive the terminal that
// launched it. Background diagnostics must not crash the main process when
// that terminal's stdout/stderr pipe closes. Action failures are still kept in
// the app's durable job/trace state; only the broken console sink is ignored.
for (const stream of [process.stdout, process.stderr]) {
  stream?.on("error", (error: NodeJS.ErrnoException) => {
    if (error.code !== "EPIPE") throw error;
  });
}

protocol.registerSchemesAsPrivileged([
  {
    scheme: OMNI_MEDIA_SCHEME,
    privileges: OMNI_MEDIA_SCHEME_PRIVILEGES
  }
]);

const moduleDirectory = dirname(fileURLToPath(import.meta.url));
const developmentRendererUrl = process.env.ELECTRON_RENDERER_URL;
let mainWindow: BrowserWindow | undefined;
let disposeIpc: (() => void) | undefined;
let engine: EngineSupervisor | undefined;
let brainRepository: BrainRepository | undefined;
let brainService: BrainService | undefined;
let sharedResources: SharedResourceRegistry | undefined;
let idleCognition: IdleCognitionScheduler | undefined;
let backgroundRuntime: BackgroundRuntimeController | undefined;
let mobileGateway: MobileGateway | undefined;
let mcpClient: McpClientService | undefined;
let mediaArtifacts: MediaArtifactRegistry | undefined;
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
      await brainService?.preflightStart(brain.id);
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
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send(IPC.brain.importFailed, {
          fileName: basename(path),
          message: error instanceof Error
            ? error.message.slice(0, 500)
            : "The Omni brain bundle could not be imported."
        });
      }
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

function installSecurityPolicy(registry: MediaArtifactRegistry): void {
  const allowedDevelopmentOrigin = rendererOrigin();
  session.defaultSession.protocol.handle(OMNI_MEDIA_SCHEME, async (request) => {
    try {
      const artifact = await registry.authorize(request.url);
      if (!artifact) {
        return new Response("Unknown or expired media capability.", {
          status: 404,
          headers: { "Content-Type": "text/plain", "Cache-Control": "no-store" }
        });
      }
      return authorizedNativeMediaResponse(
        request,
        artifact,
        (url, init) => net.fetch(url, init)
      );
    } catch {
      return new Response("Media capability failed integrity verification.", {
        status: 410,
        headers: { "Content-Type": "text/plain", "Cache-Control": "no-store" }
      });
    }
  });
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback, details) => {
    const mediaTypes = "mediaTypes" in details ? details.mediaTypes ?? [] : [];
    callback(
      allowTrustedStudioMedia({
        trustedWebContentsId: mainWindow?.webContents.id,
        requestingWebContentsId: webContents.id,
        permission,
        isMainFrame: details.isMainFrame,
        requestingUrl: details.requestingUrl,
        currentRendererUrl: mainWindow?.webContents.getURL(),
        mediaTypes
      })
    );
  });
  session.defaultSession.setPermissionCheckHandler((webContents, permission, _origin, details) =>
    allowTrustedStudioMedia({
      trustedWebContentsId: mainWindow?.webContents.id,
      requestingWebContentsId: webContents?.id,
      permission,
      isMainFrame: details.isMainFrame,
      requestingUrl: details.requestingUrl,
      currentRendererUrl: mainWindow?.webContents.getURL(),
      mediaTypes: details.mediaType ? [details.mediaType] : []
    })
  );
  session.defaultSession.setDisplayMediaRequestHandler(
    async (request, callback) => {
      const trusted = allowTrustedDisplayMedia({
        userGesture: request.userGesture,
        isMainFrame: request.frame?.top === request.frame,
        requestingUrl: request.frame?.url,
        currentRendererUrl: mainWindow?.webContents.getURL()
      });
      if (!trusted || !mainWindow || mainWindow.isDestroyed()) {
        callback({});
        return;
      }
      try {
        const sources = await desktopCapturer.getSources({
          types: ["screen"],
          thumbnailSize: { width: 0, height: 0 },
          fetchWindowIcons: false
        });
        if (sources.length === 0) {
          callback({});
          return;
        }
        const cancelId = sources.length;
        const choice = await dialog.showMessageBox(mainWindow, {
          type: "question",
          title: "Share a display with this brain",
          message: "Choose the display for visible, cancellable Live Perception.",
          detail:
            "Frames are bounded and enter the same neural observation path. Capture stops when you press Stop.",
          buttons: [...sources.map((source) => source.name), "Cancel"],
          defaultId: 0,
          cancelId,
          noLink: true
        });
        const selected = sources[choice.response];
        callback(selected ? { video: selected } : {});
      } catch {
        callback({});
      }
    },
    { useSystemPicker: true }
  );
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
      "img-src 'self' data: blob: omni-media:",
      "media-src 'self' data: blob: omni-media:",
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

async function reviewManagedBetaBrains(repository: BrainRepository): Promise<void> {
  if (await repository.betaReviewComplete()) return;
  const candidates = await repository.enumerateManagedBetaBrains();
  if (candidates.length === 0) {
    await repository.completeBetaReview("none", []);
    return;
  }
  const visible = candidates.slice(0, 20);
  const directoryList = visible
    .map((candidate) => `• ${candidate.name}\n  ${candidate.path}`)
    .join("\n");
  const hidden =
    candidates.length > visible.length
      ? `\n• …and ${candidates.length - visible.length} more app-managed beta directories.`
      : "";
  const decision = await dialog.showMessageBox({
    type: "warning",
    title: "Stable v1 found incompatible beta brains",
    message: `${candidates.length} app-managed beta brain${candidates.length === 1 ? "" : "s"} cannot be opened by stable v1.`,
    detail:
      "Keep them on disk, or permanently delete only the exact directories shown below. Omni AGI Studio does not search for or delete external .omni files.\n\n" +
      directoryList +
      hidden,
    buttons: ["Keep beta data", "Permanently delete beta data…"],
    defaultId: 0,
    cancelId: 0,
    noLink: true
  });
  if (decision.response !== 1) {
    await repository.completeBetaReview(
      "kept",
      candidates.map((candidate) => candidate.id)
    );
    return;
  }
  const confirmation = await dialog.showMessageBox({
    type: "warning",
    title: "Permanently delete beta brains?",
    message: "This cannot be undone.",
    detail:
      `Delete ${candidates.length} exact app-managed beta director${candidates.length === 1 ? "y" : "ies"} and all neural checkpoints stored inside? External files are not touched.`,
    buttons: ["Cancel", "Delete permanently"],
    defaultId: 0,
    cancelId: 0,
    noLink: true
  });
  if (confirmation.response !== 1) {
    await repository.completeBetaReview(
      "kept",
      candidates.map((candidate) => candidate.id)
    );
    return;
  }
  const deleted = await repository.deleteManagedBetaBrains(
    candidates.map((candidate) => candidate.id),
    true
  );
  await repository.completeBetaReview("deleted", deleted);
}

async function createWindow(): Promise<BrowserWindow> {
  const nativeDark = nativeTheme.shouldUseDarkColors;
  const nativeBackground = nativeDark ? "#08080d" : "#f4f3f0";
  const nativeSymbols = nativeDark ? "#f5f3fa" : "#1d1c23";
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    // Keep the desktop host usable on compact Windows tablets, split-screen
    // layouts, and mobile-sized development shells. Responsive renderer
    // navigation remains reachable down to this supported floor.
    minWidth: 360,
    minHeight: 480,
    show: false,
    title: "Omni AGI Studio",
    backgroundColor: nativeBackground,
    autoHideMenuBar: true,
    ...(process.platform === "darwin"
      ? {
          titleBarStyle: "hiddenInset" as const,
          trafficLightPosition: { x: 14, y: 15 }
        }
      : process.platform === "win32"
        ? {
          backgroundMaterial: "mica" as const,
          titleBarStyle: "hidden" as const,
          titleBarOverlay: {
            color: nativeBackground,
            symbolColor: nativeSymbols,
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

async function showOrCreateMainWindow(): Promise<void> {
  if (!mainWindow || mainWindow.isDestroyed()) {
    mainWindow = await createWindow();
  } else {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  }
  backgroundRuntime?.windowOpened();
}

async function bootstrap(): Promise<void> {
  if (process.platform === "win32") app.setAppUserModelId("ai.omniagi.studio");
  const appPath = app.getAppPath();
  sharedResources = new SharedResourceRegistry(
    join(app.getPath("userData"), "shared-resources.sqlite3")
  );
  const registry = sharedResources;
  const repository = new BrainRepository(resolveBrainDataRoot(app.getPath("userData")), {
    saved: ({ brainId, storagePoolBytes }) => registry.observeSaved(brainId, storagePoolBytes),
    removed: (brainId) => registry.observeRemoved(brainId)
  });
  brainRepository = repository;
  await repository.initialize();
  mediaArtifacts = new MediaArtifactRegistry((brainId) =>
    repository.brainDirectory(brainId)
  );
  installSecurityPolicy(mediaArtifacts);
  await reviewManagedBetaBrains(repository);
  // Older releases treated every identity as idle-enabled. Collapse that
  // legacy state before any worker or scheduler can acquire a brain.
  await repository.reconcileActiveModeLease();
  // Register inactive saved identities too. The shared limit is MAX(pool),
  // never one independent pool multiplied by the number of instances.
  const savedBrains = await repository.list();
  registry.reconcileSavedOwners(savedBrains.map((summary) => summary.id));
  for (const summary of savedBrains) {
    const saved = await repository.get(summary.id, false);
    registry.observeSaved(saved.id, saved.config.storagePoolBytes);
  }
  registry.requireSynchronized();
  const videoRuntimeCacheRoot = join(app.getPath("userData"), "video-runtime");
  engine = new EngineSupervisor({
    appPath,
    resourcesPath: process.resourcesPath,
    sharedResourceLedgerPath: registry.ledgerPath,
    videoRuntimeCacheRoot,
    prepareVideoRuntime: (signal, onProgress) => new VideoRuntimeProvisioner({
      cacheRoot: videoRuntimeCacheRoot,
      onProgress
    }).prepare(signal)
  });
  engine.on("diagnostic", (diagnostic: unknown) => {
    console.error("Neural worker diagnostic:", String(diagnostic));
  });
  const resourcePlanner = new ResourcePlanner(repository.root, {
    measurementCollector: createNativeComputeMeasurementCollector(engine)
  });
  const service = new BrainService(
    repository,
    engine,
    resourcePlanner,
    mediaArtifacts,
    registry
  );
  brainService = service;
  // Startup repeats the same physical preflight as Build. A mind that no
  // longer fits remains intact and visible, but no neural worker is started
  // for it until storage/RAM reserve requirements are restored.
  for (const summary of await repository.list()) {
    // A neural turn is committed atomically by the worker before its desktop
    // presentation document is saved. Recover that exact receipt first so a
    // stop/restart in the tiny gap between those commits cannot hide a reply.
    await service.getReconciledBrain(summary.id).catch((error: unknown) => {
      console.warn(
        `Committed chat reconciliation skipped for ${summary.id}:`,
        error instanceof Error ? error.message : error
      );
    });
    await service.preflightStart(summary.id, { selectActiveRuntime: false }).catch((error: unknown) => {
      console.warn(
        `Resource preflight blocked ${summary.id}:`,
        error instanceof Error ? error.message : error
      );
    });
    // Ownership is selected before either optional durable queue resumes.
    // Dormant imports/copies remain saved data, never a startup neural owner.
    if (summary.activeMode) service.selectCompletedActionLearningOwner(summary.id);
  }
  const jobs = new RuntimeJobManager(service, engine, mediaArtifacts);
  const buildSelections = new BuildResourceSelectionStore(
    join(app.getPath("userData"), "pending-build-resources.json")
  );
  const initialization = new BuildInitializationCoordinator(
    repository,
    service,
    jobs,
    buildSelections
  );
  const sourceRuntime = new ElectronSourceRuntimeLifecycle({
    app,
    userDataPath: app.getPath("userData"),
    resourcesPath: process.resourcesPath
  });
  const integrationSecrets = new SecureSecretStore(
    join(app.getPath("userData"), "integration-secrets.json"),
    {
      available: () =>
        safeStorage.isEncryptionAvailable() &&
        (process.platform !== "linux" || safeStorage.getSelectedStorageBackend() !== "basic_text"),
      encrypt: (value) => safeStorage.encryptString(value),
      decrypt: (value) => safeStorage.decryptString(value)
    }
  );
  const toolPreferences = new ToolPreferencesStore(
    join(app.getPath("userData"), "tool-preferences.json")
  );
  await toolPreferences.initialize();
  const teacher = new ApiTeacherTrainingService(service, integrationSecrets);
  const mcp = new McpClientService(
    join(app.getPath("userData"), "mcp-servers.json"),
    integrationSecrets,
    service
  );
  await mcp.initialize();
  mcpClient = mcp;
  const tools = new ToolExecutor(
    service,
    jobs,
    sourceRuntime,
    undefined,
    mcp,
    toolPreferences
  );
  const evolution = new EvolutionController(repository, tools, engine);
  const actions = new ChatActionController(service, tools, evolution);
  tools.setAgentChatRunner(async (brainId, objective, signal, turnId) => {
    const cancel = (): void => { actions.cancel(brainId, turnId); tools.cancel(brainId, turnId); };
    signal.addEventListener("abort", cancel, { once: true });
    try { return await actions.send(brainId, objective, signal, turnId); }
    finally { signal.removeEventListener("abort", cancel); }
  });
  mobileGateway = new MobileGateway(
    repository,
    service,
    actions,
    (brainId) => jobs.isInitializing(brainId)
  );
  idleCognition = new IdleCognitionScheduler(repository, actions, {
    intervalMs: 12_000,
    startupDelayMs: 30_000,
    minimumIdleSeconds: 6,
    maxDutyCycle: 0.12,
    isLearning: (brainId) =>
      jobs.isLearning(brainId) || service.isChatParameterLearningActive(brainId),
    isInitializationBusy: () => initialization.isBusy(),
    preemptBackground: () => engine!.claimForeground(),
    onError: (error) => console.error("Idle cognition cycle failed:", error)
  });
  evolution.setRecursiveReassessmentHandler(async (event) => {
    // Observe a completed experiment, then offer the ordinary idle scheduler
    // its next opportunity. The neural action head chooses new work or none;
    // no behavioral prompt or repeating architecture command is supplied.
    await service.learnStructuredExperience(event.brainId, {
      content: JSON.stringify(event), name: `Observed improvement ${event.parentCandidateId}`,
      sourceLabel: "completed own improvement experience", license: "Locally observed experiment"
    });
    await idleCognition?.tick();
  });
  actions.on("neural-cancelled", ({ brainId }: { brainId: string }) => {
    idleCognition?.reserveForegroundAfterCancel(brainId);
  });
  disposeIpc = registerIpcHandlers({
    repository,
    service,
    jobs,
    engine,
    tools,
    actions,
    evolution,
    buildSelections,
    initialization,
    mobile: mobileGateway,
    teacher,
    mcp,
    idleCognition,
    collectHardwareCompute: (profile) => resourcePlanner.measureNativeCompute({
      device: "cpu",
      ramBudgetBytes: profile.availableMemoryBytes,
      hardwareTier: profile.recommendedTier
    }),
    appPath
  });
  mainWindow = await createWindow();
  backgroundRuntime = new BackgroundRuntimeController({
    store: new BackgroundRuntimePreferenceStore(
      join(app.getPath("userData"), "background-runtime.json")
    ),
    scheduler: idleCognition,
    tray: new ElectronBackgroundTray(join(appPath, "build", "icon.svg")),
    hasVisibleWindows: () =>
      Boolean(mainWindow && !mainWindow.isDestroyed()),
    openStudio: showOrCreateMainWindow,
    requestAppQuit: () => app.quit(),
    resourceStatus: () => {
      const activeLearning = jobs.list().filter(
        (job) =>
          ["queued", "running", "cancelling"].includes(job.state) &&
          ["training", "ingestion", "crawl", "image", "audio", "video"].includes(
            job.kind
          )
      ).length;
      return activeLearning > 0
        ? `${activeLearning} foreground learning job${activeLearning === 1 ? "" : "s"} preempt optional cognition.`
        : "Idle neural work is capped at 12% duty; foreground work preempts it.";
    },
    permissionStatus: () =>
      "Organic external actions remain visible and use each mind's saved tool permissions."
  });
  await backgroundRuntime.initialize();
  const recoveryWindow = mainWindow;
  const recoverySequences = new Map<string, number>();
  const nextRecoverySequence = (brainId: string, candidate?: number): number => {
    const current = recoverySequences.get(brainId) ?? 0;
    const next = Math.max(current, candidate ?? current);
    recoverySequences.set(brainId, next + 1);
    return next;
  };
  void engine.start()
    .then(() => initialization.recoverAll(
      (brainId, error) => {
        console.error(
          `Initialization recovery paused for ${brainId}:`,
          error instanceof Error ? error.message : error
        );
      },
      (brainId, engineEvent) => {
        if (recoveryWindow.isDestroyed() || engineEvent.type !== "build-progress") return;
        recoveryWindow.webContents.send(IPC.brain.buildEvent, {
          brainId,
          streamId: engineEvent.streamId,
          sequence: nextRecoverySequence(brainId, engineEvent.sequence),
          phase:
            typeof (engineEvent.data as { phase?: unknown } | undefined)?.phase === "string"
              ? (engineEvent.data as { phase: string }).phase
              : "allocating",
          progress: engineEvent.progress ?? 0,
          label: engineEvent.message ?? "Recovering OmniCortex native core",
          data: (engineEvent.data as { metrics?: unknown } | undefined)?.metrics
        });
      },
      (brainId, progressEvent) => {
        if (recoveryWindow.isDestroyed()) return;
        recoveryWindow.webContents.send(
          IPC.brain.buildEvent,
          initialLearningBuildProgressEvent(
            progressEvent,
            nextRecoverySequence(brainId)
          )
        );
      }
    ))
    .catch((error: unknown) => {
      console.error(
        "Neural worker startup or initialization recovery failed:",
        error instanceof Error ? error.message : error
      );
    });
  const startupImports = [...pendingImports.splice(0), ...queuedOmniPaths(process.argv)];
  if (startupImports.length > 0) await importQueuedBundles(startupImports);
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", (_event, argv) => {
    void showOrCreateMainWindow().then(() =>
      importQueuedBundles(queuedOmniPaths(argv))
    );
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
      void showOrCreateMainWindow();
    }
  });

  app.on("window-all-closed", () => {
    if (backgroundRuntime?.handleLastWindowClosed()) return;
    app.quit();
  });
  app.on("will-quit", () => {
    // Keep handlers registered throughout asynchronous engine cleanup. The
    // renderer can still finish an in-flight health poll until its window has
    // closed; removing handlers in before-quit creates a noisy shutdown race.
    disposeIpc?.();
    disposeIpc = undefined;
  });
  app.on("before-quit", (event) => {
    if (quitAfterCleanup) return;
    event.preventDefault();
    quitAfterCleanup = true;
    backgroundRuntime?.dispose();
    idleCognition?.stop();
    mcpClient?.dispose();
    mediaArtifacts?.dispose();
    void Promise.all([
      mobileGateway?.stop() ?? Promise.resolve(),
      engine?.stop() ?? Promise.resolve()
    ]).finally(() => app.quit());
  });
}
