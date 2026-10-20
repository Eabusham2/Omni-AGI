import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import { rm } from "node:fs/promises";
import { lstatSync, mkdirSync } from "node:fs";
import {
  app,
  BrowserWindow,
  desktopCapturer,
  dialog,
  nativeTheme,
  ipcMain,
  webContents,
  Notification,
  powerMonitor,
  powerSaveBlocker,
  net,
  protocol,
  safeStorage,
  shell,
  session,
  TouchBar,
  screen
} from "electron";
import { IPC } from "../shared/ipc";
import { BrainRepository, resolveBrainDataRoot, type BrainPluginPortabilityHooks } from "./brainRepository";
import { readPortableBootstrap } from "./portableBootstrap";
import { PortableWorkspaceService } from "./portableWorkspaceService";
import { RegionalExecutionService } from "./regionalExecutionService";
import { DecoderExecutionService } from "./decoderExecutionService";
import { CachedDecoderExecutionService } from "./cachedDecoderExecutionService";
import { NativeOperationAutoService } from "./nativeOperationAutoService";
import { OsWorkLeaseService } from "./osWorkLeaseService";
import { retirePortableWorkspaceAuthority } from "./portableWorkspaceAuthority";
import { portableSavedDataReadOwned } from "./portableWorkspaceReadScope";
import { BrainService, RuntimeJobManager, safeFetch } from "./brainService";
import { EngineSupervisor } from "./engineSupervisor";
import { SparseSubstrateTransferService, SparseTransferBoundaryError } from "./sparseSubstrateTransferService";
import { sparseSubstrateRpcError } from "./sparseSubstrateRpcErrors";
import { NativeCandidateToolService, NativeCandidateToolBoundaryError } from "./nativeCandidateToolService";
import { VideoRuntimeProvisioner } from "./videoRuntimeProvisioner";
import {
  initialLearningBuildProgressEvent,
  registerIpcHandlers
} from "./ipc";
import { engineBuildProgressEvent, hostCommittedBuildProgressEvent } from "./buildProgressEvents";
import { ToolExecutor, runProcess } from "./toolExecutor";
import { ExternalAdviceAuthorityStore } from "./externalAdviceAuthority";
import { ExternalAdviceClient } from "./externalAdviceClient";
import { ExternalAdviceService } from "./externalAdviceService";
import { selectRecentAdviceContext } from "./externalAdviceContext";
import { ActionReviewer } from "./actionReviewer";
import { TaskAttachmentService } from "./taskAttachmentService";
import { TaskAttachmentStore } from "./taskAttachmentStore";
import { WebImageService } from "./webImageService";
import { webImagePermissionRevision } from "./webImageReferences";
import { declaredWebImageCapabilityPack } from "./webImageCapabilityPack";
import { FileToolService } from "./fileToolService";
import { declaredFileCapabilityPack } from "./fileCapabilityPack";
import { TaskContinuationAttachments } from "./taskContinuationAttachments";
import { InheritedTaskFiles } from "./inheritedTaskFiles";
import { executeAssembly } from "./assemblyRuntime";
import { createClockRuntime } from "./clockRuntime";
import { declaredClockCapabilityPack } from "./clockCapabilityPack";
import { declaredAutomationCapabilityPack } from "./automationCapabilityPack";
import { declaredMapCapabilityPack } from "./mapCapabilityPack";
import { declaredMapBasemapCapabilityPack } from "./mapBasemapCapabilityPack";
import { declaredAssemblyCapabilityPack } from "./assemblyCapabilityPack";
import { declaredEnvironmentCapabilityPack } from "./environmentCapabilityPack";
import { declaredHtmlArtifactCapabilityPack } from "./htmlArtifactCapabilityPack";
import { EnvironmentService } from "./environmentService";
import { ProcessMemoryService } from "./processMemoryService";
import { processMemoryHelperPids, processMemoryLaunch, runProcessMemoryHelper } from "./processMemoryBridge";
import { declaredProcessMemoryCapabilityPack } from "./processMemoryCapabilityPack";
import { ClipboardService } from "./clipboardService";
import { clipboardHostHelperPids, SystemClipboardHostAdapter } from "./clipboardHostBridge";
import { declaredClipboardCapabilityPack } from "./clipboardCapabilityPack";
import type { ClockService } from "./clockService";
import { createAutomationRuntime } from "./automationRuntime";
import type { AutomationService } from "./automationService";
import { automationResidentBytes, automationResidencySample, requireAutomationHeadroom } from "./automationResources";
import { requireCatalogMemoryHeadroom } from "./catalogMemoryAdmission";
import { UserQuestionService } from "./userQuestions";
import { UserQuestionDelivery } from "./userQuestionDelivery";
import { LearningCoordinator } from "./learningCoordinator";
import { LocalTeacherService } from "./localTeacherService";
import { declaredLocalTeacherCapabilityPack } from "./localTeacherCapabilityPack";
import { declaredTeacherStatsCapabilityPack } from "./teacherStatsCapabilityPack";
import { readSystemStatsSnapshot } from "./systemStatsService";
import { localTeacherChildPids } from "./localTeacherRunner";
import { checkpointLearningBeforeQuit } from "./learningShutdown";
import { committedNativeLearningTask, factCheckNativeTask, specificLearnNativeTask } from "./learningNativeTask";
import { requireDiskWrite } from "./diskSpace";
import { ChatActionController } from "./chatActionController";
import { SteerDirectionsService } from "./steerDirections";
import { persistChatHostSuspension } from "./chatSuspension";
import { getHeapStatistics } from "node:v8";
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
import { DeviceResourceService } from "./deviceResourceService";
import { SpillPoolAuthority } from "./spillPoolAuthority";
import { SharedResourceRegistry } from "./sharedResourceRegistry";
import { createNativeComputeMeasurementCollector } from "./nativeComputeMeasurement";
import { BuildResourceSelectionStore } from "./buildResourceSelections";
import { BuildInitializationCoordinator } from "./buildInitializationCoordinator";
import { MobileGateway } from "./mobileGateway";
import { SecureSecretStore } from "./secureSecretStore";
import { ApiTeacherTrainingService } from "./teacherTraining";
import { McpClientService } from "./mcpClient";
import { installMcpUnloadBoundary } from "./mcpLifecycle";
import { McpInteractionService } from "./mcpInteractions";
import { McpAppService } from "./mcpApps";
import { McpRootsService } from "./mcpRoots";
import { McpDiagnosticService } from "./mcpDiagnostics";
import { McpAppArtifactStore } from "./mcpAppArtifacts";
import { McpAppObservationDelivery } from "./mcpAppObservationDelivery";
import { suspendMcpForNativeOwnerRelease } from "./mcpLifecycle";
import { withBrainWrite } from "./brainWriteCoordinator";
import { MCP_APP_SCHEME } from "../shared/mcpApps";
import { MCP_APP_SCHEME_PRIVILEGES, mcpAppFrameOwner, allowMcpAppFrameNavigation, mcpAppRequestHeaders, mcpAppResponseHeaders } from "./mcpAppProxy";
import { requireStudioIpcSender } from "./studioIpcTrust";
import { cpus, freemem, getPriority, setPriority, constants as osConstants } from "node:os";
import { ToolPreferencesStore } from "./toolPreferences";
import { MacTouchBarController } from "./macTouchBar";
import { MapsService } from "./mapsService";
import { MapBasemapService } from "./mapBasemapService";
import { McpTaskStore } from "./mcpTaskStore";
import { PluginCatalog } from "./pluginCatalog";
import { PluginService } from "./pluginService";
import { PluginRuntimeService } from "./pluginRuntimeService";
import { PluginRuntimeStreamParser } from "./pluginRuntimeStreamParser";
import { ElectronPluginRuntimeProcessFactory } from "./pluginRuntimeProcess";
import { PluginRuntimeResources } from "./pluginRuntimeResources";
import { pluginRuntimeExternalBridge, publishPluginRuntimeTools, retirePluginRuntimeTools } from "./pluginRuntimeHost";
import { PLUGIN_RUNTIME_SCHEME, PLUGIN_RUNTIME_REQUEST_CHANNEL } from "../shared/pluginRuntime";
import { PLUGIN_PANEL_SCHEME, pluginPanelSecurityHeaders } from "../shared/pluginPanels";
import {
  PLUGIN_PANEL_SCHEME_PRIVILEGES,
  allowPluginPanelFrameNavigation,
  allowPluginPanelRequest,
  isPluginPanelUrl,
  pluginPanelFrameIdentity,
  pluginPanelRequestHeaders,
  pluginPanelResponseHeaders
} from "./pluginPanelProtocol";
import { PLUGIN_INTERACTIVE_PANEL_SCHEME } from "../shared/pluginInteractivePanels";
import {
  PluginInteractivePanelService,
  PLUGIN_INTERACTIVE_PANEL_SCHEME_PRIVILEGES,
  allowPluginInteractivePanelNavigation,
  allowPluginInteractivePanelRequest,
  isPluginInteractivePanelUrl,
  pluginInteractivePanelFrameOwner,
  pluginInteractivePanelRequestHeaders,
  pluginInteractivePanelResponseHeaders
} from "./pluginInteractivePanels";
import { HTML_ARTIFACT_SCHEME } from "../shared/htmlArtifacts";
import {
  HtmlArtifactService,
  HTML_ARTIFACT_SCHEME_PRIVILEGES,
  allowHtmlArtifactNavigation,
  allowHtmlArtifactRequest,
  htmlArtifactFrameOwner,
  htmlArtifactRequestHeaders,
  htmlArtifactResponseHeaders,
  isHtmlArtifactUrl
} from "./htmlArtifactService";
import { MediaArtifactRegistry } from "./mediaArtifactRegistry";
import { ImaginationCaptureService } from "./imaginationCapture";
import { ImaginationAssemblyService } from "./imaginationAssembly";
import { mainImaginationCaptureOrigin } from "./imaginationCaptureOrigin";
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
  },
  { scheme: MCP_APP_SCHEME, privileges: MCP_APP_SCHEME_PRIVILEGES },
  { scheme: PLUGIN_PANEL_SCHEME, privileges: PLUGIN_PANEL_SCHEME_PRIVILEGES },
  { scheme: PLUGIN_INTERACTIVE_PANEL_SCHEME, privileges: PLUGIN_INTERACTIVE_PANEL_SCHEME_PRIVILEGES },
  { scheme: PLUGIN_RUNTIME_SCHEME, privileges: { standard: true, secure: true, supportFetchAPI: false } },
  { scheme: HTML_ARTIFACT_SCHEME, privileges: HTML_ARTIFACT_SCHEME_PRIVILEGES }
]);

const moduleDirectory = dirname(fileURLToPath(import.meta.url));
const installedUserDataRoot = app.getPath("userData");
const portableBootstrap = readPortableBootstrap({ platform: process.platform, packaged: app.isPackaged,
  executablePath: process.execPath, installedUserDataRoot: process.platform === "win32" && process.env.LOCALAPPDATA
    ? join(resolve(process.env.LOCALAPPDATA), "OmniAGI") : installedUserDataRoot });
if (portableBootstrap) {
  mkdirSync(portableBootstrap.userDataRoot, { recursive: true, mode: 0o700 });
  if (!lstatSync(portableBootstrap.userDataRoot).isDirectory() || lstatSync(portableBootstrap.userDataRoot).isSymbolicLink()) throw new Error("Portable user data must be a real folder, not a link.");
  app.setPath("userData", portableBootstrap.userDataRoot);
  app.setPath("sessionData", portableBootstrap.userDataRoot);
}
const deviceDataRoot = portableBootstrap?.deviceDataRoot ?? installedUserDataRoot;
const developmentRendererUrl = process.env.ELECTRON_RENDERER_URL;
let mainWindow: BrowserWindow | undefined;
let trustedStudioWebContentsId: number | undefined;
let disposeIpc: (() => void) | undefined;
let engine: EngineSupervisor | undefined;
let brainRepository: BrainRepository | undefined;
let brainService: BrainService | undefined;
let sharedResources: SharedResourceRegistry | undefined;
let spillPoolAuthority: SpillPoolAuthority | undefined;
let idleCognition: IdleCognitionScheduler | undefined;
let backgroundRuntime: BackgroundRuntimeController | undefined;
let mobileGateway: MobileGateway | undefined;
let mcpClient: McpClientService | undefined;
let questionService: UserQuestionService | undefined;
let mcpInteractionService: McpInteractionService | undefined;
let mcpAppService: McpAppService | undefined;
let mcpRootsRuntime: McpRootsService | undefined;
let mcpDiagnosticsRuntime: McpDiagnosticService | undefined;
let mcpAppObservationDelivery: McpAppObservationDelivery | undefined;
let pluginService: PluginService | undefined;
let pluginInteractivePanelService: PluginInteractivePanelService | undefined;
let pluginRuntimeService: PluginRuntimeService | undefined;
let htmlArtifactService: HtmlArtifactService | undefined;
let touchBarController: MacTouchBarController | undefined;
let questionDelivery: UserQuestionDelivery | undefined;
let learningCoordinator: LearningCoordinator | undefined;
let localTeacherRuntime: LocalTeacherService | undefined;
let apiTeacherRuntime: ApiTeacherTrainingService | undefined;
let fileToolRuntime: FileToolService | undefined;
let mapBasemapRuntime: MapBasemapService | undefined;
let osWorkRuntime: OsWorkLeaseService | undefined;
let clockRuntime: ClockService | undefined;
let automationRuntime: AutomationService | undefined;
let environmentRuntime: EnvironmentService | undefined;
let processMemoryRuntime: ProcessMemoryService | undefined;
let sparseSubstrateRuntime: SparseSubstrateTransferService | undefined;
let nativeCandidateToolRuntime: NativeCandidateToolService | undefined;
let portableWorkspaceRuntime: PortableWorkspaceService | undefined;
let clipboardRuntime: ClipboardService | undefined;
let initializationCoordinator: BuildInitializationCoordinator | undefined;
let mediaArtifacts: MediaArtifactRegistry | undefined;
let imaginationCaptureRuntime: ImaginationCaptureService | undefined;
let imaginationAssemblyRuntime: ImaginationAssemblyService | undefined;
let quitAfterCleanup = false;
let quitCleanupInProgress = false;
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
      // Import/open does not grant learned-runtime residency; the workspace
      // presents the explicit Load/Auto-load fit flow.
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
  const appRequests = new Map<number, string>();
  session.defaultSession.protocol.handle(MCP_APP_SCHEME, request => mcpAppService?.document(request.url) ?? new Response("Connected app host unavailable.", { status: 410 }));
  session.defaultSession.protocol.handle(PLUGIN_PANEL_SCHEME, request =>
    pluginService?.panelDocument(request.url) ?? new Response("Static plugin panel unavailable.", {
      status: 410,
      headers: pluginPanelSecurityHeaders()
    })
  );
  session.defaultSession.protocol.handle(PLUGIN_INTERACTIVE_PANEL_SCHEME, request =>
    pluginInteractivePanelService?.document(request.url) ?? new Response("Interactive plugin panel unavailable.", {
      status: 410,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Content-Security-Policy": "default-src 'none'", "Cache-Control": "no-store" }
    })
  );
  session.defaultSession.protocol.handle(HTML_ARTIFACT_SCHEME, request =>
    htmlArtifactService?.document(request.url) ?? new Response("HTML artifact host unavailable.", {
      status: 410,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Content-Security-Policy": "default-src 'none'", "Cache-Control": "no-store" }
    })
  );
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
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
    if (isHtmlArtifactUrl(details.url) || htmlArtifactFrameOwner(details.frame)) {
      callback({ cancel: details.webContentsId !== trustedStudioWebContentsId || !allowHtmlArtifactRequest(details) });
      return;
    }
    if (isPluginInteractivePanelUrl(details.url) || pluginInteractivePanelFrameOwner(details.frame)) {
      callback({ cancel: details.webContentsId !== trustedStudioWebContentsId || !allowPluginInteractivePanelRequest(details) });
      return;
    }
    if (isPluginPanelUrl(details.url) || pluginPanelFrameIdentity(details.frame)) {
      callback({ cancel: details.webContentsId !== trustedStudioWebContentsId || !allowPluginPanelRequest(details) });
      return;
    }
    const owner = appRequests.get(details.id) ?? mcpAppFrameOwner(details.frame);
    let targetOwner: string | undefined;
    try { const url = new URL(details.url); if (url.protocol === `${MCP_APP_SCHEME}:`) targetOwner = url.hostname; } catch { /* invalid URL is rejected below */ }
    if (owner || targetOwner) {
      const appOwner = owner ?? targetOwner!; appRequests.set(details.id, appOwner);
      callback({ cancel: !mcpAppService?.allowNetwork(appOwner, details.url, details.resourceType) });
    }
    else callback({});
  });
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
    if (isHtmlArtifactUrl(details.url) || htmlArtifactFrameOwner(details.frame)) {
      callback({ cancel: details.webContentsId !== trustedStudioWebContentsId || !allowHtmlArtifactRequest(details),
        requestHeaders: htmlArtifactRequestHeaders(details.requestHeaders) });
      return;
    }
    if (isPluginInteractivePanelUrl(details.url) || pluginInteractivePanelFrameOwner(details.frame)) {
      callback({ cancel: details.webContentsId !== trustedStudioWebContentsId || !allowPluginInteractivePanelRequest(details),
        requestHeaders: pluginInteractivePanelRequestHeaders(details.requestHeaders) });
      return;
    }
    if (isPluginPanelUrl(details.url) || pluginPanelFrameIdentity(details.frame)) {
      callback({ cancel: details.webContentsId !== trustedStudioWebContentsId || !allowPluginPanelRequest(details),
        requestHeaders: pluginPanelRequestHeaders(details.requestHeaders) });
      return;
    }
    const owner = appRequests.get(details.id) ?? mcpAppFrameOwner(details.frame);
    if (!owner && !details.url.startsWith(`${MCP_APP_SCHEME}:`)) { callback({ requestHeaders: details.requestHeaders }); return; }
    callback({ cancel: Boolean(owner && !mcpAppService?.allowNetwork(owner, details.url, details.resourceType)), requestHeaders: mcpAppRequestHeaders(details.requestHeaders) });
  });
  session.defaultSession.webRequest.onCompleted(details => { appRequests.delete(details.id); });
  session.defaultSession.webRequest.onErrorOccurred(details => { appRequests.delete(details.id); });
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    if (isHtmlArtifactUrl(details.url) || htmlArtifactFrameOwner(details.frame)) {
      callback({ responseHeaders: htmlArtifactResponseHeaders(details.responseHeaders ?? {}) });
      return;
    }
    if (isPluginInteractivePanelUrl(details.url) || pluginInteractivePanelFrameOwner(details.frame)) {
      callback({ responseHeaders: pluginInteractivePanelResponseHeaders(details.responseHeaders ?? {}) });
      return;
    }
    if (isPluginPanelUrl(details.url) || pluginPanelFrameIdentity(details.frame)) {
      callback({ responseHeaders: pluginPanelResponseHeaders(details.responseHeaders ?? {}) });
      return;
    }
    if (appRequests.has(details.id) || mcpAppFrameOwner(details.frame) || details.url.startsWith(`${MCP_APP_SCHEME}:`)) {
      // Preserve the main-owned App CSP and never import ambient cookies from
      // an approved external resource into Studio's installation session.
      callback({ responseHeaders: mcpAppResponseHeaders(details.responseHeaders ?? {}) }); return;
    }
    const developmentConnect = allowedDevelopmentOrigin
      ? ` ${allowedDevelopmentOrigin} ws://${new URL(allowedDevelopmentOrigin).host}`
      : "";
    const policy = [
      "default-src 'self'",
      "base-uri 'none'",
      "object-src 'none'",
      `frame-src ${MCP_APP_SCHEME}: ${PLUGIN_PANEL_SCHEME}: ${PLUGIN_INTERACTIVE_PANEL_SCHEME}: ${HTML_ARTIFACT_SCHEME}:`,
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
  const studioWebContentsId = window.webContents.id;
  trustedStudioWebContentsId = studioWebContentsId;
  window.on("focus", () => touchBarController?.refresh());
  window.on("blur", () => touchBarController?.refresh());
  window.webContents.on("did-start-navigation", details => {
    if (details.isMainFrame && !details.isSameDocument) touchBarController?.resetOwner();
  });

  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-frame-navigate", event => {
    if (event.isMainFrame) return;
    const trustedStudioUrl = developmentRendererUrl ?? pathToFileURL(join(moduleDirectory, "../renderer/index.html")).href;
    if (!allowMcpAppFrameNavigation(event, (owner, url) => mcpAppService?.allowNetwork(owner, url, "subFrame") ?? false) &&
      !allowPluginPanelFrameNavigation(event, trustedStudioUrl) &&
      !allowPluginInteractivePanelNavigation(event, trustedStudioUrl) &&
      !allowHtmlArtifactNavigation(event, trustedStudioUrl)) event.preventDefault();
  });
  window.webContents.on("will-navigate", (event, target) => {
    let allowed = false;
    try {
      allowed = developmentRendererUrl
        ? new URL(target).origin === new URL(developmentRendererUrl).origin
        : new URL(target).href.split("#")[0] === pathToFileURL(join(moduleDirectory, "../renderer/index.html")).href;
    } catch {
      allowed = false;
    }
    if (!allowed) event.preventDefault();
  });
  window.webContents.on("will-attach-webview", (event) => event.preventDefault());
  window.once("ready-to-show", () => window.show());
  window.on("closed", () => {
    touchBarController?.resetOwner();
    if (trustedStudioWebContentsId === studioWebContentsId) trustedStudioWebContentsId = undefined;
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
    join(deviceDataRoot, "shared-resources.sqlite3")
  );
  const registry = sharedResources;
  const catalogAdmission = async (bytes: number): Promise<void> => {
    registry.requireSynchronized();
    const pids = [...app.getAppMetrics().map(metric => metric.pid), process.pid,
      ...(engine?.managedProcessIds ?? []), ...localTeacherChildPids()];
    const residentBytes = await automationResidentBytes(pids);
    // Re-read after the OS await: a newer smaller budget must govern parsing.
    registry.requireSynchronized();
    requireCatalogMemoryHeadroom(bytes, { residentBytes,
      selectedCeilingBytes: registry.status().globalRamCeilingBytes,
      freeBytes: freemem(), heapAvailableBytes: Math.max(0, getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) });
  };
  const repository = new BrainRepository(portableBootstrap?.brainRoot ?? resolveBrainDataRoot(app.getPath("userData")), {
    saved: ({ brainId, storagePoolBytes }) => registry.observeSaved(brainId, storagePoolBytes),
    removed: (brainId) => registry.observeRemoved(brainId)
  }, {
    catalogAdmission,
    ...(portableBootstrap ? { toolPermissionRoot: deviceDataRoot, bootstrapToolPermissions: false, beforeSavedDataAccess: () => {
      if (!portableSavedDataReadOwned() && !portableWorkspaceRuntime?.savedDataReady()) throw new Error("Prepare and verify this portable instance before opening its saved data. Current destination status remains available.");
    } } : {})
  });
  brainRepository = repository;
  await repository.initialize();
  mediaArtifacts = new MediaArtifactRegistry((brainId) =>
    repository.brainDirectory(brainId)
  );
  const imaginationCapture = new ImaginationCaptureService({ brainDirectory: id => repository.brainDirectory(id),
    reserve: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    admit: bytes => { if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 8) throw new Error("Capture awaits metadata transfer RAM."); },
    mediaUrl: (id, sha256, mimeType) => mediaArtifacts!.leaseCapture(id, sha256, mimeType) });
  imaginationCaptureRuntime = imaginationCapture;
  mediaArtifacts.setCaptureSink(fragment => imaginationCapture.delivered(fragment));
  installSecurityPolicy(mediaArtifacts);
  if (!portableBootstrap) await reviewManagedBetaBrains(repository);
  // Older releases treated every identity as idle-enabled. Collapse that
  // legacy state before any worker or scheduler can acquire a brain.
  if (!portableBootstrap) await repository.reconcileActiveModeLease();
  // Register inactive saved identities too. The shared limit is MAX(pool),
  // never one independent pool multiplied by the number of instances.
  const savedBrains = portableBootstrap ? [] : await repository.list();
  registry.reconcileSavedOwners(savedBrains.map((summary) => summary.id));
  for (const summary of savedBrains) {
    const saved = await repository.get(summary.id, false);
    registry.observeSaved(saved.id, saved.config.storagePoolBytes);
  }
  registry.requireSynchronized();
  const videoRuntimeCacheRoot = join(deviceDataRoot, "video-runtime");
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
  const osWork = new OsWorkLeaseService({
    platform: process.platform,
    startPowerSaveBlocker: () => powerSaveBlocker.start("prevent-app-suspension"),
    stopPowerSaveBlocker: id => { powerSaveBlocker.stop(id); },
    isPowerSaveBlockerStarted: id => powerSaveBlocker.isStarted(id),
    priorityProcessIds: () => [process.pid, ...(engine?.managedProcessIds ?? [])],
    getPriority: pid => getPriority(pid), setPriority: (pid, priority) => setPriority(pid, priority),
    requestedPriority: osConstants.priority.PRIORITY_ABOVE_NORMAL,
    diagnostic: message => console.warn(message)
  });
  osWorkRuntime = osWork;
  engine.on("activity", (event: import("./engineSupervisor").EngineActivityTransition) => {
    osWork.observeEngineActivity(event); osWork.refreshOwnedProcesses();
  });
  engine.on("worker-started", () => osWork.refreshOwnedProcesses());
  engine.on("worker-closed", ({ pid }: { pid?: number }) => { if (typeof pid === "number") osWork.retireOwnedProcess(pid); });
  spillPoolAuthority = new SpillPoolAuthority(registry.ledgerPath);
  const deviceResources = new DeviceResourceService(engine, repository.root, { spillAuthority: spillPoolAuthority });
  const resourcePlanner = new ResourcePlanner(repository.root, {
    measurementCollector: createNativeComputeMeasurementCollector(engine),
    deviceTopology: (settings) => deviceResources.topology({ storageDirectories: settings.storageDirectories }),
    spillCapacity: (settings, topology, context) => deviceResources.spillCapacity(settings, topology, context)
  });
  const service = new BrainService(
    repository,
    engine,
    resourcePlanner,
    mediaArtifacts,
    registry
  );
  service.setRuntimeAdmissionGuard(() => {
    if (portableBootstrap && !portableWorkspaceRuntime) throw new Error("Portable destination review is preparing; no neural work is admitted.");
    portableWorkspaceRuntime?.assertRuntimeAllowed();
  });
  const regionalEngine = engine;
  const regionalExecution = new RegionalExecutionService({
    requireLoaded: id => service.requireInstanceLoaded(id),
    owner: async id => {
      const brain = await repository.get(id, false);
      return { brainId: brain.id, storagePath: repository.brainDirectory(id), settings: brain.config.deviceResources,
        identitySha256: createHash("sha256").update(JSON.stringify({ id: brain.id, origin: brain.originChecksum ?? null, config: brain.config })).digest("hex") };
    },
    topology: (owner, signal, context) => deviceResources.topologyForLoadedOwner(owner.brainId, owner.storagePath, signal, context,
      { storageDirectories: owner.settings?.storageDirectories ?? [] }),
    dispatch: (method, params, signal, context) => regionalEngine.requestLoadedRegionalExecution(method, params, 30_000, signal, context)
  });
  service.addInstanceSuspendBoundary(async id => { regionalExecution.forget(id); });
  service.addInstanceRecoveryBoundary(async id => { regionalExecution.forget(id); });
  engine.on("worker-closed", () => regionalExecution.forget());
  const decoderExecutionDependencies: ConstructorParameters<typeof DecoderExecutionService>[0] = {
    requireLoaded: id => service.requireInstanceLoaded(id),
    owner: async id => {
      const brain = await repository.get(id, false);
      if (!brain.originChecksum || !/^[a-f0-9]{64}$/.test(brain.originChecksum))
        throw new Error("Decoder measurements need the current instance's verified immutable-origin metadata.");
      const configSha256 = createHash("sha256").update(JSON.stringify(brain.config)).digest("hex");
      return { brainId: brain.id, storagePath: repository.brainDirectory(id), originSha256: brain.originChecksum, configSha256,
        identitySha256: createHash("sha256").update(JSON.stringify({ id: brain.id, origin: brain.originChecksum, configSha256 })).digest("hex") };
    },
    topology: async (owner, signal, context) => {
      const brain = await repository.get(owner.brainId, false);
      if (createHash("sha256").update(JSON.stringify(brain.config)).digest("hex") !== owner.configSha256)
        throw new Error("Decoder resource settings changed before topology inspection.");
      return deviceResources.topologyForLoadedOwner(owner.brainId, owner.storagePath, signal, context,
        { storageDirectories: brain.config.deviceResources?.storageDirectories ?? [] });
    },
    dispatch: (method, params, signal, context) => regionalEngine.requestLoadedDecoderExecution(method, params, signal, context),
    control: (method, params) => regionalEngine.controlLoadedDecoderExecution(method, params)
  };
  const decoderExecution = new DecoderExecutionService(decoderExecutionDependencies);
  const cachedDecoderExecution = new CachedDecoderExecutionService({ ...decoderExecutionDependencies,
    dispatch: (method, params, signal, context) => regionalEngine.requestLoadedCachedDecoderExecution(method, params, signal, context),
    control: (method, params) => regionalEngine.controlLoadedCachedDecoderExecution(method, params)
  });
  const nativeOperationAuto = new NativeOperationAutoService({ ...decoderExecutionDependencies,
    dispatch: (method, params, signal, context) => regionalEngine.requestLoadedNativeOperationAuto(method, params, signal, context),
    control: (method, params) => regionalEngine.controlLoadedNativeOperationAuto(method, params),
    onCleanupError: (id, error) => console.warn(`Native operation Auto cleanup for ${id} did not settle:`, error)
  });
  service.addInstanceSuspendBoundary(async id => { nativeOperationAuto.forget(id); });
  service.addInstanceRecoveryBoundary(async id => { nativeOperationAuto.forget(id); });
  engine.on("worker-closed", () => nativeOperationAuto.forget());
  service.addInstanceSuspendBoundary(async id => { decoderExecution.forget(id); });
  service.addInstanceRecoveryBoundary(async id => { decoderExecution.forget(id); });
  engine.on("worker-closed", () => decoderExecution.forget());
  service.addInstanceSuspendBoundary(async id => { cachedDecoderExecution.forget(id); });
  service.addInstanceRecoveryBoundary(async id => { cachedDecoderExecution.forget(id); });
  engine.on("worker-closed", () => cachedDecoderExecution.forget());
  const taskAttachments = new TaskAttachmentService(repository);
  const continuedTaskAttachments = new TaskContinuationAttachments(repository, taskAttachments,
    id => service.currentChatAttentionEpoch(id));
  const inheritedTaskFiles = new InheritedTaskFiles({ root: id => repository.brainDirectory(id),
    context: (id, executionId) => service.inheritedTaskFileContext(id, executionId),
    permissions: id => service.listToolPermissions(id), rootEpoch: id => service.currentChatAttentionEpoch(id),
    assertActive: context => service.assertActiveInheritedTaskFileContext(context) });
  continuedTaskAttachments.setInheritedTaskFiles(inheritedTaskFiles);
  service.setTaskAttachmentService(continuedTaskAttachments);
  brainService = service;
  service.setSpillSelectionClaimer((selectionId, brainId, paths) => deviceResources.claimSpillSelection(selectionId, brainId, paths));
  const plugins = new PluginService(
    new PluginCatalog(join(app.getPath("userData"), "plugins")),
    service
  );
  pluginService = plugins;
  await plugins.initialize();
  const pluginPortability: BrainPluginPortabilityHooks = {
    cloneTrusted: (sourceBrainId, targetBrainId) =>
      plugins.cloneTrustedPortable(sourceBrainId, targetBrainId),
    exportBrain: (brainId, mode, destination) =>
      plugins.exportPortable(brainId, mode, destination),
    validateImport: (files, declaration, sourceBrainId) =>
      plugins.validatePortableImport(files, declaration, sourceBrainId),
    adoptExternal: (targetBrainId, archive) =>
      plugins.adoptExternalPortable(targetBrainId, archive),
    removeBrain: (brainId) => plugins.removePortableBrain(brainId)
  };
  repository.configurePluginPortability(pluginPortability);
  service.configureFullMergePlugins(
    (brainId, parents) => plugins.adoptFullMergePortable(brainId, parents),
    (brainId) => plugins.removePortableBrain(brainId)
  );
  service.setPluginTrainingFormatResolver(plugins);
  service.onInstanceLifecycle((state) => {
    if (state.phase === "loaded") {
      void plugins.resumeSemanticLearning(state.brainId).catch((error) =>
        console.error("Plugin semantic learning:", error)
      );
    }
  });
  // Startup repeats the same physical preflight as Build. A mind that no
  // longer fits remains intact and visible, but no neural worker is started
  // for it until storage/RAM reserve requirements are restored.
  for (const summary of portableBootstrap ? [] : await repository.list()) {
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
  jobs.on("event", ({ job }: import("../shared/types").RuntimeJobEvent) => osWork.observeRuntimeJob(job));
  const buildSelections = new BuildResourceSelectionStore(
    join(deviceDataRoot, "pending-build-resources.json")
  );
  const initialization = new BuildInitializationCoordinator(
    repository,
    service,
    jobs,
    buildSelections
  );
  initializationCoordinator = initialization;
  const sourceRuntime = new ElectronSourceRuntimeLifecycle({
    app,
    userDataPath: deviceDataRoot,
    resourcesPath: process.resourcesPath
  });
  const integrationSecrets = new SecureSecretStore(
    join(deviceDataRoot, "integration-secrets.json"),
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
  const portableWorkspace = new PortableWorkspaceService({ repository, userDataRoot: app.getPath("userData"), deviceDataRoot, bootstrap: portableBootstrap,
    io: { checkDisk: (path, bytes) => requireDiskWrite(path, { operationWriteBytes: bytes }), admit: bytes => {
      if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 8) throw new Error("Portable transfer awaits metadata RAM headroom.");
    } }, toolsPreferences: () => toolPreferences.get(), setToolsPreferences: async value => { await toolPreferences.set(value); },
    assertUnloaded: id => {
      service.requireRuntimeAdmissionOpen(); let loaded = false;
      try { service.requireInstanceLoaded(id); loaded = true; } catch { /* Unloaded is the required saved-only boundary. */ }
      if (loaded) throw new Error("Unload the selected instance before creating a portable folder. Its exact saved continuation will be exported without a model Load.");
    }, stagingRepository: root => new BrainRepository(root, undefined, { toolPermissionRoot: join(deviceDataRoot, "portable-staging-policy"), bootstrapToolPermissions: false, catalogAdmission }),
    pluginPortability, themePackage: selection => plugins.package(selection.pluginId, selection.contentSha256),
    installTheme: async value => { await plugins.installPackage(value, "Portable immutable theme dependency"); },
    planDestination: brain => resourcePlanner.plan({ mode: "manual", requestedItems: String(brain.config.workingMemorySlots), systemRamMode: brain.config.systemRamMode,
      systemRamSharePercent: brain.config.systemRamSharePercent, storagePoolMode: brain.config.storagePoolMode,
      ...(brain.config.storagePoolMode === "manual" ? { storagePoolBytes: String(brain.config.storagePoolBytes) } : {}),
      requestedContextTokens: String(brain.config.contextWindowTokens) }, { brainDirectory: repository.brainDirectory(brain.id), config: brain.config,
      allowStorageSpill: brain.config.deviceResources?.allowStorageSpill, admissionScope: "capacity" }),
    declareReservation: (id, bytes) => { registry.observeSaved(id, bytes); registry.requireSynchronized(); },
    retireCopiedAuthority: (root, id, completed, checkpoint, operation) => retirePortableWorkspaceAuthority(root, id, completed, checkpoint, operation.signal)
  });
  portableWorkspaceRuntime = portableWorkspace;
  await portableWorkspace.initialize().catch(error => console.error("Portable destination preparation:", error));
  const teacher = new ApiTeacherTrainingService(service, integrationSecrets, {
    brainRoot: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    admitMetadata: async (id, bytes, _purpose, signal) => {
      signal?.throwIfAborted();
      await repository.get(id, false);
      await catalogAdmission(bytes);
      signal?.throwIfAborted();
    },
    admitWrite: async (id, additionalBytes, signal) => {
      signal?.throwIfAborted();
      await requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: additionalBytes });
      signal?.throwIfAborted();
    }
  });
  apiTeacherRuntime = teacher;
  teacher.on("event", ({ job }: import("../shared/types").RuntimeJobEvent) => osWork.observeRuntimeJob(job));
  service.addInstanceSuspendBoundary(id => teacher.suspend(id));
  service.addInstanceRecoveryBoundary(id => teacher.suspend(id));
  const adviceAuthority = new ExternalAdviceAuthorityStore(join(deviceDataRoot, "external-advice-authority.json"));
  const advice = new ExternalAdviceService(new ExternalAdviceClient(), integrationSecrets, {
    deviceDirectory: deviceDataRoot,
    admitWrite: async (id, additionalBytes, signal) => {
      signal.throwIfAborted();
      await requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: additionalBytes });
      signal.throwIfAborted();
    },
    brainRoot: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    authority: (id, channel) => adviceAuthority.get(id, channel),
    selectContext: async (input, signal) => {
      signal.throwIfAborted();
      // Reviewer context is selected once by ActionReviewer and included in
      // its exact evidence-bound prompt. Do not append a second context set.
      if (input.channel === "reviewer") return { format: "omni-external-advice-selected-context", version: 1,
        ...input, items: [], includedCharacters: 0, omittedItems: 0, truncated: false };
      const page = await repository.conversationPage(input.brainId, undefined, 32);
      signal.throwIfAborted(); return selectRecentAdviceContext(input, page);
    }
  });
  const reviewerOwners = new Set<string>();
  let adviceCredentialEpoch = 0;
  const actionReviewer = new ActionReviewer({
    configuration: async (id, signal) => {
      reviewerOwners.add(id);
      signal.throwIfAborted();
      const authority = await adviceAuthority.get(id, "reviewer");
      if (!authority.enabled) return undefined;
      const settings = await advice.settings(id);
      signal.throwIfAborted();
      if (!settings.reviewer.enabled || !authority.enabled) return undefined;
      return { enabled: true, revision: `${settings.revision}:${adviceCredentialEpoch}`,
        ownerRevision: `${authority.ownerEpoch}:${authority.revision}`,
        reviewerId: `${settings.reviewer.provider}:${settings.reviewer.model}`,
        budget: { timeoutMs: 60_000, maxRequestBytes: 2 * 1024 * 1024,
          maxContextBytes: Math.max(1, Math.min(1024 * 1024, settings.reviewer.context.maxCharacters * 4)), maxResponseBytes: 512 * 1024 } };
    },
    review: async (request, signal) => (await advice.completeReview(request.brainId, request.prompt, signal)).text,
    onReview: (receipt, signal) => advice.recordReview(receipt, signal),
    selectContext: async (input, signal) => {
      const settings = await advice.settings(input.brainId);
      signal.throwIfAborted();
      const page = await repository.conversationPage(input.brainId, undefined, 32);
      signal.throwIfAborted();
      const selected = selectRecentAdviceContext({ brainId: input.brainId, channel: "reviewer",
        selection: { ...settings.reviewer.context, maxCharacters: Math.min(settings.reviewer.context.maxCharacters, Math.floor(input.maxBytes / 4)) } }, page);
      return selected.items.map(item => ({ id: item.id, kind: item.kind === "task" ? "task" as const : "work" as const, text: item.text }));
    }
  });
  service.addInstanceSuspendBoundary(async id => { actionReviewer.configurationChanged(id); await advice.suspend(id); });
  service.addInstanceRecoveryBoundary(async id => {
    await adviceAuthority.set(id, "assistant", false);
    await adviceAuthority.set(id, "reviewer", false);
    actionReviewer.configurationChanged(id); await advice.suspend(id);
  });
  const mcp = new McpClientService(
    join(deviceDataRoot, "mcp-servers.json"),
    integrationSecrets,
    service,
    fetch,
    { openBrowser: url => shell.openExternal(url) }
  );
  mcp.setTaskStore(new McpTaskStore({
    brainDirectory: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    admit: async bytes => { await catalogAdmission(bytes); },
    checkDisk: async (id, bytes) => { await requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: bytes }); }
  }));
  await mcp.initialize();
  mcp.enableSessionTransports();
  installMcpUnloadBoundary(service, mcp);
  mcpClient = mcp;
  service.addInstanceSuspendBoundary(id => mcp.retireTaskAuthority(id, "The instance is being suspended."));
  service.addInstanceRecoveryBoundary(id => mcp.retireTaskAuthority(id, "The instance is being recovered."));
  const mcpInteractions = new McpInteractionService({
    directory: join(deviceDataRoot, "mcp-interactions"),
    requireAccess: origin => mcp.requireInteractionOrigin(origin),
    admit: async bytes => {
      if (bytes > freemem()) throw new Error("MCP review awaits enough physical memory for this request.");
      await requireDiskWrite(deviceDataRoot, { operationWriteBytes: bytes });
    },
    openBrowser: url => shell.openExternal(url),
    sampleNative: (approved, signal) => {
      service.requireInstanceLoaded(approved.brainId, approved.originatingTurnId);
      if (!engine) throw new Error("The native sampling worker is unavailable.");
      return engine.sampleNativeMcp(approved, signal);
    }
  });
  await mcpInteractions.initialize();
  mcpInteractionService = mcpInteractions;
  mcp.setHostInteractions({ elicit: (origin, params, signal) => mcpInteractions.elicit(origin, params, signal),
    sample: (origin, params, signal) => mcpInteractions.sample(origin, params, signal),
    completeElicitation: (origin, id) => mcpInteractions.completeElicitation(origin, id) });
  const tools = new ToolExecutor(
    service,
    jobs,
    sourceRuntime,
    undefined,
    pluginRuntimeExternalBridge(mcp, () => pluginRuntimeService),
    toolPreferences,
    plugins
  );
  tools.setRuntimeAdmissionGuard(() => portableWorkspace.assertRuntimeAllowed());
  tools.setApiTeacherService(teacher);
  tools.setSystemStatsService({ snapshot: async (id, signal, requireCurrent) => {
    await requireCurrent(); signal.throwIfAborted();
    await repository.get(id, false);
    await catalogAdmission(512 * 1024);
    await requireCurrent(); signal.throwIfAborted();
    return readSystemStatsSnapshot({ dataDirectory: repository.brainDirectory(id), signal, requireCurrent,
      cachedDeviceTelemetry: () => deviceResources.cachedTopologyForDirectory(repository.root),
      managedFamilyMeasurement: checkedSignal => automationResidencySample(
        [...app.getAppMetrics().map(metric => metric.pid), process.pid, ...(engine?.managedProcessIds ?? []), ...localTeacherChildPids()],
        process.platform, checkedSignal) });
  } });
  tools.setActionReviewer(actionReviewer);
  const runtimeMemory = new PluginRuntimeResources(async id => {
    const brain = await repository.get(id, false), envelope = await resourcePlanner.reviewedRamEnvelope(brain.config);
    registry.requireSynchronized();
    const selected = registry.status().globalRamCeilingBytes;
    const pids = [...app.getAppMetrics().map(metric => metric.pid), process.pid,
      ...(engine?.managedProcessIds ?? []), ...localTeacherChildPids()];
    return { residentBytes: await automationResidentBytes(pids),
      ceilingBytes: Math.min(envelope.systemRamBudgetBytes, selected ?? envelope.systemRamBudgetBytes),
      freeBytes: envelope.availableSafeRamBytes, heapAvailableBytes: Math.max(0, getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) };
  });
  const runtimeProcesses = new ElectronPluginRuntimeProcessFactory({
    runtimePreloadPath: join(moduleDirectory, "../preload/plugin-runtime.cjs"),
    createWindow: options => new BrowserWindow(options),
    sessionFromPartition: (partition, options) => session.fromPartition(partition, options),
    allWebContents: () => webContents.getAllWebContents(),
    appMetrics: () => app.getAppMetrics(),
    document: url => pluginRuntimeService?.document(url)
  });
  const runtimeModules = new PluginRuntimeService(plugins, {
    brainRoot: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    tools, processFactory: runtimeProcesses,
    admit: input => { service.requireInstanceLoaded(input.brainId); return runtimeMemory.admit(input.brainId, input.scriptBytes); },
    admitTraining: input => runtimeMemory.admit(input.brainId, input.scriptBytes),
    checkDisk: async (id, bytes) => { await requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: bytes }); },
    admitRead: (id, bytes) => runtimeMemory.admitTransfer(id, bytes),
    publishTools: (id, definitions) => publishPluginRuntimeTools(service, id, definitions),
    retireTools: async (id, ids) => retirePluginRuntimeTools(service, tools, id, ids),
    isBrainLoaded: id => { try { service.requireInstanceLoaded(id); return true; } catch { return false; } },
    learnSchemas: async (job, signal) => {
      const packageValue = await plugins.package(job.pluginId, job.contentSha256);
      const learned = await service.learnLoadedStructuredExperience(job.brainId, {
        content: job.text, name: `${packageValue.manifest.name} tool schema`,
        sourceLabel: `Plugin ${job.pluginId}/${job.moduleId} typed tool schema`,
        provenanceUrl: packageValue.manifest.homepage, license: packageValue.manifest.license
      }, signal);
      if (learned.source.contentHash !== job.sourceSha256 || learned.source.rawTextRetained !== false) throw new Error("Plugin schema learning did not return its exact ordinary source receipt.");
      return { sourceSha256: learned.source.contentHash, receipt: { sourceId: learned.source.id, contentHash: learned.source.contentHash } };
    }
  });
  pluginRuntimeService = runtimeModules;
  plugins.setRuntimeStreamParser(new PluginRuntimeStreamParser(runtimeModules));
  ipcMain.removeHandler(PLUGIN_RUNTIME_REQUEST_CHANNEL);
  ipcMain.handle(PLUGIN_RUNTIME_REQUEST_CHANNEL, (event, message) => runtimeProcesses.request(event, message));
  service.addInstanceSuspendBoundary(id => runtimeModules.stopOwner(id));
  service.addInstanceRecoveryBoundary(id => runtimeModules.stopOwner(id));
  service.onInstanceLifecycle(state => {
    if (state.phase === "loaded") void runtimeModules.flushSchemaLearning(state.brainId).catch(error => console.error("Plugin runtime schema learning:", error));
    void runtimeModules.dispatchEvent(state.brainId, { topic: "brain.lifecycle",
      eventId: createHash("sha256").update(JSON.stringify(state)).digest("hex"), data: { brainId: state.brainId, phase: state.phase } })
      .catch(error => console.error("Plugin runtime lifecycle event:", error));
  });
  const pluginPanels = new PluginInteractivePanelService(plugins, tools, {
    brainRoot: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    checkDisk: async (id, bytes) => { await requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: bytes }); },
    admitTransfer: (id, bytes) => runtimeMemory.admitTransfer(id, bytes)
  });
  service.addInstanceSuspendBoundary(id => pluginPanels.closeOwner(id));
  service.addInstanceRecoveryBoundary(id => pluginPanels.closeOwner(id));
  pluginInteractivePanelService = pluginPanels;
  const htmlArtifacts = new HtmlArtifactService(repository, tools);
  htmlArtifactService = htmlArtifacts;
  tools.setHtmlArtifactService(htmlArtifacts);
  tools.setTaskAttachmentService(continuedTaskAttachments);
  const maps = new MapsService(join(deviceDataRoot, "map-connections.sqlite3"), integrationSecrets, {
    admit: async bytes => { if (bytes > freemem()) throw new Error("Map data awaits enough physical transfer RAM."); },
    checkDisk: async bytes => { await requireDiskWrite(deviceDataRoot, { operationWriteBytes: bytes }); }
  });
  await maps.initialize().catch((error: unknown) => {
    // An optional map-store failure must not prevent the rest of Studio from
    // opening. Map calls/settings still surface the real storage failure.
    console.warn("Private map adapter storage is unavailable:", error instanceof Error ? error.message : error);
  });
  tools.setMapsService(maps);
  const mapBasemaps = new MapBasemapService(join(deviceDataRoot, "map-imagery.sqlite3"), integrationSecrets, {
    admit: async bytes => { await catalogAdmission(bytes); },
    checkDisk: async bytes => { await requireDiskWrite(deviceDataRoot, { operationWriteBytes: bytes }); }
  });
  await mapBasemaps.initialize().catch((error: unknown) => {
    console.warn("Optional map imagery storage is unavailable:", error instanceof Error ? error.message : error);
  });
  tools.setMapBasemapService(mapBasemaps);
  mapBasemapRuntime = mapBasemaps;
  service.addInstanceSuspendBoundary(async id => { mapBasemaps.forget(id); });
  service.addInstanceRecoveryBoundary(async id => { mapBasemaps.forget(id); });
  for (const brain of await repository.list()) {
    await service.ensureToolPermission(brain.id, "maps.search", "ask");
    await service.ensureToolPermission(brain.id, "maps.display", "ask");
    await service.ensureToolPermission(brain.id, "brain.substrate", "ask");
    await service.ensureToolPermission(brain.id, "studio.clipboard", "auto");
    await service.ensureToolPermission(brain.id, "system.clipboard", "off");
    await service.ensureToolPermission(brain.id, "system.memory", "off");
    await service.ensureToolPermission(brain.id, "runtime.sandbox", "ask");
    await service.ensureToolPermission(brain.id, "runtime.vm", "ask");
    await service.ensureToolPermission(brain.id, "code.assembly", "ask");
  }
  const mcpApps = new McpAppService({
    requireAccess: origin => mcp.requireInteractionOrigin(origin),
    tool: (origin, name) => mcp.appTool(origin, name), resourceTool: origin => mcp.appResourceTool(origin),
    execute: async (invocation, requestId, viewId, signal) => {
      signal.throwIfAborted(); const cancel = (): void => { tools.cancel(invocation.brainId, requestId); };
      signal.addEventListener("abort", cancel, { once: true });
      try { return await tools.execute(invocation, undefined, requestId, undefined, { actionEventId: requestId, mcpAppViewId: viewId }); }
      finally { signal.removeEventListener("abort", cancel); }
    },
    approveTool: (origin, invocation, signal) => mcpInteractions.confirmApp(origin, { category: "tool", message: "Allow this connected-app server action?",
      toolId: invocation.toolId, arguments: invocation.arguments }, signal),
    approveNetwork: (origin, csp, signal) => mcpInteractions.confirmApp(origin, { category: "network", message: "Allow these exact connected-app network origins?", csp }, signal),
    approveLink: (origin, url, signal) => mcpInteractions.confirmApp(origin, { category: "link", message: "Open this connected-app link in your browser?", url }, signal),
    serverCapabilities: origin => mcp.serverCapabilities(origin),
    log: async (origin, params, signal, context) => {
      if (!mcpDiagnosticsRuntime) throw new Error("Connected-app diagnostics are not ready.");
      await mcpDiagnosticsRuntime.receive(origin, params, signal, "app", context.viewId, context.privateSecrets);
    },
    openBrowser: url => shell.openExternal(url),
    admit: async bytes => { if (bytes > freemem()) throw new Error("The App awaits enough physical memory for this view."); },
    theme: () => nativeTheme.shouldUseDarkColors ? "dark" : "light",
    artifacts: brainId => new McpAppArtifactStore(repository.brainDirectory(brainId), brainId, {
      admit: async bytes => { if (bytes > freemem()) throw new Error("Saved App data awaits enough physical transfer RAM."); },
      checkDisk: bytes => requireDiskWrite(repository.brainDirectory(brainId), { operationWriteBytes: bytes }).then(() => undefined)
    }),
    observe: async (origin, request, signal) => {
      const review = await mcpInteractions.reviewAppData(origin, request, signal);
      if (!review.accepted) throw new Error("The user declined this App task data. It was not admitted or learned.");
      signal.throwIfAborted(); mcp.requireInteractionOrigin(origin);
      const store = new McpAppArtifactStore(repository.brainDirectory(origin.brainId), origin.brainId, {
        signal, admit: async bytes => { if (bytes > freemem()) throw new Error("App task data awaits physical transfer RAM."); },
        checkDisk: bytes => requireDiskWrite(repository.brainDirectory(origin.brainId), { operationWriteBytes: bytes }).then(() => undefined)
      });
      // No writer reacquisition, implicit Load, fake tool result or hidden
      // future prompt. Actual ordinary native admission is a separate gate.
      const receipt = await store.queueObservation(request.view.id, request.captureSha256, {
        brainId: origin.brainId, serverId: origin.serverId, serverLabel: origin.serverLabel, endpoint: origin.endpoint
      }, review.reviewId, request.data, Math.floor(freemem() / 100));
      mcpAppObservationDelivery?.notify(origin.brainId); return receipt;
    },
    currentOrigin: (brainId, serverId) => { service.requireInstanceLoaded(brainId); return mcp.currentAppOrigin(brainId, serverId); }
  });
  mcpAppService = mcpApps; mcp.setAppHost(mcpApps);
  const evolution = new EvolutionController(repository, tools, engine);
  const questions = new UserQuestionService({
    brainDirectory: async id => { await repository.get(id); return repository.brainDirectory(id); },
    onError: (error, brainId) => console.error(`Question state for ${brainId}:`, error)
  });
  questionService = questions;
  questionDelivery = new UserQuestionDelivery(questions, service, error => console.error("Question answer delivery:", error));
  tools.setQuestionService(questions);
  const questionOwners = (await repository.list()).map(brain => brain.id);
  await questions.recover(questionOwners);
  for (const id of questionOwners) await questionDelivery.recover(id).catch(error => console.error("Question recovery:", error));
  const actions = new ChatActionController(service, tools, evolution);
  actions.on("event", (event: import("../shared/types").ActionEvent) => osWork.observeActionEvent(event));
  actions.on("event", (event: import("../shared/types").ActionEvent) => {
    // Lifecycle metadata only. Reading private content still requires the
    // module's declared host tool and the instance's current permission.
    const data = { id: event.id, kind: event.action.kind, state: event.state, updatedAt: event.updatedAt };
    void runtimeModules.dispatchEvent(event.brainId, { topic: "chat.action",
      eventId: createHash("sha256").update(JSON.stringify(data)).digest("hex"), data })
      .catch(error => console.error("Plugin runtime action event:", error));
  });
  const directions = new SteerDirectionsService({
    brainDirectory: id => repository.brainDirectory(id),
    reserveWrite: (root, bytes) => requireDiskWrite(root, { operationWriteBytes: bytes }).then(() => undefined),
    admit: async bytes => { if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 4) throw new Error("Steering history awaits enough physical transfer memory."); },
    onError: (error, id) => console.error(`Steering presentation history for ${id}:`, error)
  });
  actions.setSteerDirections(directions);
  actions.setImaginationObserver((event, job) => {
    const origin = mainImaginationCaptureOrigin(event);
    if (event.neuralActionId) mediaArtifacts!.noteCaptureOrigin(`action:${event.brainId}:${event.neuralActionId}`, origin);
    if (job && !["complete", "failed", "cancelled"].includes(job.state)) mediaArtifacts!.noteCaptureOrigin(`job:${job.brainId}:${job.id}`, origin);
  });
  service.onInstanceLifecycle(state => { if (state.phase === "loaded") { imaginationCapture.activate(state.brainId); imaginationAssemblyRuntime?.activate(state.brainId); } });
  service.addInstanceSuspendBoundary(async id => { await imaginationAssemblyRuntime?.release(id, false); await imaginationCapture.release(id); mediaArtifacts!.releaseCaptureOrigins(id); });
  service.addInstanceRecoveryBoundary(async id => { await imaginationAssemblyRuntime?.release(id); await imaginationCapture.release(id); });
  actions.setChatSuspensionRecorder(async (receipt, source) => {
    const root = repository.brainDirectory(receipt.brainId), bytes = Buffer.byteLength(JSON.stringify({ receipt, source }));
    if (bytes * 4 > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed)) throw new Error("Suspension display history awaits transfer memory.");
    await requireDiskWrite(root, { operationWriteBytes: bytes + 65536 });
    await persistChatHostSuspension(root, receipt, source);
  });
  service.addInstanceRecoveryBoundary(async id => { directions.forget(id); });
  actions.setWarmTaskAttachmentTransfer((request, signal) => continuedTaskAttachments.transfer(request, signal));
  const appObservationDelivery = new McpAppObservationDelivery({
    store: id => new McpAppArtifactStore(repository.brainDirectory(id), id, {
      admit: async bytes => { if (bytes > freemem()) throw new Error("App observation awaits physical transfer RAM."); },
      checkDisk: bytes => requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: bytes }).then(() => undefined)
    }),
    directory: id => repository.brainDirectory(id),
    requireLoaded: id => service.requireInstanceLoaded(id),
    requireSource: record => {
      const current = mcp.currentAppOrigin(record.source.brainId, record.source.serverId);
      if (current.endpoint !== record.source.endpoint) throw new Error("The App observation source changed; review it before continuing.");
    },
    withWriter: (id, task, signal) => withBrainWrite(repository, id, task, signal),
    frontier: (id, path, signal, boundary) => engine!.appObservationFrontier(id, path, signal, boundary),
    admit: (input, path, signal, boundary) => engine!.admitAppObservation(input, path, signal, boundary),
    followup: (id, input, turnId, proof, signal) => actions.send(id, input, signal, turnId, undefined, undefined, proof),
    ownerReleased: async id => {
      engine!.holdInstanceDispatch(id);
      await service.instanceLifecycle.confirmedNativeOwnerRelease(id);
      suspendMcpForNativeOwnerRelease(service, mcp, id);
    },
    publish: event => mcpApps.publishObservation(event)
  });
  mcpAppObservationDelivery = appObservationDelivery;
  service.onInstanceLifecycle(state => { if (state.phase === "loaded") appObservationDelivery.notify(state.brainId); });
  engine.on("activity", (event: import("./engineSupervisor").EngineActivityTransition) => {
    if (["complete", "failed", "cancelled"].includes(event.state) && !["app_observation_frontier", "admit_app_observation"].includes(event.method)) appObservationDelivery.retryWaiting();
  });
  actions.on("stream", (event: import("../shared/types").ChatStreamEvent) => {
    if (event.type !== "chat-reply-committed" || !event.humanMessage.inputProvenance) return;
    const store = new McpAppArtifactStore(repository.brainDirectory(event.brainId), event.brainId);
    void store.reconcileFollowup(event.humanMessage).then(receipt => {
      if (receipt) { mcpApps.publishObservation({ brainId: event.brainId, artifactId: receipt.artifactId, receipt }); appObservationDelivery.notify(event.brainId); }
    }).catch(error => console.error("App follow-up reconciliation:", error));
  });
  const learning: LearningCoordinator = new LearningCoordinator({
    brainDirectory: async id => { await repository.get(id); return repository.brainDirectory(id); },
    requireLoaded: (id, turnId) => service.requireInstanceLoaded(id, turnId),
    authorize: async (id, kind) => {
      const toolId = kind === "fact-check" ? "data.fact-check" : "data.learn";
      const permission = (await service.listToolPermissions(id)).find(row => row.toolId === toolId);
      if (!permission || permission.level === "off") throw new Error(`${toolId === "data.learn" ? "Learning" : "Fact Check"} is disabled for this instance. Change its access setting before starting or resuming work.`);
    },
    reserveWrite: async (directory, bytes) => { await requireDiskWrite(directory, { operationWriteBytes: bytes }); },
    manifest: (id, manifestId) => service.datasets.manifest(id, manifestId),
    prepare: (id, paths, signal) => service.previewDataset(id, paths, { signal }),
    ingest: (id, manifestId, epochs, control) => service.ingestManifest(id, manifestId, "encode", control.cancelled,
      (progress, label, detail) => control.progress(progress, label, detail?.coverage), false, epochs, false, control.jobId, control.signal),
    crawl: (request, control, policy) => service.crawlWeb(request, control.cancelled, control.progress, control.signal, control.jobId, policy),
    questions,
    teachCapabilities: (brainId, pack, signal) => service.teachSupplementaryCapabilities(brainId, pack, signal),
    assess: async (workspace, claim, signal) => {
      let turn = factCheckNativeTask(workspace, claim);
      const inherited = workspace.task.retainedNativeTurn && workspace.task.inheritedFrom?.executionId;
      const continuation = inherited ? await service.instanceInheritedResumePresentation(workspace.task.brainId, inherited)
        : await service.instanceResumePresentation(workspace.task.brainId);
      if (continuation) {
        const binding = await learning.executionContext(workspace.task.brainId, continuation.frontier.turnId);
        if (continuation.hostAttached || binding?.taskId !== workspace.task.id || binding.claimId !== claim.id) throw new Error("Another retained native task must resume first.");
        turn = { turnId: continuation.frontier.turnId, input: continuation.input };
      }
      await learning.bindNativeTurn(workspace.task.brainId, workspace.task.id, turn.turnId, claim.id, createHash("sha256").update(turn.input).digest("hex"));
      const result = await actions.send(workspace.task.brainId, turn.input, signal, turn.turnId, undefined, continuation ?? undefined);
      const receipt = committedNativeLearningTask(result, turn);
      await learning.recordNativeAssessment(workspace.task.brainId, workspace.task.id, claim.id, { ...receipt,
        policySha256: workspace.task.retainedNativeTurn?.policySha256 ?? workspace.task.policy.sha256 });
      // Only an actual data.fact-check.record action writes a finding. Plain
      // prose is not parsed into a claimed truth or fabricated verification.
      return undefined;
    },
    followInstructions: async (task, signal) => {
      let turn = specificLearnNativeTask(task);
      const inherited = task.retainedNativeTurn && task.inheritedFrom?.executionId;
      const continuation = inherited ? await service.instanceInheritedResumePresentation(task.brainId, inherited)
        : await service.instanceResumePresentation(task.brainId);
      if (continuation) {
        const binding = await learning.executionContext(task.brainId, continuation.frontier.turnId);
        if (continuation.hostAttached || binding?.taskId !== task.id || binding.claimId) throw new Error("Another retained native task must resume first.");
        turn = { turnId: continuation.frontier.turnId, input: continuation.input };
      }
      await learning.bindNativeTurn(task.brainId, task.id, turn.turnId, undefined, createHash("sha256").update(turn.input).digest("hex"));
      const result = await actions.send(task.brainId, turn.input, signal, turn.turnId, undefined, continuation ?? undefined);
      return { ...committedNativeLearningTask(result, turn), policySha256: task.retainedNativeTurn?.policySha256 ?? task.policy.sha256 };
    },
    onError: error => console.error("Learning workspace:", error)
  });
  learningCoordinator = learning;
  const clock = createClockRuntime({ repository, service, actions, tools,
    isLearning: id => jobs.isLearning(id) || initialization.isBusy() || learning.isLearning(id),
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    notify: occurrence => {
      if (!Notification.isSupported()) throw new Error("Desktop notifications are unavailable on this host.");
      const notification = new Notification({ title: occurrence.label, body: occurrence.input
        ? `${occurrence.input.slice(0, 160)}${occurrence.input.length > 160 ? "…" : ""}` : "Timer elapsed.", silent: false });
      notification.on("click", () => { void showOrCreateMainWindow(); }); notification.show();
    },
    onError: (error, id) => console.error(`Clock workspace ${id}:`, error)
  });
  const checkHostToolResources = async (brainId: string, bytes: number, _purpose?: string, childPids: readonly number[] = []): Promise<void> => {
      const brain = await repository.get(brainId, false), envelope = await resourcePlanner.reviewedRamEnvelope(brain.config);
      sharedResources?.requireSynchronized(); const selected = sharedResources?.status().globalRamCeilingBytes;
      const pids = [...app.getAppMetrics().map(metric => metric.pid), process.pid, ...(engine?.managedProcessIds ?? []), ...localTeacherChildPids(), ...childPids];
      requireAutomationHeadroom(bytes, { residentBytes: await automationResidentBytes(pids), ceilingBytes: Math.min(envelope.systemRamBudgetBytes, selected ?? envelope.systemRamBudgetBytes) });
  };
  const fileTools = new FileToolService({
    root: async id => { service.requireRuntimeAdmissionOpen(); await repository.get(id, false); return repository.brainDirectory(id); },
    readTask: (id, turnId, attachmentId, offset, bytes, signal) => continuedTaskAttachments.readForTask(id, turnId, attachmentId, offset, bytes, signal),
    attachmentPermission: async (id, signal) => {
      signal.throwIfAborted();
      const permission = (await service.listToolPermissions(id)).find(row => row.toolId === "task.attachments");
      signal.throwIfAborted();
      if (!permission || permission.level === "off") throw new Error("Current-task attachment access is disabled for this instance.");
      return createHash("sha256").update(JSON.stringify({ toolId: permission.toolId, level: permission.level,
        revision: permission.revision ?? 0, updatedAt: permission.updatedAt ?? null })).digest("hex");
    },
    assertTaskSource: async (id, turnId, source, signal) => {
      signal.throwIfAborted();
      const references = await continuedTaskAttachments.listForTurn(id, turnId, signal);
      const current = references.find(row => row.attachmentId === source.attachmentId);
      if (source.kind !== "task-attachment" || !current || current.contentSha256 !== source.contentSha256 ||
          current.manifestSha256 !== source.manifestSha256 || current.bytes !== source.bytes)
        throw new Error("The current task no longer owns this exact imported file.");
      signal.throwIfAborted();
    },
    checkResources: checkHostToolResources,
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); }
  });
  tools.setFileToolService(fileTools); fileToolRuntime = fileTools;
  service.onToolPermissionChanged(({ brainId }) => { fileTools.cancelBrain(brainId);
    void fileTools.settle(brainId).catch(error => console.error("File-tool permission retirement:", error)); });
  service.addInstanceSuspendBoundary(id => fileTools.forget(id));
  service.addInstanceRecoveryBoundary(id => fileTools.forget(id));
  const mcpRoots = new McpRootsService({ directory: join(deviceDataRoot, "mcp-roots"),
    connection: (id, serverId) => mcp.hostConnection(id, serverId), requireConnection: connection => mcp.requireHostConnection(connection),
    filePolicy: async (id, signal) => {
      signal.throwIfAborted(); const row = (await service.listToolPermissions(id)).find(value => value.toolId === "system.files"); signal.throwIfAborted();
      return { level: row?.level ?? "off", revision: createHash("sha256").update(JSON.stringify({ toolId: "system.files", level: row?.level ?? "off",
        revision: row?.revision ?? 0, updatedAt: row?.updatedAt ?? null })).digest("hex") };
    },
    approve: (connection, review, signal) => mcpInteractions.confirmRoots(connection, review, signal),
    admit: (bytes, id) => checkHostToolResources(id, bytes, "bounded MCP root disclosure"),
    checkDisk: async bytes => { await requireDiskWrite(deviceDataRoot, { operationWriteBytes: bytes }); },
    notify: connection => mcp.notifyRootsChanged(connection)
  });
  const mcpDiagnostics = new McpDiagnosticService({ requireAccess: origin => mcp.requireInteractionOrigin(origin),
    knownSecrets: origin => mcp.knownOperationalSecrets(origin), admit: (bytes, id) => checkHostToolResources(id, bytes, "bounded MCP diagnostics") });
  mcpRootsRuntime = mcpRoots; mcpDiagnosticsRuntime = mcpDiagnostics;
  mcp.setStandardHostServices({ roots: (origin, signal) => mcpRoots.list(origin, signal),
    log: async (origin, params, signal) => { await mcpDiagnostics.receive(origin, params, signal); },
    connectionRetired: (id, serverId) => { void mcpRoots.revokeConnection(id, serverId)
      .catch(error => console.error("MCP source root-disclosure retirement:", error)); } });
  service.onToolPermissionChanged(({ brainId, toolId }) => { if (toolId === "system.files")
    void mcpRoots.revokeBrain(brainId).catch(error => console.error("MCP root disclosure revocation:", error)); });
  service.addInstanceSuspendBoundary(id => mcpRoots.revokeBrain(id));
  service.addInstanceRecoveryBoundary(id => mcpRoots.revokeBrain(id));
  const sparseRead = <T>(id: string, method: string, params: Record<string, unknown>, signal: AbortSignal): Promise<T> =>
    engine!.request<T>(method, { ...params, brainId: id, storagePath: repository.brainDirectory(id) }, undefined, signal);
  const sparseSubstrate = new SparseSubstrateTransferService({
    brainDirectory: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    assertOwner: async id => { service.requireRuntimeAdmissionOpen(); await repository.get(id, false); },
    checkResources: checkHostToolResources,
    checkDisk: async (id, bytes) => { await requireDiskWrite(repository.brainDirectory(id), { operationWriteBytes: bytes }); },
    deferWrite: (id, operation, signal) => withBrainWrite(repository, id, operation, signal),
    assertLoadedBoundary: async id => {
      try { service.requireInstanceLoaded(id); }
      catch { throw new SparseTransferBoundaryError("Sparse import waits for explicit Load and a safe reply boundary.", "boundary-required"); }
    },
    selectedBundle: async (id, turnId, attachmentId, signal) => {
      const permission = (await service.listToolPermissions(id)).find(row => row.toolId === "task.attachments");
      if (!permission || permission.level === "off") throw new Error("Current-task attachment access is disabled.");
      const selected = await continuedTaskAttachments.resolveForTurn(id, turnId, attachmentId, signal);
      const root = repository.brainDirectory(id), path = resolve(root, ...selected.relativePath.split("/")), local = relative(root, path);
      if (selected.attachmentId !== attachmentId || !local || local.startsWith("..") || isAbsolute(local)) throw new Error("Sparse bundle attachment escaped its current task owner.");
      return { path, sha256: selected.contentSha256, bytes: selected.bytes };
    },
    native: {
      inspect: (id, query, signal) => sparseRead(id, "sparse_substrate_inspect", { query }, signal),
      export: (id, selection, destination, signal) => sparseRead(id, "sparse_substrate_export", { selection, destination }, signal),
      validateBundle: (id, bundlePath, bundleSha256, signal) => sparseRead(id, "sparse_substrate_validate_bundle", { bundlePath, bundleSha256 }, signal),
      prepare: (id, bundlePath, bundleSha256, destination, mode, signal) => sparseRead(id, "sparse_substrate_prepare", { bundlePath, bundleSha256, destination, mode }, signal),
      prepareRollback: (id, originalPlan, originalPlanSha256, receiptJson, receiptSha256, destination, signal) =>
        sparseRead(id, "sparse_substrate_prepare_rollback", { originalPlan, originalPlanSha256, receiptJson, receiptSha256, destination }, signal),
      head: (id, signal) => sparseRead(id, "sparse_substrate_head", {}, signal),
      apply: async (id, planPath, input, signal) => {
        try { return await engine!.applySparseSubstrate({ ...input, brainId: id, storagePath: repository.brainDirectory(id), planPath }, signal,
          () => service.requireInstanceLoaded(id)); }
        catch (error) { throw sparseSubstrateRpcError(error, id); }
      }
    },
    ownerReleased: async id => {
      engine!.holdInstanceDispatch(id);
      await service.instanceLifecycle.confirmedNativeOwnerRelease(id);
      suspendMcpForNativeOwnerRelease(service, mcp, id);
    }
  });
  sparseSubstrateRuntime = sparseSubstrate; tools.setSparseSubstrateService(sparseSubstrate);
  service.onToolPermissionChanged(({ brainId }) => { void sparseSubstrate.authorityChanged(brainId).catch(error => console.error("Sparse authority revocation:", error)); });
  service.addInstanceSuspendBoundary(id => sparseSubstrate.forget(id));
  service.addInstanceRecoveryBoundary(id => sparseSubstrate.forget(id));
  service.onInstanceLifecycle(state => { if (state.phase === "loaded") void sparseSubstrate.boundaryAvailable(state.brainId).catch(error => console.error("Sparse boundary:", error)); });
  engine.on("activity", (event: import("./engineSupervisor").EngineActivityTransition) => {
    if (event.brainId && ["complete", "failed", "cancelled"].includes(event.state) && !event.method.startsWith("sparse_substrate_")) {
      void sparseSubstrate.boundaryAvailable(event.brainId).catch(error => console.error("Sparse activity boundary:", error));
    }
  });
  const nativeCandidateTools = new NativeCandidateToolService({
    controller: evolution,
    deferWrite: (id, operation, signal) => withBrainWrite(repository, id, operation, signal),
    assertOwner: async id => { service.requireRuntimeAdmissionOpen(); await repository.get(id, false); },
    assertLoadedBoundary: async id => {
      service.requireRuntimeAdmissionOpen();
      // These main-only admission reads cannot start or Load a worker. Actual
      // dispatch separately binds the same child after acquiring its queue.
      if (!service.instanceLifecycle.isLoaded(id)) throw new NativeCandidateToolBoundaryError("waiting-load", "Native improvement waits for explicit Load.");
      if (!service.instanceLifecycle.canRun(id)) throw new NativeCandidateToolBoundaryError("waiting-boundary", "Continue the saved native turn before applying an improvement.");
      service.requireInstanceLoaded(id);
    },
    checkResources: checkHostToolResources
  });
  nativeCandidateToolRuntime = nativeCandidateTools; tools.setNativeCandidateToolService(nativeCandidateTools);
  service.onToolPermissionChanged(({ brainId }) => { void nativeCandidateTools.authorityChanged(brainId).catch(error => console.error("Native candidate authority:", error)); });
  service.addInstanceSuspendBoundary(id => nativeCandidateTools.forget(id));
  service.addInstanceRecoveryBoundary(id => nativeCandidateTools.forget(id));
  service.onInstanceLifecycle(state => {
    if (state.phase === "loaded") void nativeCandidateTools.boundaryAvailable(state.brainId).catch(error => console.error("Native candidate boundary:", error));
  });
  engine.on("activity", (event: import("./engineSupervisor").EngineActivityTransition) => {
    if (event.brainId && ["complete", "failed", "cancelled"].includes(event.state) && !event.method.startsWith("evolution."))
      void nativeCandidateTools.boundaryAvailable(event.brainId).catch(error => console.error("Native candidate activity boundary:", error));
  });
  const localTeacher = new LocalTeacherService({
    brainDirectory: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    currentPermission: async id => { const row = (await service.listToolPermissions(id)).find(value => value.toolId === "data.distill-local");
      return { level: row?.level ?? "off", revision: createHash("sha256").update(JSON.stringify({ id: "data.distill-local", level: row?.level ?? "off", revision: row?.revision ?? 0, updatedAt: row?.updatedAt ?? null })).digest("hex") }; },
    selectFiles: async request => {
      const runtime = await dialog.showOpenDialog({ title: `Select installed ${request.runner} · its --help will be executed without a model`, properties: ["openFile"] });
      if (runtime.canceled || !runtime.filePaths[0]) return null;
      const model = await dialog.showOpenDialog({ title: "Select local GGUF teacher · no download or Omni identity replacement", properties: ["openFile"], filters: [{ name: "GGUF", extensions: ["gguf"] }] });
      return model.canceled || !model.filePaths[0] ? null : { runtimePath: runtime.filePaths[0], modelPath: model.filePaths[0] };
    },
    requireLoaded: id => service.requireInstanceLoaded(id), isLoaded: async id => (await service.instanceStatus(id)).phase === "loaded",
    measurements: async (id, childPids = []) => { const brain = await repository.get(id, false), envelope = await resourcePlanner.reviewedRamEnvelope(brain.config);
      sharedResources?.requireSynchronized(); const selected = sharedResources?.status().globalRamCeilingBytes;
      const pids = [...app.getAppMetrics().map(metric => metric.pid), process.pid, ...(engine?.managedProcessIds ?? []), ...localTeacherChildPids(), ...childPids];
      return { appFamilyResidentBytes: await automationResidentBytes(pids), selectedRamCeilingBytes: Math.min(envelope.systemRamBudgetBytes, selected ?? envelope.systemRamBudgetBytes),
        physicalFreeBytes: Math.floor(freemem()), logicalCpuCount: cpus().length }; },
    checkMetadata: (id, bytes, pids) => checkHostToolResources(id, bytes, "local teacher source/protocol metadata", pids),
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    withBrainWrite: (id, work, signal) => withBrainWrite(repository, id, work, signal), withHostCompute: (context, signal, work) => engine!.withHostCompute(context, signal, work),
    previewDataset: (id, paths, signal) => service.previewDataset(id, paths, { signal }), learning,
    onError: (error, id) => console.error(`Local teacher ${id ?? ""}:`, error)
  });
  localTeacherRuntime = localTeacher; tools.setLocalTeacherService(localTeacher);
  localTeacher.subscribe(event => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(IPC.localTeacher.event, event); });
  localTeacher.subscribeSelection(event => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(IPC.localTeacher.selectionEvent, event); });
  service.onToolPermissionChanged(({ brainId, toolId }) => { if (toolId === "data.distill-local") localTeacher.permissionChanged(brainId); });
  service.addInstanceSuspendBoundary(async id => { await localTeacher.forget(id); });
  service.addInstanceRecoveryBoundary(id => localTeacher.forget(id));
  service.onInstanceLifecycle(state => { if (state.phase === "loaded") void localTeacher.resumeAfterLoad(state.brainId).catch(error => console.error("Local teacher Load continuation:", error)); });
  for (const id of questionOwners) await localTeacher.recover(id).catch(error => console.error("Local teacher recovery:", error));
  const automations = createAutomationRuntime({ repository, service, tools, clock,
    checkResources: checkHostToolResources,
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    onError: (error, id) => console.error(`Action files ${id}:`, error)
  });
  const imaginationAssembly = new ImaginationAssemblyService({
    brainDirectory: id => repository.brainDirectory(id),
    snapshotDatabase: (id, recordingId, revision, destination) => imaginationCapture.snapshotTimelineDatabase(id, recordingId, revision, destination),
    prepareRuntime: (signal, onProgress) => new VideoRuntimeProvisioner({ cacheRoot: videoRuntimeCacheRoot, onProgress }).prepare(signal),
    checkResources: (id, bytes, pids) => checkHostToolResources(id, bytes, "delivered timeline replay", pids),
    reserve: (root, bytes) => requireDiskWrite(root, { operationWriteBytes: bytes }).then(() => undefined),
    mediaUrl: (id, sha, mime) => mediaArtifacts!.leaseReplay(id, sha, mime),
    presentationRefreshHz: () => screen.getPrimaryDisplay().displayFrequency
  });
  imaginationAssemblyRuntime = imaginationAssembly;
  const clipboard = new ClipboardService({
    brainDirectory: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    assertBrainOwner: async id => { await repository.get(id, false); },
    requireActiveOwner: async id => { if ((await service.instanceStatus(id)).phase !== "loaded") throw new Error("Load this instance explicitly before clipboard access; no automatic Load was attempted."); },
    currentPermission: async (id, toolId) => {
      const row = (await service.listToolPermissions(id)).find(permission => permission.toolId === toolId);
      return { level: row?.level ?? "off", revision: createHash("sha256").update(JSON.stringify({ toolId,
        level: row?.level ?? "off", updatedAt: row?.updatedAt ?? null, revision: row?.revision ?? 0 })).digest("hex") };
    },
    checkResources: checkHostToolResources,
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    host: new SystemClipboardHostAdapter(),
    onError: (error, id) => console.error(`Clipboard holder ${id}:`, error)
  });
  tools.setClipboardService(clipboard); clipboardRuntime = clipboard;
  service.onToolPermissionChanged(({ brainId, toolId }) => clipboard.permissionChanged(brainId, toolId));
  service.addInstanceSuspendBoundary(async brainId => { await clipboard.forget(brainId); });
  service.addInstanceRecoveryBoundary(id => clipboard.forget(id));
  const processMemory = new ProcessMemoryService({
    installationDirectory: join(deviceDataRoot, "host-process-access"),
    assertBrainOwner: async id => { await repository.get(id, false); },
    requireActiveOwner: async id => { if ((await service.instanceStatus(id)).phase !== "loaded") throw new Error("Load this instance explicitly before process-memory access; no automatic Load was attempted."); },
    currentPermission: async id => {
      const row = (await service.listToolPermissions(id)).find(permission => permission.toolId === "system.memory");
      return { level: row?.level ?? "off", revision: createHash("sha256").update(JSON.stringify({ toolId: "system.memory",
        level: row?.level ?? "off", updatedAt: row?.updatedAt ?? null, revision: row?.revision ?? 0 })).digest("hex") };
    },
    protectedPids: () => [...app.getAppMetrics().map(metric => metric.pid), process.pid, ...(engine?.managedProcessIds ?? []), ...clipboardHostHelperPids(), ...processMemoryHelperPids()],
    checkResources: checkHostToolResources,
    reserveWrite: async bytes => { await requireDiskWrite(deviceDataRoot, { operationWriteBytes: bytes }); },
    run: (request, context) => runProcessMemoryHelper(processMemoryLaunch({ appPath, resourcesPath: process.resourcesPath,
      packagedRequired: app.isPackaged || process.env.OMNI_PACKAGED_ENGINE_REQUIRED === "1" }), request, context)
  });
  tools.setProcessMemoryService(processMemory); processMemoryRuntime = processMemory;
  service.onToolPermissionChanged(({ brainId, toolId }) => { if (toolId === "system.memory") processMemory.permissionChanged(brainId); });
  service.addInstanceSuspendBoundary(async brainId => { await processMemory.forget(brainId); });
  service.addInstanceRecoveryBoundary(id => processMemory.forget(id));
  const environments = new EnvironmentService({
    brainDirectory: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    run: runProcess,
    checkResources: checkHostToolResources,
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    vmRuntimeDirectory: async (brainId, environmentId) => join(deviceDataRoot, "environment-runtimes", brainId, environmentId),
    deleteVmRuntime: async brainId => {
      const root = resolve(deviceDataRoot, "environment-runtimes"), target = resolve(root, brainId), fromRoot = relative(root, target);
      if (!fromRoot || fromRoot === ".." || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot)) throw new Error("VM runtime deletion escaped its installation owner.");
      await rm(target, { recursive: true, force: true });
    },
    resolveVmImage: async (brainId, turnId, attachmentId, expectedSha256, architecture, signal) => {
      const attachmentPermission = (await service.listToolPermissions(brainId)).find(row => row.toolId === "task.attachments");
      if (!attachmentPermission || attachmentPermission.level === "off") throw new Error("Current-task attachment access is disabled for this instance.");
      const resolved = await continuedTaskAttachments.resolveForTurn(brainId, turnId, attachmentId, signal);
      if (resolved.attachmentId !== attachmentId || resolved.contentSha256 !== expectedSha256) throw new Error("The reviewed VM image attachment changed.");
      const brainRoot = repository.brainDirectory(brainId), path = resolve(brainRoot, ...resolved.relativePath.split("/")), fromRoot = relative(brainRoot, path);
      if (!fromRoot || fromRoot.startsWith("..") || isAbsolute(fromRoot)) throw new Error("VM image attachment escaped its owning brain.");
      return { attachmentId, path, sha256: resolved.contentSha256, bytes: resolved.bytes, architecture };
    },
    onError: (error, id) => console.error(`Managed environment ${id}:`, error)
  });
  tools.setEnvironmentService(environments);
  environmentRuntime = environments;
  service.addInstanceSuspendBoundary(async (brainId, report) => {
    if (environments.activeCount(brainId)) report({ kind: "external-effect", label: "Cancelling the current sandbox/VM operation at its receipt boundary; external VM state will require explicit inspection" });
    await environments.forget(brainId);
  });
  service.addInstanceRecoveryBoundary(id => environments.forget(id));
  tools.setAssemblyService({ execute: async (brainId, action, args, _requestId, signal) => {
    await repository.get(brainId, false);
    return executeAssembly({ root: repository.brainDirectory(brainId), run: runProcess,
      checkResources: (bytes, purpose, childPids) => checkHostToolResources(brainId, bytes, purpose, childPids),
      reserveWrite: async bytes => { await requireDiskWrite(repository.brainDirectory(brainId), { operationWriteBytes: bytes }); }
    }, action, args, signal);
  } });
  const webImages = new WebImageService({
    root: async id => { await repository.get(id, false); return repository.brainDirectory(id); },
    initializeAttachments: id => new TaskAttachmentStore(repository.brainDirectory(id), id).initialize(),
    attachments: taskAttachments,
    requireLoaded: async id => { if ((await service.instanceStatus(id)).phase !== "loaded") throw new Error("Load this instance explicitly before native web-image inspection."); },
    permission: async id => { const row = (await service.listToolPermissions(id)).find(value => value.toolId === "web.fetch");
      return row ? { level: row.level, revision: webImagePermissionRevision(row) } : { level: "off", revision: "0".repeat(64) }; },
    learningContext: (id, turnId) => learning.executionContext(id, turnId),
    checkResources: checkHostToolResources,
    reserveWrite: async (root, bytes) => { await requireDiskWrite(root, { operationWriteBytes: bytes }); },
    fetchPublic: safeFetch
  });
  tools.setWebImageService(webImages);
  service.setWebImageObservationValidator((id, turnId, selected, signal) => webImages.validateObservation(id, turnId, selected, signal));
  automationRuntime = automations; await automations.recover(questionOwners);
  clockRuntime = clock; await clock.recover(questionOwners); clock.start();
  powerMonitor.on("suspend", () => clockRuntime?.systemSuspending());
  powerMonitor.on("resume", () => { void clockRuntime?.systemResumed().catch(error => console.error("Clock resume:", error)); });
  tools.setLearningCoordinator(learning);
  actions.setLearningContextResolver((brainId, turnId) => learning.executionContext(brainId, turnId));
  learning.subscribe(event => actions.observeLearningTask(event.task));
  service.addInstanceSuspendBoundary(async (brainId, report) => {
    if (learning.isLearning(brainId)) report({ kind: "native-boundary", label: "Waiting for this learning source's committed boundary" });
    await learning.pauseAtBoundary(brainId);
  });
  service.onInstanceLifecycle(state => {
    if (state.phase === "loaded") void learning.resumeAfterLoad(state.brainId).catch(error => console.error("Learning continuation after Load:", error));
  });
  service.addInstanceRecoveryBoundary(async id => {
    await questions.pauseForRecovery(id); await questionDelivery?.settle(id); await learning.detachAndForget(id);
  });
  for (const id of questionOwners) await learning.recover(id).catch(error => console.error("Learning recovery:", error));
  actions.setCapabilityPreparation(async (brainId, signal) => {
    try {
      const supplement = await learning.installCapabilities(brainId);
      await service.teachSupplementaryCapabilities(brainId, declaredLocalTeacherCapabilityPack(), signal);
      await service.teachSupplementaryCapabilities(brainId, declaredTeacherStatsCapabilityPack(), signal);
      const clockTeaching = await service.teachSupplementaryCapabilities(brainId, declaredClockCapabilityPack(), signal);
      const actionTeaching = await service.teachSupplementaryCapabilities(brainId, declaredAutomationCapabilityPack(), signal);
      const mapTeaching = await service.teachSupplementaryCapabilities(brainId, declaredMapCapabilityPack(), signal);
      await service.teachSupplementaryCapabilities(brainId, declaredMapBasemapCapabilityPack(), signal);
      const assemblyTeaching = await service.teachSupplementaryCapabilities(brainId, declaredAssemblyCapabilityPack(), signal);
      const environmentTeaching = await service.teachSupplementaryCapabilities(brainId, declaredEnvironmentCapabilityPack(), signal);
      const memoryTeaching = await service.teachSupplementaryCapabilities(brainId, declaredProcessMemoryCapabilityPack(), signal);
      const clipboardTeaching = await service.teachSupplementaryCapabilities(brainId, declaredClipboardCapabilityPack(), signal);
      const artifactTeaching = await service.teachSupplementaryCapabilities(brainId, declaredHtmlArtifactCapabilityPack(), signal);
      await service.teachSupplementaryCapabilities(brainId, declaredWebImageCapabilityPack(), signal);
      await service.teachSupplementaryCapabilities(brainId, declaredFileCapabilityPack(), signal);
      const servers = await mcp.learnCapabilities(brainId, signal);
      signal.throwIfAborted();
      return [...(clipboardTeaching?.candidateAccepted === true ? [] : ["Clipboard-tool teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(memoryTeaching?.candidateAccepted === true ? [] : ["Process-memory teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(artifactTeaching?.candidateAccepted === true ? [] : ["HTML-artifact teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(environmentTeaching?.candidateAccepted === true ? [] : ["Sandbox/VM teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(assemblyTeaching?.candidateAccepted === true ? [] : ["Assembly-tool teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(mapTeaching?.candidateAccepted === true ? [] : ["Private-map teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(actionTeaching?.candidateAccepted === true ? [] : ["Action-file teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(clockTeaching?.candidateAccepted === true ? [] : ["Clock/schedule teaching settled without a verified accepted candidate; learned use remains unproven."]), ...(supplement.capabilityPack?.receipt?.candidateAccepted === true ? [] : [supplement.state === "complete"
        ? "Declared supplementary tool teaching settled without an accepted candidate; learned tool use remains unverified."
        : "Supplementary learning tool semantics are queued; learned use is not yet verified."]),
        ...servers.filter((server) => server.effectiveEnabled && server.schemaLearning === "pending")
        .map((server) => server.error ?? `Learning ${server.label} capabilities is pending.`)];
    } catch (error) {
      signal.throwIfAborted();
      return [`MCP capability learning is pending: ${error instanceof Error ? error.message : String(error)}`];
    }
  });
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
    isLearning: (brainId) =>
      jobs.isLearning(brainId) || learning.isLearning(brainId) || service.isChatParameterLearningActive(brainId),
    isInitializationBusy: () => initialization.isBusy(),
    preemptBackground: () => engine!.claimForeground(),
    onError: (error) => console.error("Idle cognition cycle failed:", error)
  });
  evolution.setRecursiveReassessmentHandler(async (event, admission) => {
    // Observe a completed experiment, then offer the ordinary idle scheduler
    // its next opportunity. The neural action head chooses new work or none;
    // no behavioral prompt or repeating architecture command is supplied.
    const experience = {
      content: JSON.stringify(event), name: `Observed improvement ${event.parentCandidateId}`,
      sourceLabel: "completed own improvement experience", license: "Locally observed experiment"
    };
    if (admission?.loadedOnly) {
      await service.learnLoadedStructuredExperience(event.brainId, experience, admission.signal, admission.assertCurrent);
      // The observed experiment is already committed. A revoked source may
      // prevent an optional next wake, never erase that ACK or retrain it.
      try { admission.signal.throwIfAborted(); await admission.assertCurrent(); service.requireInstanceLoaded(event.brainId); }
      catch { return; }
    } else await service.learnStructuredExperience(event.brainId, experience);
    await idleCognition?.tick();
  });
  actions.on("neural-cancelled", ({ brainId }: { brainId: string }) => {
    idleCognition?.reserveForegroundAfterCancel(brainId);
  });
  touchBarController = new MacTouchBarController({
    isDestroyed: () => !mainWindow || mainWindow.isDestroyed(),
    isFocused: () => Boolean(mainWindow && !mainWindow.isDestroyed() && mainWindow.isFocused()),
    setTouchBar: value => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.setTouchBar(value as TouchBar | null); }
  }, {
    button: options => new TouchBar.TouchBarButton(options),
    spacer: () => new TouchBar.TouchBarSpacer({ size: "small" }),
    bar: items => new TouchBar({ items: items as NonNullable<Electron.TouchBarConstructorOptions["items"]> })
  }, event => {
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.isFocused()) {
      mainWindow.webContents.send(IPC.window.touchBarAction, event);
    }
  });
  disposeIpc = registerIpcHandlers({
    osWork,
    decoderExecution,
    cachedDecoderExecution,
    nativeOperationAuto,
    regionalExecution,
    portableWorkspace,
    advice,
    adviceAuthority,
    actionReviewer,
    adviceCredentialsChanged: () => {
      adviceCredentialEpoch += 1;
      for (const id of reviewerOwners) actionReviewer.configurationChanged(id);
    },
    repository,
    service,
    jobs,
    engine,
    tools,
    sparseSubstrate,
    actions,
    taskAttachments,
    inheritedTaskFiles,
    processMemory,
    localTeacher,
    directions,
    imaginationCapture,
    imaginationAssembly,
    evolution,
    buildSelections,
    initialization,
    mobile: mobileGateway,
    teacher,
    mcp,
    mcpInteractions,
    mcpApps,
    mcpRoots,
    mcpDiagnostics,
    maps,
    mapBasemaps,
    plugins,
    pluginPanels,
    pluginRuntime: runtimeModules,
    htmlArtifacts,
    publishPluginEvent: event => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send(IPC.plugins.event, event);
      }
    },
    requireTrustedSender: event => requireStudioIpcSender({ senderId: event.sender.id, trustedSenderId: mainWindow?.webContents.id,
      senderFrame: event.senderFrame, mainFrame: event.sender.mainFrame, frameUrl: event.senderFrame?.url,
      rendererUrl: developmentRendererUrl ?? pathToFileURL(join(moduleDirectory, "../renderer/index.html")).href }),
    studioWebContentsId: () => mainWindow && !mainWindow.isDestroyed() ? mainWindow.webContents.id : undefined,
    questions,
    learning,
    clock,
    automation: automations,
    idleCognition,
    environment: environments,
    touchBar: touchBarController,
    collectHardwareCompute: (profile) => resourcePlanner.measureNativeCompute({
      device: "cpu",
      ramBudgetBytes: profile.availableMemoryBytes,
      hardwareTier: profile.recommendedTier
    }),
    deviceResources,
    appPath
  });
  mainWindow = await createWindow();
  backgroundRuntime = new BackgroundRuntimeController({
    store: new BackgroundRuntimePreferenceStore(
      join(deviceDataRoot, "background-runtime.json")
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
  const recoveryStarted = new Map<string, { wall: string; monotonic: number }>();
  const recoveryTiming = (brainId: string) => {
    const current = recoveryStarted.get(brainId) ?? { wall: new Date().toISOString(), monotonic: performance.now() };
    recoveryStarted.set(brainId, current);
    return { startedAt: current.wall, updatedAt: new Date().toISOString(), elapsedMs: Math.max(0, performance.now() - current.monotonic) };
  };
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
        recoveryWindow.webContents.send(IPC.brain.buildEvent, engineBuildProgressEvent({
          event: engineEvent,
          brainId,
          sequence: nextRecoverySequence(brainId, engineEvent.sequence),
          fallbackLabel: "Recovering OmniCortex native core",
          timing: recoveryTiming(brainId)
        }));
      },
      (brainId, progressEvent) => {
        if (recoveryWindow.isDestroyed()) return;
        const event = initialLearningBuildProgressEvent(progressEvent, nextRecoverySequence(brainId));
        const timing = recoveryTiming(brainId);
        recoveryWindow.webContents.send(IPC.brain.buildEvent, { ...timing, ...event, elapsedMs: event.elapsedMs ?? timing.elapsedMs });
      },
      (brainId, brain) => {
        if (recoveryWindow.isDestroyed() || brain.readiness.state !== "ready") return;
        recoveryWindow.webContents.send(IPC.brain.buildEvent, hostCommittedBuildProgressEvent({
          brainId,
          sequence: nextRecoverySequence(brainId),
          timing: recoveryTiming(brainId)
        }));
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
    touchBarController?.resetOwner();
    touchBarController = undefined;
    spillPoolAuthority?.close();
    spillPoolAuthority = undefined;
  });
  app.on("before-quit", (event) => {
    if (quitAfterCleanup) return;
    event.preventDefault();
    if (quitCleanupInProgress) return;
    quitCleanupInProgress = true;
    const buildBusy = initializationCoordinator?.isBusy() ?? false;
    const releaseShutdownAdmission = buildBusy ? undefined : brainService?.beginRuntimeShutdown();
    void (async () => {
      if (buildBusy) throw new Error("Initial Build is still running. Finish it or pause it through its own controls before quitting; no unsupported Build checkpoint has been claimed.");
      await clockRuntime?.checkpointBeforeQuit();
      await automationRuntime?.checkpointBeforeQuit();
      await environmentRuntime?.shutdown();
      await clipboardRuntime?.shutdown();
      await processMemoryRuntime?.shutdown();
      await sparseSubstrateRuntime?.shutdown();
      await nativeCandidateToolRuntime?.shutdown();
      await localTeacherRuntime?.close();
      await apiTeacherRuntime?.close();
      await fileToolRuntime?.close();
      mapBasemapRuntime?.close();
      await mcpRootsRuntime?.close(); mcpDiagnosticsRuntime?.close();
      if (brainRepository && brainService) await checkpointLearningBeforeQuit({
        owners: () => brainRepository!.list(), status: id => brainService!.instanceStatus(id),
        unload: id => brainService!.unloadInstance(id), isLearning: id => learningCoordinator?.isLearning(id) ?? false,
        closeLearning: async () => { await learningCoordinator?.close(); }, stopWorker: async () => { await engine?.stop(); }
      });
      else { await learningCoordinator?.close(); await engine?.stop(); }
      await clockRuntime?.flushRateHistory();
      backgroundRuntime?.dispose(); idleCognition?.stop(); mcpClient?.dispose();
      await mcpClient?.settleTaskWrites();
      questionDelivery?.close(); questionService?.close(); await imaginationAssemblyRuntime?.dispose(); await imaginationCaptureRuntime?.dispose(); mediaArtifacts?.dispose();
      mcpInteractionService?.close();
      mcpAppObservationDelivery?.close();
      mcpAppService?.dispose();
      await pluginInteractivePanelService?.dispose();
      await pluginRuntimeService?.dispose();
      ipcMain.removeHandler(PLUGIN_RUNTIME_REQUEST_CHANNEL);
      await htmlArtifactService?.dispose();
      await mobileGateway?.stop();
      osWorkRuntime?.close();
      quitAfterCleanup = true; app.quit();
    })().catch(error => {
      releaseShutdownAdmission?.();
      clockRuntime?.start();
      automationRuntime?.resumeAdmissions();
      quitCleanupInProgress = false;
      console.error("Quit paused; saved execution boundary not confirmed:", error);
      dialog.showErrorBox("Work has not reached a saved boundary", error instanceof Error ? error.message : String(error));
    });
  });
}
