import { randomUUID } from "node:crypto";
import { cpus, freemem, totalmem } from "node:os";
import { lstat, readFile, readdir } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeTheme,
  shell,
  systemPreferences,
  webContents,
  type IpcMainInvokeEvent
} from "electron";
import type {
  BrainConfig,
  BrainExportMode,
  BrainStorageOperationEvent,
  BuildProgressEvent,
  BuildResourceSelection,
  BuildResourceStartRequest,
  CatalogEntry,
  ChatDeliveryReceiptRequest,
  ChatTurnMetadata,
  CreateBrainRequest,
  DatasetPreviewProgress,
  DatasetPreviewRequest,
  DatasetStartRequest,
  DeleteInstanceRequest,
  FeedbackRequest,
  HardwareProfile,
  EvolutionApprovalRequest,
  EvolutionRollbackRequest,
  EvolutionStartRequest,
  InstallModalityPackUrlRequest,
  ImportUrlRequest,
  IngestFilesRequest,
  IngestWebRequest,
  LiveObservationControlResolution,
  LiveObservationControlRequest,
  LiveObservationPacket,
  LiveObservationSessionStartRequest,
  MobileGatewayStartRequest,
  ModalityGenerateRequest,
  NativeAppearanceRequest,
  SubstrateQuery,
  ToolInvocation,
  ToolPermissionLevel,
  TraceQuery,
  WebCrawlRequest,
  WorkingMemoryPlanRequest,
  ApiTeacherCredentialRequest,
  ApiTeacherModelDiscoveryRequest,
  ApiTeacherProvider,
  ApiTeacherTrainingRequest,
  McpServerRegistrationRequest,
  McpRegistryUpdateRequest,
  ToolRuntimePreferences
} from "../shared/types";
import type { CortexQuery, CortexActivityQuery } from "../shared/cortexInspection";
import type { RouterSpikeQuery } from "../shared/routerActivity";
import { IPC } from "../shared/ipc";
import { createPortableWorkspaceFromIntent } from "./portableWorkspaceIpc";
import { requireMcpRootCommand, selectMcpRootsFromIntent } from "./mcpRootsIntent";
import type { FullMergeStartRequest } from "../shared/fullMerge";
import { BRAIN_EXPORT_CONFIRMATION_DETAIL } from "../shared/brainExportDisclosure";
import {
  EXPERIENCE_UPLOADS,
  isExperienceUploadKind,
  type ExperienceUploadKind
} from "../shared/uploadSupport";
import type { BrainRepository } from "./brainRepository";
import {
  BrainService,
  requireNewDataIngestionPolicy,
  type RuntimeJobManager
} from "./brainService";
import type { EngineSupervisor } from "./engineSupervisor";
import type { ToolExecutor } from "./toolExecutor";
import type { ChatActionController } from "./chatActionController";
import type { EvolutionController } from "./evolutionController";
import type { BuildResourceSelectionStore } from "./buildResourceSelections";
import type {
  BuildInitializationCoordinator,
  InitialLearningProgress
} from "./buildInitializationCoordinator";
import type { MobileGateway } from "./mobileGateway";
import type { ApiTeacherTrainingService } from "./teacherTraining";
import type { McpClientService } from "./mcpClient";
import type { McpInteractionService } from "./mcpInteractions";
import type { McpAppService } from "./mcpApps";
import { requireWorkingMemoryPlanRequest } from "./resourcePlanRequest";
import { BrainStorageOperationManager } from "./brainStorageOperations";
import type { IdleCognitionScheduler } from "./idleCognitionScheduler";
import type { DeviceResourceService, DeviceTopologyRequest } from "./deviceResourceService";
import type { UserQuestionService } from "./userQuestions";
import type { UserQuestionListQuery, UserQuestionRequest, UserQuestionRespondRequest, UserQuestionPreferences } from "../shared/userQuestions";
import type { LearningCoordinator } from "./learningCoordinator";
import type { SpecificLearnRequest, FactCheckRequest, FactCheckReviewRequest, LearningTaskSteerRequest } from "../shared/learning";
import type { PluginService } from "./pluginService";
import type { PluginEvent, PluginInstanceAccessRequest } from "../shared/plugins";
import type { PluginInteractivePanelService } from "./pluginInteractivePanels";
import type { PluginInteractiveReviewRequest, PluginInteractiveToolRequest } from "../shared/pluginInteractivePanels";
import { InstanceVoiceService } from "./voiceService";
import type { InstanceVoiceLegacyRequest, InstanceVoiceUpdateRequest } from "../shared/voiceRouting";
import type { ChatTaskAttachmentDroppedRequest, ChatTaskAttachmentStageRequest } from "../shared/taskAttachments";
import type { TaskAttachmentService } from "./taskAttachmentService";
import type { SteerDirectionsService } from "./steerDirections";
import type { SteerDirectionDismissRequest, SteerDirectionPageQuery } from "../shared/steerDirections";
import { NativeAudioChatPreparations, revalidateNativeAudioChat, type NativeAudioCaptureLeaseRequest } from "./nativeAudioChat";
import type { HtmlArtifactService } from "./htmlArtifactService";
import type { HtmlArtifactActivationRequest, HtmlArtifactOpenRequest, HtmlArtifactReviewRequest, HtmlArtifactToolRequest } from "../shared/htmlArtifacts";
import { engineBuildProgressEvent, hostCommittedBuildProgressEvent } from "./buildProgressEvents";

export interface SparseUiSubscriptionRecord { senderId: number; brainId: string; subscriptionId: string }

/** Main-only routing index. A renderer cannot move one subscription identity between owners. */
export class SparseUiSubscriptionRegistry {
  private readonly records = new Map<string, SparseUiSubscriptionRecord>();
  private key(senderId: number, subscriptionId: string): string { return `${senderId}\0${subscriptionId}`; }
  subscribe(record: SparseUiSubscriptionRecord): void {
    const key = this.key(record.senderId, record.subscriptionId), prior = this.records.get(key);
    if (prior && prior.brainId !== record.brainId) throw new Error("Sparse subscription identity already belongs to another brain owner.");
    this.records.set(key, { ...record });
  }
  unsubscribe(senderId: number, brainId: string, subscriptionId: string): boolean {
    const key = this.key(senderId, subscriptionId), prior = this.records.get(key);
    if (prior && prior.brainId !== brainId) throw new Error("Sparse subscription belongs to another brain owner.");
    return this.records.delete(key);
  }
  matching(brainId: string): Array<{ key: string; record: SparseUiSubscriptionRecord }> {
    return [...this.records].filter(([, value]) => value.brainId === brainId)
      .map(([key, record]) => ({ key, record: { ...record } }));
  }
  deleteKey(key: string): void { this.records.delete(key); }
  clear(): void { this.records.clear(); }
}

export interface IpcDependencies {
  nativeOperationAuto?: import("../shared/nativeOperationAuto").NativeOperationAutoApi;
  osWork?: Pick<import("./osWorkLeaseService").OsWorkLeaseService, "status" | "observeStorageOperation">;
  decoderExecution?: import("../shared/decoderExecution").DecoderExecutionApi;
  cachedDecoderExecution?: import("../shared/cachedDecoderExecution").CachedDecoderExecutionApi;
  regionalExecution?: import("../shared/regionalExecution").RegionalExecutionApi;
  portableWorkspace?: Pick<import("./portableWorkspaceService").PortableWorkspaceService, "create" | "prepare" | "status" | "reviewDestination" | "approveDestination">;
  advice?: import("./externalAdviceService").ExternalAdviceService;
  adviceAuthority?: import("./externalAdviceAuthority").ExternalAdviceAuthorityStore;
  actionReviewer?: import("./actionReviewer").ActionReviewer;
  adviceCredentialsChanged?(): void;
  localTeacher?: import("./localTeacherService").LocalTeacherService;
  processMemory?: Pick<import("./processMemoryService").ProcessMemoryService, "preferences" | "update">;
  environment?: Pick<import("./environmentService").EnvironmentService, "forget" | "deleteInstallationRuntime">;
  repository: BrainRepository;
  service: BrainService;
  jobs: RuntimeJobManager;
  engine: EngineSupervisor;
  tools: ToolExecutor;
  actions: ChatActionController;
  taskAttachments: TaskAttachmentService;
  inheritedTaskFiles?: import("./inheritedTaskFiles").InheritedTaskFiles;
  directions?: SteerDirectionsService;
  imaginationCapture?: import("./imaginationCapture").ImaginationCaptureService;
  imaginationAssembly?: import("./imaginationAssembly").ImaginationAssemblyService;
  evolution: EvolutionController;
  buildSelections: BuildResourceSelectionStore;
  initialization: BuildInitializationCoordinator;
  mobile: MobileGateway;
  teacher: ApiTeacherTrainingService;
  mcp: McpClientService;
  mcpInteractions?: McpInteractionService;
  mcpApps?: McpAppService;
  mcpRoots?: import("./mcpRoots").McpRootsService;
  mcpDiagnostics?: import("./mcpDiagnostics").McpDiagnosticService;
  maps?: Pick<import("./mapsService").MapsService, "list" | "save" | "remove">;
  mapBasemaps?: import("../shared/mapBasemaps").MapBasemapsApi;
  plugins: PluginService;
  pluginPanels: PluginInteractivePanelService;
  pluginRuntime?: import("./pluginRuntimeService").PluginRuntimeService;
  sparseSubstrate?: Pick<import("./sparseSubstrateTransferService").SparseSubstrateTransferService,
    "uiJobs" | "uiJob" | "uiPlan" | "userStop" | "onUiEvent">;
  htmlArtifacts: HtmlArtifactService;
  publishPluginEvent(event: PluginEvent): void;
  requireTrustedSender(event: IpcMainInvokeEvent): void;
  /** Production broadcasts must exclude isolated plugin renderers. */
  studioWebContentsId?(): number | undefined;
  questions?: UserQuestionService;
  learning?: LearningCoordinator;
  clock?: import("./clockService").ClockService;
  automation?: import("./automationService").AutomationService;
  idleCognition?: Pick<IdleCognitionScheduler, "cancelActive">;
  touchBar?: Pick<import("./macTouchBar").MacTouchBarController, "update">;
  /** Main-owned primitive measurement; no renderer model/shape arguments. */
  collectHardwareCompute?: (profile: HardwareProfile) => Promise<unknown>;
  deviceResources?: DeviceResourceService;
  appPath: string;
}

function senderWindow(event: IpcMainInvokeEvent): BrowserWindow {
  const window = BrowserWindow.fromWebContents(event.sender);
  if (!window || window.isDestroyed()) throw new Error("The application window is unavailable.");
  return window;
}

function safeName(value: string): string {
  return value
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "-")
    .replace(/[. ]+$/g, "")
    .slice(0, 100) || "OmniCortex";
}

async function measureSelectedPaths(paths: readonly string[]): Promise<{
  bytes: number;
  fileCount: number;
}> {
  const queue = [...paths.map((value) => resolve(value))];
  let bytes = 0;
  let fileCount = 0;
  while (queue.length > 0) {
    const path = queue.pop();
    if (!path) continue;
    let info;
    try {
      info = await lstat(path);
    } catch {
      continue;
    }
    if (info.isSymbolicLink()) continue;
    if (info.isFile()) {
      bytes = Math.min(Number.MAX_SAFE_INTEGER, bytes + info.size);
      fileCount += 1;
      continue;
    }
    if (!info.isDirectory()) continue;
    try {
      for (const name of await readdir(path)) queue.push(join(path, name));
    } catch {
      // Manifest creation reports inaccessible entries precisely. This early
      // measurement remains a conservative planning hint rather than coverage.
    }
  }
  return { bytes, fileCount };
}

function requireId(value: unknown, label = "id"): string {
  if (typeof value !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value)) {
    throw new Error(`Invalid ${label}.`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireNewLearningRequestPolicy(value: { policy?: unknown }): void {
  if (value.policy !== undefined) requireNewDataIngestionPolicy(value.policy);
}

export function initialLearningBuildProgressEvent(
  event: InitialLearningProgress,
  sequence: number
): BuildProgressEvent {
  const output = isRecord(event.job.output)
    ? event.job.output as BuildProgressEvent["data"]
    : undefined;
  return {
    brainId: event.job.brainId,
    sequence,
    phase: "initial-materials",
    progress: 0.9 + Math.max(0, Math.min(1, event.progress)) * 0.1,
    label: event.label,
    startedAt: event.job.createdAt,
    updatedAt: event.job.updatedAt,
    ...(event.job.telemetry?.elapsedMs !== undefined ? { elapsedMs: event.job.telemetry.elapsedMs } : {}),
    ...(event.job.telemetry?.totalEtaMs !== undefined ? { etaMs: event.job.telemetry.totalEtaMs } : {}),
    job: event.job,
    ...(output ? { data: output } : {})
  };
}

function requireChatTurnMetadata(value: unknown): ChatTurnMetadata | undefined {
  if (value === undefined) return undefined;
  if (
    !isRecord(value) ||
    !["steer", "task"].includes(String(value.kind)) ||
    value.source !== "human" ||
    typeof value.createdAt !== "string" ||
    !Number.isFinite(Date.parse(value.createdAt))
  ) {
    throw new Error("Invalid chat turn metadata.");
  }
  const attachmentIds = value.attachmentIds === undefined
    ? undefined
    : Array.isArray(value.attachmentIds)
      ? value.attachmentIds.map((id) => requireId(id, "attachment id"))
      : (() => { throw new Error("Invalid chat attachment identities."); })();
  if (attachmentIds && new Set(attachmentIds).size !== attachmentIds.length) {
    throw new Error("Chat attachment identities are duplicated.");
  }
  if (value.kind === "task") {
    if (!attachmentIds?.length) throw new Error("A task attachment turn requires at least one attachment.");
    return {
      kind: "task",
      source: "human",
      attachmentIds,
      createdAt: new Date(value.createdAt).toISOString()
    };
  }
  return {
    kind: "steer",
    source: "human",
    replacesTurnId: requireId(value.replacesTurnId, "replaced turn id"),
    createdAt: new Date(value.createdAt).toISOString(),
    ...(attachmentIds?.length ? { attachmentIds } : {})
  };
}

function requireChatDeliveryReceipt(value: unknown): ChatDeliveryReceiptRequest {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !["pending", "queued", "steered", "stopped", "cancelled", "failed"].includes(String(value.state)) ||
    typeof value.content !== "string" ||
    typeof value.createdAt !== "string" ||
    !Number.isFinite(Date.parse(value.createdAt))
  ) {
    throw new Error("Invalid chat delivery receipt.");
  }
  const content = value.content.replace(/\0/g, "").trim();
  if (!content || content.length > 100_000) {
    throw new Error("Invalid chat delivery receipt content.");
  }
  const attachmentIds = value.attachmentIds === undefined
    ? undefined
    : Array.isArray(value.attachmentIds)
      ? value.attachmentIds.map((id) => requireId(id, "delivery attachment id"))
      : (() => { throw new Error("Invalid delivery attachment identities."); })();
  if (attachmentIds && new Set(attachmentIds).size !== attachmentIds.length) {
    throw new Error("Delivery attachment identities are duplicated.");
  }
  return {
    schemaVersion: 1,
    turnId: requireId(value.turnId, "delivery receipt turn id"),
    content,
    createdAt: new Date(value.createdAt).toISOString(),
    state: value.state as ChatDeliveryReceiptRequest["state"],
    ...(attachmentIds?.length ? { attachmentIds } : {})
  };
}

function requireNativeAppearance(value: unknown): NativeAppearanceRequest {
  if (!isRecord(value) || value.schemaVersion !== 1) {
    throw new Error("Invalid appearance request.");
  }
  if (!["system", "light", "dark"].includes(String(value.mode))) {
    throw new Error("Invalid appearance mode.");
  }
  if (!["light", "dark"].includes(String(value.resolvedColorScheme))) {
    throw new Error("Invalid resolved color scheme.");
  }
  if (!["standard", "classic", "expressive", "glass"].includes(String(value.layout))) {
    throw new Error("Invalid appearance layout.");
  }
  return value as unknown as NativeAppearanceRequest;
}

function uploadDescriptor(value: unknown): {
  kind: ExperienceUploadKind;
  descriptor: (typeof EXPERIENCE_UPLOADS)[ExperienceUploadKind];
} {
  if (value !== undefined && !isExperienceUploadKind(value)) {
    throw new Error("Invalid experience upload kind.");
  }
  const kind = value ?? "files";
  return { kind, descriptor: EXPERIENCE_UPLOADS[kind] };
}

async function loadCatalog(appPath: string): Promise<CatalogEntry[]> {
  const path = join(catalogRoot(appPath), "catalog.json");
  const document = JSON.parse(await readFile(path, "utf8")) as unknown;
  if (!isRecord(document) || document.schemaVersion !== 1 || !Array.isArray(document.entries)) {
    throw new Error("The bundled catalog manifest is invalid.");
  }
  return document.entries.map((entry): CatalogEntry => {
    if (
      !isRecord(entry) ||
      typeof entry.id !== "string" ||
      typeof entry.name !== "string" ||
      typeof entry.description !== "string" ||
      typeof entry.sourceUrl !== "string" ||
      typeof entry.license !== "string" ||
      !["brain", "recipe", "dataset", "modality-pack"].includes(String(entry.kind))
    ) {
      throw new Error("The bundled catalog contains an invalid entry.");
    }
    return {
      id: entry.id,
      name: entry.name,
      description: entry.description,
      sourceUrl: entry.sourceUrl,
      license: entry.license,
      sha256: typeof entry.sha256 === "string" ? entry.sha256 : undefined,
      kind: entry.kind as CatalogEntry["kind"]
    };
  });
}

function catalogRoot(appPath: string): string {
  return app.isPackaged ? join(process.resourcesPath, "catalog") : join(appPath, "catalog");
}

async function hardwareProfile(): Promise<HardwareProfile> {
  const logicalCpus = cpus().length;
  const totalMemoryBytes = totalmem();
  const availableMemoryBytes = freemem();
  let details: Record<string, unknown> = {};
  try {
    details = (await app.getGPUInfo("basic")) as unknown as Record<string, unknown>;
  } catch {
    details = {};
  }
  const devices = Array.isArray(details.gpuDevice)
    ? (details.gpuDevice as Array<Record<string, unknown>>)
    : [];
  const primary = devices[0];
  const vendor = typeof primary?.vendorString === "string" ? primary.vendorString : undefined;
  const device = typeof primary?.deviceString === "string" ? primary.deviceString : undefined;
  const driver = typeof details.driverVersion === "string" ? details.driverVersion : undefined;
  const description = `${vendor ?? ""} ${device ?? ""}`.toLocaleLowerCase();
  const gpuAvailable =
    Boolean(primary) &&
    !description.includes("swiftshader") &&
    !description.includes("microsoft basic") &&
    !description.includes("software");
  const gib = totalMemoryBytes / 1024 ** 3;
  const recommendedTier =
    gib >= 48 || (gib >= 32 && logicalCpus >= 24)
      ? "workstation"
      : gpuAvailable && gib >= 12
        ? "gpu"
        : gib >= 16
          ? "personal"
          : "micro";
  const recommendations = {
    micro: "Use the Micro architecture, small batches, gradient accumulation, and disk offload.",
    personal: "Use the Personal architecture with adaptive retention and compact modalities.",
    gpu: "Use the GPU architecture with CUDA, Metal/MPS, or DirectML acceleration and checkpointing.",
    workstation: "Use the Workstation architecture with larger experts and concurrent modality training."
  } as const;
  return {
    platform: process.platform,
    architecture: process.arch,
    logicalCpus,
    totalMemoryBytes,
    availableMemoryBytes,
    gpu: {
      available: gpuAvailable,
      vendor,
      device,
      driver,
      details
    },
    recommendedTier,
    recommendation: recommendations[recommendedTier]
  };
}

export function registerIpcHandlers(dependencies: IpcDependencies): () => void {
  const {
    repository,
    service,
    jobs,
    engine,
    tools,
    actions,
    taskAttachments,
    evolution,
    buildSelections,
    initialization,
    mobile,
    teacher,
    mcp,
    mcpInteractions,
    mcpApps,
    plugins,
    questions,
    learning,
    idleCognition,
    appPath
  } = dependencies;
  const channels: string[] = [];
  const studioWindows = (): BrowserWindow[] => BrowserWindow.getAllWindows().filter(window =>
    !window.isDestroyed() && (!dependencies.studioWebContentsId || window.webContents.id === dependencies.studioWebContentsId()));
  const previewControllers = new Map<
    string,
    { brainId: string; controller: AbortController }
  >();
  const substrateQueryControllers = new Map<string, AbortController>();
  const sparseSubscriptions = new SparseUiSubscriptionRegistry();
  const storageOperations = new BrainStorageOperationManager();
  const voice = new InstanceVoiceService(repository);
  const audioLease = (request: NativeAudioCaptureLeaseRequest) => {
    if (process.platform === "darwin" || process.platform === "win32") {
      const status = systemPreferences.getMediaAccessStatus("microphone");
      if (status === "denied" || status === "restricted") throw new Error("Operating-system microphone access has been revoked.");
    }
    return jobs.requireNativeAudioCaptureLease(request);
  };
  service.setNativeAudioAdmissionGuard(async (brainId, input) => {
    if (!taskAttachments) throw new Error("Native audio attachment storage is unavailable.");
    await revalidateNativeAudioChat(taskAttachments, audioLease, brainId, input);
  });
  const audioPreparations = taskAttachments ? new NativeAudioChatPreparations(taskAttachments, audioLease) : undefined;
  service.configureFullMergeAccess(async (brainId) => {
    await mcp.initializeMergedIdentity(brainId);
  });
  const handle = <T extends unknown[]>(
    channel: string,
    listener: (event: IpcMainInvokeEvent, ...args: T) => unknown
  ): void => {
    ipcMain.removeHandler(channel);
    ipcMain.handle(channel, (event, ...args: T) => { dependencies.requireTrustedSender(event); return listener(event, ...args); });
    channels.push(channel);
  };

  const advice = (): import("./externalAdviceService").ExternalAdviceService => {
    if (!dependencies.advice) throw new Error("The optional advice service is unavailable.");
    return dependencies.advice;
  };
  const sparse = (): NonNullable<IpcDependencies["sparseSubstrate"]> => {
    if (!dependencies.sparseSubstrate) throw new Error("Sparse substrate job status is unavailable.");
    return dependencies.sparseSubstrate;
  };
  handle(IPC.substrate.jobs, (_event, brainId: string, query?: import("../shared/sparseSubstrateTransfer").SparseSubstrateJobQuery) =>
    sparse().uiJobs(requireId(brainId), query));
  handle(IPC.substrate.job, (_event, brainId: string, jobId: string) => sparse().uiJob(requireId(brainId), requireId(jobId, "sparse job id")));
  handle(IPC.substrate.plan, (_event, brainId: string, planId: string) => sparse().uiPlan(requireId(brainId), requireId(planId, "sparse plan id")));
  handle(IPC.substrate.stop, (_event, brainId: string, jobId: string) => sparse().userStop(requireId(brainId), requireId(jobId, "sparse job id")));
  handle(IPC.substrate.subscribe, async (event, brainId: string, subscriptionId: string) => {
    const owner = requireId(brainId), id = requireId(subscriptionId, "sparse subscription id");
    const service = sparse();
    sparseSubscriptions.subscribe({ senderId: event.sender.id, brainId: owner, subscriptionId: id });
    try {
      const initial = await service.uiJobs(owner, { limit: 40 });
      const ack: import("../shared/sparseSubstrateTransfer").SparseSubstrateUiSubscriptionAck = {
        brainId: owner, subscriptionId: id, subscribed: true, initial, observedAt: new Date().toISOString()
      };
      return ack;
    } catch (error) {
      sparseSubscriptions.unsubscribe(event.sender.id, owner, id);
      throw error;
    }
  });
  handle(IPC.substrate.unsubscribe, (event, brainId: string, subscriptionId: string) => {
    return sparseSubscriptions.unsubscribe(event.sender.id, requireId(brainId), requireId(subscriptionId, "sparse subscription id"));
  });
  handle(IPC.advice.settings, (_event, id: string) => advice().settings(id));
  handle(IPC.advice.configure, async (_event, id: string, input: import("../shared/externalAdvice").ExternalAdviceConfigureInput) => {
    if (!dependencies.adviceAuthority) throw new Error("Device-owned advice activation is unavailable.");
    // Fail-safe settings transaction: revoke first, validate/write next, then
    // mint current-device consent only for an explicitly successful save.
    await repository.get(id, false);
    await dependencies.adviceAuthority.set(id, input?.channel, false);
    dependencies.actionReviewer?.configurationChanged(id);
    await advice().suspend(id);
    await advice().configure(id, input);
    await dependencies.adviceAuthority.set(id, input.channel, input.configuration.enabled);
    dependencies.actionReviewer?.configurationChanged(id);
    return advice().settings(id);
  });
  handle(IPC.advice.preview, (_event, id: string, input: import("../shared/externalAdvice").ExternalAdvicePreviewInput) => advice().preview(id, input));
  handle(IPC.advice.start, (_event, id: string, input: import("../shared/externalAdvice").ExternalAdviceStartInput) => advice().start(id, input));
  handle(IPC.advice.cancel, (_event, id: string, requestId: string) => advice().cancel(id, requestId));
  handle(IPC.advice.history, (_event, id: string, query?: import("../shared/externalAdvice").ExternalAdviceHistoryQuery) => advice().history(id, query));
  handle(IPC.advice.discoverModels, (_event, id: string, input: import("../shared/externalAdvice").ExternalAdviceModelDiscoveryInput) =>
    advice().discoverModels(requireId(id), input));
  handle(IPC.advice.cancelModelDiscovery, (_event, id: string, input: Pick<import("../shared/externalAdvice").ExternalAdviceModelDiscoveryInput, "channel" | "requestId">) =>
    advice().cancelModelDiscovery(requireId(id), input));
  handle(IPC.advice.credentialStatus, () => advice().credentialStatus());
  handle(IPC.advice.saveCredential, async (_event, input: import("../shared/externalAdvice").ExternalAdviceCredentialSaveInput) => {
    dependencies.adviceCredentialsChanged?.();
    try { return await advice().saveCredential(input); } finally { dependencies.adviceCredentialsChanged?.(); }
  });
  handle(IPC.advice.removeCredential, async (_event, provider: import("../shared/externalAdvice").ExternalAdviceProvider) => {
    dependencies.adviceCredentialsChanged?.();
    try { return await advice().removeCredential(provider); } finally { dependencies.adviceCredentialsChanged?.(); }
  });

  handle(IPC.window.minimize, (event) => senderWindow(event).minimize());
  handle(IPC.window.workStatus, () => {
    if (!dependencies.osWork) throw new Error("Operating-system work status is unavailable.");
    return dependencies.osWork.status();
  });
  handle(IPC.window.maximize, (event) => {
    const window = senderWindow(event);
    if (window.isMaximized()) window.unmaximize();
    else window.maximize();
  });
  handle(IPC.window.close, (event) => senderWindow(event).close());
  handle(IPC.window.isMaximized, (event) => senderWindow(event).isMaximized());
  handle(IPC.window.platform, () => process.platform);
  handle(IPC.window.touchBarUpdate, (event, state: import("../shared/touchBar").TouchBarState | null) =>
    dependencies.touchBar?.update(event.sender.id, state) ?? { supported: false, accepted: false });
  handle(IPC.window.setAppearance, (event, value: unknown) => {
    const request = requireNativeAppearance(value);
    const window = senderWindow(event);
    nativeTheme.themeSource = request.mode;
    const dark = request.resolvedColorScheme === "dark";
    const backgroundColor = dark ? "#08080d" : "#f4f3f0";
    const symbolColor = dark ? "#f5f3fa" : "#1d1c23";
    window.setBackgroundColor(backgroundColor);
    if (process.platform === "win32") {
      window.setTitleBarOverlay({
        color: request.layout === "glass" ? `${backgroundColor}cc` : backgroundColor,
        symbolColor,
        height: 46
      });
      window.setBackgroundMaterial(request.layout === "glass" ? "acrylic" : "mica");
    }
    return { ...request, backgroundColor, symbolColor };
  });
  handle(IPC.window.openExternal, async (_event, rawUrl: string) => {
    if (typeof rawUrl !== "string" || rawUrl.length > 16_000) throw new Error("Invalid URL.");
    const url = new URL(rawUrl);
    if (!["https:", "http:", "mailto:"].includes(url.protocol)) {
      throw new Error("Only web and email links may be opened externally.");
    }
    if (url.username || url.password) throw new Error("URLs containing credentials are not allowed.");
    await shell.openExternal(url.toString());
  });
  handle(IPC.window.revealDataFolder, async () => {
    await repository.initialize();
    const error = await shell.openPath(repository.root);
    if (error) throw new Error(error);
  });

  handle(IPC.brain.list, () => repository.list());
  handle(IPC.brain.get, async (_event, id: string) => {
    const brain = await service.getReconciledBrain(requireId(id));
    await mcp.attachBrain(brain.id);
    if (brain.readiness.state === "ready") service.selectCompletedActionLearningOwner(brain.id);
    return brain;
  });
  handle(IPC.brain.create, async (event, request: CreateBrainRequest) => {
    service.requireRuntimeAdmissionOpen();
    if (!isRecord(request) || !isRecord(request.config)) throw new Error("Invalid build request.");
    if (
      request.initialResources !== undefined &&
      (!Array.isArray(request.initialResources) ||
        request.initialResources.some((resource) =>
          !isRecord(resource) ||
          (resource.kind !== "selection" && resource.kind !== "web") ||
          (resource.kind === "selection" &&
            (typeof resource.selectionId !== "string" ||
              !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(resource.selectionId))) ||
          (resource.kind === "web" && typeof resource.url !== "string")
        ))
    ) {
      throw new Error("Invalid initial learning resources.");
    }
    const window = senderWindow(event);
    let buildSequence = 0;
    const buildStartedAt = new Date().toISOString();
    const buildStartedAtMs = performance.now();
    const nextBuildSequence = (candidate?: number): number => {
      const observed = Number.isSafeInteger(candidate) && Number(candidate) >= 0
        ? Number(candidate)
        : buildSequence;
      const sequence = Math.max(buildSequence, observed);
      buildSequence = sequence + 1;
      return sequence;
    };
    const buildTiming = () => ({
      startedAt: buildStartedAt,
      updatedAt: new Date().toISOString(),
      elapsedMs: Math.max(0, performance.now() - buildStartedAtMs)
    });
    const brain = await initialization.create(request, (engineEvent) => {
      if (window.isDestroyed() || engineEvent.type !== "build-progress") return;
      window.webContents.send(IPC.brain.buildEvent, engineBuildProgressEvent({
        event: engineEvent,
        sequence: nextBuildSequence(engineEvent.sequence),
        fallbackLabel: "Building OmniCortex native core",
        timing: buildTiming()
      }));
    }, (progressEvent) => {
      if (window.isDestroyed()) return;
      window.webContents.send(
        IPC.brain.buildEvent,
        initialLearningBuildProgressEvent(progressEvent, nextBuildSequence())
      );
    });
    if (brain.readiness.state !== "ready") {
      throw new Error("Initialization returned before durable readiness was committed.");
    }
    if (!window.isDestroyed()) {
      window.webContents.send(IPC.brain.buildEvent, hostCommittedBuildProgressEvent({
        brainId: brain.id,
        sequence: nextBuildSequence(),
        timing: buildTiming()
      }));
    }
    return brain;
  });
  handle(IPC.brain.completeInitialization, async (_event, id: string) => {
    service.requireRuntimeAdmissionOpen();
    return initialization.complete(requireId(id));
  });
  handle(IPC.brain.retryInitialization, async (event, id: string) => {
    service.requireRuntimeAdmissionOpen();
    const brainId = requireId(id);
    const window = senderWindow(event);
    let sequence = 0;
    const buildStartedAt = new Date().toISOString();
    const buildStartedAtMs = performance.now();
    const nextBuildSequence = (candidate?: number): number => {
      const observed = Number.isSafeInteger(candidate) && Number(candidate) >= 0
        ? Number(candidate)
        : sequence;
      const next = Math.max(sequence, observed);
      sequence = next + 1;
      return next;
    };
    const buildTiming = () => ({
      startedAt: buildStartedAt,
      updatedAt: new Date().toISOString(),
      elapsedMs: Math.max(0, performance.now() - buildStartedAtMs)
    });
    const brain = await initialization.retry(
      brainId,
      (engineEvent) => {
        if (window.isDestroyed() || engineEvent.type !== "build-progress") return;
        window.webContents.send(IPC.brain.buildEvent, engineBuildProgressEvent({
          event: engineEvent,
          brainId,
          sequence: nextBuildSequence(engineEvent.sequence),
          fallbackLabel: "Recovering OmniCortex native core",
          timing: buildTiming()
        }));
      },
      (progressEvent) => {
        if (window.isDestroyed()) return;
        window.webContents.send(
          IPC.brain.buildEvent,
          initialLearningBuildProgressEvent(progressEvent, nextBuildSequence())
        );
      }
    );
    if (brain.readiness.state !== "ready") {
      throw new Error("Initialization retry returned before durable readiness was committed.");
    }
    if (!window.isDestroyed()) {
      window.webContents.send(IPC.brain.buildEvent, hostCommittedBuildProgressEvent({
        brainId,
        sequence: nextBuildSequence(),
        timing: buildTiming()
      }));
    }
    return brain;
  });
  handle(IPC.brain.update, async (_event, id: string, config: BrainConfig) => {
    const brainId = requireId(id);
    return service.updateConfig(brainId, config);
  });
  handle(IPC.brain.lifecycle, (_event, id: string) => service.instanceStatus(requireId(id)));
  handle(IPC.brain.resumePresentation, (_event, id: string) => service.instanceResumePresentation(requireId(id)));
  handle(IPC.brain.stopSavedTurn, (_event, request: { brainId: string; turnId: string; checkpointId?: string; executionId?: string }) => {
    if (!isRecord(request)) throw new Error("Invalid explicit saved-turn Stop.");
    return service.stopSavedTurn({ brainId: requireId(request.brainId), turnId: requireId(request.turnId, "saved turn id"),
      ...(request.checkpointId !== undefined ? { checkpointId: requireId(request.checkpointId, "saved checkpoint id") } : {}),
      ...(request.executionId !== undefined ? { executionId: requireId(request.executionId, "saved execution id") } : {}) });
  });
  handle(IPC.brain.inheritedResumePresentation, (_event, id: string, executionId: string) =>
    service.instanceInheritedResumePresentation(requireId(id), requireId(executionId, "inherited execution id")));
  handle(IPC.brain.prepareInheritedTaskFiles, (_event, request: { brainId: string; executionId: string }) => {
    if (!isRecord(request) || !dependencies.inheritedTaskFiles) throw new Error("Inherited task file review is unavailable.");
    return dependencies.inheritedTaskFiles.prepare({ brainId: requireId(request.brainId), executionId: requireId(request.executionId, "inherited execution id") });
  });
  for (const [channel, method] of [[IPC.brain.approveInheritedTaskFiles, "approve"], [IPC.brain.revokeInheritedTaskFiles, "revoke"]] as const) {
    handle(channel, (_event, request: { brainId: string; reviewId: string; reviewSha256: string }) => {
      if (!isRecord(request) || !dependencies.inheritedTaskFiles || typeof request.reviewSha256 !== "string" || !/^[a-f0-9]{64}$/.test(request.reviewSha256)) throw new Error("Invalid current inherited task file review.");
      return dependencies.inheritedTaskFiles[method]({ brainId: requireId(request.brainId), reviewId: requireId(request.reviewId, "inherited review id"), reviewSha256: request.reviewSha256 });
    });
  }
  handle(IPC.brain.resumeInheritedTurn, async (_event, request: { brainId: string; executionId: string }) => {
    if (!isRecord(request)) throw new Error("Invalid inherited resume request.");
    const brainId = requireId(request.brainId), executionId = requireId(request.executionId, "inherited execution id");
    const attachment = await service.instanceInheritedResumePresentation(brainId, executionId);
    return dependencies.learning ? dependencies.learning.resumeNativeContinuation(brainId, attachment,
      signal => actions.send(brainId, attachment.input, signal, attachment.frontier.turnId, undefined, attachment))
      : actions.resumeInheritedInstanceTurn(brainId, executionId);
  });
  handle(IPC.brain.resumeTurn, async (_event, request: { brainId: string; checkpointId: string }) => {
    if (!isRecord(request) || typeof request.checkpointId !== "string" || !request.checkpointId || request.checkpointId.length > 256) throw new Error("Invalid saved-turn reattachment.");
    const brainId = requireId(request.brainId), attachment = await service.instanceResumePresentation(brainId);
    if (!attachment || attachment.frontier.checkpointId !== request.checkpointId) throw new Error("The exact saved turn continuation is unavailable.");
    return dependencies.learning ? dependencies.learning.resumeNativeContinuation(brainId, attachment,
      signal => actions.send(brainId, attachment.input, signal, attachment.frontier.turnId, undefined, attachment))
      : actions.resumeInstanceTurn(brainId, request.checkpointId);
  });
  handle(IPC.brain.loadFit, (_event, id: string) => service.instanceLoadFit(requireId(id)));
  handle(IPC.brain.load, (_event, request: import("../shared/instanceLifecycle").InstanceLoadRequest) => {
    if (!isRecord(request) || request.expectedCheckpointId !== undefined && (typeof request.expectedCheckpointId !== "string" || request.expectedCheckpointId.length > 256)) throw new Error("Invalid instance load request.");
    return service.loadInstance(requireId(request.brainId), request.expectedCheckpointId);
  });
  handle(IPC.brain.unload, (_event, request: import("../shared/instanceLifecycle").InstanceUnloadRequest) => {
    if (!isRecord(request)) throw new Error("Invalid instance unload request.");
    return service.unloadInstance(requireId(request.brainId));
  });
  handle(IPC.brain.setAutoLoad, (_event, request: import("../shared/instanceLifecycle").InstanceAutoLoadRequest) => {
    if (!isRecord(request) || typeof request.enabled !== "boolean") throw new Error("Invalid Auto-load preference.");
    return service.setInstanceAutoLoad(requireId(request.brainId), request.enabled);
  });
  handle(IPC.brain.setOnlineLearning, async (_event, id: string, enabled: boolean) => {
    const brainId = requireId(id);
    return service.setOnlineLearning(brainId, enabled);
  });
  handle(IPC.brain.setActiveMode, async (
    _event,
    id: string,
    enabled: boolean
  ) => {
    const brainId = requireId(id);
    if (typeof enabled !== "boolean") {
      throw new Error("Active Mode must be enabled or disabled.");
    }
    const current = await repository.get(brainId);
    if (current.config.idleCognition !== enabled) {
      // A visible switch waits until the prior optional cycle has relinquished
      // the one neural worker before persisting the new owner.
      await idleCognition?.cancelActive();
    }
    return repository.setActiveMode(brainId, enabled);
  });
  handle(IPC.brain.duplicate, async (
    _event,
    id: string,
    name?: string,
    requestedOperationId?: string
  ) => {
    const brainId = requireId(id);
    const operationId = requestedOperationId
      ? requireId(requestedOperationId, "storage operation id")
      : randomUUID();
    const session = storageOperations.create(operationId, "duplicate", brainId);
    try {
      const brain = await repository.duplicate(
        brainId,
        typeof name === "string" ? name : undefined,
        session.hooks
      );
      session.complete({ targetBrainId: brain.id });
      return brain;
    } catch (error) {
      session.fail(error);
      throw error;
    } finally {
      storageOperations.finish(operationId);
    }
  });
  handle(IPC.brain.fork, async (
    _event,
    id: string,
    name?: string,
    requestedOperationId?: string
  ) => {
    const brainId = requireId(id);
    const operationId = requestedOperationId
      ? requireId(requestedOperationId, "storage operation id")
      : randomUUID();
    const session = storageOperations.create(operationId, "fork", brainId);
    try {
      const brain = await repository.fork(
        brainId,
        typeof name === "string" ? name : undefined,
        session.hooks
      );
      session.complete({ targetBrainId: brain.id });
      return brain;
    } catch (error) {
      session.fail(error);
      throw error;
    } finally {
      storageOperations.finish(operationId);
    }
  });
  handle(IPC.brain.pauseStorageOperation, (_event, operationId: string) =>
    storageOperations.pause(requireId(operationId, "storage operation id"))
  );
  handle(IPC.brain.startFullMerge, async (_event, request: FullMergeStartRequest) => {
    if (!request || !Array.isArray(request.parentIds)) throw new Error("Invalid full merge request.");
    requireId(request.operationId, "full merge operation id");
    for (const id of request.parentIds) {
      requireId(id);
      if (actions.isBusy(id) || jobs.list(id).some((job) => ["queued", "running", "cancelling"].includes(job.state))) {
        throw new Error("Pause the parent's active chat or learning job before starting a full merge.");
      }
    }
    await idleCognition?.cancelActive();
    return service.startFullMerge(request);
  });
  handle(IPC.brain.listFullMerges, () => service.listFullMerges());
  handle(IPC.brain.fullMergeStatus, (_event, id: string) => service.fullMergeStatus(requireId(id, "full merge operation id")));
  handle(IPC.brain.pauseFullMerge, (_event, id: string) => service.pauseFullMerge(requireId(id, "full merge operation id")));
  handle(IPC.brain.resumeFullMerge, (_event, id: string) => service.resumeFullMerge(requireId(id, "full merge operation id")));
  handle(IPC.brain.cancelFullMerge, (_event, id: string) => service.cancelFullMerge(requireId(id, "full merge operation id")));
  handle(IPC.brain.fullMergeHistory, (_event, id: string, after?: number, limit?: number) => service.fullMergeHistory(requireId(id), after, limit));
  handle(IPC.brain.fullMergePendingIngestions, (_event, id: string) => service.fullMergePendingIngestions(requireId(id)));
  handle(IPC.brain.resumeFullMergeIngestion, async (event, request: import("../shared/fullMerge").FullMergeResumeIngestionRequest) => {
    requireId(request?.brainId);
    if (typeof request.pendingId !== "string" || !/^[a-f0-9]{64}$/.test(request.pendingId) || typeof request.sourceName !== "string" || !request.sourceName || request.sourceName.includes("\0")) throw new Error("Invalid retained source selection.");
    const selected = await dialog.showOpenDialog(senderWindow(event), { title: "Locate the original retained learning source", properties: ["openFile"] });
    if (selected.canceled || !selected.filePaths[0]) return null;
    return jobs.resumeFullMergeIngestion(request, selected.filePaths[0]);
  });
  handle(IPC.brain.resumeStorageOperation, (_event, operationId: string) =>
    storageOperations.resume(requireId(operationId, "storage operation id"))
  );
  handle(IPC.brain.cancelStorageOperation, (_event, operationId: string) =>
    storageOperations.cancel(requireId(operationId, "storage operation id"))
  );
  handle(IPC.brain.remove, async (_event, value: DeleteInstanceRequest) => {
    if (
      !isRecord(value) ||
      value.acknowledgedIrreversible !== true ||
      typeof value.typedName !== "string" ||
      typeof value.finalConfirmation !== "string"
    ) {
      throw new Error("Invalid permanent instance deletion request.");
    }
    const brainId = requireId(value.brainId);
    const current = await repository.get(brainId);
    if (
      value.typedName !== current.name ||
      value.finalConfirmation !== `PERMANENTLY DELETE ${current.name}`
    ) {
      throw new Error("Permanent deletion confirmations do not match this instance.");
    }
    const unloaded = await service.unloadInstance(brainId);
    if (unloaded.learnedRuntimeResident || !["unloaded", "suspended"].includes(unloaded.phase)) throw new Error("Deletion waits for the current native boundary; the saved instance has not been removed.");
    await service.detachInstanceHostContinuations(brainId);
    await dependencies.learning?.detachAndForget(brainId);
    await dependencies.clock?.forget(brainId);
    await dependencies.automation?.forget(brainId);
    await dependencies.environment?.forget(brainId);
    await dependencies.environment?.deleteInstallationRuntime(brainId);
    return repository.permanentlyDeleteInstance({
      brainId,
      acknowledgedIrreversible: true,
      typedName: value.typedName,
      finalConfirmation: value.finalConfirmation
    });
  });
  handle(IPC.brain.snapshot, async (
    _event,
    id: string,
    label?: string,
    requestedOperationId?: string
  ) => {
    const brainId = requireId(id);
    const operationId = requestedOperationId
      ? requireId(requestedOperationId, "storage operation id")
      : randomUUID();
    const session = storageOperations.create(operationId, "snapshot", brainId);
    try {
      const snapshot = await service.createRecoveryPoint(
        brainId,
        typeof label === "string" ? label : undefined,
        operationId,
        session.hooks
      );
      const storage = snapshot.storage;
      session.complete(storage
        ? {
            targetBrainId: brainId,
            filesCompleted: storage.files,
            filesTotal: storage.files,
            logicalBytesCompleted: storage.logicalBytes,
            logicalBytesTotal: storage.logicalBytes,
            physicalBytesAdded: storage.physicalBytesAdded,
            sharedBytes: storage.sharedBytes
          }
        : { targetBrainId: brainId });
      return snapshot;
    } catch (error) {
      session.fail(error);
      throw error;
    } finally {
      storageOperations.finish(operationId);
    }
  });
  handle(IPC.brain.listSnapshots, (_event, id: string) =>
    repository.listSnapshots(requireId(id))
  );
  handle(IPC.brain.restoreSnapshot, async (
    _event,
    id: string,
    snapshotId: string,
    requestedOperationId?: string
  ) => {
    const brainId = requireId(id);
    const recoveryId = requireId(snapshotId, "snapshot id");
    const operationId = requestedOperationId
      ? requireId(requestedOperationId, "storage operation id")
      : randomUUID();
    const session = storageOperations.create(operationId, "restore", brainId);
    try {
      const restored = await service.restoreRecoveryPoint(
        brainId,
        recoveryId,
        operationId,
        session.hooks
      );
      session.complete({ targetBrainId: brainId });
      return restored;
    } catch (error) {
      session.fail(error);
      throw error;
    } finally {
      storageOperations.finish(operationId);
    }
  });
  const requireDecoderExecution = (): import("../shared/decoderExecution").DecoderExecutionApi => {
    if (!dependencies.decoderExecution) throw new Error("Loaded decoder performance measurements are unavailable.");
    return dependencies.decoderExecution;
  };
  handle(IPC.decoderExecution.captureNext, (_event, request: Parameters<import("../shared/decoderExecution").DecoderExecutionApi["captureNext"]>[0]) => requireDecoderExecution().captureNext(request));
  handle(IPC.decoderExecution.prepare, (_event, request: Parameters<import("../shared/decoderExecution").DecoderExecutionApi["prepare"]>[0]) => requireDecoderExecution().prepare(request));
  handle(IPC.decoderExecution.calibrate, (_event, request: Parameters<import("../shared/decoderExecution").DecoderExecutionApi["calibrate"]>[0]) => requireDecoderExecution().calibrate(request));
  handle(IPC.decoderExecution.apply, (_event, request: Parameters<import("../shared/decoderExecution").DecoderExecutionApi["apply"]>[0]) => requireDecoderExecution().apply(request));
  handle(IPC.decoderExecution.status, (_event, brainId: string) => requireDecoderExecution().status(requireId(brainId, "brain id")));
  handle(IPC.decoderExecution.stop, (_event, request: Parameters<import("../shared/decoderExecution").DecoderExecutionApi["stop"]>[0]) => requireDecoderExecution().stop(request));
  const requireNativeOperationAuto = (): import("../shared/nativeOperationAuto").NativeOperationAutoApi => {
    if (!dependencies.nativeOperationAuto) throw new Error("Native perception Auto is unavailable.");
    return dependencies.nativeOperationAuto;
  };
  handle(IPC.nativeOperationAuto.prepare, (_event, request: Parameters<import("../shared/nativeOperationAuto").NativeOperationAutoApi["prepare"]>[0]) => requireNativeOperationAuto().prepare(request));
  handle(IPC.nativeOperationAuto.arm, (_event, request: Parameters<import("../shared/nativeOperationAuto").NativeOperationAutoApi["arm"]>[0]) => requireNativeOperationAuto().arm(request));
  handle(IPC.nativeOperationAuto.status, (_event, brainId: string) => requireNativeOperationAuto().status(requireId(brainId, "brain id")));
  handle(IPC.nativeOperationAuto.plans, (_event, request: Parameters<import("../shared/nativeOperationAuto").NativeOperationAutoApi["plans"]>[0]) => requireNativeOperationAuto().plans(request));
  handle(IPC.nativeOperationAuto.stop, (_event, request: Parameters<import("../shared/nativeOperationAuto").NativeOperationAutoApi["stop"]>[0]) => requireNativeOperationAuto().stop(request));
  const requireCachedDecoderExecution = (): import("../shared/cachedDecoderExecution").CachedDecoderExecutionApi => {
    if (!dependencies.cachedDecoderExecution) throw new Error("Retained-context decoder measurements are unavailable.");
    return dependencies.cachedDecoderExecution;
  };
  handle(IPC.cachedDecoderExecution.prepare, (_event, request: Parameters<import("../shared/cachedDecoderExecution").CachedDecoderExecutionApi["prepare"]>[0]) => requireCachedDecoderExecution().prepare(request));
  handle(IPC.cachedDecoderExecution.arm, (_event, request: Parameters<import("../shared/cachedDecoderExecution").CachedDecoderExecutionApi["arm"]>[0]) => requireCachedDecoderExecution().arm(request));
  handle(IPC.cachedDecoderExecution.status, (_event, brainId: string) => requireCachedDecoderExecution().status(requireId(brainId, "brain id")));
  handle(IPC.cachedDecoderExecution.stop, (_event, request: Parameters<import("../shared/cachedDecoderExecution").CachedDecoderExecutionApi["stop"]>[0]) => requireCachedDecoderExecution().stop(request));
  handle(IPC.regionalExecution.prepare, async (_event, request: Parameters<import("../shared/regionalExecution").RegionalExecutionApi["prepare"]>[0]) => {
    if (!dependencies.regionalExecution) throw new Error("Manual regional execution is unavailable.");
    return dependencies.regionalExecution.prepare(request);
  });
  handle(IPC.regionalExecution.status, async (_event, brainId: string) => {
    if (!dependencies.regionalExecution) throw new Error("Manual regional execution is unavailable.");
    return dependencies.regionalExecution.status(requireId(brainId, "brain id"));
  });
  handle(IPC.regionalExecution.activate, async (_event, request: Parameters<import("../shared/regionalExecution").RegionalExecutionApi["activate"]>[0]) => {
    if (!dependencies.regionalExecution) throw new Error("Manual regional execution is unavailable.");
    return dependencies.regionalExecution.activate(request);
  });
  handle(IPC.regionalExecution.deactivate, async (_event, request: Parameters<import("../shared/regionalExecution").RegionalExecutionApi["deactivate"]>[0]) => {
    if (!dependencies.regionalExecution) throw new Error("Manual regional execution is unavailable.");
    return dependencies.regionalExecution.deactivate(request);
  });
  handle(IPC.portable.status, async () => dependencies.portableWorkspace?.status() ?? { portable: false, state: "installed" });
  handle(IPC.portable.prepare, async (_event, request: unknown) => {
    if (!dependencies.portableWorkspace || !isRecord(request) || Object.keys(request).join(",") !== "requestId") throw new Error("Portable preparation needs an exact current request ID.");
    const operationId = requireId(request.requestId, "portable preparation"), status = await dependencies.portableWorkspace.status();
    const session = storageOperations.create(operationId, "import", status.brainId);
    try {
      const result = await dependencies.portableWorkspace.prepare({ requestId: operationId }, session.hooks);
      session.complete(); return result;
    } catch (error) { session.fail(error); throw error; }
    finally { storageOperations.finish(operationId); }
  });
  handle(IPC.portable.reviewDestination, async () => {
    if (!dependencies.portableWorkspace) throw new Error("Portable destination review is unavailable.");
    return dependencies.portableWorkspace.reviewDestination();
  });
  handle(IPC.portable.approveDestination, async (_event, request: unknown) => {
    if (!dependencies.portableWorkspace) throw new Error("Portable destination review is unavailable.");
    if (!isRecord(request) || Object.keys(request).sort().join(",") !== "reviewId,reviewSha256" ||
        typeof request.reviewSha256 !== "string" || !/^[a-f0-9]{64}$/.test(request.reviewSha256)) throw new Error("Review the exact current portable destination before approving it.");
    return dependencies.portableWorkspace.approveDestination({ reviewId: requireId(request.reviewId, "destination review"), reviewSha256: request.reviewSha256 });
  });
  handle(IPC.portable.create, async (event, request: unknown) => {
    if (!dependencies.portableWorkspace) throw new Error("Portable workspace creation is unavailable.");
    return createPortableWorkspaceFromIntent({
      portable: dependencies.portableWorkspace,
      brain: id => repository.get(id, false),
      confirm: async name => (await dialog.showMessageBox(senderWindow(event), {
        type: "warning", title: "Create a portable Windows workspace?",
        message: `Include the complete saved ${name} instance?`, detail: BRAIN_EXPORT_CONFIRMATION_DETAIL + "\n\nChoose an Omni Windows app folder and a new output folder. Device credentials and live grants are not transferred. Empty reservation capacity is planned on the destination drive, not shipped as padding.",
        buttons: ["Cancel", "Create private portable copy"], defaultId: 0, cancelId: 0, noLink: true
      })).response === 1,
      chooseRuntime: async () => {
        if (process.platform === "win32" && app.isPackaged) return dirname(process.execPath);
        const chosen = await dialog.showOpenDialog(senderWindow(event), { title: "Select the Windows Omni app folder containing Omni AGI Studio.exe", properties: ["openDirectory"] });
        return chosen.canceled ? null : chosen.filePaths[0] ?? null;
      },
      chooseDestination: async name => {
        const chosen = await dialog.showSaveDialog(senderWindow(event), { title: "Choose a new portable workspace folder", defaultPath: `${safeName(name)}-Windows-Portable`, buttonLabel: "Create folder" });
        return chosen.canceled ? null : chosen.filePath ?? null;
      },
      session: (id, brainId) => storageOperations.create(id, "export", brainId),
      finish: id => storageOperations.finish(id)
    }, request);
  });
  handle(IPC.brain.export, async (
    event,
    id: string,
    mode: BrainExportMode = "current",
    requestedOperationId?: string
  ) => {
    const brain = await repository.get(requireId(id));
    if (!["current", "origin", "private-archive", "referenced"].includes(mode)) {
      throw new Error("Invalid export mode.");
    }
    const confirmation = await dialog.showMessageBox(senderWindow(event), {
      type: "warning",
      title: "Export saved mind?",
      message: "This backup preserves private saved content without sanitizing it.",
      detail: BRAIN_EXPORT_CONFIRMATION_DETAIL,
      buttons: ["Cancel", "Export backup"],
      defaultId: 0,
      cancelId: 0,
      noLink: true
    });
    if (confirmation.response !== 1) return null;
    const suffix =
      mode === "origin"
        ? "-Origin"
        : mode === "private-archive"
          ? "-Private"
          : mode === "referenced"
            ? "-Local-Reference"
            : "";
    const choice = await dialog.showSaveDialog(senderWindow(event), {
      title: "Export portable OmniCortex brain",
      defaultPath: `${safeName(brain.name)}${suffix}.omni`,
      filters: [{ name: "Omni brain", extensions: ["omni"] }]
    });
    if (choice.canceled || !choice.filePath) return null;
    const destination = choice.filePath.toLocaleLowerCase().endsWith(".omni")
      ? choice.filePath
      : `${choice.filePath}.omni`;
    const operationId = requestedOperationId
      ? requireId(requestedOperationId, "storage operation id")
      : randomUUID();
    const session = storageOperations.create(operationId, "export", brain.id);
    try {
      if (mode !== "origin") {
        await service.flushNeuralCheckpoint(
          brain.id,
          operationId,
          session.hooks
        );
      }
      await repository.exportBundle(
        brain.id,
        destination,
        mode,
        session.hooks
      );
      session.complete();
      return destination;
    } catch (error) {
      session.fail(error);
      throw error;
    } finally {
      storageOperations.finish(operationId);
    }
  });
  handle(IPC.brain.importFile, async (event, requestedOperationId?: string) => {
    const choice = await dialog.showOpenDialog(senderWindow(event), {
      title: "Import an OmniCortex brain",
      properties: ["openFile"],
      filters: [{ name: "Omni brain", extensions: ["omni"] }]
    });
    if (choice.canceled || !choice.filePaths[0]) return null;
    const operationId = requestedOperationId
      ? requireId(requestedOperationId, "storage operation id")
      : randomUUID();
    const session = storageOperations.create(operationId, "import");
    let brain: Awaited<ReturnType<BrainRepository["importBundle"]>>;
    try {
      brain = await repository.importBundle(
        choice.filePaths[0],
        {},
        session.hooks
      );
      session.complete({ targetBrainId: brain.id });
    } catch (error) {
      session.fail(error);
      throw error;
    } finally {
      storageOperations.finish(operationId);
    }
    // Import preserves data; explicit Load owns neural residency.
    return brain;
  });
  handle(IPC.brain.health, () => engine.health());
  handle(IPC.brain.persistedSubstrateOverview, (_event, id: string) =>
    service.persistedSubstrateOverview(requireId(id))
  );
  handle(IPC.brain.queryConceptIds, async (_event, id: string, view: unknown, sourceTurnId: string, offset?: number) =>
    service.queryConceptIds(id, view, sourceTurnId, offset));

  handle(IPC.brain.querySubstrate, async (event, id: string, query?: SubstrateQuery) => {
    const brainId = requireId(id);
    if (query !== undefined && !isRecord(query)) {
      throw new Error("Invalid substrate query.");
    }
    const key = `${event.sender.id}:${brainId}`;
    substrateQueryControllers.get(key)?.abort();
    const controller = new AbortController();
    substrateQueryControllers.set(key, controller);
    const abortOnDestroyed = (): void => controller.abort();
    event.sender.once("destroyed", abortOnDestroyed);
    try {
      return await service.querySubstrate(brainId, query, controller.signal);
    } finally {
      event.sender.removeListener("destroyed", abortOnDestroyed);
      if (substrateQueryControllers.get(key) === controller) {
        substrateQueryControllers.delete(key);
      }
    }
  });
  handle(IPC.wholeBrainInspection.query, async (event, id: string, query?: import("../shared/wholeBrainInspection").WholeBrainInspectionQuery) => {
    const brainId = requireId(id);
    if (query !== undefined && !isRecord(query)) throw new Error("Invalid whole-brain inspection query.");
    const key = `${event.sender.id}:${brainId}:whole-brain`;
    substrateQueryControllers.get(key)?.abort();
    const controller = new AbortController(); substrateQueryControllers.set(key, controller);
    const abortOnDestroyed = (): void => controller.abort(); event.sender.once("destroyed", abortOnDestroyed);
    try { return await service.queryWholeBrain(brainId, query, controller.signal); }
    finally { event.sender.removeListener("destroyed", abortOnDestroyed);
      if (substrateQueryControllers.get(key) === controller) substrateQueryControllers.delete(key); }
  });
  handle(IPC.brain.queryCortex, async (event, id: string, query?: CortexQuery) => {
    const brainId = requireId(id);
    if (query !== undefined && !isRecord(query)) throw new Error("Invalid cortical query.");
    const key = `${event.sender.id}:${brainId}:cortex`;
    substrateQueryControllers.get(key)?.abort();
    const controller = new AbortController();
    substrateQueryControllers.set(key, controller);
    const abortOnDestroyed = (): void => controller.abort();
    event.sender.once("destroyed", abortOnDestroyed);
    try {
      return await service.queryCortex(brainId, query, controller.signal);
    } finally {
      event.sender.removeListener("destroyed", abortOnDestroyed);
      if (substrateQueryControllers.get(key) === controller) substrateQueryControllers.delete(key);
    }
  });
  handle(IPC.brain.cortexActivity, (_event, id: string, query: CortexActivityQuery) => {
    if (!isRecord(query)) throw new Error("Invalid cortical activity query.");
    return service.cortexActivity(requireId(id), query);
  });
  handle(IPC.brain.routerActivity, (_event, id: string, query: RouterSpikeQuery) => {
    if (!isRecord(query)) throw new Error("Invalid router spike query.");
    return service.routerActivity(requireId(id), query);
  });
  handle(IPC.brain.workspace, (_event, id: string) =>
    service.workspace(requireId(id))
  );
  handle(IPC.brain.freshAttention, async (_event, id: string) => {
    const brainId = requireId(id), result = await service.startFreshAttention(brainId);
    await dependencies.imaginationCapture?.release(brainId, true); dependencies.imaginationCapture?.activate(brainId); return result;
  });
  handle(
    IPC.brain.journalPage,
    (_event, id: string, cursor?: string, limit?: number) =>
      repository.journalPage(
        requireId(id),
        typeof cursor === "string" ? cursor : undefined,
        typeof limit === "number" ? limit : undefined
      )
  );

  const requireDirections = (): SteerDirectionsService => { if (!dependencies.directions) throw new Error("Steering presentation history is unavailable."); return dependencies.directions; };
  handle(IPC.chat.directionsList, (_event, id: string, query?: SteerDirectionPageQuery) => requireDirections().list(requireId(id), query));
  handle(IPC.chat.directionsGet, (_event, id: string, directionId: string) => requireDirections().get(requireId(id), requireId(directionId)));
  handle(IPC.chat.directionsForTurns, (_event, id: string, turnIds: string[]) => {
    if (!Array.isArray(turnIds)) throw new Error("Invalid visible turn lookup.");
    return requireDirections().forTurns(requireId(id), turnIds.map(value => requireId(value, "visible turn id")));
  });
  handle(IPC.chat.directionsDismiss, (_event, request: SteerDirectionDismissRequest) => {
    if (!request || typeof request !== "object") throw new Error("Invalid direction display request.");
    return requireDirections().dismiss({ ...request, brainId: requireId(request.brainId), id: requireId(request.id) });
  });
  handle(IPC.chat.send, async (
    _event,
    id: string,
    input: string,
    turnId?: string,
    turnMetadata?: unknown
  ) => {
    const brainId = requireId(id);
    const persistedBrain = await repository.get(brainId);
    if (
      persistedBrain.readiness.state !== "ready" ||
      jobs.isInitializing(brainId)
    ) {
      throw new Error(
        "This mind is still completing its initial learning and cannot chat yet."
      );
    }
    // Deterministic test latency proves the renderer shows a pending human turn
    // before the worker reply. It is ignored outside the test environment.
    const requestedTestDelay = process.env.NODE_ENV === "test"
      ? Number.parseInt(process.env.OMNI_E2E_CHAT_DELAY_MS ?? "0", 10)
      : 0;
    const testDelay = Number.isFinite(requestedTestDelay)
      ? Math.max(0, Math.min(requestedTestDelay, 10_000))
      : 0;
    if (testDelay > 0) {
      await new Promise<void>((resolveDelay) => setTimeout(resolveDelay, testDelay));
    }
    return actions.send(
      brainId,
      input,
      undefined,
      typeof turnId === "string" ? requireId(turnId, "turn id") : undefined,
      requireChatTurnMetadata(turnMetadata)
    );
  });
  handle(IPC.chat.stageAttachments, async (event, request: ChatTaskAttachmentStageRequest) => {
    if (!isRecord(request) || request.selection !== "folder" && !isExperienceUploadKind(request.selection)) {
      throw new Error("Invalid task attachment selection.");
    }
    const brainId = requireId(request.brainId);
    const requestId = requireId(request.requestId, "attachment request id");
    const prior = await taskAttachments.requestHandles(brainId, requestId);
    if (prior) return prior;
    const descriptor = request.selection === "folder" ? undefined : EXPERIENCE_UPLOADS[request.selection];
    const choice = await dialog.showOpenDialog(senderWindow(event), {
      title: `Attach ${descriptor?.shortLabel ?? "folder"} to this task`,
      properties: request.selection === "folder" ? ["openDirectory"] : ["openFile", "multiSelections"],
      ...(descriptor ? { filters: [
        { name: descriptor.filterName, extensions: [...descriptor.extensions] },
        { name: "All files", extensions: ["*"] }
      ] } : {})
    });
    if (choice.canceled || choice.filePaths.length === 0) return null;
    return taskAttachments.stagePaths(
      brainId,
      requestId,
      choice.filePaths.map((path) => ({ path: resolve(path) })),
      request.selection,
      undefined,
      traversal => {
        if (!event.sender.isDestroyed()) event.sender.send(IPC.chat.attachmentStageProgress,
          { brainId, requestId, phase: traversal.complete ? "sealing" : "copying", traversal });
      }
    );
  });
  handle(IPC.chat.stageDroppedAttachments, async (
    _event,
    request: ChatTaskAttachmentDroppedRequest,
    paths: unknown
  ) => {
    if (!isRecord(request) || !Array.isArray(paths) ||
        paths.some((path) => typeof path !== "string" || !isAbsolute(path))) {
      throw new Error("Invalid dropped task attachments.");
    }
    const brainId = requireId(request.brainId);
    const requestId = requireId(request.requestId, "attachment request id");
    const prior = await taskAttachments.requestHandles(brainId, requestId);
    if (prior) return prior;
    return taskAttachments.stagePaths(
      brainId,
      requestId,
      paths.map((path) => ({ path: resolve(path as string) })),
      "files"
    );
  });
  handle(IPC.chat.cancelAttachmentStage, (_event, brainId: string, requestId: string) =>
    taskAttachments.cancelStage(requireId(brainId), requireId(requestId, "attachment request id")));
  handle(IPC.chat.listAttachments, (_event, brainId: string, state?: unknown) => {
    if (state !== undefined && state !== "staged") throw new Error("Invalid task attachment state filter.");
    return taskAttachments.list(requireId(brainId), state);
  });
  handle(IPC.chat.attachmentsForTurn, (_event, brainId: string, turnId: string) =>
    taskAttachments.handlesForTurn(requireId(brainId), requireId(turnId, "attachment turn id")));
  handle(IPC.chat.attachmentsByIds, async (_event, brainId: string, attachmentIds: unknown) => {
    if (!Array.isArray(attachmentIds)) throw new Error("Invalid task attachment identities.");
    const owner = requireId(brainId), ids = attachmentIds.map(id => requireId(id, "attachment id"));
    if (new Set(ids).size !== ids.length) throw new Error("Task attachment identities are duplicated.");
    const handles: import("../shared/taskAttachments").ChatTaskAttachmentHandle[] = [];
    for (const id of ids) {
      try { handles.push(...await taskAttachments.handlesByIds(owner, [id])); }
      catch (error) {
        if (!dependencies.inheritedTaskFiles) throw error;
        // Trusted-user presentation only; this does not enter any model file scope or byte-reading API.
        handles.push(...await dependencies.inheritedTaskFiles.historicalFileHandlesForUser(owner, [id]));
      }
    }
    return handles;
  });
  handle(IPC.chat.removeAttachment, (_event, brainId: string, attachmentId: string) =>
    taskAttachments.remove(requireId(brainId), requireId(attachmentId, "attachment id")));
  handle(IPC.chat.cancel, async (_event, id: string, turnId?: string) => {
    const brainId = requireId(id), ownerTurn = typeof turnId === "string" ? requireId(turnId, "turn id") : undefined;
    const staging = audioPreparations?.cancelAndWait(brainId, ownerTurn) ?? Promise.resolve(0);
    const running = actions.cancelAndWait(brainId, ownerTurn);
    const [stagedCount, activeCount] = await Promise.all([staging, running]);
    return stagedCount + activeCount;
  });
  handle(IPC.chat.audioInputCapabilities, async (_event, id: string) => {
    const brainId = requireId(id); service.requireInstanceLoaded(brainId);
    const capabilities = await service.modalityCapabilities(brainId);
    return { brainId, loaded: true as const, available: Boolean(taskAttachments && capabilities.audioRegionAvailable),
      sameBrainSubstrate: true as const, transcriptRequired: false as const,
      detail: "Direct raw-audio conversation uses this brain's sensory and language regions. Paired training and intelligibility are not verified by connectivity." };
  });
  handle(IPC.chat.sendAudio, async (_event, id: string, input: unknown, turnId?: string, turnMetadata?: unknown) => {
    const brainId = requireId(id), ownerTurn = turnId === undefined ? randomUUID() : requireId(turnId, "turn id");
    const brain = await repository.get(brainId);
    if (brain.readiness.state !== "ready" || jobs.isInitializing(brainId)) throw new Error("This mind is still completing initial learning.");
    service.requireInstanceLoaded(brainId);
    if (!audioPreparations) throw new Error("Native audio attachment storage is unavailable.");
    const audio = await audioPreparations.prepare(brainId, ownerTurn, input);
    return actions.send(brainId, "", undefined, ownerTurn, requireChatTurnMetadata(turnMetadata), undefined, undefined, audio);
  });
  handle(IPC.chat.recordDeliveryReceipt, async (
    _event,
    id: string,
    receipt: unknown
  ) => {
    const brainId = requireId(id), checked = requireChatDeliveryReceipt(receipt);
    if (checked.state === "pending" && checked.attachmentIds?.length) {
      await taskAttachments.bindToTurn(brainId, checked.turnId, checked.attachmentIds);
    }
    await repository.appendChatDeliveryReceipt(brainId, checked);
  });
  handle(IPC.chat.cancelInlineAction, (_event, id: string, turnId: string, actionEventId: string) =>
    actions.cancelInlineAction(requireId(id), requireId(turnId, "turn id"),
      requireId(actionEventId, "action event id"))
  );
  handle(IPC.chat.approveAction, (_event, value: unknown) => {
    if (!isRecord(value)) throw new Error("Invalid approved chat action request.");
    return actions.approveAction({
      brainId: requireId(value.brainId, "brain id"),
      actionEventId: requireId(value.actionEventId, "action event id"),
      approvalToken: requireId(value.approvalToken, "approval token")
    });
  });
  handle(IPC.chat.declineAction, (_event, value: unknown) => {
    if (!isRecord(value)) throw new Error("Invalid declined chat action request.");
    return actions.declineAction({ brainId: requireId(value.brainId, "brain id"),
      actionEventId: requireId(value.actionEventId, "action event id"), approvalToken: requireId(value.approvalToken, "approval token") });
  });
  handle(IPC.chat.list, async (_event, id: string) => {
    const brainId = requireId(id);
    await service.getReconciledBrain(brainId);
    return (await repository.conversationPage(brainId, undefined, 200)).entries
      .flatMap((entry) => entry.message ? [entry.message] : []);
  });
  handle(IPC.chat.listPage, async (
    _event,
    id: string,
    beforeSequence?: number,
    limit?: number
  ) => {
    const brainId = requireId(id);
    await service.getReconciledBrain(brainId);
    return repository.conversationPage(brainId, beforeSequence, limit);
  });
  handle(IPC.chat.feedback, (_event, request: FeedbackRequest) => service.feedback(request));

  handle(IPC.mobile.status, () => mobile.status());
  handle(IPC.mobile.startPairing, (_event, value: MobileGatewayStartRequest) => {
    if (!isRecord(value) || typeof value.allowLan !== "boolean") {
      throw new Error("Invalid mobile pairing request.");
    }
    return mobile.startPairing({ allowLan: value.allowLan });
  });
  handle(IPC.mobile.stop, () => mobile.stop());
  handle(IPC.mobile.revoke, (_event, deviceId: string) =>
    mobile.revoke(requireId(deviceId, "mobile device id"))
  );

  handle(IPC.train.cancel, (_event, jobId: string) => jobs.cancel(requireId(jobId, "job id")));
  handle(IPC.train.list, (_event, id?: string) =>
    jobs.list(typeof id === "string" ? requireId(id) : undefined)
  );

  handle(IPC.teacher.status, () => teacher.status());
  handle(IPC.teacher.saveCredential, (_event, request: ApiTeacherCredentialRequest) => teacher.saveCredential(request));
  handle(IPC.teacher.removeCredential, (_event, provider: ApiTeacherProvider) => teacher.removeCredential(provider));
  handle(IPC.teacher.discoverModels, (_event, request: ApiTeacherModelDiscoveryRequest) => teacher.discoverModels(request));
  handle(IPC.teacher.cancelModelDiscovery, (_event, request: ApiTeacherModelDiscoveryRequest) => teacher.cancelModelDiscovery(request));
  handle(IPC.teacher.start, (_event, request: ApiTeacherTrainingRequest) => teacher.start(request));
  handle(IPC.teacher.pause, (_event, request: import("../shared/types").ApiTeacherJobControlRequest) =>
    teacher.pause(requireId(request?.brainId), requireId(request?.jobId, "teacher job id")));
  handle(IPC.teacher.resume, (_event, request: import("../shared/types").ApiTeacherJobControlRequest) =>
    teacher.resume(requireId(request?.brainId), requireId(request?.jobId, "teacher job id")));
  handle(IPC.teacher.stop, (_event, request: import("../shared/types").ApiTeacherJobControlRequest) =>
    teacher.stop(requireId(request?.brainId), requireId(request?.jobId, "teacher job id")));
  handle(IPC.teacher.details, (_event, request: import("../shared/types").ApiTeacherJobControlRequest) =>
    teacher.details(requireId(request?.brainId), requireId(request?.jobId, "teacher job id")));
  handle(IPC.teacher.resolveUncertain, (_event, request: import("../shared/types").ApiTeacherUncertainResolutionRequest) => {
    if (!Number.isSafeInteger(request?.itemIndex) || request.itemIndex < 0 || !["retry", "skip"].includes(request.decision))
      throw new Error("Choose the exact uncertain teacher item and Retry or Skip.");
    return teacher.resolveUncertain(requireId(request.brainId), requireId(request.jobId, "teacher job id"), request.itemIndex, request.decision);
  });
  handle(IPC.teacher.cancel, (_event, jobId: string) =>
    teacher.cancel(requireId(jobId, "teacher job id"))
  );
  handle(IPC.teacher.list, (_event, brainId?: string) =>
    teacher.list(typeof brainId === "string" ? requireId(brainId) : undefined)
  );

  handle(IPC.mcp.list, (_event, brainId: string) => mcp.list(requireId(brainId)));
  const requireMcpRoots = () => {
    if (!dependencies.mcpRoots) throw new Error("MCP filesystem root disclosure is unavailable."); return dependencies.mcpRoots;
  };
  handle(IPC.mcp.rootsStatus, async (_event, brainId: string, serverId: string) => {
    const id = requireId(brainId); await repository.get(id, false); return requireMcpRoots().status(id, requireId(serverId));
  });
  handle(IPC.mcp.rootsSelect, (_event, request: unknown) => selectMcpRootsFromIntent(request, {
    current: async command => { service.requireRuntimeAdmissionOpen(); await repository.get(command.brainId, false); return mcp.hostConnection(command.brainId, command.serverId); },
    pick: async (_command, connection) => {
      const value = await dialog.showOpenDialog({ title: `Choose filesystem roots for ${connection.serverLabel}`,
        properties: ["openDirectory", "multiSelections"], buttonLabel: "Review disclosure" });
      return value.canceled ? null : value.filePaths;
    },
    select: (command, paths) => requireMcpRoots().select(command.brainId, command.serverId, paths, new AbortController().signal),
    status: command => requireMcpRoots().status(command.brainId, command.serverId)
  }));
  for (const [channel, operation] of [[IPC.mcp.rootsReview, "review"], [IPC.mcp.rootsClear, "clear"]] as const) handle(channel, async (_event, request: unknown) => {
    const command = requireMcpRootCommand(request); service.requireRuntimeAdmissionOpen(); await repository.get(command.brainId, false);
    return requireMcpRoots()[operation](command.brainId, command.serverId, new AbortController().signal);
  });
  handle(IPC.mcp.diagnosticsPage, async (_event, brainId: string, serverId: string, afterSequence?: number, limit?: number) => {
    if (!dependencies.mcpDiagnostics) throw new Error("MCP diagnostics are unavailable.");
    const id = requireId(brainId), server = requireId(serverId); await repository.get(id, false);
    mcp.requireInteractionOrigin(mcp.hostConnection(id, server));
    return dependencies.mcpDiagnostics.page(id, server, afterSequence, limit);
  });
  handle(IPC.mcp.interactions, (_event, brainId: string) => mcpInteractions?.list(requireId(brainId)) ?? []);
  handle(IPC.mcp.respondInteraction, (_event, response: import("../shared/mcpInteractions").McpInteractionResponse) => {
    if (!mcpInteractions) throw new Error("MCP review requests are unavailable.");
    return mcpInteractions.respond(response);
  });
  handle(IPC.mcp.appDescribe, (_event, brainId: string, viewId: string) => {
    if (!mcpApps) throw new Error("Interactive connected apps are unavailable.");
    return mcpApps.describe(requireId(brainId), requireId(viewId));
  });
  handle(IPC.mcp.appOpen, (_event, brainId: string, viewId: string, selection?: { captureSha256: string; reviewedMapping: true }) => {
    if (!mcpApps) throw new Error("Interactive connected apps are unavailable.");
    return mcpApps.open(requireId(brainId), requireId(viewId), selection);
  });
  handle(IPC.mcp.appRequest, (_event, request: import("../shared/mcpApps").McpAppRequest) => {
    if (!mcpApps) throw new Error("Interactive connected apps are unavailable.");
    return mcpApps.request(request);
  });
  handle(IPC.mcp.appClose, (_event, brainId: string, viewId: string, token: string) => {
    if (!mcpApps) throw new Error("Interactive connected apps are unavailable.");
    mcpApps.close(requireId(brainId), requireId(viewId), token);
  });
  handle(IPC.mcp.appDisplayModeAcknowledge, (_event, request: import("../shared/mcpApps").McpAppDisplayAcknowledgement) => {
    if (!mcpApps) throw new Error("Interactive connected apps are unavailable.");
    mcpApps.acknowledgeDisplayMode(request);
  });
  handle(IPC.mcp.registry, (_event, brainId: string) => mcp.registry(requireId(brainId)));
  handle(IPC.mcp.tasksLocalPage, (_event, brainId: string, after?: string, limit?: number) =>
    mcp.tasksLocalPage(requireId(brainId), after, limit));
  handle(IPC.mcp.tasksStopWait, (_event, brainId: string, taskRef: string) =>
    mcp.stopTaskWait(requireId(brainId), requireId(taskRef)));
  handle(IPC.maps.list, () => {
    if (!dependencies.maps) throw new Error("Map connection settings are unavailable."); return dependencies.maps.list();
  });
  handle(IPC.maps.basemapList, () => {
    if (!dependencies.mapBasemaps) throw new Error("Optional map imagery settings are unavailable."); return dependencies.mapBasemaps.list();
  });
  handle(IPC.maps.basemapSave, (_event, value: import("../shared/mapBasemaps").MapBasemapSave) => {
    if (!dependencies.mapBasemaps) throw new Error("Optional map imagery settings are unavailable."); return dependencies.mapBasemaps.save(value);
  });
  handle(IPC.maps.basemapRemove, (_event, id: string, expectedRevision: number) => {
    if (!dependencies.mapBasemaps) throw new Error("Optional map imagery settings are unavailable."); return dependencies.mapBasemaps.remove(requireId(id, "map imagery profile id"), expectedRevision);
  });
  handle(IPC.maps.save, (_event, value: import("../shared/maps").MapConnectionSave) => {
    if (!dependencies.maps) throw new Error("Map connection settings are unavailable."); return dependencies.maps.save(value);
  });
  handle(IPC.maps.remove, (_event, id: string, expectedRevision: number) => {
    if (!dependencies.maps) throw new Error("Map connection settings are unavailable."); return dependencies.maps.remove(requireId(id, "map connection id"), expectedRevision);
  });
  handle(IPC.mcp.configure, (_event, request: McpRegistryUpdateRequest) => mcp.configure(request));
  handle(IPC.mcp.authStatus, (_event, brainId: string, serverId: string) => mcp.authStatus(requireId(brainId), requireId(serverId, "MCP server id")));
  handle(IPC.mcp.authorize, (_event, brainId: string, serverId: string, input: import("../shared/mcpAuth").McpOAuthStart) =>
    mcp.authorize(requireId(brainId), requireId(serverId, "MCP server id"), input));
  handle(IPC.mcp.cancelAuthorization, (_event, brainId: string, serverId: string) => mcp.cancelAuthorization(requireId(brainId), requireId(serverId, "MCP server id")));
  handle(IPC.mcp.disconnectAuthorization, (_event, brainId: string, serverId: string) => mcp.disconnectAuthorization(requireId(brainId), requireId(serverId, "MCP server id")));
  handle(IPC.mcp.updateCredentials, (_event, brainId: string, serverId: string, input: import("../shared/mcpAuth").McpCredentialUpdate) =>
    mcp.updateCredentials(requireId(brainId), requireId(serverId, "MCP server id"), input));
  handle(IPC.mcp.add, (_event, request: McpServerRegistrationRequest) => mcp.add(request));
  handle(IPC.mcp.remove, (_event, brainId: string, serverId: string) =>
    mcp.remove(requireId(brainId), requireId(serverId, "MCP server id"))
  );
  handle(IPC.mcp.refresh, (_event, brainId: string, serverId: string) =>
    mcp.refresh(requireId(brainId), requireId(serverId, "MCP server id"))
  );

  handle(
    IPC.data.selectBuildResources,
    async (
      event,
      kind: BuildResourceSelection["kind"],
      requestedSelection?: ExperienceUploadKind
    ) => {
      if (!["files", "folder"].includes(kind)) {
        throw new Error("Invalid build resource kind.");
      }
      const folder = kind === "folder";
      const { descriptor } = uploadDescriptor(requestedSelection);
      const choice = await dialog.showOpenDialog(senderWindow(event), {
        title: folder
          ? "Choose a dataset folder"
          : descriptor.title,
        properties: folder ? ["openDirectory"] : ["openFile", "multiSelections"],
        filters: folder
          ? undefined
          : [
              {
                name: descriptor.filterName,
                extensions: [...descriptor.extensions]
              },
              { name: "All files", extensions: ["*"] }
            ]
      });
      if (choice.canceled || choice.filePaths.length === 0) return null;
      const id = randomUUID();
      const paths = choice.filePaths.map((path) => resolve(path));
      const measured = await measureSelectedPaths(paths);
      const selection = {
        id,
        kind,
        label:
          folder
            ? basename(paths[0]!)
            : paths.length === 1
              ? basename(paths[0]!)
              : `${paths.length} selected ${descriptor.shortLabel}`,
        itemCount: paths.length,
        bytes: measured.bytes,
        fileCount: measured.fileCount,
        paths,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        state: "selected" as const
      };
      await buildSelections.put(selection);
      const { paths: _paths, createdAt: _createdAt, ...publicSelection } = selection;
      return publicSelection;
    }
  );
  handle(IPC.data.listBuildResources, () => buildSelections.list());
  handle(IPC.data.discardBuildResource, (_event, selectionId: string) =>
    buildSelections.delete(requireId(selectionId, "build resource selection id"))
  );
  handle(
    IPC.data.startBuildResource,
    async (_event, request: BuildResourceStartRequest) => {
      if (!isRecord(request)) throw new Error("Invalid build resource request.");
      requireNewLearningRequestPolicy(request);
      const brainId = requireId(request.brainId);
      const selectionId = requireId(
        request.selectionId,
        "build resource selection id"
      );
      const selection = await buildSelections.get(selectionId);
      if (!selection) {
        throw new Error("The selected build resource is no longer available.");
      }
      await buildSelections.claim(selectionId, brainId);
      const job = jobs.startBuildResource(
        { ...request, brainId, selectionId },
        selection.paths,
        (manifestId) => buildSelections.commitManifest(selectionId, manifestId),
        () => buildSelections.complete(selectionId)
      );
      await buildSelections.attachRuntimeJob(selectionId, job.id);
      return job;
    }
  );

  handle(IPC.data.preview, async (event, request: DatasetPreviewRequest) => {
    if (!isRecord(request)) throw new Error("Invalid dataset preview request.");
    requireNewLearningRequestPolicy(request);
    const brainId = requireId(request.brainId);
    const requestId = request.requestId === undefined
      ? randomUUID()
      : requireId(request.requestId, "dataset preview request id");
    if (previewControllers.has(requestId)) {
      throw new Error("That dataset preview is already running.");
    }
    const folder = request.selection === "folder";
    const { descriptor } = uploadDescriptor(
      folder ? undefined : request.selection
    );
    const choice = await dialog.showOpenDialog(senderWindow(event), {
      title: folder ? "Choose a dataset folder" : descriptor.title,
      properties: folder ? ["openDirectory"] : ["openFile", "multiSelections"],
      filters: folder
        ? undefined
        : [
            {
              name: descriptor.filterName,
              extensions: [...descriptor.extensions]
            },
            { name: "All files", extensions: ["*"] }
          ]
    });
    if (choice.canceled || choice.filePaths.length === 0) return null;
    const controller = new AbortController();
    previewControllers.set(requestId, { brainId, controller });
    const startedAt = new Date().toISOString();
    let latestCounts = {
      discoveredFiles: 0,
      discoveredBytes: 0,
      hashedFiles: 0,
      hashedBytes: 0
    };
    const publish = (
      value: Omit<DatasetPreviewProgress, "schemaVersion" | "requestId" | "brainId" | "startedAt" | "updatedAt">
    ): void => {
      if (event.sender.isDestroyed()) return;
      latestCounts = {
        discoveredFiles: value.discoveredFiles,
        discoveredBytes: value.discoveredBytes,
        hashedFiles: value.hashedFiles,
        hashedBytes: value.hashedBytes
      };
      event.sender.send(IPC.data.previewEvent, {
        schemaVersion: 1,
        requestId,
        brainId,
        startedAt,
        updatedAt: new Date().toISOString(),
        ...value
      } satisfies DatasetPreviewProgress);
    };
    publish({
      phase: "discovering",
      discoveredFiles: 0,
      discoveredBytes: 0,
      hashedFiles: 0,
      hashedBytes: 0,
      message: "Discovering files and committing their content hashes…"
    });
    try {
      const manifest = await service.previewDataset(brainId, choice.filePaths, {
        signal: controller.signal,
        ...(request.parserSelection ? { parserSelection: request.parserSelection } : {}),
        onProgress: (progress) => {
          const current = progress.currentFile ? ` · ${progress.currentFile}` : "";
          publish({
            ...progress,
            message:
              progress.phase === "committing"
                ? "Committing the deterministic dataset snapshot…"
                : progress.phase === "hashing"
                  ? `Hashing source ${progress.hashedFiles + 1}${current}`
                  : `Discovered ${progress.discoveredFiles} source${progress.discoveredFiles === 1 ? "" : "s"}${current}`
          });
        }
      });
      publish({
        phase: "complete",
        discoveredFiles: manifest.discoveredFiles,
        discoveredBytes: manifest.discoveredBytes,
        hashedFiles: latestCounts.hashedFiles,
        hashedBytes: latestCounts.hashedBytes,
        message: `Committed ${manifest.discoveredFiles} source${manifest.discoveredFiles === 1 ? "" : "s"} to a resumable manifest.`
      });
      return manifest;
    } catch (error) {
      const cancelled = controller.signal.aborted ||
        (error instanceof Error && error.name === "AbortError");
      publish({
        phase: cancelled ? "cancelled" : "failed",
        ...latestCounts,
        message: cancelled
          ? "Dataset discovery and hashing were cancelled; no partial manifest was kept."
          : error instanceof Error
            ? error.message
            : "Dataset preview failed."
      });
      throw error;
    } finally {
      previewControllers.delete(requestId);
    }
  });
  handle(IPC.data.cancelPreview, (_event, requestId: string) => {
    const preview = previewControllers.get(requireId(requestId, "dataset preview request id"));
    if (!preview) return false;
    preview.controller.abort();
    return true;
  });
  handle(IPC.data.start, (_event, request: DatasetStartRequest) => {
    if (!isRecord(request)) throw new Error("Invalid dataset start request.");
    requireNewLearningRequestPolicy(request);
    requireId(request.brainId);
    requireId(request.manifestId, "dataset manifest id");
    return jobs.startIngestion({ ...request, resume: request.resume ?? true });
  });
  handle(IPC.data.pause, (_event, jobId: string) =>
    jobs.cancel(requireId(jobId, "job id"))
  );
  handle(IPC.data.resume, (_event, request: DatasetStartRequest) => {
    if (!isRecord(request)) throw new Error("Invalid dataset resume request.");
    requireNewLearningRequestPolicy(request);
    requireId(request.brainId);
    requireId(request.manifestId, "dataset manifest id");
    return jobs.resumeIngestion({ ...request, resume: true });
  });
  handle(IPC.data.resumable, async (_event, brainId: string) =>
    (await service.datasets.latestResumable(requireId(brainId))) ?? null
  );
  handle(IPC.data.coverage, (_event, brainId: string, manifestId: string) =>
    service.datasets.coverage(
      requireId(brainId),
      requireId(manifestId, "dataset manifest id")
    )
  );

  handle(IPC.data.ingestFiles, async (event, request: IngestFilesRequest) => {
    requireNewLearningRequestPolicy(request);
    const brainId = requireId(request.brainId);
    const { descriptor } = uploadDescriptor(request.selection);
    const choice = await dialog.showOpenDialog(senderWindow(event), {
      title: descriptor.title,
      properties: ["openFile", "multiSelections"],
      filters: [
        {
          name: descriptor.filterName,
          extensions: [...descriptor.extensions]
        },
        { name: "All files", extensions: ["*"] }
      ]
    });
    if (choice.canceled) return [];
    return service.ingestPaths(brainId, choice.filePaths, request.policy);
  });
  handle(IPC.data.ingestFolder, async (event, request: IngestFilesRequest) => {
    requireNewLearningRequestPolicy(request);
    const brainId = requireId(request.brainId);
    const choice = await dialog.showOpenDialog(senderWindow(event), {
      title: "Choose a folder to learn",
      properties: ["openDirectory"]
    });
    if (choice.canceled || !choice.filePaths[0]) return [];
    return service.ingestPaths(brainId, [choice.filePaths[0]], request.policy);
  });
  handle(
    IPC.data.ingestDropped,
    (_event, request: IngestFilesRequest, rawPaths: unknown) => {
      requireNewLearningRequestPolicy(request);
      const brainId = requireId(request.brainId);
      if (!Array.isArray(rawPaths)) {
        throw new Error("Invalid dropped-file selection.");
      }
      const paths = rawPaths.map((path) => {
        if (
          typeof path !== "string" ||
          path.includes("\0") ||
          path.length > 32_000 ||
          !isAbsolute(path)
        ) {
          throw new Error("A dropped file has an invalid local path.");
        }
        return resolve(path);
      });
      return service.ingestPaths(brainId, paths, request.policy);
    }
  );
  handle(IPC.data.ingestWeb, (_event, request: IngestWebRequest) => {
    requireNewLearningRequestPolicy(request);
    return service.ingestWeb(request);
  });
  handle(IPC.data.crawlWeb, (_event, request: WebCrawlRequest) => {
    requireNewLearningRequestPolicy(request);
    requireId(request.brainId);
    return jobs.startCrawl(request);
  });
  handle(IPC.data.cancel, (_event, jobId: string) =>
    jobs.cancel(requireId(jobId, "job id"))
  );
  handle(
    IPC.data.sources,
    (_event, brainId: string, cursor?: string, limit?: number) =>
      repository.trainingSourcePage(
        requireId(brainId),
        typeof cursor === "string" ? cursor : undefined,
        typeof limit === "number" ? limit : undefined
      )
  );

  handle(IPC.modality.capabilities, (_event, brainId: string) =>
    service.modalityCapabilities(requireId(brainId, "brain id"))
  );
  const capture = () => { if (!dependencies.imaginationCapture) throw new Error("Imagination capture is unavailable."); return dependencies.imaginationCapture; };
  const assembly = () => { if (!dependencies.imaginationAssembly) throw new Error("Delivered timeline encoding is unavailable."); return dependencies.imaginationAssembly; };
  handle(IPC.modality.assemblyPlan, (_event, request: import("../shared/imaginationAssembly").ImaginationAssemblyPlanRequest) => { if (!isRecord(request)) throw new Error("Invalid replay plan."); requireId(request.brainId); requireId(request.recordingId);requireId(request.requestId); return assembly().plan(request); });
  handle(IPC.modality.assemblyPlanCancel,(_event,request:{brainId:string;requestId:string})=>{if(!isRecord(request))throw new Error("Invalid preparation cancellation.");requireId(request.brainId);requireId(request.requestId);return assembly().cancelPlan(request);});
  handle(IPC.modality.assemblyCreate, (_event, request: import("../shared/imaginationAssembly").ImaginationAssemblyRequest) => { if (!isRecord(request)) throw new Error("Invalid replay request."); requireId(request.brainId); requireId(request.recordingId); requireId(request.requestId); return assembly().assemble(request); });
  handle(IPC.modality.assemblyGet, (_event, id: string, assemblyId: string) => assembly().get(requireId(id), requireId(assemblyId)));
  handle(IPC.modality.assemblyManifest, (_event, id: string, assemblyId: string) => assembly().manifest(requireId(id), requireId(assemblyId)));
  handle(IPC.modality.assemblyManifestPage, (_event, id: string, assemblyId: string, query: { section: "fragments" | "tracks" | "spans"; afterIndex?: number; limit?: number }) => assembly().manifestPage(requireId(id), requireId(assemblyId), query));
  handle(IPC.modality.assemblyList, (_event, id: string, query?: { afterId?: string; limit?: number; recordingId?: string }) => assembly().list(requireId(id), query));
  for (const [channel, method] of [[IPC.modality.assemblyCancel, "cancel"], [IPC.modality.assemblyResume, "resume"]] as const) handle(channel, (_event, request: { brainId: string; assemblyId: string; expectedRevision: number }) => { if (!isRecord(request)) throw new Error("Invalid replay control."); requireId(request.brainId); requireId(request.assemblyId); return assembly()[method](request); });
  handle(IPC.modality.captureStatus, (_event, id: string) => capture().status(requireId(id)));
  handle(IPC.modality.captureSettings, (_event, request: { brainId: string; expectedRevision: number; bufferSeconds: number; alwaysSave: boolean }) => {
    if (!isRecord(request)) throw new Error("Invalid capture settings."); return capture().settings({ brainId: requireId(request.brainId), expectedRevision: request.expectedRevision, bufferSeconds: request.bufferSeconds, alwaysSave: request.alwaysSave }); });
  handle(IPC.modality.captureRecord, (_event, request: { brainId: string; requestId: string; lookbackSeconds: number }) => {
    if (!isRecord(request)) throw new Error("Invalid capture record request."); return capture().record({ brainId: requireId(request.brainId), requestId: requireId(request.requestId), lookbackSeconds: request.lookbackSeconds }); });
  for (const [channel, method] of [[IPC.modality.captureStop, "stop"], [IPC.modality.captureCancel, "cancel"], [IPC.modality.captureResume, "resume"]] as const) handle(channel, (_event, request: { brainId: string; recordingId: string; expectedRevision: number }) => {
    if (!isRecord(request)) throw new Error("Invalid capture boundary."); return capture()[method]({ brainId: requireId(request.brainId), recordingId: requireId(request.recordingId), expectedRevision: request.expectedRevision }); });
  handle(IPC.modality.captureRecordings, (_event, id: string, query?: { beforeSequence?: number; limit?: number }) => capture().recordings(requireId(id), query));
  handle(IPC.modality.captureFragments, (_event, id: string, recordingId: string, query?: { afterIndex?: number; limit?: number }) => capture().fragments(requireId(id), requireId(recordingId), query));
  const localTeacher = (): NonNullable<IpcDependencies["localTeacher"]> => {
    if (!dependencies.localTeacher) throw new Error("Local GGUF teacher jobs are unavailable."); return dependencies.localTeacher;
  };
  handle(IPC.localTeacher.select, (_event, request: import("../shared/localTeacher").LocalTeacherSelectRequest) => { requireId(request?.brainId); return localTeacher().select(request); });
  handle(IPC.localTeacher.cancelSelection, (_event, brainId: string, requestId: string) => localTeacher().cancelSelection(requireId(brainId), requireId(requestId)));
  handle(IPC.localTeacher.selectionProgress, (_event, brainId: string, requestId: string) => localTeacher().selectionProgress(requireId(brainId), requireId(requestId)));
  handle(IPC.localTeacher.selections, (_event, brainId: string) => localTeacher().selections(requireId(brainId)));
  handle(IPC.localTeacher.plan, (_event, request: { brainId: string; teacherId: string; prompts?: string[] }) => { requireId(request?.brainId); return localTeacher().plan(request); });
  handle(IPC.localTeacher.start, (_event, request: import("../shared/localTeacher").LocalTeacherStartRequest) => { requireId(request?.brainId); return localTeacher().start(request); });
  handle(IPC.localTeacher.jobs, (_event, brainId: string) => localTeacher().jobs(requireId(brainId)));
  handle(IPC.localTeacher.get, (_event, brainId: string, jobId: string) => localTeacher().get(requireId(brainId), requireId(jobId)));
  handle(IPC.localTeacher.items, (_event, brainId: string, jobId: string, after?: number, limit?: number) => localTeacher().items(requireId(brainId), requireId(jobId), after, limit));
  handle(IPC.localTeacher.pause, (_event, brainId: string, jobId: string) => localTeacher().pause(requireId(brainId), requireId(jobId)));
  handle(IPC.localTeacher.resume, (_event, brainId: string, jobId: string) => localTeacher().resume(requireId(brainId), requireId(jobId)));
  handle(IPC.localTeacher.stop, (_event, brainId: string, jobId: string) => localTeacher().stop(requireId(brainId), requireId(jobId)));
  handle(IPC.processMemory.preferences, (_event, brainId: string) => {
    if (!dependencies.processMemory) throw new Error("OS process-memory settings are unavailable.");
    return dependencies.processMemory.preferences(requireId(brainId));
  });
  handle(IPC.processMemory.update, (_event, request: import("../shared/processMemory").ProcessMemoryPreferenceUpdate) => {
    if (!dependencies.processMemory) throw new Error("OS process-memory settings are unavailable.");
    requireId(request?.brainId);
    return dependencies.processMemory.update(request);
  });
  handle(IPC.voice.preferences, (_event, brainId: string) => voice.preferences(requireId(brainId)));
  handle(IPC.voice.update, (_event, request: InstanceVoiceUpdateRequest) => {
    requireId(request?.brainId); return voice.update(request);
  });
  handle(IPC.voice.migrateLegacy, (_event, request: InstanceVoiceLegacyRequest) => {
    requireId(request?.brainId); return voice.migrateLegacy(request);
  });
  handle(IPC.modality.artifacts, (
    _event,
    brainId: string,
    cursor?: string,
    limit?: number
  ) =>
    jobs.listArtifacts(requireId(brainId, "brain id"), cursor, limit)
  );
  handle(IPC.modality.generate, (_event, request: ModalityGenerateRequest) => {
    requireId(request.brainId);
    const job = jobs.generate(request); service.mediaArtifacts?.noteCaptureOrigin(`job:${job.brainId}:${job.id}`, { kind: "user-requested", authority: "main-ui", requestId: job.id }); return job;
  });
  handle(IPC.modality.generateSpeech, (_event, request: unknown) => jobs.generateSpeech(request));
  handle(
    IPC.modality.selectInput,
    async (
      event,
      request: Omit<ModalityGenerateRequest, "inputPath">
    ) => {
      requireId(request.brainId);
      const extensions =
        request.modality === "audio"
          ? EXPERIENCE_UPLOADS.audio.extensions
          : request.modality === "video"
            ? EXPERIENCE_UPLOADS.video.extensions
            : EXPERIENCE_UPLOADS.images.extensions;
      const choice = await dialog.showOpenDialog(senderWindow(event), {
        title: `Choose ${request.modality} input`,
        properties: ["openFile"],
        filters: [{ name: `${request.modality} input`, extensions: [...extensions] }]
      });
      if (choice.canceled || !choice.filePaths[0]) return null;
      const job = jobs.generate({ ...request, inputPath: choice.filePaths[0] }); service.mediaArtifacts?.noteCaptureOrigin(`job:${job.brainId}:${job.id}`, { kind: "user-requested", authority: "main-ui", requestId: job.id }); return job;
    }
  );
  handle(IPC.modality.cancel, (_event, jobId: string) =>
    jobs.cancel(requireId(jobId, "job id"))
  );
  handle(
    IPC.modality.startObservation,
    (_event, request: LiveObservationSessionStartRequest) => {
      if (!isRecord(request)) throw new Error("Invalid live observation request.");
      requireId(request.brainId, "brain id");
      return jobs.startObservation(request);
    }
  );
  handle(
    IPC.modality.pushObservation,
    (_event, packet: LiveObservationPacket) => {
      if (!isRecord(packet)) throw new Error("Invalid live observation packet.");
      requireId(packet.sessionId, "observation session id");
      return jobs.pushObservation(packet);
    }
  );
  handle(IPC.modality.stopObservation, (_event, sessionId: string) =>
    jobs.stopObservation(requireId(sessionId, "observation session id"))
  );
  handle(IPC.modality.cancelObservation, (_event, sessionId: string) =>
    jobs.cancelObservation(requireId(sessionId, "observation session id"))
  );
  handle(
    IPC.modality.requestObservationControl,
    (_event, request: LiveObservationControlRequest) => {
      if (!isRecord(request)) {
        throw new Error("Invalid live observation control request.");
      }
      requireId(request.sessionId, "observation session id");
      return jobs.requestObservationControl(request);
    }
  );
  handle(
    IPC.modality.resolveObservationControl,
    (_event, resolution: LiveObservationControlResolution) => {
      if (!isRecord(resolution)) {
        throw new Error("Invalid live observation control resolution.");
      }
      requireId(resolution.sessionId, "observation session id");
      requireId(resolution.controlId, "observation control id");
      return jobs.resolveObservationControl(resolution);
    }
  );

  handle(IPC.trace.list, async (_event, brainId: string, query?: TraceQuery) => {
    const traces = (await service.getReconciledBrain(requireId(brainId))).traces;
    const before = query?.before ? Date.parse(query.before) : Number.POSITIVE_INFINITY;
    const limit = Math.max(1, Math.min(1_000, Math.round(query?.limit ?? 100)));
    return traces
      .filter((trace) => Date.parse(trace.createdAt) < before)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .slice(0, limit);
  });

  handle(IPC.tool.descriptions, (_event, brainId: string) => service.describeTools(requireId(brainId)));
  handle(IPC.tool.listPermissions, (_event, brainId: string) =>
    service.listToolPermissions(requireId(brainId))
  );
  handle(
    IPC.tool.setPermission,
    (
      _event,
      brainId: string,
      toolId: string,
      level: ToolPermissionLevel,
      requestId?: string,
      expectedRevision?: number
    ) => service.setToolPermission(
      requireId(brainId),
      toolId,
      level,
      typeof requestId === "string" ? requireId(requestId, "permission request id") : undefined,
      expectedRevision
    )
  );
  handle(IPC.tool.execute, (_event, invocation: ToolInvocation) => tools.execute(invocation));
  handle(IPC.tool.declineApproval, (_event, id: string, token: string) =>
    tools.invalidateApproval(requireId(id), requireId(token, "approval token")));
  handle(IPC.tool.cancel, (_event, brainId: string, requestId?: string) =>
    tools.cancel(
      requireId(brainId),
      typeof requestId === "string" ? requireId(requestId, "tool request id") : undefined
    )
  );
  handle(IPC.tool.preferences, () => tools.preferences());
  handle(IPC.tool.setPreferences, (_event, value: ToolRuntimePreferences) => tools.setPreferences(value));

  handle(IPC.plugins.list, () => plugins.list());
  handle(IPC.plugins.package, (_event, pluginId: string, contentSha256?: string) =>
    plugins.package(requireId(pluginId, "plugin id"), contentSha256)
  );
  handle(IPC.plugins.installFile, async (event) => {
    const result = await dialog.showOpenDialog(senderWindow(event), {
      title: "Install an Omni plugin package",
      properties: ["openFile"],
      filters: [
        { name: "Omni plugin", extensions: ["omni-plugin", "json"] },
        { name: "All files", extensions: ["*"] }
      ]
    });
    if (result.canceled || !result.filePaths[0]) return null;
    return plugins.installFile(resolve(result.filePaths[0]));
  });
  handle(IPC.plugins.installPackage, (
    _event,
    value: unknown,
    sourceLabel?: string
  ) => plugins.installPackage(
    value,
    typeof sourceLabel === "string" ? sourceLabel : "Plugin maker"
  ));
  handle(IPC.plugins.edit, (
    _event,
    pluginId: string,
    baseContentSha256: string,
    value: unknown
  ) => plugins.edit(
    requireId(pluginId, "plugin id"),
    baseContentSha256,
    value
  ));
  handle(IPC.plugins.activate, (
    _event,
    pluginId: string,
    contentSha256: string
  ) => plugins.activate(requireId(pluginId, "plugin id"), contentSha256));
  handle(IPC.plugins.setEnabled, (
    _event,
    pluginId: string,
    enabled: boolean,
    expectedContentSha256: string,
    expectedRevision: number
  ) => plugins.setEnabled(
    requireId(pluginId, "plugin id"), enabled, expectedContentSha256, expectedRevision
  ));
  handle(IPC.plugins.exportFile, async (
    event,
    pluginId: string,
    contentSha256?: string
  ) => {
    const id = requireId(pluginId, "plugin id");
    const packageValue = await plugins.package(id, contentSha256);
    const result = await dialog.showSaveDialog(senderWindow(event), {
      title: "Export Omni plugin package",
      defaultPath: `${packageValue.manifest.id}-${packageValue.manifest.version}.omni-plugin`,
      filters: [{ name: "Omni plugin", extensions: ["omni-plugin"] }]
    });
    if (result.canceled || !result.filePath) return null;
    return plugins.exportFile(id, resolve(result.filePath), contentSha256, true);
  });
  handle(IPC.plugins.instance, (
    _event,
    brainId: string,
    pluginId: string
  ) => plugins.instance(requireId(brainId), requireId(pluginId, "plugin id")));
  handle(IPC.plugins.setInstanceAccess, (
    _event,
    request: PluginInstanceAccessRequest
  ) => plugins.setInstanceAccess(request));
  handle(IPC.plugins.portable, (_event, brainId: string) =>
    plugins.portable(requireId(brainId)));
  const runtime = (): import("./pluginRuntimeService").PluginRuntimeService => {
    if (!dependencies.pluginRuntime) throw new Error("Plugin feature runtime is unavailable.");
    return dependencies.pluginRuntime;
  };
  handle(IPC.pluginRuntime.activate, (_event, request: import("../shared/pluginRuntime").PluginRuntimeActivationRequest) => runtime().activate(request));
  handle(IPC.pluginRuntime.stop, (_event, brainId: string, sessionId: string) => runtime().stop(requireId(brainId), requireId(sessionId)));
  handle(IPC.pluginRuntime.list, (_event, brainId: string, page?: Parameters<import("../shared/pluginRuntime").PluginRuntimeApi["list"]>[1]) => runtime().list(requireId(brainId), page));
  handle(IPC.pluginRuntime.history, (_event, brainId: string, page?: Parameters<import("../shared/pluginRuntime").PluginRuntimeApi["history"]>[1]) => runtime().history(requireId(brainId), page));
  handle(IPC.pluginRuntime.readHistoryEvent, (_event, brainId: string, sequence: number, offset: number, length: number, sourceSha256?: string) =>
    runtime().readHistoryEvent(requireId(brainId), sequence, offset, length, sourceSha256));
  handle(IPC.pluginRuntime.stateSources, (_event, brainId: string, page?: Parameters<import("../shared/pluginRuntime").PluginRuntimeApi["stateSources"]>[1]) => runtime().stateSources(requireId(brainId), page));
  handle(IPC.pluginRuntime.stateConflicts, (_event, brainId: string, page?: Parameters<import("../shared/pluginRuntime").PluginRuntimeApi["stateConflicts"]>[1]) => runtime().stateConflicts(requireId(brainId), page));
  handle(IPC.pluginRuntime.readSourceState, (_event, brainId: string, sourceSha256: string, pluginHash: string, moduleId: string, key: string, maxBytes?: number) =>
    runtime().readSourceState(requireId(brainId), sourceSha256, pluginHash, moduleId, key, maxBytes));
  handle(IPC.pluginRuntime.readSourceStateBytes, (_event, brainId: string, sourceSha256: string, pluginHash: string, moduleId: string, key: string, offset: number, length: number) =>
    runtime().readSourceStateBytes(requireId(brainId), sourceSha256, pluginHash, moduleId, key, offset, length));
  handle(IPC.pluginRuntime.selectSourceState, (_event, request: import("../shared/pluginRuntime").PluginRuntimeStateSelection) => runtime().selectSourceState(request));
  handle(IPC.pluginRuntime.contributions, (_event, brainId: string) => runtime().contributions(requireId(brainId)));
  handle(IPC.pluginRuntime.invokeContribution, (_event, brainId: string, sessionId: string, contributionId: string, eventId: string, data?: unknown) =>
    runtime().invokeContribution(requireId(brainId), requireId(sessionId), requireId(contributionId), requireId(eventId), data));
  handle(IPC.pluginRuntime.approvals, (_event, brainId: string) => runtime().approvals(requireId(brainId)));
  handle(IPC.pluginRuntime.review, (_event, request: import("../shared/pluginRuntime").PluginRuntimeReviewRequest) => runtime().review(request));
  handle(IPC.plugins.openPanel, (
    _event,
    brainId: string,
    pluginId: string,
    contentSha256: string,
    panelId: string
  ) => dependencies.pluginPanels.open(
    requireId(brainId),
    requireId(pluginId, "plugin id"),
    contentSha256,
    requireId(panelId, "panel id")
  ));
  handle(IPC.plugins.panelRequest, (_event, request: PluginInteractiveToolRequest) =>
    dependencies.pluginPanels.request(request));
  handle(IPC.plugins.reviewPanelRequest, (_event, request: PluginInteractiveReviewRequest) =>
    dependencies.pluginPanels.review(request));
  handle(IPC.plugins.closePanel, (
    _event,
    brainId: string,
    viewId: string,
    channelNonce: string
  ) => dependencies.pluginPanels.close(
    requireId(brainId), requireId(viewId, "panel view id"), channelNonce
  ));
  handle(IPC.plugins.panelHistory, (_event, brainId: string, page?: Parameters<import("../shared/plugins").PluginsApi["panelHistory"]>[1]) =>
    dependencies.pluginPanels.history(requireId(brainId), page));
  handle(IPC.plugins.panelHistoryBytes, (_event, brainId: string, sequence: number, offset: number, length: number, sourceSha256?: string) =>
    dependencies.pluginPanels.historyBytes(requireId(brainId), sequence, offset, length, sourceSha256));
  handle(IPC.plugins.panelHistorySources, (_event, brainId: string, page?: Parameters<import("../shared/plugins").PluginsApi["panelHistorySources"]>[1]) =>
    dependencies.pluginPanels.historySources(requireId(brainId), page));
  handle(IPC.artifacts.open, (_event, request: HtmlArtifactOpenRequest) =>
    dependencies.htmlArtifacts.open(request));
  handle(IPC.artifacts.activate, (_event, request: HtmlArtifactActivationRequest) =>
    dependencies.htmlArtifacts.activate(request));
  handle(IPC.artifacts.request, (_event, request: HtmlArtifactToolRequest) =>
    dependencies.htmlArtifacts.request(request));
  handle(IPC.artifacts.review, (_event, request: HtmlArtifactReviewRequest) =>
    dependencies.htmlArtifacts.review(request));
  handle(IPC.artifacts.close, (_event, brainId: string, viewId: string, channelNonce: string) =>
    dependencies.htmlArtifacts.close(requireId(brainId), requireId(viewId, "HTML artifact view id"), channelNonce));

  handle(IPC.agent.fork, (_event, brainId: string, name?: string) =>
    repository.fork(requireId(brainId), typeof name === "string" ? name : undefined)
  );
  handle(
    IPC.agent.previewMerge,
    (_event, sourceBrainId: string, targetBrainId: string) =>
      service.previewMerge(requireId(sourceBrainId), requireId(targetBrainId))
  );
  handle(
    IPC.agent.merge,
    (
      _event,
      sourceBrainId: string,
      targetBrainId: string,
      reviewToken: string
    ) =>
      service.merge(
        requireId(sourceBrainId),
        requireId(targetBrainId),
        requireId(reviewToken, "merge review token")
      )
  );

  handle(IPC.evolution.start, (_event, request: EvolutionStartRequest) => {
    if (!isRecord(request)) throw new Error("Invalid evolution request.");
    requireId(request.brainId);
    return evolution.start(request);
  });
  handle(IPC.evolution.stop, (_event, brainId: string, runId: string) =>
    evolution.stop(requireId(brainId), requireId(runId, "evolution run id"))
  );
  handle(
    IPC.evolution.listCandidates,
    (_event, brainId: string, runId?: string) =>
      evolution.listCandidates(
        requireId(brainId),
        typeof runId === "string" ? requireId(runId, "evolution run id") : undefined
      )
  );
  handle(IPC.evolution.approve, (_event, request: EvolutionApprovalRequest) => {
    if (!isRecord(request)) throw new Error("Invalid evolution approval request.");
    requireId(request.brainId);
    requireId(request.candidateId, "evolution candidate id");
    return evolution.approve(request);
  });
  handle(IPC.evolution.rollback, (_event, request: EvolutionRollbackRequest) => {
    if (!isRecord(request)) throw new Error("Invalid evolution rollback request.");
    requireId(request.brainId);
    requireId(request.candidateId, "evolution candidate id");
    return evolution.rollback(request);
  });

  handle(IPC.catalog.list, () => loadCatalog(appPath));
  handle(IPC.catalog.importUrl, async (_event, request: ImportUrlRequest) => {
    const brain = await service.importUrl(request);
    return brain;
  });
  handle(IPC.catalog.loadRecipeUrl, (_event, request: ImportUrlRequest) =>
    service.loadRecipeUrl(request)
  );
  handle(IPC.catalog.loadRecipeEntry, async (_event, entryId: string) => {
    const id = requireId(entryId, "catalog entry id");
    const entry = (await loadCatalog(appPath)).find((item) => item.id === id);
    if (!entry || entry.kind !== "recipe") throw new Error("Catalog recipe entry was not found.");
    const root = resolve(catalogRoot(appPath));
    const source = resolve(root, entry.sourceUrl.replace(/^catalog[\\/]/, ""));
    if (source !== root && !source.startsWith(`${root}/`) && !source.startsWith(`${root}\\`)) {
      throw new Error("Catalog recipe path escapes the bundled catalog.");
    }
    const recipe = await service.loadRecipeBuffer(await readFile(source), `catalog:${entry.id}`);
    if (entry.sha256 && recipe.sha256 !== entry.sha256.toLocaleLowerCase()) {
      throw new Error("Bundled recipe checksum verification failed.");
    }
    return { ...recipe, license: entry.license };
  });
  handle(IPC.catalog.loadRecipeFile, async (event) => {
    const result = await dialog.showOpenDialog(senderWindow(event), {
      title: "Open an Omni build recipe",
      properties: ["openFile"],
      filters: [
        { name: "Omni build recipe", extensions: ["json"] },
        { name: "All files", extensions: ["*"] }
      ]
    });
    if (result.canceled || !result.filePaths[0]) return null;
    const path = resolve(result.filePaths[0]);
    return service.loadRecipeBuffer(await readFile(path), path);
  });
  handle(
    IPC.catalog.installModalityPackUrl,
    (_event, request: InstallModalityPackUrlRequest) =>
      service.installModalityPackUrl(requireId(request.brainId, "brain id"), request)
  );
  handle(IPC.catalog.installModalityPackFile, async (event, brainId: string) => {
    const id = requireId(brainId, "brain id");
    const result = await dialog.showOpenDialog(senderWindow(event), {
      title: "Install an Omni modality pack",
      properties: ["openFile"],
      filters: [
        { name: "Omni modality pack", extensions: ["omnipack"] },
        { name: "All files", extensions: ["*"] }
      ]
    });
    if (result.canceled || !result.filePaths[0]) return null;
    const path = resolve(result.filePaths[0]);
    return service.installModalityPackBuffer(id, await readFile(path), path);
  });
  handle(IPC.catalog.listModalityPacks, (_event, brainId: string) =>
    service.listModalityPacks(requireId(brainId, "brain id"))
  );
  handle(IPC.catalog.hardwareProfile, async () => {
    const profile = await hardwareProfile();
    // An unsupported/resource-deferred primitive is not a brain readiness quiz
    // or a reason to prevent ordinary setup. The collector validates/caches it.
    // Optional calibration is never a readiness gate for opening Build. Its
    // background collector defers when the one neural worker is occupied.
    void dependencies.collectHardwareCompute?.(profile).catch(() => undefined);
    return profile;
  });
  handle(IPC.catalog.deviceTopology, async (_event, request: DeviceTopologyRequest | undefined) => {
    if (!dependencies.deviceResources) throw new Error("Device resource discovery is unavailable.");
    return dependencies.deviceResources.topology(request);
  });
  handle(IPC.catalog.selectSpillPools, async (event, brainId?: string) => {
    if (!dependencies.deviceResources) throw new Error("Device resource discovery is unavailable.");
    const id = brainId === undefined ? undefined : requireId(brainId, "brain id");
    if (id) await repository.get(id);
    const result = await dialog.showOpenDialog(senderWindow(event), {
      title: "Choose drives or folders for temporary neural storage",
      properties: ["openDirectory", "multiSelections", "createDirectory"]
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return dependencies.deviceResources.selectedSpillDirectories(result.filePaths, id);
  });
  handle(IPC.catalog.resourcePlan, async (_event, value: WorkingMemoryPlanRequest) => {
    const request = requireWorkingMemoryPlanRequest(value);
    return service.planWorkingMemory(request, {
      hardwareTier: request.hardwareTier,
      brainId: request.brainId
    });
  });

  const requireQuestions = (): UserQuestionService => {
    if (!questions) throw new Error("Questions are unavailable in this runtime.");
    return questions;
  };
  const requireClock = (): import("./clockService").ClockService => {
    if (!dependencies.clock) throw new Error("Host clock workspace is unavailable."); return dependencies.clock;
  };
  handle(IPC.clock.snapshot, (_event, id: string) => requireClock().snapshot(requireId(id)));
  handle(IPC.clock.now, (_event, timeZone?: string) => requireClock().now(timeZone));
  handle(IPC.clock.schedules, (_event, id: string, query?: import("../shared/clock").ClockPageQuery) => requireClock().schedules(requireId(id), query));
  handle(IPC.clock.stopwatches, (_event, id: string, query?: import("../shared/clock").ClockPageQuery) => requireClock().stopwatches(requireId(id), query));
  handle(IPC.clock.occurrences, (_event, id: string, query?: import("../shared/clock").ClockPageQuery) => requireClock().occurrences(requireId(id), query));
  handle(IPC.clock.laps, (_event, id: string, stopwatchId: string, after?: number) => requireClock().laps(requireId(id), stopwatchId, after));
  handle(IPC.clock.rates, (_event, id: string, query?: import("../shared/clock").ClockPageQuery) => requireClock().rates(requireId(id), query));
  const unsubscribeClock = dependencies.clock?.subscribe(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.clock.event, event);
  });
  const requireAutomations = (): import("./automationService").AutomationService => {
    if (!dependencies.automation) throw new Error("Independent action files are unavailable."); return dependencies.automation;
  };
  handle(IPC.automation.snapshot, (_event, id: string) => requireAutomations().snapshot(requireId(id)));
  handle(IPC.automation.definitions, (_event, id: string, query?: import("../shared/automation").AutomationPageQuery) => requireAutomations().definitions(requireId(id), query));
  handle(IPC.automation.program, (_event, id: string, definitionId: string, revision?: string) => requireAutomations().program(requireId(id), definitionId, revision));
  handle(IPC.automation.runs, (_event, id: string, query?: import("../shared/automation").AutomationPageQuery) => requireAutomations().runs(requireId(id), query));
  handle(IPC.automation.steps, (_event, id: string, runId: string, query?: import("../shared/automation").AutomationPageQuery) => requireAutomations().steps(requireId(id), runId, query));
  handle(IPC.automation.currentStep, (_event, id: string, runId: string) => requireAutomations().currentStep(requireId(id), runId));
  const unsubscribeAutomations = dependencies.automation?.subscribe(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.automation.event, event);
  });
  const requireLearning = (): LearningCoordinator => {
    if (!learning) throw new Error("Learning workspaces are unavailable in this runtime.");
    return learning;
  };
  handle(IPC.learning.start, (_event, request: SpecificLearnRequest) => requireLearning().start(request));
  handle(IPC.learning.startFactCheck, (_event, request: FactCheckRequest) => requireLearning().startFactCheck(request));
  handle(IPC.learning.list, (_event, id: string) => requireLearning().list(requireId(id)));
  handle(IPC.learning.get, (_event, id: string, taskId: string) => requireLearning().get(requireId(id), taskId));
  handle(IPC.learning.factCheck, (_event, id: string, taskId: string) => requireLearning().factCheck(requireId(id), taskId));
  handle(IPC.learning.pause, (_event, id: string, taskId: string) => requireLearning().pause(requireId(id), taskId));
  handle(IPC.learning.resume, (_event, id: string, taskId: string) => requireLearning().resume(requireId(id), taskId));
  handle(IPC.learning.stop, (_event, id: string, taskId: string) => requireLearning().stop(requireId(id), taskId));
  handle(IPC.learning.steer, (_event, request: LearningTaskSteerRequest) => requireLearning().steer(request));
  handle(IPC.learning.review, (_event, request: FactCheckReviewRequest) => requireLearning().review(request, "user"));
  handle(IPC.learning.acceptCorrection, (_event, id: string, taskId: string, claimId: string, findingId: string) => requireLearning().acceptCorrection(requireId(id), taskId, claimId, findingId));
  const unsubscribeLearning = learning?.subscribe(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.learning.event, event);
  });
  handle(IPC.questions.request, async (_event, request: UserQuestionRequest) => {
    await repository.get(requireId(request?.brainId, "brain id"));
    return requireQuestions().request({ ...request, source: { kind: "host" } });
  });
  handle(IPC.questions.list, (_event, id: string, query?: UserQuestionListQuery) => requireQuestions().list(requireId(id), query));
  handle(IPC.questions.respond, (_event, request: UserQuestionRespondRequest) => requireQuestions().respond(request));
  handle(IPC.questions.cancel, (_event, id: string, questionId: string, reason?: string) => requireQuestions().cancel(requireId(id), questionId, reason));
  handle(IPC.questions.resume, (_event, id: string, questionId?: string) => requireQuestions().resume(requireId(id), questionId));
  handle(IPC.questions.preferences, (_event, id: string) => requireQuestions().preferences(requireId(id)));
  handle(IPC.questions.setPreferences, (_event, request: UserQuestionPreferences & { brainId: string }) => requireQuestions().setPreferences(request));
  const unsubscribeQuestions = questions?.subscribe(event => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.questions.event, event);
    }
  });

  const jobListener = (event: unknown): void => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.train.event, event);
    }
  };
  jobs.on("event", jobListener);
  const storageOperationListener = (event: BrainStorageOperationEvent): void => {
    dependencies.osWork?.observeStorageOperation(event);
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) {
        window.webContents.send(IPC.brain.storageOperationEvent, event);
      }
    }
  };
  storageOperations.on("event", storageOperationListener);
  const unsubscribeFullMerge = service.onFullMerge((progress) => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.brain.fullMergeEvent, progress);
    }
  });
  const unsubscribeLifecycle = service.onInstanceLifecycle((state) => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.brain.lifecycleEvent, state);
    }
  });
  const unsubscribeMcpCapabilities = mcp.onCapabilityEvent((event) => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.mcp.capabilityEvent, event);
    }
  });
  const unsubscribeMcpInteractions = mcpInteractions?.subscribe((event) => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.mcp.interactionEvent, event);
    }
  });
  const unsubscribeMcpAppObservations = mcpApps?.subscribeObservations(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.mcp.appObservation, event);
  });
  const unsubscribeMcpAppServerNotifications = mcpApps?.subscribeServerNotifications(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.mcp.appServerNotification, event);
  });
  const unsubscribeMcpAppDisplayModes = mcpApps?.subscribeDisplayModes(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.mcp.appDisplayMode, event);
  });
  const unsubscribePluginEvents = plugins.onEvent((event) => {
    dependencies.publishPluginEvent(event);
  });
  const unsubscribeRuntimeEvents = dependencies.pluginRuntime?.onEvent(event => {
    for (const window of studioWindows()) window.webContents.send(IPC.pluginRuntime.event, event);
  });
  const teacherListener = (event: unknown): void => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.teacher.event, event);
    }
  };
  teacher.on("event", teacherListener);
  const adviceListener = (event: import("../shared/externalAdvice").ExternalAdviceEvent): void => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.advice.event, event);
  };
  dependencies.advice?.on("event", adviceListener);
  const observationListener = (event: unknown): void => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) {
        window.webContents.send(IPC.modality.observationEvent, event);
      }
    }
  };
  jobs.on("observation", observationListener);
  const actionListener = (event: unknown): void => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.chat.actionEvent, event);
    }
  };
  actions.on("event", actionListener);
  const streamListener = (event: unknown): void => {
    for (const window of studioWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.chat.streamEvent, event);
    }
  };
  actions.on("stream", streamListener);
  const unsubscribeDirections = dependencies.directions?.subscribe(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.chat.directionsEvent, event);
  });
  const unsubscribeSparseSubstrate = dependencies.sparseSubstrate?.onUiEvent(event => {
    for (const { key, record: subscription } of sparseSubscriptions.matching(event.brainId)) {
      const sender = webContents.fromId(subscription.senderId);
      if (!sender || sender.isDestroyed()) { sparseSubscriptions.deleteKey(key); continue; }
      const envelope: import("../shared/sparseSubstrateTransfer").SparseSubstrateUiEventEnvelope = {
        subscriptionId: subscription.subscriptionId, brainId: subscription.brainId, event
      };
      try { sender.send(IPC.substrate.event, envelope); }
      catch { sparseSubscriptions.deleteKey(key); }
    }
  });
  const unsubscribeImaginationCapture = dependencies.imaginationCapture?.subscribe(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.modality.captureEvent, event);
  });
  const unsubscribeImaginationAssembly = dependencies.imaginationAssembly?.subscribe(event => {
    for (const window of studioWindows()) if (!window.isDestroyed()) window.webContents.send(IPC.modality.assemblyEvent, event);
  });
  const unsubscribeImaginationPlans=dependencies.imaginationAssembly?.subscribePlans(event=>{for(const window of BrowserWindow.getAllWindows())if(!window.isDestroyed())window.webContents.send(IPC.modality.assemblyPlanEvent,event);});

  return () => {
    unsubscribeDirections?.();
    unsubscribeSparseSubstrate?.();
    sparseSubscriptions.clear();
    unsubscribeImaginationCapture?.();
    unsubscribeImaginationAssembly?.();
    unsubscribeImaginationPlans?.();
    unsubscribeLearning?.();
    unsubscribeClock?.();
    unsubscribeAutomations?.();
    unsubscribeQuestions?.();
    unsubscribeFullMerge();
    unsubscribeLifecycle();
    unsubscribeMcpCapabilities();
    unsubscribeMcpInteractions?.();
    unsubscribeMcpAppObservations?.();
    unsubscribeMcpAppServerNotifications?.();
    unsubscribeMcpAppDisplayModes?.();
    unsubscribePluginEvents();
    unsubscribeRuntimeEvents?.();
    for (const preview of previewControllers.values()) preview.controller.abort();
    previewControllers.clear();
    for (const controller of substrateQueryControllers.values()) controller.abort();
    substrateQueryControllers.clear();
    storageOperations.cancelAll();
    teacher.closeModelDiscovery();
    jobs.off("event", jobListener);
    storageOperations.off("event", storageOperationListener);
    teacher.off("event", teacherListener);
    dependencies.advice?.off("event", adviceListener);
    dependencies.advice?.close();
    dependencies.actionReviewer?.dispose();
    jobs.off("observation", observationListener);
    actions.off("event", actionListener);
    actions.off("stream", streamListener);
    for (const channel of channels) ipcMain.removeHandler(channel);
  };
}
