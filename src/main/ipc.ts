import { randomUUID } from "node:crypto";
import { cpus, freemem, totalmem } from "node:os";
import { lstat, readFile, readdir } from "node:fs/promises";
import { basename, isAbsolute, join, resolve } from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeTheme,
  shell,
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
  ApiTeacherProvider,
  ApiTeacherTrainingRequest,
  McpServerRegistrationRequest,
  ToolRuntimePreferences
} from "../shared/types";
import type { CortexQuery, CortexActivityQuery } from "../shared/cortexInspection";
import { IPC } from "../shared/ipc";
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
import { requireWorkingMemoryPlanRequest } from "./resourcePlanRequest";
import { BrainStorageOperationManager } from "./brainStorageOperations";
import type { IdleCognitionScheduler } from "./idleCognitionScheduler";

export interface IpcDependencies {
  repository: BrainRepository;
  service: BrainService;
  jobs: RuntimeJobManager;
  engine: EngineSupervisor;
  tools: ToolExecutor;
  actions: ChatActionController;
  evolution: EvolutionController;
  buildSelections: BuildResourceSelectionStore;
  initialization: BuildInitializationCoordinator;
  mobile: MobileGateway;
  teacher: ApiTeacherTrainingService;
  mcp: McpClientService;
  idleCognition?: Pick<IdleCognitionScheduler, "cancelActive">;
  /** Main-owned primitive measurement; no renderer model/shape arguments. */
  collectHardwareCompute?: (profile: HardwareProfile) => Promise<unknown>;
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
    job: event.job,
    ...(output ? { data: output } : {})
  };
}

function requireChatTurnMetadata(value: unknown): ChatTurnMetadata | undefined {
  if (value === undefined) return undefined;
  if (
    !isRecord(value) ||
    value.kind !== "steer" ||
    value.source !== "human" ||
    typeof value.createdAt !== "string" ||
    !Number.isFinite(Date.parse(value.createdAt))
  ) {
    throw new Error("Invalid chat turn metadata.");
  }
  return {
    kind: "steer",
    source: "human",
    replacesTurnId: requireId(value.replacesTurnId, "replaced turn id"),
    createdAt: new Date(value.createdAt).toISOString()
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
  return {
    schemaVersion: 1,
    turnId: requireId(value.turnId, "delivery receipt turn id"),
    content,
    createdAt: new Date(value.createdAt).toISOString(),
    state: value.state as ChatDeliveryReceiptRequest["state"]
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
    evolution,
    buildSelections,
    initialization,
    mobile,
    teacher,
    mcp,
    idleCognition,
    appPath
  } = dependencies;
  const channels: string[] = [];
  const previewControllers = new Map<
    string,
    { brainId: string; controller: AbortController }
  >();
  const substrateQueryControllers = new Map<string, AbortController>();
  const storageOperations = new BrainStorageOperationManager();
  const handle = <T extends unknown[]>(
    channel: string,
    listener: (event: IpcMainInvokeEvent, ...args: T) => unknown
  ): void => {
    ipcMain.removeHandler(channel);
    ipcMain.handle(channel, listener);
    channels.push(channel);
  };

  handle(IPC.window.minimize, (event) => senderWindow(event).minimize());
  handle(IPC.window.maximize, (event) => {
    const window = senderWindow(event);
    if (window.isMaximized()) window.unmaximize();
    else window.maximize();
  });
  handle(IPC.window.close, (event) => senderWindow(event).close());
  handle(IPC.window.isMaximized, (event) => senderWindow(event).isMaximized());
  handle(IPC.window.platform, () => process.platform);
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
  handle(IPC.brain.get, (_event, id: string) =>
    service.getReconciledBrain(requireId(id))
  );
  handle(IPC.brain.create, async (event, request: CreateBrainRequest) => {
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
    const brain = await initialization.create(request, (engineEvent) => {
      if (window.isDestroyed() || engineEvent.type !== "build-progress") return;
      window.webContents.send(IPC.brain.buildEvent, {
        brainId: engineEvent.brainId,
        streamId: engineEvent.streamId,
        sequence: engineEvent.sequence ?? 0,
        phase:
          typeof (engineEvent.data as { phase?: unknown } | undefined)?.phase === "string"
            ? (engineEvent.data as { phase: string }).phase
            : "allocating",
        progress: engineEvent.progress ?? 0,
        label: engineEvent.message ?? "Building OmniCortex native core",
        data: (engineEvent.data as { metrics?: unknown } | undefined)?.metrics
      });
    }, (progressEvent) => {
      if (window.isDestroyed()) return;
      window.webContents.send(
        IPC.brain.buildEvent,
        initialLearningBuildProgressEvent(progressEvent, buildSequence++)
      );
    });
    return brain;
  });
  handle(IPC.brain.completeInitialization, async (_event, id: string) => {
    return initialization.complete(requireId(id));
  });
  handle(IPC.brain.retryInitialization, async (event, id: string) => {
    const brainId = requireId(id);
    const window = senderWindow(event);
    let sequence = 0;
    return initialization.retry(
      brainId,
      (engineEvent) => {
        if (window.isDestroyed() || engineEvent.type !== "build-progress") return;
        window.webContents.send(IPC.brain.buildEvent, {
          brainId,
          streamId: engineEvent.streamId,
          sequence: engineEvent.sequence ?? sequence++,
          phase:
            typeof (engineEvent.data as { phase?: unknown } | undefined)?.phase === "string"
              ? (engineEvent.data as { phase: string }).phase
              : "allocating",
          progress: engineEvent.progress ?? 0,
          label: engineEvent.message ?? "Recovering OmniCortex native core",
          data: (engineEvent.data as { metrics?: unknown } | undefined)?.metrics
        });
      },
      (progressEvent) => {
        if (window.isDestroyed()) return;
        window.webContents.send(
          IPC.brain.buildEvent,
          initialLearningBuildProgressEvent(progressEvent, sequence++)
        );
      }
    );
  });
  handle(IPC.brain.update, async (_event, id: string, config: BrainConfig) => {
    const brainId = requireId(id);
    return service.updateConfig(brainId, config);
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
    await engine.tryRequest("unload", { brainId }, 30_000);
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
    await service.preflightStart(brain.id);
    await engine.tryRequest("unload", { brainId: brain.id }, 30_000);
    await engine.tryRequest(
      "load",
      {
        brainId: brain.id,
        config: brain.config,
        storagePath: repository.brainDirectory(brain.id)
      },
      300_000
    );
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
  handle(IPC.brain.workspace, (_event, id: string) =>
    service.workspace(requireId(id))
  );
  handle(IPC.brain.freshAttention, (_event, id: string) =>
    service.startFreshAttention(requireId(id))
  );
  handle(
    IPC.brain.journalPage,
    (_event, id: string, cursor?: string, limit?: number) =>
      repository.journalPage(
        requireId(id),
        typeof cursor === "string" ? cursor : undefined,
        typeof limit === "number" ? limit : undefined
      )
  );

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
  handle(IPC.chat.cancel, (_event, id: string, turnId?: string) =>
    actions.cancelAndWait(
      requireId(id),
      typeof turnId === "string" ? requireId(turnId, "turn id") : undefined
    )
  );
  handle(IPC.chat.recordDeliveryReceipt, async (
    _event,
    id: string,
    receipt: unknown
  ) => {
    await repository.appendChatDeliveryReceipt(
      requireId(id),
      requireChatDeliveryReceipt(receipt)
    );
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
  handle(IPC.teacher.start, (_event, request: ApiTeacherTrainingRequest) => teacher.start(request));
  handle(IPC.teacher.cancel, (_event, jobId: string) =>
    teacher.cancel(requireId(jobId, "teacher job id"))
  );
  handle(IPC.teacher.list, (_event, brainId?: string) =>
    teacher.list(typeof brainId === "string" ? requireId(brainId) : undefined)
  );

  handle(IPC.mcp.list, (_event, brainId: string) => mcp.list(requireId(brainId)));
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
    return jobs.generate(request);
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
      return jobs.generate({ ...request, inputPath: choice.filePaths[0] });
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

  handle(IPC.tool.listPermissions, (_event, brainId: string) =>
    service.listToolPermissions(requireId(brainId))
  );
  handle(
    IPC.tool.setPermission,
    (_event, brainId: string, toolId: string, level: ToolPermissionLevel) =>
      service.setToolPermission(requireId(brainId), toolId, level)
  );
  handle(IPC.tool.execute, (_event, invocation: ToolInvocation) => tools.execute(invocation));
  handle(IPC.tool.cancel, (_event, brainId: string, requestId?: string) =>
    tools.cancel(
      requireId(brainId),
      typeof requestId === "string" ? requireId(requestId, "tool request id") : undefined
    )
  );
  handle(IPC.tool.preferences, () => tools.preferences());
  handle(IPC.tool.setPreferences, (_event, value: ToolRuntimePreferences) => tools.setPreferences(value));

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
    await service.preflightStart(brain.id);
    await engine.tryRequest("unload", { brainId: brain.id }, 30_000);
    await engine.tryRequest(
      "load",
      {
        brainId: brain.id,
        config: brain.config,
        storagePath: repository.brainDirectory(brain.id)
      },
      300_000
    );
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
  handle(IPC.catalog.resourcePlan, async (_event, value: WorkingMemoryPlanRequest) => {
    const request = requireWorkingMemoryPlanRequest(value);
    return service.planWorkingMemory(request, {
      hardwareTier: request.hardwareTier,
      brainId: request.brainId
    });
  });

  const jobListener = (event: unknown): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.train.event, event);
    }
  };
  jobs.on("event", jobListener);
  const storageOperationListener = (event: BrainStorageOperationEvent): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) {
        window.webContents.send(IPC.brain.storageOperationEvent, event);
      }
    }
  };
  storageOperations.on("event", storageOperationListener);
  const teacherListener = (event: unknown): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.teacher.event, event);
    }
  };
  teacher.on("event", teacherListener);
  const observationListener = (event: unknown): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) {
        window.webContents.send(IPC.modality.observationEvent, event);
      }
    }
  };
  jobs.on("observation", observationListener);
  const actionListener = (event: unknown): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.chat.actionEvent, event);
    }
  };
  actions.on("event", actionListener);
  const streamListener = (event: unknown): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.chat.streamEvent, event);
    }
  };
  actions.on("stream", streamListener);

  return () => {
    for (const preview of previewControllers.values()) preview.controller.abort();
    previewControllers.clear();
    for (const controller of substrateQueryControllers.values()) controller.abort();
    substrateQueryControllers.clear();
    storageOperations.cancelAll();
    jobs.off("event", jobListener);
    storageOperations.off("event", storageOperationListener);
    teacher.off("event", teacherListener);
    jobs.off("observation", observationListener);
    actions.off("event", actionListener);
    actions.off("stream", streamListener);
    for (const channel of channels) ipcMain.removeHandler(channel);
  };
}
