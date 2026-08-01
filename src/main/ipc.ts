import { randomUUID } from "node:crypto";
import { cpus, freemem, totalmem } from "node:os";
import { readFile } from "node:fs/promises";
import { basename, isAbsolute, join, resolve } from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  shell,
  type IpcMainInvokeEvent
} from "electron";
import type {
  BrainConfig,
  BrainExportMode,
  BuildResourceSelection,
  BuildResourceStartRequest,
  CatalogEntry,
  CreateBrainRequest,
  DatasetPreviewRequest,
  DatasetStartRequest,
  FeedbackRequest,
  HardwareProfile,
  EvolutionApprovalRequest,
  EvolutionRollbackRequest,
  EvolutionStartRequest,
  InstallModalityPackUrlRequest,
  ImportUrlRequest,
  IngestFilesRequest,
  IngestWebRequest,
  ModalityGenerateRequest,
  StartTrainingRequest,
  SubstrateQuery,
  ToolInvocation,
  ToolPermissionLevel,
  TraceQuery,
  WebCrawlRequest
} from "../shared/types";
import { IPC } from "../shared/ipc";
import {
  EXPERIENCE_UPLOADS,
  isExperienceUploadKind,
  type ExperienceUploadKind
} from "../shared/uploadSupport";
import type { BrainRepository } from "./brainRepository";
import {
  BrainService,
  type RuntimeJobManager
} from "./brainService";
import type { EngineSupervisor } from "./engineSupervisor";
import type { ToolExecutor } from "./toolExecutor";
import type { ChatActionController } from "./chatActionController";
import type { EvolutionController } from "./evolutionController";

export interface IpcDependencies {
  repository: BrainRepository;
  service: BrainService;
  jobs: RuntimeJobManager;
  engine: EngineSupervisor;
  tools: ToolExecutor;
  actions: ChatActionController;
  evolution: EvolutionController;
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

function requireId(value: unknown, label = "id"): string {
  if (typeof value !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value)) {
    throw new Error(`Invalid ${label}.`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
    personal: "Use the Personal architecture with background consolidation and compact modalities.",
    gpu: "Use the GPU architecture with CUDA or DirectML acceleration and checkpointing.",
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
  const { repository, service, jobs, engine, tools, actions, evolution, appPath } = dependencies;
  const channels: string[] = [];
  const buildSelections = new Map<
    string,
    BuildResourceSelection & { paths: string[] }
  >();
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
  handle(IPC.brain.get, (_event, id: string) => repository.get(requireId(id)));
  handle(IPC.brain.create, (_event, request: CreateBrainRequest) => {
    if (!isRecord(request) || !isRecord(request.config)) throw new Error("Invalid build request.");
    return service.create(request);
  });
  handle(IPC.brain.update, async (_event, id: string, config: BrainConfig) => {
    const brainId = requireId(id);
    const brain = await repository.updateConfig(brainId, config);
    await engine.tryRequest(
      "update_config",
      {
        brainId,
        config: brain.config,
        storagePath: repository.brainDirectory(brainId)
      },
      300_000
    );
    return brain;
  });
  handle(IPC.brain.duplicate, (_event, id: string, name?: string) =>
    repository.duplicate(requireId(id), typeof name === "string" ? name : undefined)
  );
  handle(IPC.brain.fork, (_event, id: string, name?: string) =>
    repository.fork(requireId(id), typeof name === "string" ? name : undefined)
  );
  handle(IPC.brain.remove, async (_event, id: string) => {
    const brainId = requireId(id);
    await engine.tryRequest("unload", { brainId }, 30_000);
    return repository.remove(brainId);
  });
  handle(IPC.brain.snapshot, async (_event, id: string, label?: string) => {
    const brainId = requireId(id);
    await engine.tryRequest(
      "snapshot",
      {
        brainId,
        label,
        storagePath: repository.brainDirectory(brainId)
      },
      300_000
    );
    return repository.snapshot(brainId, typeof label === "string" ? label : undefined);
  });
  handle(IPC.brain.listSnapshots, (_event, id: string) =>
    repository.listSnapshots(requireId(id))
  );
  handle(IPC.brain.restoreSnapshot, async (_event, id: string, snapshotId: string) => {
    const brainId = requireId(id);
    const restored = await repository.restoreSnapshot(
      brainId,
      requireId(snapshotId, "snapshot id")
    );
    await engine.tryRequest(
      "reload",
      {
        brainId,
        storagePath: repository.brainDirectory(brainId)
      },
      300_000
    );
    return restored;
  });
  handle(IPC.brain.export, async (event, id: string, mode: BrainExportMode = "current") => {
    const brain = await repository.get(requireId(id));
    if (!["current", "origin", "private-archive", "referenced"].includes(mode)) {
      throw new Error("Invalid export mode.");
    }
    if (mode === "private-archive") {
      const confirmation = await dialog.showMessageBox(senderWindow(event), {
        type: "warning",
        title: "Export private training archive?",
        message: "This export can contain retained source material.",
        detail:
          "Tool grants and known credential patterns are removed, and detected credentials in retained text block export. This cannot prove arbitrary weights or binary data are secret-free; inspect the archive before sharing.",
        buttons: ["Cancel", "Export private archive"],
        defaultId: 0,
        cancelId: 0,
        noLink: true
      });
      if (confirmation.response !== 1) return null;
    }
    if (mode !== "origin") {
      await engine.tryRequest(
        "snapshot",
        {
          brainId: brain.id,
          label: "export-candidate",
          config: brain.config,
          storagePath: repository.brainDirectory(brain.id)
        },
        300_000
      );
    }
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
    await repository.exportBundle(brain.id, destination, mode);
    return destination;
  });
  handle(IPC.brain.importFile, async (event) => {
    const choice = await dialog.showOpenDialog(senderWindow(event), {
      title: "Import an OmniCortex brain",
      properties: ["openFile"],
      filters: [{ name: "Omni brain", extensions: ["omni"] }]
    });
    if (choice.canceled || !choice.filePaths[0]) return null;
    const brain = await repository.importBundle(choice.filePaths[0]);
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
  handle(IPC.brain.querySubstrate, (_event, id: string, query?: SubstrateQuery) => {
    const brainId = requireId(id);
    if (query !== undefined && !isRecord(query)) {
      throw new Error("Invalid substrate query.");
    }
    return service.querySubstrate(brainId, query);
  });
  handle(IPC.brain.workspace, (_event, id: string) =>
    service.workspace(requireId(id))
  );

  handle(IPC.chat.send, (_event, id: string, input: string, turnId?: string) =>
    actions.send(
      requireId(id),
      input,
      undefined,
      typeof turnId === "string" ? requireId(turnId, "turn id") : undefined
    )
  );
  handle(IPC.chat.cancel, (_event, id: string, turnId?: string) =>
    actions.cancel(
      requireId(id),
      typeof turnId === "string" ? requireId(turnId, "turn id") : undefined
    )
  );
  handle(IPC.chat.list, async (_event, id: string) => (await repository.get(requireId(id))).messages);
  handle(IPC.chat.feedback, (_event, request: FeedbackRequest) => service.feedback(request));

  handle(IPC.train.start, (_event, request: StartTrainingRequest) => {
    if (!isRecord(request)) throw new Error("Invalid training request.");
    requireId(request.brainId);
    return jobs.startTraining(request);
  });
  handle(IPC.train.consolidate, (_event, id: string) => service.consolidate(requireId(id)));
  handle(IPC.train.cancel, (_event, jobId: string) => jobs.cancel(requireId(jobId, "job id")));
  handle(IPC.train.list, (_event, id?: string) =>
    jobs.list(typeof id === "string" ? requireId(id) : undefined)
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
      const selection: BuildResourceSelection & { paths: string[] } = {
        id,
        kind,
        label:
          folder
            ? basename(paths[0]!)
            : paths.length === 1
              ? basename(paths[0]!)
              : `${paths.length} selected ${descriptor.shortLabel}`,
        itemCount: paths.length,
        paths
      };
      buildSelections.set(id, selection);
      const { paths: _paths, ...publicSelection } = selection;
      return publicSelection;
    }
  );
  handle(IPC.data.discardBuildResource, (_event, selectionId: string) =>
    buildSelections.delete(requireId(selectionId, "build resource selection id"))
  );
  handle(
    IPC.data.startBuildResource,
    async (_event, request: BuildResourceStartRequest) => {
      if (!isRecord(request)) throw new Error("Invalid build resource request.");
      const brainId = requireId(request.brainId);
      const selectionId = requireId(
        request.selectionId,
        "build resource selection id"
      );
      const selection = buildSelections.get(selectionId);
      if (!selection) {
        throw new Error("The selected build resource is no longer available.");
      }
      const manifest = await service.previewDataset(brainId, selection.paths);
      buildSelections.delete(selectionId);
      return jobs.startIngestion({
        brainId,
        manifestId: manifest.id,
        policy: request.policy ?? "pretrain",
        epochs: request.epochs ?? 1,
        resume: true
      });
    }
  );

  handle(IPC.data.preview, async (event, request: DatasetPreviewRequest) => {
    if (!isRecord(request)) throw new Error("Invalid dataset preview request.");
    const brainId = requireId(request.brainId);
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
    return service.previewDataset(brainId, choice.filePaths);
  });
  handle(IPC.data.start, (_event, request: DatasetStartRequest) => {
    if (!isRecord(request)) throw new Error("Invalid dataset start request.");
    requireId(request.brainId);
    requireId(request.manifestId, "dataset manifest id");
    return jobs.startIngestion({ ...request, resume: request.resume ?? true });
  });
  handle(IPC.data.pause, (_event, jobId: string) =>
    jobs.cancel(requireId(jobId, "job id"))
  );
  handle(IPC.data.resume, (_event, request: DatasetStartRequest) => {
    if (!isRecord(request)) throw new Error("Invalid dataset resume request.");
    requireId(request.brainId);
    requireId(request.manifestId, "dataset manifest id");
    return jobs.startIngestion({ ...request, resume: true });
  });
  handle(IPC.data.coverage, (_event, brainId: string, manifestId: string) =>
    service.datasets.coverage(
      requireId(brainId),
      requireId(manifestId, "dataset manifest id")
    )
  );

  handle(IPC.data.ingestFiles, async (event, request: IngestFilesRequest) => {
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
  handle(IPC.data.ingestWeb, (_event, request: IngestWebRequest) => service.ingestWeb(request));
  handle(IPC.data.crawlWeb, (_event, request: WebCrawlRequest) => {
    requireId(request.brainId);
    return jobs.startCrawl(request);
  });
  handle(IPC.data.cancel, (_event, jobId: string) =>
    jobs.cancel(requireId(jobId, "job id"))
  );

  handle(IPC.modality.generate, (_event, request: ModalityGenerateRequest) => {
    requireId(request.brainId);
    return jobs.generate(request);
  });
  handle(
    IPC.modality.selectInput,
    async (
      event,
      request: Omit<ModalityGenerateRequest, "inputPath">
    ) => {
      requireId(request.brainId);
      const extensions =
        request.modality === "audio"
          ? ["wav", "mp3", "flac", "m4a", "ogg"]
          : request.modality === "video"
            ? ["mp4", "webm", "mov", "mkv"]
            : ["png", "jpg", "jpeg", "webp", "bmp", "gif"];
      const choice = await dialog.showOpenDialog(senderWindow(event), {
        title: `Choose ${request.modality} input`,
        properties: ["openFile"],
        filters: [{ name: `${request.modality} input`, extensions }]
      });
      if (choice.canceled || !choice.filePaths[0]) return null;
      return jobs.generate({ ...request, inputPath: choice.filePaths[0] });
    }
  );
  handle(IPC.modality.cancel, (_event, jobId: string) =>
    jobs.cancel(requireId(jobId, "job id"))
  );

  handle(IPC.trace.list, async (_event, brainId: string, query?: TraceQuery) => {
    const traces = (await repository.get(requireId(brainId))).traces;
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
  handle(IPC.tool.cancel, (_event, brainId: string) => tools.cancel(requireId(brainId)));

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
  handle(IPC.catalog.hardwareProfile, () => hardwareProfile());

  const jobListener = (event: unknown): void => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(IPC.train.event, event);
    }
  };
  jobs.on("event", jobListener);
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
    buildSelections.clear();
    jobs.off("event", jobListener);
    actions.off("event", actionListener);
    actions.off("stream", streamListener);
    for (const channel of channels) ipcMain.removeHandler(channel);
  };
}
