import { contextBridge, ipcRenderer, webUtils } from "electron";
import type {
  ActionEvent,
  ChatStreamEvent,
  OmniApi,
  RuntimeJobEvent
} from "../shared/types";
import { IPC } from "../shared/ipc";

const invoke = <T>(channel: string, ...args: unknown[]): Promise<T> =>
  ipcRenderer.invoke(channel, ...args) as Promise<T>;

const windowApi: OmniApi["window"] = {
  minimize: () => invoke(IPC.window.minimize),
  maximize: () => invoke(IPC.window.maximize),
  close: () => invoke(IPC.window.close),
  isMaximized: () => invoke(IPC.window.isMaximized),
  openExternal: (url) => invoke(IPC.window.openExternal, url),
  revealDataFolder: () => invoke(IPC.window.revealDataFolder),
  platform: () => invoke(IPC.window.platform),
  setAppearance: (request) => invoke(IPC.window.setAppearance, request)
};

const api: OmniApi = {
  window: windowApi,
  brain: {
    list: () => invoke(IPC.brain.list),
    get: (id) => invoke(IPC.brain.get, id),
    create: (request) => invoke(IPC.brain.create, request),
    update: (id, config) => invoke(IPC.brain.update, id, config),
    duplicate: (id, name) => invoke(IPC.brain.duplicate, id, name),
    fork: (id, name) => invoke(IPC.brain.fork, id, name),
    remove: (id) => invoke(IPC.brain.remove, id),
    snapshot: (id, label) => invoke(IPC.brain.snapshot, id, label),
    listSnapshots: (id) => invoke(IPC.brain.listSnapshots, id),
    restoreSnapshot: (id, snapshotId) =>
      invoke(IPC.brain.restoreSnapshot, id, snapshotId),
    export: (id, mode) => invoke(IPC.brain.export, id, mode),
    importFile: () => invoke(IPC.brain.importFile),
    onImported: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, brain: Parameters<typeof listener>[0]): void =>
        listener(brain);
      ipcRenderer.on(IPC.brain.imported, wrapped);
      return () => ipcRenderer.removeListener(IPC.brain.imported, wrapped);
    },
    health: (id) => invoke(IPC.brain.health, id),
    querySubstrate: (id, query) => invoke(IPC.brain.querySubstrate, id, query),
    workspace: (id) => invoke(IPC.brain.workspace, id)
  },
  chat: {
    send: (id, input, turnId) => invoke(IPC.chat.send, id, input, turnId),
    cancel: (id, turnId) => invoke(IPC.chat.cancel, id, turnId),
    list: (id) => invoke(IPC.chat.list, id),
    feedback: (request) => invoke(IPC.chat.feedback, request),
    onAction: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: ActionEvent): void =>
        listener(value);
      ipcRenderer.on(IPC.chat.actionEvent, wrapped);
      return () => ipcRenderer.removeListener(IPC.chat.actionEvent, wrapped);
    },
    onStream: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: ChatStreamEvent): void =>
        listener(value);
      ipcRenderer.on(IPC.chat.streamEvent, wrapped);
      return () => ipcRenderer.removeListener(IPC.chat.streamEvent, wrapped);
    }
  },
  train: {
    start: (request) => invoke(IPC.train.start, request),
    consolidate: (id) => invoke(IPC.train.consolidate, id),
    cancel: (jobId) => invoke(IPC.train.cancel, jobId),
    list: (id) => invoke(IPC.train.list, id),
    onEvent: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: RuntimeJobEvent): void =>
        listener(value);
      ipcRenderer.on(IPC.train.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.train.event, wrapped);
    }
  },
  data: {
    selectBuildResources: (kind, selection) =>
      invoke(IPC.data.selectBuildResources, kind, selection),
    discardBuildResource: (selectionId) =>
      invoke(IPC.data.discardBuildResource, selectionId),
    startBuildResource: (request) => invoke(IPC.data.startBuildResource, request),
    preview: (request) => invoke(IPC.data.preview, request),
    start: (request) => invoke(IPC.data.start, request),
    pause: (jobId) => invoke(IPC.data.pause, jobId),
    resume: (request) => invoke(IPC.data.resume, request),
    coverage: (brainId, manifestId) =>
      invoke(IPC.data.coverage, brainId, manifestId),
    ingestFiles: (request) => invoke(IPC.data.ingestFiles, request),
    ingestFolder: (request) => invoke(IPC.data.ingestFolder, request),
    ingestDropped: (request, files) => {
      const paths = files
        .map((file) => {
          try {
            return webUtils.getPathForFile(file as File);
          } catch {
            return "";
          }
        })
        .filter(Boolean);
      return invoke(IPC.data.ingestDropped, request, paths);
    },
    ingestWeb: (request) => invoke(IPC.data.ingestWeb, request),
    crawlWeb: (request) => invoke(IPC.data.crawlWeb, request),
    cancel: (jobId) => invoke(IPC.data.cancel, jobId)
  },
  modality: {
    generate: (request) => invoke(IPC.modality.generate, request),
    selectInput: (request) => invoke(IPC.modality.selectInput, request),
    cancel: (jobId) => invoke(IPC.modality.cancel, jobId)
  },
  trace: {
    list: (brainId, query) => invoke(IPC.trace.list, brainId, query)
  },
  tool: {
    listPermissions: (brainId) => invoke(IPC.tool.listPermissions, brainId),
    setPermission: (brainId, toolId, level) =>
      invoke(IPC.tool.setPermission, brainId, toolId, level),
    execute: (request) => invoke(IPC.tool.execute, request),
    cancel: (brainId) => invoke(IPC.tool.cancel, brainId)
  },
  agent: {
    fork: (brainId, name) => invoke(IPC.agent.fork, brainId, name),
    previewMerge: (sourceBrainId, targetBrainId) =>
      invoke(IPC.agent.previewMerge, sourceBrainId, targetBrainId),
    merge: (sourceBrainId, targetBrainId, reviewToken) =>
      invoke(IPC.agent.merge, sourceBrainId, targetBrainId, reviewToken)
  },
  evolution: {
    start: (request) => invoke(IPC.evolution.start, request),
    stop: (brainId, runId) => invoke(IPC.evolution.stop, brainId, runId),
    listCandidates: (brainId, runId) =>
      invoke(IPC.evolution.listCandidates, brainId, runId),
    approve: (request) => invoke(IPC.evolution.approve, request),
    rollback: (request) => invoke(IPC.evolution.rollback, request)
  },
  catalog: {
    list: () => invoke(IPC.catalog.list),
    importUrl: (request) => invoke(IPC.catalog.importUrl, request),
    loadRecipeEntry: (id) => invoke(IPC.catalog.loadRecipeEntry, id),
    loadRecipeUrl: (request) => invoke(IPC.catalog.loadRecipeUrl, request),
    loadRecipeFile: () => invoke(IPC.catalog.loadRecipeFile),
    installModalityPackUrl: (request) =>
      invoke(IPC.catalog.installModalityPackUrl, request),
    installModalityPackFile: (brainId) =>
      invoke(IPC.catalog.installModalityPackFile, brainId),
    listModalityPacks: (brainId) => invoke(IPC.catalog.listModalityPacks, brainId),
    hardwareProfile: () => invoke(IPC.catalog.hardwareProfile)
  }
};

contextBridge.exposeInMainWorld("omni", api);
