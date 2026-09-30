import { contextBridge, ipcRenderer, webUtils } from "electron";
import type {
  ActionEvent,
  BrainImportFailure,
  BrainStorageOperationEvent,
  ChatStreamEvent,
  DatasetPreviewProgress,
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
    completeInitialization: (id) =>
      invoke(IPC.brain.completeInitialization, id),
    retryInitialization: (id) =>
      invoke(IPC.brain.retryInitialization, id),
    update: (id, config) => invoke(IPC.brain.update, id, config),
    setOnlineLearning: (id, enabled) => invoke(IPC.brain.setOnlineLearning, id, enabled),
    setActiveMode: (id, enabled) =>
      invoke(IPC.brain.setActiveMode, id, enabled),
    duplicate: (id, name, operationId) =>
      invoke(IPC.brain.duplicate, id, name, operationId),
    fork: (id, name, operationId) =>
      invoke(IPC.brain.fork, id, name, operationId),
    pauseStorageOperation: (operationId) =>
      invoke(IPC.brain.pauseStorageOperation, operationId),
    resumeStorageOperation: (operationId) =>
      invoke(IPC.brain.resumeStorageOperation, operationId),
    cancelStorageOperation: (operationId) =>
      invoke(IPC.brain.cancelStorageOperation, operationId),
    onStorageOperation: (listener) => {
      const wrapped = (
        _event: Electron.IpcRendererEvent,
        value: BrainStorageOperationEvent
      ): void => listener(value);
      ipcRenderer.on(IPC.brain.storageOperationEvent, wrapped);
      return () =>
        ipcRenderer.removeListener(IPC.brain.storageOperationEvent, wrapped);
    },
    remove: (request) => invoke(IPC.brain.remove, request),
    snapshot: (id, label, operationId) =>
      invoke(IPC.brain.snapshot, id, label, operationId),
    listSnapshots: (id) => invoke(IPC.brain.listSnapshots, id),
    restoreSnapshot: (id, snapshotId, operationId) =>
      invoke(IPC.brain.restoreSnapshot, id, snapshotId, operationId),
    export: (id, mode, operationId) =>
      invoke(IPC.brain.export, id, mode, operationId),
    importFile: (operationId) => invoke(IPC.brain.importFile, operationId),
    onImported: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, brain: Parameters<typeof listener>[0]): void =>
        listener(brain);
      ipcRenderer.on(IPC.brain.imported, wrapped);
      return () => ipcRenderer.removeListener(IPC.brain.imported, wrapped);
    },
    onImportFailed: (listener) => {
      const wrapped = (
        _event: Electron.IpcRendererEvent,
        failure: BrainImportFailure
      ): void => listener(failure);
      ipcRenderer.on(IPC.brain.importFailed, wrapped);
      return () => ipcRenderer.removeListener(IPC.brain.importFailed, wrapped);
    },
    onBuild: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: Parameters<typeof listener>[0]): void =>
        listener(value);
      ipcRenderer.on(IPC.brain.buildEvent, wrapped);
      return () => ipcRenderer.removeListener(IPC.brain.buildEvent, wrapped);
    },
    health: (id) => invoke(IPC.brain.health, id),
    persistedSubstrateOverview: (id) =>
      invoke(IPC.brain.persistedSubstrateOverview, id),
    querySubstrate: (id, query) => invoke(IPC.brain.querySubstrate, id, query),
    queryConceptIds: (id, view, sourceTurnId, offset) => invoke(IPC.brain.queryConceptIds, id, view, sourceTurnId, offset),
    queryCortex: (id, query) => invoke(IPC.brain.queryCortex, id, query),
    cortexActivity: (id, query) => invoke(IPC.brain.cortexActivity, id, query),
    workspace: (id) => invoke(IPC.brain.workspace, id),
    freshAttention: (id) => invoke(IPC.brain.freshAttention, id),
    journalPage: (id, cursor, limit) =>
      invoke(IPC.brain.journalPage, id, cursor, limit)
  },
  chat: {
    send: (id, input, turnId, turnMetadata) =>
      invoke(IPC.chat.send, id, input, turnId, turnMetadata),
    cancel: (id, turnId) => invoke(IPC.chat.cancel, id, turnId),
    cancelInlineAction: (id, turnId, actionEventId) =>
      invoke(IPC.chat.cancelInlineAction, id, turnId, actionEventId),
    recordDeliveryReceipt: (id, receipt) =>
      invoke(IPC.chat.recordDeliveryReceipt, id, receipt),
    approveAction: (request) => invoke(IPC.chat.approveAction, request),
    list: (id) => invoke(IPC.chat.list, id),
    listPage: (id, beforeSequence, limit) =>
      invoke(IPC.chat.listPage, id, beforeSequence, limit),
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
  mobile: {
    status: () => invoke(IPC.mobile.status),
    startPairing: (request) => invoke(IPC.mobile.startPairing, request),
    stop: () => invoke(IPC.mobile.stop),
    revoke: (deviceId) => invoke(IPC.mobile.revoke, deviceId)
  },
  train: {
    cancel: (jobId) => invoke(IPC.train.cancel, jobId),
    list: (id) => invoke(IPC.train.list, id),
    onEvent: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: RuntimeJobEvent): void =>
        listener(value);
      ipcRenderer.on(IPC.train.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.train.event, wrapped);
    }
  },
  teacher: {
    status: () => invoke(IPC.teacher.status),
    saveCredential: (request) => invoke(IPC.teacher.saveCredential, request),
    removeCredential: (provider) => invoke(IPC.teacher.removeCredential, provider),
    start: (request) => invoke(IPC.teacher.start, request),
    cancel: (jobId) => invoke(IPC.teacher.cancel, jobId),
    list: (brainId) => invoke(IPC.teacher.list, brainId),
    onEvent: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: RuntimeJobEvent): void =>
        listener(value);
      ipcRenderer.on(IPC.teacher.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.teacher.event, wrapped);
    }
  },
  mcp: {
    list: (brainId) => invoke(IPC.mcp.list, brainId),
    add: (request) => invoke(IPC.mcp.add, request),
    remove: (brainId, serverId) => invoke(IPC.mcp.remove, brainId, serverId),
    refresh: (brainId, serverId) => invoke(IPC.mcp.refresh, brainId, serverId)
  },
  data: {
    selectBuildResources: (kind, selection) =>
      invoke(IPC.data.selectBuildResources, kind, selection),
    listBuildResources: () => invoke(IPC.data.listBuildResources),
    discardBuildResource: (selectionId) =>
      invoke(IPC.data.discardBuildResource, selectionId),
    startBuildResource: (request) => invoke(IPC.data.startBuildResource, request),
    preview: (request) => invoke(IPC.data.preview, request),
    cancelPreview: (requestId) => invoke(IPC.data.cancelPreview, requestId),
    onPreviewProgress: (listener) => {
      const wrapped = (
        _event: Electron.IpcRendererEvent,
        value: DatasetPreviewProgress
      ): void => listener(value);
      ipcRenderer.on(IPC.data.previewEvent, wrapped);
      return () => ipcRenderer.removeListener(IPC.data.previewEvent, wrapped);
    },
    start: (request) => invoke(IPC.data.start, request),
    pause: (jobId) => invoke(IPC.data.pause, jobId),
    resume: (request) => invoke(IPC.data.resume, request),
    resumable: (brainId) => invoke(IPC.data.resumable, brainId),
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
    cancel: (jobId) => invoke(IPC.data.cancel, jobId),
    sources: (brainId, cursor, limit) =>
      invoke(IPC.data.sources, brainId, cursor, limit)
  },
  modality: {
    capabilities: (brainId) => invoke(IPC.modality.capabilities, brainId),
    artifacts: (brainId, cursor, limit) =>
      invoke(IPC.modality.artifacts, brainId, cursor, limit),
    generate: (request) => invoke(IPC.modality.generate, request),
    generateSpeech: (request) => invoke(IPC.modality.generateSpeech, request),
    onGeneration: (listener) => {
      const handler = (_event: Electron.IpcRendererEvent, value: RuntimeJobEvent) => listener(value);
      ipcRenderer.on(IPC.train.event, handler);
      return () => ipcRenderer.removeListener(IPC.train.event, handler);
    },
    selectInput: (request) => invoke(IPC.modality.selectInput, request),
    cancel: (jobId) => invoke(IPC.modality.cancel, jobId),
    startObservation: (request) =>
      invoke(IPC.modality.startObservation, request),
    pushObservation: (packet) =>
      invoke(IPC.modality.pushObservation, packet),
    stopObservation: (sessionId) =>
      invoke(IPC.modality.stopObservation, sessionId),
    cancelObservation: (sessionId) =>
      invoke(IPC.modality.cancelObservation, sessionId),
    requestObservationControl: (request) =>
      invoke(IPC.modality.requestObservationControl, request),
    resolveObservationControl: (resolution) =>
      invoke(IPC.modality.resolveObservationControl, resolution),
    onObservation: (listener) => {
      const wrapped = (
        _event: Electron.IpcRendererEvent,
        value: Parameters<typeof listener>[0]
      ): void => listener(value);
      ipcRenderer.on(IPC.modality.observationEvent, wrapped);
      return () =>
        ipcRenderer.removeListener(IPC.modality.observationEvent, wrapped);
    }
  },
  trace: {
    list: (brainId, query) => invoke(IPC.trace.list, brainId, query)
  },
  tool: {
    listPermissions: (brainId) => invoke(IPC.tool.listPermissions, brainId),
    setPermission: (brainId, toolId, level) =>
      invoke(IPC.tool.setPermission, brainId, toolId, level),
    execute: (request) => invoke(IPC.tool.execute, request),
    cancel: (brainId, requestId) => invoke(IPC.tool.cancel, brainId, requestId),
    preferences: () => invoke(IPC.tool.preferences),
    setPreferences: (value) => invoke(IPC.tool.setPreferences, value)
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
    hardwareProfile: () => invoke(IPC.catalog.hardwareProfile),
    resourcePlan: (request) => invoke(IPC.catalog.resourcePlan, request)
  }
};

contextBridge.exposeInMainWorld("omni", api);
