import { contextBridge, ipcRenderer, webUtils } from "electron";
import { randomUUID } from "node:crypto";
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
import type { UserQuestionEvent } from "../shared/userQuestions";

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
  workStatus: () => invoke(IPC.window.workStatus),
  setAppearance: (request) => invoke(IPC.window.setAppearance, request),
  touchBar: {
    update: state => invoke(IPC.window.touchBarUpdate, state),
    onAction: listener => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/touchBar").TouchBarActionEvent): void => listener(value);
      ipcRenderer.on(IPC.window.touchBarAction, wrapped);
      return () => { ipcRenderer.removeListener(IPC.window.touchBarAction, wrapped); };
    }
  }
};

const substrateApi: OmniApi["substrate"] = {
  jobs: (brainId, query) => invoke(IPC.substrate.jobs, brainId, query),
  job: (brainId, jobId) => invoke(IPC.substrate.job, brainId, jobId),
  plan: (brainId, planId) => invoke(IPC.substrate.plan, brainId, planId),
  stop: (brainId, jobId) => invoke(IPC.substrate.stop, brainId, jobId),
  onEvent: (brainId, listener) => {
    const subscriptionId = randomUUID();
    let ready = false, disposed = false;
    const buffered = new Map<string, Extract<import("../shared/sparseSubstrateTransfer").SparseSubstrateUiEvent, { kind: "job" }>>();
    const seen = new Map<string, number>();
    const deliver = (event: import("../shared/sparseSubstrateTransfer").SparseSubstrateUiEvent): void => {
      if (disposed) return;
      if (event.kind === "job") {
        const revision = seen.get(event.job.id) ?? 0;
        if (event.job.revision <= revision) return;
        seen.set(event.job.id, event.job.revision);
      }
      try { listener(event); } catch { /* A renderer observer cannot affect custody or subscription state. */ }
    };
    const wrapped = (_event: Electron.IpcRendererEvent, envelope: import("../shared/sparseSubstrateTransfer").SparseSubstrateUiEventEnvelope): void => {
      if (disposed || envelope.subscriptionId !== subscriptionId || envelope.brainId !== brainId || envelope.event.brainId !== brainId) return;
      if (!ready) {
        const prior = buffered.get(envelope.event.job.id);
        if (!prior || envelope.event.job.revision > prior.job.revision) buffered.set(envelope.event.job.id, envelope.event);
      } else deliver(envelope.event);
    };
    // Listener first: main may publish immediately after accepting subscribe.
    ipcRenderer.on(IPC.substrate.event, wrapped);
    void invoke<import("../shared/sparseSubstrateTransfer").SparseSubstrateUiSubscriptionAck>(IPC.substrate.subscribe, brainId, subscriptionId).then(ack => {
      if (ack.brainId !== brainId || ack.subscriptionId !== subscriptionId || ack.subscribed !== true) throw new Error("Sparse subscription acknowledgement is foreign or malformed.");
      if (disposed) { void invoke(IPC.substrate.unsubscribe, brainId, subscriptionId).catch(error => console.error("Late sparse unsubscribe failed:", error)); return; }
      if (!ack.initial || !Array.isArray(ack.initial.entries) || !Number.isSafeInteger(ack.initial.total) || ack.initial.total < ack.initial.entries.length ||
          ack.initial.entries.some(job => job.brainId !== brainId || !Number.isSafeInteger(job.revision) || job.revision < 1) ||
          !Number.isFinite(Date.parse(ack.observedAt))) {
        throw new Error("Sparse subscription initial owner snapshot is malformed.");
      }
      ready = true;
      for (const job of ack.initial.entries) deliver({ format: "omni-sparse-substrate-ui-event", version: 1,
        brainId, kind: "job", job, observedAt: ack.observedAt });
      for (const event of buffered.values()) deliver(event);
      buffered.clear();
    }).catch(error => {
      if (disposed) return;
      ready = true; deliver({ format: "omni-sparse-substrate-ui-event", version: 1, brainId, kind: "error",
        error: error instanceof Error ? error.message : String(error), observedAt: new Date().toISOString() });
      disposed = true; ipcRenderer.removeListener(IPC.substrate.event, wrapped);
    });
    return () => {
      if (disposed) return; disposed = true; buffered.clear(); ipcRenderer.removeListener(IPC.substrate.event, wrapped);
      void invoke(IPC.substrate.unsubscribe, brainId, subscriptionId).catch(error => console.error("Sparse unsubscribe failed:", error));
    };
  }
};

const api: OmniApi = {
  nativeOperationAuto: {
    prepare: request => invoke(IPC.nativeOperationAuto.prepare, request),
    arm: request => invoke(IPC.nativeOperationAuto.arm, request),
    status: brainId => invoke(IPC.nativeOperationAuto.status, brainId),
    plans: request => invoke(IPC.nativeOperationAuto.plans, request),
    stop: request => invoke(IPC.nativeOperationAuto.stop, request)
  },
  wholeBrainInspection: { query: (brainId, query) => invoke(IPC.wholeBrainInspection.query, brainId, query) },
  cachedDecoderExecution: {
    prepare: request => invoke(IPC.cachedDecoderExecution.prepare, request),
    arm: request => invoke(IPC.cachedDecoderExecution.arm, request),
    status: brainId => invoke(IPC.cachedDecoderExecution.status, brainId),
    stop: request => invoke(IPC.cachedDecoderExecution.stop, request)
  },
  decoderExecution: {
    captureNext: request => invoke(IPC.decoderExecution.captureNext, request),
    prepare: request => invoke(IPC.decoderExecution.prepare, request),
    calibrate: request => invoke(IPC.decoderExecution.calibrate, request),
    apply: request => invoke(IPC.decoderExecution.apply, request),
    status: brainId => invoke(IPC.decoderExecution.status, brainId),
    stop: request => invoke(IPC.decoderExecution.stop, request)
  },
  regionalExecution: {
    prepare: request => invoke(IPC.regionalExecution.prepare, request), status: brainId => invoke(IPC.regionalExecution.status, brainId),
    activate: request => invoke(IPC.regionalExecution.activate, request), deactivate: request => invoke(IPC.regionalExecution.deactivate, request)
  },
  portable: {
    create: request => invoke(IPC.portable.create, request),
    prepare: request => invoke(IPC.portable.prepare, request),
    status: () => invoke(IPC.portable.status),
    reviewDestination: () => invoke(IPC.portable.reviewDestination),
    approveDestination: request => invoke(IPC.portable.approveDestination, request)
  },
  advice: {
    settings: id => invoke(IPC.advice.settings, id), configure: (id, input) => invoke(IPC.advice.configure, id, input),
    preview: (id, input) => invoke(IPC.advice.preview, id, input), start: (id, input) => invoke(IPC.advice.start, id, input),
    cancel: (id, requestId) => invoke(IPC.advice.cancel, id, requestId), history: (id, query) => invoke(IPC.advice.history, id, query),
    discoverModels: (id, input) => invoke(IPC.advice.discoverModels, id, input),
    cancelModelDiscovery: (id, input) => invoke(IPC.advice.cancelModelDiscovery, id, input),
    credentialStatus: () => invoke(IPC.advice.credentialStatus), saveCredential: input => invoke(IPC.advice.saveCredential, input),
    removeCredential: provider => invoke(IPC.advice.removeCredential, provider),
    onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/externalAdvice").ExternalAdviceEvent): void => listener(value);
      ipcRenderer.on(IPC.advice.event, wrapped); return () => { ipcRenderer.removeListener(IPC.advice.event, wrapped); }; }
  },
  localTeacher: {
    select: request => invoke(IPC.localTeacher.select, request), cancelSelection: (id, requestId) => invoke(IPC.localTeacher.cancelSelection, id, requestId),
    selectionProgress: (id, requestId) => invoke(IPC.localTeacher.selectionProgress, id, requestId), selections: id => invoke(IPC.localTeacher.selections, id),
    plan: request => invoke(IPC.localTeacher.plan, request), start: request => invoke(IPC.localTeacher.start, request), jobs: id => invoke(IPC.localTeacher.jobs, id),
    get: (id, jobId) => invoke(IPC.localTeacher.get, id, jobId), items: (id, jobId, after, limit) => invoke(IPC.localTeacher.items, id, jobId, after, limit),
    pause: (id, jobId) => invoke(IPC.localTeacher.pause, id, jobId), resume: (id, jobId) => invoke(IPC.localTeacher.resume, id, jobId), stop: (id, jobId) => invoke(IPC.localTeacher.stop, id, jobId),
    onSelectionProgress: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/localTeacher").LocalTeacherSelectionProgress): void => listener(value);
      ipcRenderer.on(IPC.localTeacher.selectionEvent, wrapped); return () => { ipcRenderer.removeListener(IPC.localTeacher.selectionEvent, wrapped); }; },
    onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/localTeacher").LocalTeacherEvent): void => listener(value);
      ipcRenderer.on(IPC.localTeacher.event, wrapped); return () => { ipcRenderer.removeListener(IPC.localTeacher.event, wrapped); }; }
  },
  processMemory: {
    preferences: brainId => invoke(IPC.processMemory.preferences, brainId),
    update: request => invoke(IPC.processMemory.update, request)
  },
  substrate: substrateApi,
  voice: {
    preferences: brainId => invoke(IPC.voice.preferences, brainId),
    update: request => invoke(IPC.voice.update, request),
    migrateLegacy: request => invoke(IPC.voice.migrateLegacy, request)
  },
  automation: {
    snapshot: id => invoke(IPC.automation.snapshot, id), definitions: (id, query) => invoke(IPC.automation.definitions, id, query),
    program: (id, definitionId, revision) => invoke(IPC.automation.program, id, definitionId, revision), runs: (id, query) => invoke(IPC.automation.runs, id, query),
    steps: (id, runId, query) => invoke(IPC.automation.steps, id, runId, query),
    currentStep: (id, runId) => invoke(IPC.automation.currentStep, id, runId),
    onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/automation").AutomationWorkspaceEvent): void => listener(value);
      ipcRenderer.on(IPC.automation.event, wrapped); return () => { ipcRenderer.removeListener(IPC.automation.event, wrapped); }; }
  },
  clock: {
    now: timeZone => invoke(IPC.clock.now, timeZone),
    snapshot: id => invoke(IPC.clock.snapshot, id), schedules: (id, query) => invoke(IPC.clock.schedules, id, query),
    stopwatches: (id, query) => invoke(IPC.clock.stopwatches, id, query), occurrences: (id, query) => invoke(IPC.clock.occurrences, id, query),
    laps: (id, stopwatchId, after) => invoke(IPC.clock.laps, id, stopwatchId, after), rates: (id, query) => invoke(IPC.clock.rates, id, query),
    onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/clock").ClockWorkspaceEvent): void => listener(value);
      ipcRenderer.on(IPC.clock.event, wrapped); return () => { ipcRenderer.removeListener(IPC.clock.event, wrapped); }; }
  },
  window: windowApi,
  brain: {
    lifecycle: (id) => invoke(IPC.brain.lifecycle, id),
    loadFit: (id) => invoke(IPC.brain.loadFit, id),
    load: (request) => invoke(IPC.brain.load, request),
    unload: (request) => invoke(IPC.brain.unload, request),
    setAutoLoad: (request) => invoke(IPC.brain.setAutoLoad, request),
    resumePresentation: (id) => invoke(IPC.brain.resumePresentation, id),
    resumeTurn: (request) => invoke(IPC.brain.resumeTurn, request),
    stopSavedTurn: (request) => invoke(IPC.brain.stopSavedTurn, request),
    inheritedResumePresentation: (id, executionId) => invoke(IPC.brain.inheritedResumePresentation, id, executionId),
    resumeInheritedTurn: (request) => invoke(IPC.brain.resumeInheritedTurn, request),
    prepareInheritedTaskFiles: request => invoke(IPC.brain.prepareInheritedTaskFiles, request),
    approveInheritedTaskFiles: request => invoke(IPC.brain.approveInheritedTaskFiles, request),
    revokeInheritedTaskFiles: request => invoke(IPC.brain.revokeInheritedTaskFiles, request),
    onLifecycle: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, state: import("../shared/instanceLifecycle").InstanceLifecycleState): void => listener(state);
      ipcRenderer.on(IPC.brain.lifecycleEvent, wrapped);
      return () => { ipcRenderer.removeListener(IPC.brain.lifecycleEvent, wrapped); };
    },
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
    startFullMerge: (request) => invoke(IPC.brain.startFullMerge, request),
    listFullMerges: () => invoke(IPC.brain.listFullMerges),
    fullMergeStatus: (id) => invoke(IPC.brain.fullMergeStatus, id),
    pauseFullMerge: (id) => invoke(IPC.brain.pauseFullMerge, id),
    resumeFullMerge: (id) => invoke(IPC.brain.resumeFullMerge, id),
    cancelFullMerge: (id) => invoke(IPC.brain.cancelFullMerge, id),
    fullMergeHistory: (id, after, limit) => invoke(IPC.brain.fullMergeHistory, id, after, limit),
    fullMergePendingIngestions: (id) => invoke(IPC.brain.fullMergePendingIngestions, id),
    resumeFullMergeIngestion: (request) => invoke(IPC.brain.resumeFullMergeIngestion, request),
    onFullMerge: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/fullMerge").FullMergeProgress): void => listener(value);
      ipcRenderer.on(IPC.brain.fullMergeEvent, wrapped);
      return () => { ipcRenderer.removeListener(IPC.brain.fullMergeEvent, wrapped); };
    },
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
    routerActivity: (id, query) => invoke(IPC.brain.routerActivity, id, query),
    workspace: (id) => invoke(IPC.brain.workspace, id),
    freshAttention: (id) => invoke(IPC.brain.freshAttention, id),
    journalPage: (id, cursor, limit) =>
      invoke(IPC.brain.journalPage, id, cursor, limit)
  },
  chat: {
    directions: {
      list: (brainId, query) => invoke(IPC.chat.directionsList, brainId, query),
      get: (brainId, id) => invoke(IPC.chat.directionsGet, brainId, id),
      forTurns: (brainId, turnIds) => invoke(IPC.chat.directionsForTurns, brainId, turnIds),
      dismiss: request => invoke(IPC.chat.directionsDismiss, request),
      onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/steerDirections").SteerDirectionEvent): void => listener(value);
        ipcRenderer.on(IPC.chat.directionsEvent, wrapped); return () => { ipcRenderer.removeListener(IPC.chat.directionsEvent, wrapped); }; }
    },
    sendAudio: (id, input, turnId, turnMetadata) => invoke(IPC.chat.sendAudio, id, input, turnId, turnMetadata),
    audioInputCapabilities: (id) => invoke(IPC.chat.audioInputCapabilities, id),
    send: (id, input, turnId, turnMetadata) =>
      invoke(IPC.chat.send, id, input, turnId, turnMetadata),
    attachments: {
      stage: request => invoke(IPC.chat.stageAttachments, request),
      stageDropped: (request, files) => {
        const paths = Array.from(files, file => webUtils.getPathForFile(file as File)).filter(Boolean);
        return invoke(IPC.chat.stageDroppedAttachments, request, paths);
      },
      cancelStage: (brainId, requestId) =>
        invoke(IPC.chat.cancelAttachmentStage, brainId, requestId),
      list: (brainId, state) => state === undefined
        ? invoke(IPC.chat.listAttachments, brainId)
        : invoke(IPC.chat.listAttachments, brainId, state),
      forTurn: (brainId, turnId) => invoke(IPC.chat.attachmentsForTurn, brainId, turnId),
      byIds: (brainId, attachmentIds) => invoke(IPC.chat.attachmentsByIds, brainId, attachmentIds),
      remove: (brainId, attachmentId) =>
        invoke(IPC.chat.removeAttachment, brainId, attachmentId)
    },
    cancel: (id, turnId) => invoke(IPC.chat.cancel, id, turnId),
    cancelInlineAction: (id, turnId, actionEventId) =>
      invoke(IPC.chat.cancelInlineAction, id, turnId, actionEventId),
    recordDeliveryReceipt: (id, receipt) =>
      invoke(IPC.chat.recordDeliveryReceipt, id, receipt),
    approveAction: (request) => invoke(IPC.chat.approveAction, request),
    declineAction: (request) => invoke(IPC.chat.declineAction, request),
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
  questions: {
    request: request => invoke(IPC.questions.request, request),
    list: (id, query) => invoke(IPC.questions.list, id, query),
    respond: request => invoke(IPC.questions.respond, request),
    cancel: (id, questionId, reason) => invoke(IPC.questions.cancel, id, questionId, reason),
    resume: (id, questionId) => invoke(IPC.questions.resume, id, questionId),
    preferences: id => invoke(IPC.questions.preferences, id),
    setPreferences: request => invoke(IPC.questions.setPreferences, request),
    onEvent: listener => {
      const wrapped = (_event: Electron.IpcRendererEvent, event: UserQuestionEvent) => listener(event);
      ipcRenderer.on(IPC.questions.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.questions.event, wrapped);
    }
  },
  learning: {
    start: request => invoke(IPC.learning.start, request),
    startFactCheck: request => invoke(IPC.learning.startFactCheck, request),
    list: id => invoke(IPC.learning.list, id), get: (id, taskId) => invoke(IPC.learning.get, id, taskId),
    factCheck: (id, taskId) => invoke(IPC.learning.factCheck, id, taskId),
    pause: (id, taskId) => invoke(IPC.learning.pause, id, taskId), resume: (id, taskId) => invoke(IPC.learning.resume, id, taskId),
    stop: (id, taskId) => invoke(IPC.learning.stop, id, taskId), steer: request => invoke(IPC.learning.steer, request),
    review: request => invoke(IPC.learning.review, request),
    acceptCorrection: (id, taskId, claimId, findingId) => invoke(IPC.learning.acceptCorrection, id, taskId, claimId, findingId),
    onEvent: listener => {
      const wrapped = (_event: Electron.IpcRendererEvent, event: import("../shared/learning").LearningTaskEvent) => listener(event);
      ipcRenderer.on(IPC.learning.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.learning.event, wrapped);
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
    discoverModels: (request) => invoke(IPC.teacher.discoverModels, request),
    cancelModelDiscovery: (request) => invoke(IPC.teacher.cancelModelDiscovery, request),
    start: (request) => invoke(IPC.teacher.start, request),
    cancel: (jobId) => invoke(IPC.teacher.cancel, jobId),
    pause: request => invoke(IPC.teacher.pause, request),
    resume: request => invoke(IPC.teacher.resume, request),
    stop: request => invoke(IPC.teacher.stop, request),
    details: request => invoke(IPC.teacher.details, request),
    resolveUncertain: request => invoke(IPC.teacher.resolveUncertain, request),
    list: (brainId) => invoke(IPC.teacher.list, brainId),
    onEvent: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: RuntimeJobEvent): void =>
        listener(value);
      ipcRenderer.on(IPC.teacher.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.teacher.event, wrapped);
    }
  },
  maps: {
    basemaps: {
      list: () => invoke(IPC.maps.basemapList),
      save: value => invoke(IPC.maps.basemapSave, value),
      remove: (id, expectedRevision) => invoke(IPC.maps.basemapRemove, id, expectedRevision)
    },
    list: () => invoke(IPC.maps.list),
    save: value => invoke(IPC.maps.save, value),
    remove: (id, expectedRevision) => invoke(IPC.maps.remove, id, expectedRevision)
  },
  mcp: {
    tasks: {
      localPage: (brainId, after, limit) => invoke(IPC.mcp.tasksLocalPage, brainId, after, limit),
      stopWait: (brainId, taskRef) => invoke(IPC.mcp.tasksStopWait, brainId, taskRef)
    },
    roots: { status: (brainId, serverId) => invoke(IPC.mcp.rootsStatus, brainId, serverId),
      select: request => invoke(IPC.mcp.rootsSelect, request), review: request => invoke(IPC.mcp.rootsReview, request), clear: request => invoke(IPC.mcp.rootsClear, request) },
    diagnostics: { page: (brainId, serverId, afterSequence, limit) => invoke(IPC.mcp.diagnosticsPage, brainId, serverId, afterSequence, limit) },
    list: (brainId) => invoke(IPC.mcp.list, brainId),
    registry: (brainId) => invoke(IPC.mcp.registry, brainId),
    configure: (request) => invoke(IPC.mcp.configure, request),
    add: (request) => invoke(IPC.mcp.add, request),
    remove: (brainId, serverId) => invoke(IPC.mcp.remove, brainId, serverId),
    refresh: (brainId, serverId) => invoke(IPC.mcp.refresh, brainId, serverId),
    authStatus: (brainId, serverId) => invoke(IPC.mcp.authStatus, brainId, serverId),
    authorize: (brainId, serverId, input) => invoke(IPC.mcp.authorize, brainId, serverId, input),
    cancelAuthorization: (brainId, serverId) => invoke(IPC.mcp.cancelAuthorization, brainId, serverId),
    disconnectAuthorization: (brainId, serverId) => invoke(IPC.mcp.disconnectAuthorization, brainId, serverId),
    updateCredentials: (brainId, serverId, input) => invoke(IPC.mcp.updateCredentials, brainId, serverId, input),
    onCapabilityEvent: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, event: import("../shared/mcpCapabilities").McpCapabilityEvent): void => listener(event);
      ipcRenderer.on(IPC.mcp.capabilityEvent, wrapped);
      return () => { ipcRenderer.removeListener(IPC.mcp.capabilityEvent, wrapped); };
    },
    interactions: (brainId) => invoke(IPC.mcp.interactions, brainId),
    respondInteraction: (response) => invoke(IPC.mcp.respondInteraction, response),
    onInteractionEvent: (listener) => {
      const wrapped = (_event: Electron.IpcRendererEvent, event: import("../shared/mcpInteractions").McpInteractionEvent): void => listener(event);
      ipcRenderer.on(IPC.mcp.interactionEvent, wrapped);
      return () => { ipcRenderer.removeListener(IPC.mcp.interactionEvent, wrapped); };
    },
    apps: {
      describe: (brainId, viewId) => invoke(IPC.mcp.appDescribe, brainId, viewId),
      open: (brainId, viewId, selection) => invoke(IPC.mcp.appOpen, brainId, viewId, selection),
      request: request => invoke(IPC.mcp.appRequest, request),
      close: (brainId, viewId, viewToken) => invoke(IPC.mcp.appClose, brainId, viewId, viewToken),
      onObservation: listener => {
        const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/mcpApps").McpAppObservationEvent): void => listener(value);
        ipcRenderer.on(IPC.mcp.appObservation, wrapped);
        return () => ipcRenderer.removeListener(IPC.mcp.appObservation, wrapped);
      },
      onServerNotification: listener => {
        const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/mcpHostServices").McpAppServerNotification): void => listener(value);
        ipcRenderer.on(IPC.mcp.appServerNotification, wrapped);
        return () => ipcRenderer.removeListener(IPC.mcp.appServerNotification, wrapped);
      },
      onDisplayMode: listener => {
        const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/mcpApps").McpAppDisplayEvent): void => listener(value);
        ipcRenderer.on(IPC.mcp.appDisplayMode, wrapped);
        return () => ipcRenderer.removeListener(IPC.mcp.appDisplayMode, wrapped);
      },
      acknowledgeDisplayMode: request => invoke(IPC.mcp.appDisplayModeAcknowledge, request)
    }
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
    capture: {
      assembly: {
        plan: request => invoke(IPC.modality.assemblyPlan, request),
        cancelPlan: request => invoke(IPC.modality.assemblyPlanCancel, request),
        onPlanEvent: listener => { const wrapped=(_event:Electron.IpcRendererEvent,value:import("../shared/imaginationAssembly").ImaginationPlanEvent)=>listener(value);ipcRenderer.on(IPC.modality.assemblyPlanEvent,wrapped);return()=>{ipcRenderer.removeListener(IPC.modality.assemblyPlanEvent,wrapped);}; },
        assemble: request => invoke(IPC.modality.assemblyCreate, request),
        get: (id, assemblyId) => invoke(IPC.modality.assemblyGet, id, assemblyId),
        list: (id, query) => invoke(IPC.modality.assemblyList, id, query),
        manifest: (id, assemblyId) => invoke(IPC.modality.assemblyManifest, id, assemblyId),
        manifestPage: (id, assemblyId, query) => invoke(IPC.modality.assemblyManifestPage, id, assemblyId, query),
        cancel: request => invoke(IPC.modality.assemblyCancel, request),
        resume: request => invoke(IPC.modality.assemblyResume, request),
        onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/imaginationAssembly").ImaginationAssemblyEvent) => listener(value);
          ipcRenderer.on(IPC.modality.assemblyEvent, wrapped); return () => { ipcRenderer.removeListener(IPC.modality.assemblyEvent, wrapped); }; }
      },
      status: id => invoke(IPC.modality.captureStatus, id),
      settings: request => invoke(IPC.modality.captureSettings, request),
      record: request => invoke(IPC.modality.captureRecord, request),
      stop: request => invoke(IPC.modality.captureStop, request),
      cancel: request => invoke(IPC.modality.captureCancel, request),
      resume: request => invoke(IPC.modality.captureResume, request),
      recordings: (id, query) => invoke(IPC.modality.captureRecordings, id, query),
      fragments: (id, recordingId, query) => invoke(IPC.modality.captureFragments, id, recordingId, query),
      onEvent: listener => { const wrapped = (_event: Electron.IpcRendererEvent, value: import("../shared/imaginationCapture").ImaginationCaptureEvent) => listener(value);
        ipcRenderer.on(IPC.modality.captureEvent, wrapped); return () => { ipcRenderer.removeListener(IPC.modality.captureEvent, wrapped); }; }
    },
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
    descriptions: (brainId) => invoke(IPC.tool.descriptions, brainId),
    listPermissions: (brainId) => invoke(IPC.tool.listPermissions, brainId),
    setPermission: (brainId, toolId, level, requestId, expectedRevision) =>
      invoke(IPC.tool.setPermission, brainId, toolId, level, requestId, expectedRevision),
    execute: (request) => invoke(IPC.tool.execute, request),
    declineApproval: (id, token) => invoke(IPC.tool.declineApproval, id, token),
    cancel: (brainId, requestId) => invoke(IPC.tool.cancel, brainId, requestId),
    preferences: () => invoke(IPC.tool.preferences),
    setPreferences: (value) => invoke(IPC.tool.setPreferences, value)
  },
  plugins: {
    list: () => invoke(IPC.plugins.list),
    package: (pluginId, contentSha256) =>
      invoke(IPC.plugins.package, pluginId, contentSha256),
    installFile: () => invoke(IPC.plugins.installFile),
    installPackage: (value, sourceLabel) =>
      invoke(IPC.plugins.installPackage, value, sourceLabel),
    edit: (pluginId, baseContentSha256, value) =>
      invoke(IPC.plugins.edit, pluginId, baseContentSha256, value),
    activate: (pluginId, contentSha256) =>
      invoke(IPC.plugins.activate, pluginId, contentSha256),
    setEnabled: (pluginId, enabled, expectedContentSha256, expectedRevision) =>
      invoke(IPC.plugins.setEnabled, pluginId, enabled, expectedContentSha256, expectedRevision),
    exportFile: (pluginId, contentSha256) =>
      invoke(IPC.plugins.exportFile, pluginId, contentSha256),
    instance: (brainId, pluginId) =>
      invoke(IPC.plugins.instance, brainId, pluginId),
    setInstanceAccess: (request) =>
      invoke(IPC.plugins.setInstanceAccess, request),
    portable: (brainId) => invoke(IPC.plugins.portable, brainId),
    openPanel: (brainId, pluginId, contentSha256, panelId) =>
      invoke(IPC.plugins.openPanel, brainId, pluginId, contentSha256, panelId),
    panelRequest: request => invoke(IPC.plugins.panelRequest, request),
    reviewPanelRequest: request => invoke(IPC.plugins.reviewPanelRequest, request),
    closePanel: (brainId, viewId, channelNonce) =>
      invoke(IPC.plugins.closePanel, brainId, viewId, channelNonce),
    panelHistory: (brainId, page) => invoke(IPC.plugins.panelHistory, brainId, page),
    panelHistoryBytes: (brainId, sequence, offset, length, sourceSha256) =>
      invoke(IPC.plugins.panelHistoryBytes, brainId, sequence, offset, length, sourceSha256),
    panelHistorySources: (brainId, page) => invoke(IPC.plugins.panelHistorySources, brainId, page),
    onEvent: (listener) => {
      const wrapped = (
        _event: Electron.IpcRendererEvent,
        value: Parameters<typeof listener>[0]
      ): void => listener(value);
      ipcRenderer.on(IPC.plugins.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.plugins.event, wrapped);
    }
  },
  pluginRuntime: {
    activate: request => invoke(IPC.pluginRuntime.activate, request),
    stop: (brainId, sessionId) => invoke(IPC.pluginRuntime.stop, brainId, sessionId),
    list: (brainId, page) => invoke(IPC.pluginRuntime.list, brainId, page),
    history: (brainId, page) => invoke(IPC.pluginRuntime.history, brainId, page),
    readHistoryEvent: (brainId, sequence, offset, length, sourceSha256) =>
      invoke(IPC.pluginRuntime.readHistoryEvent, brainId, sequence, offset, length, sourceSha256),
    stateSources: (brainId, page) => invoke(IPC.pluginRuntime.stateSources, brainId, page),
    stateConflicts: (brainId, page) => invoke(IPC.pluginRuntime.stateConflicts, brainId, page),
    readSourceState: (brainId, sourceSha256, pluginHash, moduleId, key, maxBytes) =>
      invoke(IPC.pluginRuntime.readSourceState, brainId, sourceSha256, pluginHash, moduleId, key, maxBytes),
    readSourceStateBytes: (brainId, sourceSha256, pluginHash, moduleId, key, offset, length) =>
      invoke(IPC.pluginRuntime.readSourceStateBytes, brainId, sourceSha256, pluginHash, moduleId, key, offset, length),
    selectSourceState: request => invoke(IPC.pluginRuntime.selectSourceState, request),
    contributions: brainId => invoke(IPC.pluginRuntime.contributions, brainId),
    invokeContribution: (brainId, sessionId, contributionId, eventId, data) =>
      invoke(IPC.pluginRuntime.invokeContribution, brainId, sessionId, contributionId, eventId, data),
    approvals: brainId => invoke(IPC.pluginRuntime.approvals, brainId),
    review: request => invoke(IPC.pluginRuntime.review, request),
    onEvent: listener => {
      const wrapped = (_event: Electron.IpcRendererEvent, value: Parameters<typeof listener>[0]): void => listener(value);
      ipcRenderer.on(IPC.pluginRuntime.event, wrapped);
      return () => ipcRenderer.removeListener(IPC.pluginRuntime.event, wrapped);
    }
  },
  artifacts: {
    open: request => invoke(IPC.artifacts.open, request),
    activate: request => invoke(IPC.artifacts.activate, request),
    request: request => invoke(IPC.artifacts.request, request),
    review: request => invoke(IPC.artifacts.review, request),
    close: (brainId, viewId, channelNonce) => invoke(IPC.artifacts.close, brainId, viewId, channelNonce)
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
    deviceTopology: (request) => invoke(IPC.catalog.deviceTopology, request),
    selectSpillPools: (brainId) => invoke(IPC.catalog.selectSpillPools, brainId),
    resourcePlan: (request) => invoke(IPC.catalog.resourcePlan, request)
  }
};

// Electron preloads can be run in subframes on some configurations. Neither
// the trusted sandbox proxy nor its untrusted child gets the Studio bridge.
if (process.isMainFrame !== false) contextBridge.exposeInMainWorld("omni", api);
