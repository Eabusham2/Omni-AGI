export const IPC = {
  window: {
    minimize: "omni:window:minimize",
    maximize: "omni:window:maximize",
    close: "omni:window:close",
    isMaximized: "omni:window:is-maximized",
    openExternal: "omni:window:open-external",
    revealDataFolder: "omni:window:reveal-data-folder",
    platform: "omni:window:platform",
    setAppearance: "omni:window:set-appearance"
  },
  brain: {
    list: "omni:brain:list",
    get: "omni:brain:get",
    create: "omni:brain:create",
    completeInitialization: "omni:brain:complete-initialization",
    retryInitialization: "omni:brain:retry-initialization",
    update: "omni:brain:update",
    setOnlineLearning: "omni:brain:set-online-learning",
    setActiveMode: "omni:brain:set-active-mode",
    duplicate: "omni:brain:duplicate",
    fork: "omni:brain:fork",
    pauseStorageOperation: "omni:brain:pause-storage-operation",
    resumeStorageOperation: "omni:brain:resume-storage-operation",
    cancelStorageOperation: "omni:brain:cancel-storage-operation",
    storageOperationEvent: "omni:brain:storage-operation-event",
    remove: "omni:brain:remove",
    snapshot: "omni:brain:snapshot",
    listSnapshots: "omni:brain:list-snapshots",
    restoreSnapshot: "omni:brain:restore-snapshot",
    export: "omni:brain:export",
    importFile: "omni:brain:import-file",
    imported: "omni:brain:imported",
    importFailed: "omni:brain:import-failed",
    buildEvent: "omni:brain:build-event",
    health: "omni:brain:health",
    persistedSubstrateOverview: "omni:brain:persisted-substrate-overview",
    querySubstrate: "omni:brain:query-substrate",
    workspace: "omni:brain:workspace",
    freshAttention: "omni:brain:fresh-attention",
    journalPage: "omni:brain:journal-page"
  },
  chat: {
    send: "omni:chat:send",
    cancel: "omni:chat:cancel",
    recordDeliveryReceipt: "omni:chat:record-delivery-receipt",
    approveAction: "omni:chat:approve-action",
    list: "omni:chat:list",
    listPage: "omni:chat:list-page",
    feedback: "omni:chat:feedback",
    actionEvent: "omni:chat:action-event",
    streamEvent: "omni:chat:stream-event"
  },
  mobile: {
    status: "omni:mobile:status",
    startPairing: "omni:mobile:start-pairing",
    stop: "omni:mobile:stop",
    revoke: "omni:mobile:revoke"
  },
  train: {
    cancel: "omni:train:cancel",
    list: "omni:train:list",
    event: "omni:train:event"
  },
  teacher: {
    status: "omni:teacher:status",
    saveCredential: "omni:teacher:save-credential",
    removeCredential: "omni:teacher:remove-credential",
    start: "omni:teacher:start",
    cancel: "omni:teacher:cancel",
    list: "omni:teacher:list",
    event: "omni:teacher:event"
  },
  mcp: {
    list: "omni:mcp:list",
    add: "omni:mcp:add",
    remove: "omni:mcp:remove",
    refresh: "omni:mcp:refresh"
  },
  data: {
    selectBuildResources: "omni:data:select-build-resources",
    listBuildResources: "omni:data:list-build-resources",
    discardBuildResource: "omni:data:discard-build-resource",
    startBuildResource: "omni:data:start-build-resource",
    preview: "omni:data:preview",
    previewEvent: "omni:data:preview-event",
    cancelPreview: "omni:data:cancel-preview",
    start: "omni:data:start",
    pause: "omni:data:pause",
    resume: "omni:data:resume",
    resumable: "omni:data:resumable",
    coverage: "omni:data:coverage",
    ingestFiles: "omni:data:ingest-files",
    ingestFolder: "omni:data:ingest-folder",
    ingestDropped: "omni:data:ingest-dropped",
    ingestWeb: "omni:data:ingest-web",
    crawlWeb: "omni:data:crawl-web",
    cancel: "omni:data:cancel",
    sources: "omni:data:sources"
  },
  modality: {
    capabilities: "omni:modality:capabilities",
    artifacts: "omni:modality:artifacts",
    generate: "omni:modality:generate",
    selectInput: "omni:modality:select-input",
    cancel: "omni:modality:cancel",
    startObservation: "omni:modality:start-observation",
    pushObservation: "omni:modality:push-observation",
    stopObservation: "omni:modality:stop-observation",
    cancelObservation: "omni:modality:cancel-observation",
    requestObservationControl: "omni:modality:request-observation-control",
    resolveObservationControl: "omni:modality:resolve-observation-control",
    observationEvent: "omni:modality:observation-event"
  },
  trace: {
    list: "omni:trace:list"
  },
  tool: {
    listPermissions: "omni:tool:list-permissions",
    setPermission: "omni:tool:set-permission",
    execute: "omni:tool:execute",
    cancel: "omni:tool:cancel",
    preferences: "omni:tool:preferences",
    setPreferences: "omni:tool:set-preferences"
  },
  agent: {
    fork: "omni:agent:fork",
    previewMerge: "omni:agent:preview-merge",
    merge: "omni:agent:merge"
  },
  evolution: {
    start: "omni:evolution:start",
    stop: "omni:evolution:stop",
    listCandidates: "omni:evolution:list-candidates",
    approve: "omni:evolution:approve",
    rollback: "omni:evolution:rollback"
  },
  catalog: {
    list: "omni:catalog:list",
    importUrl: "omni:catalog:import-url",
    loadRecipeEntry: "omni:catalog:load-recipe-entry",
    loadRecipeUrl: "omni:catalog:load-recipe-url",
    loadRecipeFile: "omni:catalog:load-recipe-file",
    installModalityPackUrl: "omni:catalog:install-modality-pack-url",
    installModalityPackFile: "omni:catalog:install-modality-pack-file",
    listModalityPacks: "omni:catalog:list-modality-packs",
    hardwareProfile: "omni:catalog:hardware-profile",
    resourcePlan: "omni:catalog:resource-plan"
  }
} as const;
