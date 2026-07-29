export const IPC = {
  window: {
    minimize: "omni:window:minimize",
    maximize: "omni:window:maximize",
    close: "omni:window:close",
    isMaximized: "omni:window:is-maximized",
    openExternal: "omni:window:open-external",
    revealDataFolder: "omni:window:reveal-data-folder",
    platform: "omni:window:platform"
  },
  brain: {
    list: "omni:brain:list",
    get: "omni:brain:get",
    create: "omni:brain:create",
    update: "omni:brain:update",
    duplicate: "omni:brain:duplicate",
    fork: "omni:brain:fork",
    remove: "omni:brain:remove",
    snapshot: "omni:brain:snapshot",
    listSnapshots: "omni:brain:list-snapshots",
    restoreSnapshot: "omni:brain:restore-snapshot",
    export: "omni:brain:export",
    importFile: "omni:brain:import-file",
    imported: "omni:brain:imported",
    health: "omni:brain:health",
    querySubstrate: "omni:brain:query-substrate",
    workspace: "omni:brain:workspace"
  },
  chat: {
    send: "omni:chat:send",
    cancel: "omni:chat:cancel",
    list: "omni:chat:list",
    feedback: "omni:chat:feedback",
    actionEvent: "omni:chat:action-event",
    streamEvent: "omni:chat:stream-event"
  },
  train: {
    start: "omni:train:start",
    consolidate: "omni:train:consolidate",
    cancel: "omni:train:cancel",
    list: "omni:train:list",
    event: "omni:train:event"
  },
  data: {
    selectBuildResources: "omni:data:select-build-resources",
    discardBuildResource: "omni:data:discard-build-resource",
    startBuildResource: "omni:data:start-build-resource",
    preview: "omni:data:preview",
    start: "omni:data:start",
    pause: "omni:data:pause",
    resume: "omni:data:resume",
    coverage: "omni:data:coverage",
    ingestFiles: "omni:data:ingest-files",
    ingestFolder: "omni:data:ingest-folder",
    ingestDropped: "omni:data:ingest-dropped",
    ingestWeb: "omni:data:ingest-web",
    crawlWeb: "omni:data:crawl-web",
    cancel: "omni:data:cancel"
  },
  modality: {
    generate: "omni:modality:generate",
    selectInput: "omni:modality:select-input",
    cancel: "omni:modality:cancel"
  },
  trace: {
    list: "omni:trace:list"
  },
  tool: {
    listPermissions: "omni:tool:list-permissions",
    setPermission: "omni:tool:set-permission",
    execute: "omni:tool:execute",
    cancel: "omni:tool:cancel"
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
    hardwareProfile: "omni:catalog:hardware-profile"
  },
  legacy: {
    runtimeHealth: "omni:legacy:runtime-health"
  }
} as const;
