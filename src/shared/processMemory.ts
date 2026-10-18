/** Host process memory is deliberately separate from Omni's learned substrate. */
export const PROCESS_MEMORY_FORMAT = "omni-process-memory-1" as const;
export const PROCESS_MEMORY_MAX_BYTES = 1024 * 1024;
export const PROCESS_MEMORY_MAX_REGIONS = 200;

export type ProcessMemoryAccess = "read" | "read-write";
export type ProcessMemoryOperation = "attach" | "regions" | "read" | "write";
export interface ProcessMemoryIdentity {
  platform: "darwin" | "linux" | "win32";
  pid: number;
  /** OS process-creation identity, not a rounded JavaScript timestamp. */
  creation: string;
  executable: string;
  executableIdentity: string;
}
export interface ProcessMemoryCapabilities {
  platform: string;
  adapter: "mach-task-port" | "proc-mem-fd" | "windows-process-handle" | "unsupported";
  implemented: boolean;
  targetAccess: "not-probed";
  maxTransferBytes: number;
  maxRegionPage: number;
  environment: "host";
  physicalRamAccess: false;
  elevationPerformed: false;
  protectionChanges: false;
  detail: string;
}
export interface ProcessMemoryPreferences {
  brainId: string;
  enabled: boolean;
  revision: number;
  updatedAt: string | null;
  /** No process lease, approval or OS handle is persisted with the instance. */
  authority: "current-installation-only";
  capabilities: ProcessMemoryCapabilities;
}
export interface ProcessMemoryPreferenceUpdate {
  brainId: string;
  expectedRevision: number;
  enabled: boolean;
}
export interface ProcessMemoryTarget {
  id: string;
  brainId: string;
  identity: ProcessMemoryIdentity;
  access: ProcessMemoryAccess;
  preferenceRevision: number;
  attachedAt: string;
  lease: "current-host-process-only";
}
export interface ProcessMemoryRegion {
  start: string;
  end: string;
  readable: boolean;
  writable: boolean;
  executable: boolean;
}
export interface ProcessMemoryNativeRequest {
  format: typeof PROCESS_MEMORY_FORMAT;
  operationId: string;
  operation: ProcessMemoryOperation;
  pid: number;
  access: ProcessMemoryAccess;
  expectedIdentity?: ProcessMemoryIdentity;
  address?: string;
  length?: number;
  dataBase64?: string;
  expectedBeforeSha256?: string;
  afterAddress?: string;
  regionLimit?: number;
}
export type ProcessMemoryOutcome = "complete" | "partial" | "denied" | "protected" | "unsupported" |
  "identity-changed" | "preimage-mismatch" | "cancelled" | "failed" | "write-uncertain";
export interface ProcessMemoryReceipt {
  format: typeof PROCESS_MEMORY_FORMAT;
  operationId: string;
  operation: ProcessMemoryOperation;
  requestSha256: string;
  state: ProcessMemoryOutcome;
  identity?: ProcessMemoryIdentity;
  requestedBytes: number;
  transferredBytes: number | null;
  writeMayHaveOccurred: boolean;
  retrySafe: boolean;
  compareAtomic: false;
  beforeSha256?: string;
  afterSha256?: string;
  dataBase64?: string;
  dataSha256?: string;
  regions?: ProcessMemoryRegion[];
  nextAddress?: string;
  endOfRegions?: boolean;
  error?: string;
  osCode?: number;
}
export interface ProcessMemoryReady {
  format: typeof PROCESS_MEMORY_FORMAT;
  type: "ready";
  operationId: string;
  requestSha256: string;
  nonce: string;
  identity: ProcessMemoryIdentity;
}
export interface ProcessMemoryApi {
  preferences(brainId: string): Promise<ProcessMemoryPreferences>;
  update(request: ProcessMemoryPreferenceUpdate): Promise<ProcessMemoryPreferences>;
}
