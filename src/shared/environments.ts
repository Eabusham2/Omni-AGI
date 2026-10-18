export const ENVIRONMENT_RECORD_FORMAT = "omni-managed-environment" as const;
export const ENVIRONMENT_OPERATION_FORMAT = "omni-environment-operation" as const;

export type ManagedEnvironmentKind = "sandbox" | "vm";
export type ManagedEnvironmentState =
  | "creating"
  | "ready"
  | "loading"
  | "loaded"
  | "busy"
  | "stopping"
  | "stopped"
  | "dormant"
  | "review-required"
  | "error";

export type EnvironmentOperationState =
  | "intent"
  | "running"
  | "complete"
  | "cancelled"
  | "failed"
  | "interrupted";

export interface ManagedEnvironmentRecord {
  format: typeof ENVIRONMENT_RECORD_FORMAT;
  version: 1;
  id: string;
  brainId: string;
  kind: ManagedEnvironmentKind;
  name: string;
  adapter: "darwin-sandbox-exec" | "linux-bwrap" | "windows-sandbox-cli" | "lima";
  state: ManagedEnvironmentState;
  revision: number;
  network: "none" | "guest-outbound";
  createdAt: string;
  updatedAt: string;
  /** Adapter-owned identity. It is data, never a command or permission token. */
  externalId?: string;
  imageSha256?: string;
  guestArchitecture?: "aarch64" | "x86_64";
  resourcePlan?: { cpus: number; memoryBytes: number; diskBytes: number };
  /** A prior sandbox child/guest command may still exist and must be disposed before reuse. */
  orphanedProcessRisk?: boolean;
  lastOperationId?: string;
  warning?: string;
  inheritedFrom?: { brainId: string; environmentId: string };
}

export interface EnvironmentProcessReceipt {
  exitCode: number;
  stdout: string;
  stderr: string;
  truncated: boolean;
  /** Windows Sandbox exec returns an exit code but exposes no guest process I/O. */
  guestOutput?: "guest-output-unavailable";
  /** Exact Windows Sandbox Stop outcome; never implies another session was touched. */
  sandboxDisposition?: "already-absent" | "stopped" | "unconfirmed";
}

export interface EnvironmentOperationReceipt {
  format: typeof ENVIRONMENT_OPERATION_FORMAT;
  version: 1;
  id: string;
  brainId: string;
  environmentId: string;
  requestId: string;
  action: "create" | "load" | "run" | "stop";
  state: EnvironmentOperationState;
  argumentsSha256: string;
  dispatchStarted: boolean;
  createdAt: string;
  updatedAt: string;
  process?: EnvironmentProcessReceipt;
  error?: string;
}

export interface EnvironmentAdapterCapability {
  id: ManagedEnvironmentRecord["adapter"];
  kind: ManagedEnvironmentKind;
  available: boolean;
  executable?: string;
  version?: string;
  supportsCreate: boolean;
  supportsRun: boolean;
  requiresLocalImage: boolean;
  deprecated?: boolean;
  warning?: string;
}

export interface EnvironmentPage {
  entries: ManagedEnvironmentRecord[];
  total: number;
  nextAfter?: string;
}

export interface EnvironmentToolResult {
  environment: ManagedEnvironmentRecord;
  operation: EnvironmentOperationReceipt;
}
