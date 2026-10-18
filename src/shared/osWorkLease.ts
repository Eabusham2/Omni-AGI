export type OsWorkLeaseSource = "engine" | "runtime-job" | "storage-operation" | "build" | "host";
export type OsWorkCapabilityOutcome =
  | "not-attempted"
  | "active"
  | "partial"
  | "restored"
  | "unsupported"
  | "denied"
  | "error"
  | "restore-failed";

export interface OsWorkLeaseStatus {
  format: "omni-os-work-lease-status";
  version: 1;
  platform: string;
  active: boolean;
  activeLeaseCount: number;
  activeReferenceCount: number;
  activeBySource: Record<OsWorkLeaseSource, number>;
  sleepPrevention: {
    mode: "prevent-app-suspension";
    active: boolean;
    outcome: OsWorkCapabilityOutcome;
    detail: string;
  };
  processPriority: {
    requested: "above-normal";
    requestedValue?: number;
    activeTargets: number;
    deniedTargets: number;
    errorTargets: number;
    restoreFailedTargets: number;
    outcome: OsWorkCapabilityOutcome;
    detail: string;
  };
  limitations: string[];
  updatedAt: string;
}

export interface OsWorkLeaseEvent { status: OsWorkLeaseStatus }
