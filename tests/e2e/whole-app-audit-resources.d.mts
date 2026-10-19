export interface AuditResourceSnapshot {
  capturedAt: string; diskReserveBytes: number; availableDiskBytes?: number; platform: string;
  osMemoryPressureLevel?: number; swapUsageRaw?: string; canContinue: boolean; reason?: string;
}
export class AuditResourcePauseError extends Error { snapshot: AuditResourceSnapshot; constructor(snapshot: AuditResourceSnapshot); }
export function auditResourceSnapshot(repository: string): AuditResourceSnapshot;
export function assertAuditResources(repository: string, output: string, label: string): AuditResourceSnapshot;
