export interface AppearanceAuditSnapshot {
  rootPresent: boolean;
  rootConnected: boolean;
  rootVisible: boolean;
  popoverOpen: boolean;
  expanded: string | null;
  rootBounds: { left: number; top: number; width: number; height: number } | null;
  activeElement: Record<string, unknown>;
  visibleModals: Array<Record<string, unknown>>;
}
export interface AppearanceAuditObservations {
  capacity: number;
  sequence: number;
  dropped: number;
  timeOriginMs: number;
  events: Array<{ sequence: number; atMs: number; kind: string; [key: string]: unknown }>;
  current: AppearanceAuditSnapshot | null;
}
export function createAppearanceAuditObservation(options?: { capacity?: number; install?: boolean }): {
  record(kind: string, detail?: Record<string, unknown>): void;
  read(): AppearanceAuditObservations;
  snapshot(): AppearanceAuditSnapshot | null;
  disconnect(): void;
};
export function appearanceAuditFailureMessage(snapshot: AppearanceAuditSnapshot | null | undefined, phase: string): string | null;
