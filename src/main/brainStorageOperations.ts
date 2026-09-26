import { EventEmitter } from "node:events";
import type {
  BrainStorageOperationEvent,
  BrainStorageOperationKind,
  DiskSpaceTelemetry
} from "../shared/types";
import { DiskReservePauseError, requireDiskWrite } from "./diskSpace";

export interface BrainStorageProgressUpdate {
  phase?: string;
  label?: string;
  targetBrainId?: string;
  filesCompleted?: number;
  filesTotal?: number;
  logicalBytesCompleted?: number;
  logicalBytesTotal?: number;
  physicalBytesAdded?: number;
  sharedBytes?: number;
  diskSpace?: DiskSpaceTelemetry;
}

export interface BrainStorageOperationHooks {
  readonly signal: AbortSignal;
  checkpoint(update?: BrainStorageProgressUpdate): Promise<void>;
  checkDisk(path: string, operationWriteBytes: number): Promise<DiskSpaceTelemetry>;
}

function abortError(message = "Storage operation cancelled."): Error {
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

function safeCount(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isSafeInteger(value) && value >= 0
    ? value
    : fallback;
}

export class BrainStorageOperationSession extends EventEmitter {
  readonly controller = new AbortController();
  private readonly startedAtMs = Date.now();
  private pauseRequested = false;
  private resumeWaiters = new Set<() => void>();
  private terminal = false;
  private snapshot: BrainStorageOperationEvent;

  constructor(
    operationId: string,
    kind: BrainStorageOperationKind,
    sourceBrainId?: string,
    private readonly diskCheck: (
      path: string,
      operationWriteBytes: number
    ) => Promise<DiskSpaceTelemetry> = (path, operationWriteBytes) =>
      requireDiskWrite(path, { operationWriteBytes })
  ) {
    super();
    const updatedAt = new Date().toISOString();
    this.snapshot = {
      schemaVersion: 1,
      operationId,
      kind,
      state: "queued",
      phase: "planning",
      label: `Planning ${kind}`,
      ...(sourceBrainId ? { sourceBrainId } : {}),
      filesCompleted: 0,
      filesTotal: 0,
      logicalBytesCompleted: 0,
      logicalBytesTotal: 0,
      physicalBytesAdded: 0,
      sharedBytes: 0,
      bytesPerSecond: 0,
      elapsedMs: 0,
      updatedAt
    };
  }

  current(): BrainStorageOperationEvent {
    return structuredClone(this.snapshot);
  }

  private publish(
    state: BrainStorageOperationEvent["state"],
    update: BrainStorageProgressUpdate = {},
    error?: string
  ): BrainStorageOperationEvent {
    const elapsedMs = Math.max(0, Date.now() - this.startedAtMs);
    const filesCompleted = safeCount(
      update.filesCompleted,
      this.snapshot.filesCompleted
    );
    const filesTotal = safeCount(update.filesTotal, this.snapshot.filesTotal);
    const logicalBytesCompleted = safeCount(
      update.logicalBytesCompleted,
      this.snapshot.logicalBytesCompleted
    );
    const logicalBytesTotal = safeCount(
      update.logicalBytesTotal,
      this.snapshot.logicalBytesTotal
    );
    const bytesPerSecond = elapsedMs > 0
      ? Math.round((logicalBytesCompleted * 1_000) / elapsedMs)
      : 0;
    const remaining = Math.max(0, logicalBytesTotal - logicalBytesCompleted);
    const etaMs = bytesPerSecond > 0 && remaining > 0
      ? Math.ceil((remaining / bytesPerSecond) * 1_000)
      : undefined;
    this.snapshot = {
      ...this.snapshot,
      state,
      phase: update.phase ?? this.snapshot.phase,
      label: update.label ?? this.snapshot.label,
      ...(update.targetBrainId
        ? { targetBrainId: update.targetBrainId }
        : {}),
      filesCompleted,
      filesTotal,
      logicalBytesCompleted,
      logicalBytesTotal,
      physicalBytesAdded: safeCount(
        update.physicalBytesAdded,
        this.snapshot.physicalBytesAdded
      ),
      sharedBytes: safeCount(update.sharedBytes, this.snapshot.sharedBytes),
      bytesPerSecond,
      elapsedMs,
      ...(etaMs === undefined ? {} : { etaMs }),
      ...(update.diskSpace ? { diskSpace: update.diskSpace } : {}),
      ...(error ? { error } : {}),
      updatedAt: new Date().toISOString()
    };
    if (etaMs === undefined) delete this.snapshot.etaMs;
    this.emit("event", this.current());
    return this.current();
  }

  readonly hooks: BrainStorageOperationHooks = {
    signal: this.controller.signal,
    checkDisk: async (path, operationWriteBytes) => {
      while (true) {
        if (this.controller.signal.aborted) throw abortError();
        try {
          const diskSpace = await this.diskCheck(path, operationWriteBytes);
          this.publish("running", { diskSpace });
          return diskSpace;
        } catch (error) {
          if (!(error instanceof DiskReservePauseError)) throw error;
          this.pauseRequested = true;
          this.publish("paused", {
            diskSpace: error.diskSpace,
            label: "Paused before entering the mandatory free-space reserve"
          });
          await new Promise<void>((resolve, reject) => {
            const resume = (): void => {
              this.resumeWaiters.delete(resume);
              if (this.controller.signal.aborted) reject(abortError());
              else resolve();
            };
            this.resumeWaiters.add(resume);
            if (this.controller.signal.aborted) resume();
          });
        }
      }
    },
    checkpoint: async (update = {}) => {
      if (this.controller.signal.aborted) throw abortError();
      if (this.pauseRequested) {
        this.publish("paused", {
          ...update,
          label: update.label ?? "Paused after the current materialization"
        });
        await new Promise<void>((resolve, reject) => {
          const resume = (): void => {
            this.resumeWaiters.delete(resume);
            if (this.controller.signal.aborted) reject(abortError());
            else resolve();
          };
          this.resumeWaiters.add(resume);
          if (this.controller.signal.aborted) resume();
        });
      }
      if (this.controller.signal.aborted) throw abortError();
      this.publish("running", update);
    }
  };

  start(): BrainStorageOperationEvent {
    return this.publish("running");
  }

  pause(): BrainStorageOperationEvent {
    if (this.terminal || this.controller.signal.aborted) return this.current();
    this.pauseRequested = true;
    return this.publish("paused", {
      label: "Pause requested; finishing the current materialization"
    });
  }

  resume(): BrainStorageOperationEvent {
    if (this.terminal || this.controller.signal.aborted) return this.current();
    this.pauseRequested = false;
    for (const resume of this.resumeWaiters) resume();
    this.resumeWaiters.clear();
    return this.publish("running", { label: "Resuming storage operation" });
  }

  cancel(): BrainStorageOperationEvent {
    if (this.terminal) return this.current();
    this.controller.abort();
    for (const resume of this.resumeWaiters) resume();
    this.resumeWaiters.clear();
    return this.publish("cancelling", {
      label: "Cancelling after in-flight materialization"
    });
  }

  complete(update: BrainStorageProgressUpdate = {}): BrainStorageOperationEvent {
    this.terminal = true;
    return this.publish("complete", {
      phase: "complete",
      label: "Storage operation complete",
      ...update
    });
  }

  fail(error: unknown): BrainStorageOperationEvent {
    this.terminal = true;
    const cancelled = this.controller.signal.aborted;
    return this.publish(
      cancelled ? "cancelled" : "failed",
      {
        phase: cancelled ? "cancelled" : "failed",
        label: cancelled ? "Storage operation cancelled" : "Storage operation failed"
      },
      cancelled
        ? undefined
        : error instanceof Error
          ? error.message
          : String(error)
    );
  }
}

export class BrainStorageOperationManager extends EventEmitter {
  private readonly sessions = new Map<string, BrainStorageOperationSession>();

  create(
    operationId: string,
    kind: BrainStorageOperationKind,
    sourceBrainId?: string
  ): BrainStorageOperationSession {
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(operationId)) {
      throw new Error("Invalid storage operation id.");
    }
    if (this.sessions.has(operationId)) {
      throw new Error("This storage operation id is already in use.");
    }
    const session = new BrainStorageOperationSession(
      operationId,
      kind,
      sourceBrainId
    );
    session.on("event", (event: BrainStorageOperationEvent) => {
      this.emit("event", event);
    });
    this.sessions.set(operationId, session);
    session.start();
    return session;
  }

  pause(operationId: string): BrainStorageOperationEvent {
    return this.require(operationId).pause();
  }

  resume(operationId: string): BrainStorageOperationEvent {
    return this.require(operationId).resume();
  }

  cancel(operationId: string): BrainStorageOperationEvent {
    return this.require(operationId).cancel();
  }

  cancelAll(): void {
    for (const session of this.sessions.values()) session.cancel();
  }

  finish(operationId: string): void {
    const session = this.sessions.get(operationId);
    if (!session) return;
    const current = session.current();
    if (current.state !== "complete" && current.state !== "failed" && current.state !== "cancelled") {
      return;
    }
    while (this.sessions.size > 256) {
      const oldest = this.sessions.keys().next().value;
      if (!oldest || oldest === operationId) break;
      this.sessions.delete(oldest);
    }
  }

  private require(operationId: string): BrainStorageOperationSession {
    const session = this.sessions.get(operationId);
    if (!session) throw new Error("Storage operation was not found.");
    return session;
  }
}
