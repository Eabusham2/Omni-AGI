/** Retires a host attachment after acknowledged unload, never a native Stop. */
export class InstanceContinuationDetachedError extends Error {
  readonly code = "INSTANCE_CONTINUATION_DETACHED";
  constructor(readonly brainId: string, readonly turnId: string,
    readonly frontier?: import("../shared/instanceLifecycle").InstanceResumeFrontier,
    readonly inheritedExecutionId?: string) {
    super("The unloaded turn's host attachment was detached before restoring a recovery point. Its durable native frontier was not cancelled or replayed.");
    this.name = "InstanceContinuationDetachedError";
  }
}
