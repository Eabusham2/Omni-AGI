import { BackgroundRequestDeferredError, EngineRequestError } from "./engineSupervisor";
import { SparseTransferBoundaryError } from "./sparseSubstrateTransferService";

/** Preserve uncertainty unless the exact native owner supplies proof. */
export function sparseSubstrateRpcError(error: unknown, brainId: string): unknown {
  if (error instanceof BackgroundRequestDeferredError || error instanceof EngineRequestError && error.code === -32032) {
    return new SparseTransferBoundaryError("Sparse import is queued until the current native activity reaches its boundary.", "boundary-required");
  }
  if (!(error instanceof EngineRequestError) || !error.data || typeof error.data !== "object") return error;
  const data = error.data as Record<string, unknown>;
  if (data.nativeSparseTransfer !== true || data.brainId !== brainId) return error;
  if (error.code === -32800 && data.safeBoundary === true && data.cancelled === true && data.committed === false && data.nativeOwnerReleased === false) {
    return new SparseTransferBoundaryError(error.message, "cancelled-unapplied", false);
  }
  if (error.code === -32033 && data.needsReview === true) return new SparseTransferBoundaryError(error.message, "needs-review");
  if (error.code === -32031 && data.reloadRequired === true && data.nativeOwnerReleased === true) {
    return new SparseTransferBoundaryError(error.message, "reload-required", typeof data.committed === "boolean" ? data.committed : null);
  }
  return error;
}
