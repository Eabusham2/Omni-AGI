import type { ActionEvent } from "../shared/types";
import type { ImaginationCaptureOrigin } from "../shared/imaginationCapture";

/** Caller supplies an existing main-owned action, not prose or renderer flags. */
export function mainImaginationCaptureOrigin(event: ActionEvent): ImaginationCaptureOrigin {
  if (event.action.source === "human") return { kind: "user-requested", authority: "main-human-action", requestId: event.id };
  if (event.action.source === "organic") return { kind: "spontaneous", authority: "main-organic-action", actionEventId: event.id };
  return { kind: "unclassified", authority: "native-brain-action" };
}
