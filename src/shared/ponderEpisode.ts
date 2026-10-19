import type { StructuredAction } from "./types";

/** Measured operation only, not hidden chain-of-thought or execution authority. */
export interface NativePonderEpisodeFrame {
  state: "started" | "finished";
  actionId: string;
  action: StructuredAction;
}

export function measuredPonderTrace(action: StructuredAction): Record<string, unknown> | undefined {
  const value = action.arguments.ponderTrace;
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const trace = value as Record<string, unknown>;
  const measured = typeof trace.passes === "number" && Number.isSafeInteger(trace.passes) && trace.passes >= 0 &&
    typeof trace.final_delta === "number" && Number.isFinite(trace.final_delta) && trace.final_delta >= 0;
  const incompleteFailure = trace.measurementIncomplete === true && trace.completed === false &&
    ["cancelled", "failed"].includes(String(trace.stop_reason)) && trace.passes === null && trace.final_delta === null;
  return action.kind === "ponder" && action.arguments.handledInTurn === true &&
    trace.activated === true && trace.activated_by === "learned-action-head" &&
    trace.private === true && trace.visibleMagicTags === false && typeof trace.completed === "boolean" &&
    action.arguments.completedInTurn === trace.completed && trace.phase === action.selectionPhase &&
    (trace.measurementIncomplete === true ? incompleteFailure : measured) &&
    typeof trace.elapsed_ms === "number" && Number.isFinite(trace.elapsed_ms) && trace.elapsed_ms >= 0 &&
    ["converged", "learned-action-transition", "steered", "suspended", "resource-pressure", "cancelled", "failed"].includes(String(trace.stop_reason)) &&
    trace.completed === ["converged", "learned-action-transition"].includes(String(trace.stop_reason))
      ? trace : undefined;
}
