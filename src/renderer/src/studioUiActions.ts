import type { ActionEvent } from "../../shared/types";

export type StudioUiDestination = "imagine" | "tools";

function valueRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/**
 * Translate only a completed, typed, trusted studio capability result into a
 * renderer route. No chat prose, slash command, model tag, or arbitrary output
 * string is interpreted as navigation.
 */
export function studioUiDestination(
  event: ActionEvent
): StudioUiDestination | null {
  if (event.state !== "complete" || event.action.kind !== "tool" || event.execution?.state !== "complete") {
    return null;
  }
  const creativity =
    event.action.toolId === "studio.ui" &&
    event.action.action === "open-creativity" &&
    event.execution.toolId === "studio.ui" &&
    event.execution.action === "open-creativity";
  const permissions =
    event.action.toolId === "studio.settings" &&
    event.action.action === "open-permissions" &&
    event.execution.toolId === "studio.settings" &&
    event.execution.action === "open-permissions";
  if (!creativity && !permissions) return null;
  const output = valueRecord(event.execution.output);
  // Active-turn streams intentionally omit execution output to avoid moving
  // large tool payloads twice. The exact completed protocol tuple is already
  // authoritative; when the final compact result is present, validate it too.
  if (!output) return creativity ? "imagine" : "tools";
  if (creativity) {
    return output.workspace === "creativity" &&
      output.view === "imagine" &&
      output.local === true &&
      output.reversible === true
      ? "imagine"
      : null;
  }
  return output.workspace === "tools" &&
    output.view === "permissions" &&
    output.grantsChanged === false &&
    output.local === true &&
    output.reversible === true
    ? "tools"
    : null;
}
