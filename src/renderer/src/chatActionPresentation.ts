import type { ActionEvent, StructuredAction } from "../../shared/types";
import { conciseUiMessage } from "./uiPresentation";

const stateLabels: Record<ActionEvent["state"], string> = {
  proposed: "Ready",
  "approval-required": "Needs approval",
  running: "In progress",
  complete: "Done",
  failed: "Failed",
  stopped: "Stopped"
};

const kindLabels: Record<StructuredAction["kind"], string> = {
  talk: "Message",
  tool: "Tool action",
  imagine: "Imagination",
  agent: "Agent",
  ponder: "Pondering",
  learn: "Learning",
  evolve: "Self-improvement",
  stop: "Stopped"
};

function modalityLabel(action: StructuredAction): string | null {
  const modality = action.arguments.modality;
  if (modality === "image") return "Image imagination";
  if (modality === "audio") return "Sound imagination";
  if (modality === "video") return "Video imagination";
  return null;
}

/** Plain-language copy for the compact chat timeline. Protocol identifiers
 * remain available in Trace, but never leak into the basic conversation UI. */
export function chatActionTitle(action: StructuredAction): string {
  if (action.kind === "imagine") return modalityLabel(action) ?? "Imagination";
  if (action.toolId === "studio.ui" && action.action === "open-creativity") {
    return "Creativity";
  }
  if (action.kind === "tool") {
    if (action.toolId === "brain.history") return "Conversation history";
    if (action.toolId?.startsWith("web.")) return "Web access";
    if (
      action.toolId === "system.files" ||
      action.toolId === "windows.files" ||
      action.toolId?.startsWith("files.")
    ) return "File access";
    if (action.toolId?.startsWith("browser.")) return "Browser action";
    if (
      action.toolId === "system.shell" ||
      action.toolId === "windows.powershell"
    ) return "System shell";
    if (action.toolId?.startsWith("code.") || action.toolId?.startsWith("shell.")) {
      return "Computer action";
    }
  }
  return kindLabels[action.kind] ?? "Action";
}

export function chatActionStateLabel(state: ActionEvent["state"]): string {
  return stateLabels[state] ?? "Updated";
}

/** Keep the concrete web request visible in the collapsed timeline card so a
 * person can see what the brain is searching without opening raw JSON. */
export function chatActionSearchActivity(
  action: StructuredAction,
  state: ActionEvent["state"]
): string | null {
  if (action.kind !== "tool" || !action.toolId?.startsWith("web.")) return null;
  const candidate = [
    action.arguments.query,
    action.arguments.q,
    action.arguments.search,
    action.arguments.url
  ].find((value) => typeof value === "string" && value.trim().length > 0);
  if (typeof candidate !== "string") return null;
  const value = candidate.replace(/\s+/g, " ").trim();
  const bounded = value.length > 120 ? `${value.slice(0, 119).trimEnd()}…` : value;
  const verb = state === "running" || state === "proposed" || state === "approval-required"
    ? "Searching"
    : state === "complete"
      ? "Searched"
      : "Search";
  return `${verb} “${bounded}”`;
}

/** Remove the two legacy protocol strings that were most visible during live
 * testing. Unknown diagnostic identifiers are replaced with plain action copy
 * instead of being prettified and accidentally exposed. */
export function cleanChatActionStatus(
  value: string | undefined,
  action?: StructuredAction
): string {
  if (!value) return action ? chatActionTitle(action) : "Action update";
  const cleaned = conciseUiMessage(value, action ? chatActionTitle(action) : "Action update")
    .replace(/\bponder\.ponder\b/gi, "Pondering")
    .replace(/\bstudio\.ui[/.]open-creativity\b/gi, "Creativity")
    .replace(/\bopen-creativity\b/gi, "Creativity")
    .trim();
  if (/\b[a-z][\w-]*(?:\.[a-z][\w-]*)+(?:\s*[·:/]\s*[\w-]+)?\b/i.test(cleaned)) {
    return action ? chatActionTitle(action) : "Action update";
  }
  return cleaned;
}
