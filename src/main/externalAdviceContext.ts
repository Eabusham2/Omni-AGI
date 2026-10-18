import { createHash } from "node:crypto";
import type { ExternalAdviceChannel, ExternalAdviceContextItem, ExternalAdviceContextSelection, ExternalAdviceSelectedContext } from "../shared/externalAdvice";
import type { ConversationLedgerPage } from "../shared/types";

/** A read-only window of recent recorded work, not the neural model or its hidden activity. */
export function selectRecentAdviceContext(input: {
  brainId: string;
  channel: ExternalAdviceChannel;
  selection: ExternalAdviceContextSelection;
}, page: ConversationLedgerPage): ExternalAdviceSelectedContext {
  if (page.brainId !== input.brainId || !Number.isSafeInteger(input.selection.maxCharacters) ||
      input.selection.maxCharacters < 0) throw new Error("Advice context owner or budget is invalid.");
  const selected: Array<Omit<ExternalAdviceContextItem, "contentSha256">> = [];
  const seen = new Set<string>();
  const newest = [...page.entries].sort((a, b) => b.sequence - a.sequence);
  // Current task is the latest actual user entry in this bounded host window.
  const task = input.selection.includeTask ? newest.find(item => item.message?.role === "human" && !item.message.inputProvenance) : undefined;
  if (task?.message) selected.push({ id: `message:${task.message.id}`, kind: "task", label: "Most recent recorded user task",
    text: task.message.content, createdAt: task.createdAt });
  if (input.selection.includeActivity) for (const item of newest) {
    if (item.action && !seen.has(item.action.id)) {
      seen.add(item.action.id);
      const action = item.action;
      selected.push({ id: `action:${action.id}`, kind: "activity", label: "Recorded action status",
        text: JSON.stringify({ kind: action.action.kind, tool: action.action.toolId, action: action.action.action,
          state: action.state, status: action.statusLabel, error: action.error }), createdAt: item.createdAt });
    } else if (item.message?.role === "brain") selected.push({ id: `message:${item.message.id}`, kind: "activity",
      label: "Recorded reply (not hidden thoughts)", text: item.message.content, createdAt: item.createdAt });
  }
  let available = input.selection.maxCharacters, includedCharacters = 0, truncated = false;
  const items: ExternalAdviceContextItem[] = [];
  for (const item of selected) {
    if (available === 0) { truncated = true; break; }
    const text = item.text.slice(0, available);
    truncated ||= text.length !== item.text.length;
    if (text) items.push({ ...item, text, contentSha256: createHash("sha256").update(text).digest("hex") });
    available -= text.length; includedCharacters += text.length;
  }
  return { format: "omni-external-advice-selected-context", version: 1, ...input, items,
    includedCharacters, omittedItems: Math.max(0, page.totalEntries - items.length),
    truncated: truncated || page.hasOlder };
}
