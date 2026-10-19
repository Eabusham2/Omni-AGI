import type { ActionEvent } from "../../shared/types";

/** Host-issued presentation provenance, never inferred from neural scores/text. */
export interface PonderLinkedAction extends ActionEvent {
  chatTurnId?: string;
  parentPonderActionEventId?: string;
}

export interface PonderActionGroups {
  roots: ActionEvent[];
  childrenByParent: Map<string, ActionEvent[]>;
}

const nativeId = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{32}$/.test(value);

/**
 * Compose only an explicit same-owner native episode relationship. Old history,
 * missing parents, ambiguous IDs and unbound human actions remain visible flat.
 * A state, timestamp, selectionPhase or selectionStep is not ancestry evidence.
 */
export function groupPonderActions(events: readonly PonderLinkedAction[]): PonderActionGroups {
  const byId = new Map<string, PonderLinkedAction>();
  const ambiguous = new Set<string>();
  for (const event of events) {
    if (byId.has(event.id)) ambiguous.add(event.id);
    else byId.set(event.id, event);
  }
  const childrenByParent = new Map<string, ActionEvent[]>();
  const nested = new Set<PonderLinkedAction>();
  for (const event of events) {
    const parentId = event.parentPonderActionEventId;
    if (!parentId || ambiguous.has(parentId) || ambiguous.has(event.id)) continue;
    const parent = byId.get(parentId);
    if (!parent || parent === event || parent.action.kind !== "ponder" || event.action.kind === "ponder" ||
        event.action.source === "human" || parent.action.source === "human" ||
        !nativeId(parent.neuralActionId) || !nativeId(event.neuralActionId) ||
        event.brainId !== parent.brainId || typeof event.chatTurnId !== "string" ||
        event.chatTurnId.length === 0 || event.chatTurnId !== parent.chatTurnId) continue;
    const children = childrenByParent.get(parentId) ?? [];
    children.push(event);
    childrenByParent.set(parentId, children);
    nested.add(event);
  }
  return { roots: events.filter(event => !nested.has(event)), childrenByParent };
}
