import type { ExternalAdviceChannel, ExternalAdviceRequest } from "../../shared/externalAdvice";
export interface AdvicePanelState {
  entries: ExternalAdviceRequest[];
  live: Record<string, boolean>;
  persisted: Record<string, number>;
}
export const emptyAdvicePanel: AdvicePanelState = { entries: [], live: {}, persisted: {} };
const pending = (entry: ExternalAdviceRequest) => entry.state === "queued" || entry.state === "streaming";
export function mergeAdvicePanel(state: AdvicePanelState, updates: Array<{
  request: ExternalAdviceRequest; active: boolean; persistedCharacters?: number;
}>): AdvicePanelState {
  const entries = new Map(state.entries.map(entry => [entry.id, entry]));
  const live = { ...state.live }, persisted = { ...state.persisted };
  for (const update of updates) {
    const next = update.request, previous = entries.get(next.id);
    if (previous && (previous.updatedAt > next.updatedAt || previous.updatedAt === next.updatedAt &&
        (pending(next) && !pending(previous) || previous.state === "streaming" && next.state === "queued" ||
          pending(next) && previous.response.length > next.response.length || pending(next) && state.live[next.id] && !update.active))) continue;
    entries.set(next.id, next);
    live[next.id] = update.active && pending(next);
    persisted[next.id] = update.persistedCharacters ?? next.response.length;
  }
  return { entries: [...entries.values()].sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id)), live, persisted };
}
export function advicePanelActive(state: AdvicePanelState, channel: ExternalAdviceChannel): boolean {
  return state.entries.some(entry => entry.channel === channel && state.live[entry.id] === true && pending(entry));
}
