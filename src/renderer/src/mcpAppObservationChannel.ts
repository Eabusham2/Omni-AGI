import type { McpAppsApi, McpAppObservationEvent } from "../../shared/mcpApps";

const channels = new WeakMap<McpAppsApi, { listeners: Set<(event: McpAppObservationEvent) => void>; close(): void }>();
/** Many historical App cards share one read-only preload listener. */
export function subscribeAppObservations(api: McpAppsApi, listener: (event: McpAppObservationEvent) => void): () => void {
  if (!api.onObservation) return () => undefined;
  let channel = channels.get(api);
  if (!channel) {
    const listeners = new Set<(event: McpAppObservationEvent) => void>();
    const close = api.onObservation(event => { for (const receive of listeners) receive(event); });
    channel = { listeners, close }; channels.set(api, channel);
  }
  const current = channel; current.listeners.add(listener);
  return () => { current.listeners.delete(listener); if (!current.listeners.size && channels.get(api) === current) { channels.delete(api); current.close(); } };
}
