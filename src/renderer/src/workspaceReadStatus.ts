/** A connection is not evidence that a failed initial read is still pending.
 * Callers use this only while their authoritative snapshot is absent. */
export function workspaceReadStatus(connected: boolean, failed: boolean, text: {
  pending: string; unavailable: string; disconnected: string;
}): string {
  return !connected ? text.disconnected : failed ? text.unavailable : text.pending;
}
