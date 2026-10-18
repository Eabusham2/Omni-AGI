const guards = new Map<string, Set<() => boolean>>(), snapshots = new Set<string>();
export function registerLocalTeacherBoundary(root: string, busy: () => boolean): () => void {
  let entries = guards.get(root); if (!entries) { entries = new Set(); guards.set(root, entries); } entries.add(busy);
  return () => { entries!.delete(busy); if (!entries!.size) guards.delete(root); };
}
export function requireLocalTeacherMutation(root: string): void {
  if (snapshots.has(root)) throw new Error("Teacher workspace is being copied at its quiescent boundary; retry this explicit operation afterward.");
}
/** Never snapshots an active writer/child or silently cancels it. Pause first. */
export async function withLocalTeacherSnapshot<T>(root: string, operation: () => Promise<T>): Promise<T> {
  if (snapshots.has(root) || [...(guards.get(root) ?? [])].some(busy => busy()))
    throw new Error("Pause active local teacher generation/selection before copying. Committed source data and pending job progress will be preserved.");
  snapshots.add(root); try { return await operation(); } finally { snapshots.delete(root); }
}
