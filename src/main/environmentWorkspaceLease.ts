type Mode = "mutation" | "snapshot";
interface Waiter { mode: Mode; resolve(release: () => void): void }
interface State { mutations: number; snapshot: boolean; waitingSnapshots: number; queue: Waiter[] }
const states = new Map<string, State>();

function state(root: string): State {
  let value = states.get(root);
  if (!value) { value = { mutations: 0, snapshot: false, waitingSnapshots: 0, queue: [] }; states.set(root, value); }
  return value;
}

function drain(root: string, value: State): void {
  if (value.snapshot || value.mutations) return;
  const snapshotIndex = value.queue.findIndex(waiter => waiter.mode === "snapshot");
  if (snapshotIndex >= 0) {
    const [waiter] = value.queue.splice(snapshotIndex, 1); value.waitingSnapshots--; value.snapshot = true;
    waiter!.resolve(() => release(root, "snapshot")); return;
  }
  const mutations = value.queue.splice(0).filter(waiter => waiter.mode === "mutation");
  for (const waiter of mutations) { value.mutations++; waiter.resolve(() => release(root, "mutation")); }
}
function release(root: string, mode: Mode): void {
  const value = states.get(root); if (!value) throw new Error("Environment workspace lease lost its owner.");
  if (mode === "snapshot") value.snapshot = false; else value.mutations--;
  if (value.mutations < 0) throw new Error("Environment workspace lease underflow.");
  drain(root, value);
  if (!value.snapshot && value.mutations === 0 && value.queue.length === 0) states.delete(root);
}
function acquire(root: string, mode: Mode): Promise<() => void> {
  const value = state(root);
  if (mode === "mutation" && !value.snapshot && value.waitingSnapshots === 0) { value.mutations++; return Promise.resolve(() => release(root, mode)); }
  if (mode === "snapshot" && !value.snapshot && value.mutations === 0) { value.snapshot = true; return Promise.resolve(() => release(root, mode)); }
  if (mode === "snapshot") value.waitingSnapshots++;
  return new Promise(resolve => value.queue.push({ mode, resolve }));
}
async function withLease<T>(root: string, mode: Mode, action: () => Promise<T>): Promise<T> {
  const releaseLease = await acquire(root, mode);
  try { return await action(); } finally { releaseLease(); }
}

/** Normal lifecycle operations may coexist; a portable snapshot waits for all
 * of them and prevents a new writer from starting until capture is complete. */
export function withEnvironmentMutation<T>(root: string, action: () => Promise<T>): Promise<T> { return withLease(root, "mutation", action); }
export function withEnvironmentSnapshot<T>(root: string, action: () => Promise<T>): Promise<T> { return withLease(root, "snapshot", action); }
