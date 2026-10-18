interface Lease { tail: Promise<void>; active: number; snapshot: boolean }
const leases = new Map<string, Lease>();
const lease = (root: string): Lease => { let row = leases.get(root); if (!row) { row = { tail: Promise.resolve(), active: 0, snapshot: false }; leases.set(root, row); } return row; };

async function exclusive<T>(root: string, snapshot: boolean, work: () => Promise<T>): Promise<T> {
  const row = lease(root); let release!: () => void; const previous = row.tail;
  row.tail = new Promise<void>(resolve => { release = resolve; }); await previous;
  if (row.active || row.snapshot) { release(); throw new Error("Clipboard workspace lease ownership is inconsistent."); }
  row.active = 1; row.snapshot = snapshot;
  try { return await work(); }
  finally { row.active = 0; row.snapshot = false; release(); }
}
export const withClipboardMutation = <T>(root: string, work: () => Promise<T>): Promise<T> => exclusive(root, false, work);
export const withClipboardSnapshot = <T>(root: string, work: () => Promise<T>): Promise<T> => exclusive(root, true, work);
