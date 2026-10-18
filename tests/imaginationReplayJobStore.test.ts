/** Tiny actual SQLite cursor fixtures only: no neural constructor, recording, codec or media execution. */
import { createHash } from "node:crypto";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { ImaginationAssembly } from "../src/shared/imaginationAssembly";
import { ReplayJobStore, rekeyCopiedReplayJobStore, replayJobStatePath, type ReplayJobSegment } from "../src/main/imaginationReplayJobStore";

const roots: string[] = [];
const opened = new Set<ReplayJobStore>();
const job = "a".repeat(32), attempt = "b".repeat(32), sourceSha = "c".repeat(64);
async function root(): Promise<string> { const value = await mkdtemp(join(tmpdir(), "omni-replay-cursor-")); roots.push(value); return value; }
const bytesHash = (value: Buffer | string): string => createHash("sha256").update(value).digest("hex");
const initial = (overrides: Partial<ImaginationAssembly> = {}): ImaginationAssembly => ({
  format: "omni-imagination-assembly", version: 1, id: job, brainId: "parent", requestId: "request-1", revision: 1,
  recordingId: "recording-1", recordingRevision: 2, state: "queued", createdAt: "2026-10-13T01:00:00.000Z", updatedAt: "2026-10-13T01:00:00.000Z",
  completedSpans: 0, totalSpans: 1_000_000, durationMs: 1000, manifestSha256: sourceSha,
  label: "Delivered timeline replay — presentation timing, not native synchronization", ...overrides
});
const segment = (index: number, overrides: Partial<ReplayJobSegment> = {}): ReplayJobSegment => ({ index, path: `work/${attempt}/segment-${index}.mkv`, sha256: bytesHash(`segment ${index}`), bytes: 16, ...overrides });
const next = (view: ImaginationAssembly, overrides: Partial<ImaginationAssembly> = {}): ImaginationAssembly => ({ ...view, revision: view.revision + 1, state: "encoding", ...overrides });
async function create(directory: string, options: Parameters<typeof ReplayJobStore.open>[3] = {}): Promise<ReplayJobStore> {
  const store = await ReplayJobStore.open(directory, job, "parent", { create: initial(), ...options }); opened.add(store); return store;
}
function close(store: ReplayJobStore): void { store.close(); opened.delete(store); }
async function edit(directory: string, work: (db: DatabaseSync) => void): Promise<void> {
  const db = new DatabaseSync(replayJobStatePath(directory, job)); try { work(db); } finally { db.close(); }
}
afterEach(async () => { for (const store of opened) store.close(); opened.clear(); for (const directory of roots.splice(0)) await rm(directory, { recursive: true, force: true }); });

describe("bounded, exact replay-job cursor storage", () => {
  it("persists only committed contiguous spans, preserves large declared length, and reads the sealed file without modification", async () => {
    const directory = await root(), store = await create(directory);
    const one = next(store.view(), { completedSpans: 1 }); store.commitSegment(segment(0), one, 1);
    const two = next(one, { completedSpans: 2 }); store.commitSegment(segment(1), two, 2);
    expect(store.count()).toBe(2); expect(store.view().totalSpans).toBe(1_000_000);
    expect(store.segment(1)).toEqual(segment(1)); expect(store.segment(2)).toBeUndefined();
    expect(() => store.commitSegment(segment(3), next(two, { completedSpans: 4 }), 3)).toThrow(/skip/);
    expect(() => store.saveView(next(two, { completedSpans: 3 }), 3)).toThrow(/segment count/);
    store.validate(); close(store);
    const before = bytesHash(await readFile(replayJobStatePath(directory, job)));
    const read = await ReplayJobStore.open(directory, job, "parent", { readOnly: true }); opened.add(read);
    expect([...read.segments()]).toEqual([segment(0), segment(1)]); expect(read.head().sequence).toBe(3); read.validate();
    expect(() => read.saveView(next(read.view()), 3)).toThrow(/Read-only/);
    expect(() => read.rekey("child")).toThrow(/Read-only/); close(read);
    expect(bytesHash(await readFile(replayJobStatePath(directory, job)))).toBe(before);
  });

  it("rolls back inserted event/segment rows if head admission fails, then accepts one exact idempotent retry", async () => {
    const directory = await root(); let failAtHead = false, headBytes = 0;
    const store = await create(directory, { admitBytes(bytes) { if (failAtHead && bytes === headBytes) throw new Error("test head admission paused"); } });
    headBytes = Buffer.byteLength(JSON.stringify(store.head())); const before = store.head(), one = next(store.view(), { completedSpans: 1 });
    failAtHead = true;
    expect(() => store.commitSegment(segment(0), one, 1)).toThrow(/test head admission paused/);
    failAtHead = false;
    expect(store.head()).toEqual(before); expect(store.segment(0)).toBeUndefined(); store.validate();
    store.commitSegment(segment(0), one, 1); const committed = store.head();
    expect(store.commitSegment(segment(0), one, 1)).toEqual(one); expect(store.head()).toEqual(committed);
    expect(() => store.commitSegment(segment(0, { sha256: "d".repeat(64) }), one, 1)).toThrow(/Repeated/);
    expect(() => store.commitSegment(segment(0), { ...one, error: "different result" }, 1)).toThrow(/Repeated/);
    expect(store.count()).toBe(1); store.validate();
  });

  it("uses current-revision CAS across independent connections and leaves source/manifest identity immutable", async () => {
    const directory = await root(), a = await create(directory), b = await ReplayJobStore.open(directory, job, "parent"); opened.add(b);
    const view = a.view(), changed = next(view, { state: "preparing", error: "Unsanitized private detail: user/example" }); b.saveView(changed, 1);
    expect(() => a.saveView(next(view), 1)).toThrow(/revision changed/);
    expect(() => a.saveView(next(changed, { manifestSha256: "e".repeat(64) }), 2)).toThrow(/immutable/);
    expect(() => a.saveView(next(changed, { totalSpans: 5 }), 2)).toThrow(/immutable/);
    expect(() => a.saveView(next(changed, { brainId: "other" }), 2)).toThrow(/inheritance/);
    expect(a.view().error).toBe(changed.error); a.validate();
    await expect(ReplayJobStore.open(directory, job, "other", { readOnly: true })).rejects.toThrow(/another exact/);
  });

  it("refuses missing heads, schema alterations, and older damaged event history without silently creating empty state", async () => {
    const missing = await root(), first = await create(missing); close(first);
    await edit(missing, db => { db.exec("DELETE FROM head"); });
    await expect(ReplayJobStore.open(missing, job, "parent", { create: initial() })).rejects.toThrow(/head is missing/);
    const db = new DatabaseSync(replayJobStatePath(missing, job)); try { expect(db.prepare("SELECT COUNT(*) AS n FROM events").get()?.n).toBe(1); } finally { db.close(); }

    const history = await root(), second = await create(history);
    second.saveView(next(second.view(), { state: "preparing" }), 1); second.saveView(next(second.view()), 2); close(second);
    await edit(history, handle => { handle.prepare("UPDATE events SET body=? WHERE sequence=2").run("{\"invalid\":true}"); });
    const damaged = await ReplayJobStore.open(history, job, "parent", { readOnly: true }); opened.add(damaged);
    expect(damaged.view().revision).toBe(3); // Bounded status is not a full custody-history attestation.
    expect(() => damaged.validate()).toThrow(/checksum/); close(damaged);
    await edit(history, handle => { handle.exec("DELETE FROM events WHERE sequence=2"); });
    const gap = await ReplayJobStore.open(history, job, "parent", { readOnly: true }); opened.add(gap);
    expect(() => gap.validate()).toThrow(/missing/); close(gap);

    const schema = await root(), third = await create(schema); close(third);
    await edit(schema, handle => { handle.exec("CREATE TRIGGER unauthorized AFTER INSERT ON segments BEGIN DELETE FROM events; END"); });
    await expect(ReplayJobStore.open(schema, job, "parent", { readOnly: true })).rejects.toThrow(/schema objects/);
  });

  it("checks body admission before fetching/parsing a changed large stored row", async () => {
    const directory = await root(), store = await create(directory); close(store);
    await edit(directory, db => { db.prepare("UPDATE events SET body=? WHERE sequence=1").run("x".repeat(8192)); });
    let rejectedBytes = 0;
    await expect(ReplayJobStore.open(directory, job, "parent", { readOnly: true, admitBytes(bytes) {
      if (bytes > 2048) { rejectedBytes = bytes; throw new Error("bounded row admission paused"); }
    } })).rejects.toThrow(/bounded row admission paused/);
    expect(rejectedBytes).toBe(8192);
  });

  it("admits schema names and SQL bytes before retrieving an unsupported oversized schema object", async () => {
    const directory = await root(), store = await create(directory); close(store);
    await edit(directory, db => { db.exec(`CREATE TABLE "${"u".repeat(4096)}"(value TEXT)`); });
    let rejected = 0;
    await expect(ReplayJobStore.open(directory, job, "parent", { readOnly: true, admitBytes(bytes) {
      if (bytes > 2048) { rejected = bytes; throw new Error("bounded schema admission paused"); }
    } })).rejects.toThrow(/bounded schema admission paused/);
    expect(rejected).toBeGreaterThan(8192);
  });

  it("iterates verified one-row cursors with cancellation, without a count-sized JSON array or full-history status query", async () => {
    const directory = await root(), store = await create(directory);
    for (let index = 0; index < 4; index++) { const view = store.view(); store.commitSegment(segment(index), next(view, { completedSpans: index + 1 }), view.revision); }
    store.validate(); close(store);
    const controller = new AbortController(), admissions: number[] = [];
    const read = await ReplayJobStore.open(directory, job, "parent", { readOnly: true, signal: controller.signal, admitBytes: bytes => { admissions.push(bytes); } }); opened.add(read);
    const iterator = read.segments(); expect(iterator.next().value).toEqual(segment(0));
    expect(Math.max(...admissions)).toBeLessThan(2048); controller.abort(new Error("explicit cursor cancellation"));
    expect(() => iterator.next()).toThrow(/explicit cursor cancellation/); close(read);
    const reopened = await ReplayJobStore.open(directory, job, "parent", { readOnly: true }); opened.add(reopened);
    expect(reopened.count()).toBe(4); reopened.validate();
    const database = new DatabaseSync(replayJobStatePath(directory, job), { readOnly: true });
    try {
      for (const row of database.prepare("SELECT body FROM events").iterate()) expect(String(row.body)).not.toContain('"segments":[');
      for (const sql of ["SELECT sequence FROM events ORDER BY sequence DESC LIMIT 1", "SELECT idx FROM segments ORDER BY idx DESC LIMIT 1"]) {
        const detail = [...database.prepare(`EXPLAIN QUERY PLAN ${sql}`).iterate()].map(row => String(row.detail)).join(" ");
        expect(detail).not.toContain("TEMP B-TREE");
      }
    } finally { database.close(); }
  });

  it("rekeys only a private copied DB to dormant while preserving original rows, intent, hashes, and full unsanitized history", async () => {
    const source = await root(), target = await root(), secondTarget = await root(), store = await create(source);
    const saved = next(store.view(), { error: "Private original owner information is preserved, not sanitized", currentAttempt: attempt, currentSpan: 0 }); store.saveView(saved, 1); close(store);
    const sourcePath = replayJobStatePath(source, job), sourceBytes = bytesHash(await readFile(sourcePath)), copyId = "d".repeat(64);
    const targetPath = replayJobStatePath(target, copyId); await mkdir(dirname(targetPath), { recursive: true }); await copyFile(sourcePath, targetPath);
    const intent = "original exact intent bytes; source owner parent and source assembly unchanged";
    await writeFile(join(dirname(targetPath), "intent.json"), intent);
    await expect(ReplayJobStore.open(target, copyId, "parent")).rejects.toThrow(/another exact/);
    await rekeyCopiedReplayJobStore(target, copyId, "parent", job, "child");
    const copied = await ReplayJobStore.open(target, copyId, "child", { readOnly: true }); opened.add(copied);
    const projected = copied.view(); expect(projected.state).toBe("dormant"); expect(projected.currentAttempt).toBeUndefined(); expect(projected.currentSpan).toBeUndefined();
    expect(projected.inheritedFrom).toEqual({ brainId: "parent", assemblyId: job }); expect(projected.error).toBe(saved.error); copied.validate(); close(copied);
    expect(await readFile(join(dirname(targetPath), "intent.json"), "utf8")).toBe(intent);
    expect(bytesHash(await readFile(sourcePath))).toBe(sourceBytes);
    const original = new DatabaseSync(sourcePath, { readOnly: true }), adopted = new DatabaseSync(targetPath, { readOnly: true });
    try { for (const row of original.prepare("SELECT * FROM events ORDER BY sequence").iterate()) expect(adopted.prepare("SELECT * FROM events WHERE sequence=?").get(Number(row.sequence))).toEqual(row); }
    finally { original.close(); adopted.close(); }
    const twicePath = replayJobStatePath(secondTarget, copyId); await mkdir(dirname(twicePath), { recursive: true }); await copyFile(targetPath, twicePath);
    await rekeyCopiedReplayJobStore(secondTarget, copyId, "child", copyId, "grandchild");
    const twice = await ReplayJobStore.open(secondTarget, copyId, "grandchild", { readOnly: true }); opened.add(twice);
    expect(twice.view().inheritedFrom).toEqual({ brainId: "parent", assemblyId: job }); expect(twice.view().state).toBe("dormant"); twice.validate();
  });
});
