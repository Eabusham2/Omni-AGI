/** Actual small-file/SQLite reads; no corpus, native model, capture or learning. */
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { copyFile, lstat, mkdtemp, readFile, rename, rm, symlink, utimes, writeFile, type FileHandle } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BrainRepository } from "../src/main/brainRepository";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { TaskAttachmentReadVerifier } from "../src/main/taskAttachmentReadVerifier";

const roots: string[] = [];
const cachePossible = process.platform !== "win32" && typeof constants.O_NOFOLLOW === "number" && constants.O_NOFOLLOW !== 0;
const hash = (value: Buffer) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { vi.restoreAllMocks(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture(options: ConstructorParameters<typeof TaskAttachmentReadVerifier>[0] = {}) {
  const root = await mkdtemp(join(tmpdir(), "omni-task-read-proof-")); roots.push(root);
  const brainId = "read-owner", turnId = "read-turn", source = Buffer.from("0123456789abcdef".repeat(5120)); //80KiB
  const verifier = new TaskAttachmentReadVerifier({ strongLocalChangeTimes: async () => true, ...options });
  const store = new TaskAttachmentStore(root, brainId, verifier);
  const attachment = await store.stageBytes("read-source", { data: source, displayName: "source.txt", mimeType: "text/plain", kind: "file",
    source: { kind: "local-file", selectedAt: new Date().toISOString() } });
  await store.bindToTurn(turnId, [attachment.id]);
  const repository = { get: async (id: string) => ({ id }), brainDirectory: () => root } as unknown as BrainRepository;
  const service = new TaskAttachmentService(repository, verifier);
  const path = join(root, "attachments", "blobs", `${attachment.contentSha256}.data`);
  return { root, brainId, turnId, source, verifier, store, service, attachment, path };
}

describe("exact-FD current-task attachment verification", () => {
  it("uses the actual production filesystem classifier without inventing eligibility on other filesystems", async () => {
    const f = await fixture({ strongLocalChangeTimes: undefined });
    await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7);
    const eligible = f.verifier.statistics.proofs === 1;
    await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 7, 7);
    expect(f.verifier.statistics.fullHashes).toBe(eligible ? 1 : 2);
    expect(f.verifier.statistics.cacheHits).toBe(eligible ? 1 : 0);
    console.info("Production attachment proof policy:", { eligible, fullHashes: f.verifier.statistics.fullHashes, cacheHits: f.verifier.statistics.cacheHits });
  });

  it("cancellation after a cached actual page read returns no result and removes its selected proof", async () => {
    const f = await fixture(), binding = { authorityKey: "a".repeat(64), path: f.path,
      contentSha256: f.attachment.contentSha256, bytes: f.source.length }, controller = new AbortController();
    await f.verifier.withVerifiedFile(binding, async () => undefined, async file => {
      const bytes = Buffer.alloc(7); await file.read(bytes, 0, 7, 0); return bytes;
    });
    await expect(f.verifier.withVerifiedFile(binding, async () => undefined, async file => {
      const bytes = Buffer.alloc(7); await file.read(bytes, 0, 7, 7); controller.abort(); return bytes;
    }, controller.signal)).rejects.toThrow(/abort/i);
    expect(f.verifier.statistics.cacheHits).toBe(cachePossible ? 1 : 0);
    expect(f.verifier.statistics.proofs).toBe(0);
  });

  it("retains a bounded full-SHA proof across successive Service-created stores without rehashing each page", async () => {
    const f = await fixture(), originalHash = hash(await readFile(f.path));
    const pages = [];
    for (let offset = 0; offset < 28; offset += 7) pages.push(await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, offset, 7));
    expect(pages.map(page => page.content).join("")).toBe(f.source.subarray(0, 28).toString());
    expect(pages.every(page => page.attachment.contentSha256 === f.attachment.contentSha256 && page.attachment.ingestionStarted === false)).toBe(true);
    expect(f.verifier.statistics.fullHashes).toBe(cachePossible ? 1 : 4);
    expect(f.verifier.statistics.cacheHits).toBe(cachePossible ? 3 : 0);
    expect(f.verifier.statistics.hashedBytes).toBe(f.source.length * (cachePossible ? 1 : 4));
    expect(f.verifier.statistics.proofBytes).toBeLessThanOrEqual(256 * 1024);
    expect(hash(await readFile(f.path))).toBe(originalHash);
  });

  it("rehashes a replaced same-content inode, and detects changed bytes even after mtime restoration", async () => {
    const f = await fixture(); await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7);
    const original = await lstat(f.path), moved = `${f.path}.old`;
    await rename(f.path, moved); await writeFile(f.path, f.source);
    expect((await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 7, 7)).content).toBe(f.source.subarray(7, 14).toString());
    expect(f.verifier.statistics.fullHashes).toBe(2);
    await writeFile(f.path, Buffer.alloc(f.source.length, 120)); await utimes(f.path, original.atime, original.mtime);
    await expect(f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7)).rejects.toThrow(/integrity validation/);
    expect(f.verifier.statistics.fullHashAttempts).toBe(3); expect(f.verifier.statistics.proofs).toBe(0);
  });

  it.skipIf(process.platform === "win32")("refuses even a same-byte symbolic link instead of following it", async () => {
    const f = await fixture(); await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7);
    const real = `${f.path}.real`; await rename(f.path, real); await symlink(real, f.path);
    await expect(f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7)).rejects.toThrow(/safe regular file|symbolic|ELOOP/);
  });

  it("closes and refuses a pathname replacement between the real full hash and page read", async () => {
    const f = await fixture(); let opened: FileHandle | undefined;
    const realHash = f.verifier.hashOpened.bind(f.verifier);
    vi.spyOn(f.verifier, "hashOpened").mockImplementation(async (binding, file, identity, signal) => {
      opened = file;
      const result = await realHash(binding, file, identity, signal);
      await rename(f.path, `${f.path}.hashed`); await writeFile(f.path, Buffer.alloc(f.source.length, 120));
      return result;
    });
    await expect(f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7)).rejects.toThrow(/changed|replaced/);
    expect(f.verifier.statistics.proofs).toBe(0);
    await expect(opened!.stat()).rejects.toThrow(/closed|EBADF/);
  });

  it("rejects a source change during the actual page operation, never returns its read result", async () => {
    const f = await fixture(); const authority = "a".repeat(64);
    await expect(f.verifier.withVerifiedFile({ authorityKey: authority, path: f.path,
      contentSha256: f.attachment.contentSha256, bytes: f.source.length }, async () => undefined, async file => {
      const output = Buffer.alloc(7); await file.read(output, 0, 7, 0);
      await writeFile(f.path, Buffer.alloc(f.source.length, 120)); return output;
    })).rejects.toThrow(/changed/);
    expect(f.verifier.statistics.proofs).toBe(0);
  });

  it("rebinds recovery/replaced registry identity and never reuses a proof for another owner", async () => {
    const f = await fixture(); await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7);
    const registry = TaskAttachmentStore.databasePath(f.root), replacement = `${registry}.replacement`;
    await copyFile(registry, replacement); await rename(replacement, registry);
    expect((await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7)).content).toBe("0123456");
    expect(f.verifier.statistics.fullHashes).toBe(2);
    await expect(f.service.readForTurn("wrong-owner", f.turnId, f.attachment.id, 0, 7)).rejects.toThrow(/another brain/);
    expect(f.verifier.statistics.fullHashes).toBe(2);
  });

  it("checks the current exact row/source SHA after hashing and before returning or learning bytes", async () => {
    const f = await fixture(), realHash = f.verifier.hashOpened.bind(f.verifier);
    vi.spyOn(f.verifier, "hashOpened").mockImplementation(async (...args) => {
      const result = await realHash(...args), database = new DatabaseSync(TaskAttachmentStore.databasePath(f.root));
      try { database.prepare("UPDATE attachments SET source_sha256=? WHERE id=?").run("0".repeat(64), f.attachment.id); }
      finally { database.close(); }
      return result;
    });
    await expect(f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7)).rejects.toThrow(/provenance checksum/);
    expect(f.verifier.statistics.proofs).toBe(0);
  });

  it("honors cancellation mid-hash, closes the actual FD and publishes no partial proof", async () => {
    const f = await fixture(), controller = new AbortController(), realHash = f.verifier.hashOpened.bind(f.verifier);
    let opened: FileHandle | undefined;
    vi.spyOn(f.verifier, "hashOpened").mockImplementation(async (binding, file, identity, signal) => {
      opened = file;
      const typed = file as unknown as { read(...args: unknown[]): Promise<{ bytesRead: number }> }, realRead = typed.read.bind(typed);
      vi.spyOn(typed, "read").mockImplementation(async (...args) => { const result = await realRead(...args); controller.abort(); return result; });
      return realHash(binding, file, identity, signal);
    });
    await expect(f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7, controller.signal)).rejects.toThrow(/abort/i);
    expect(f.verifier.statistics.fullHashAttempts).toBe(1); expect(f.verifier.statistics.fullHashes).toBe(0); expect(f.verifier.statistics.proofs).toBe(0);
    await expect(opened!.stat()).rejects.toThrow(/closed|EBADF/);
  });

  it("preserves Windows/unknown-FS reads with a full exact-FD SHA every time and no cache claim", async () => {
    for (const options of [{ platform: "win32" as const }, { strongLocalChangeTimes: async () => false }]) {
      const f = await fixture(options);
      await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 0, 7);
      await f.service.readForTurn(f.brainId, f.turnId, f.attachment.id, 7, 7);
      expect(f.verifier.statistics.fullHashes).toBe(2); expect(f.verifier.statistics.cacheHits).toBe(0); expect(f.verifier.statistics.proofs).toBe(0);
    }
  });

  it("resource refusal occurs before hash/page buffers, and metadata cache eviction never caps sources", async () => {
    const refused = await fixture({ reserveMemory: (_bytes, label) => { if (label === "Attachment SHA read buffer") throw new Error("fixture RAM reserve"); } });
    await expect(refused.service.readForTurn(refused.brainId, refused.turnId, refused.attachment.id, 0, 7)).rejects.toThrow(/RAM reserve/);
    expect(refused.verifier.statistics.fullHashes).toBe(0); expect(refused.verifier.statistics.proofs).toBe(0);
    const uncached = await fixture({ cacheBytes: 0 });
    await uncached.service.readForTurn(uncached.brainId, uncached.turnId, uncached.attachment.id, 0, 7);
    await uncached.service.readForTurn(uncached.brainId, uncached.turnId, uncached.attachment.id, 7, 7);
    expect(uncached.verifier.statistics.fullHashes).toBe(2); expect(uncached.verifier.statistics.proofBytes).toBe(0);
    const tiny = await fixture({ cacheBytes: 1024 });
    for (let index = 0; index < 3; index++) {
      const selected = await tiny.store.stageBytes(`extra-${index}`, { data: Buffer.from(`other source ${index}`), displayName: `extra-${index}.txt`, mimeType: "text/plain", kind: "file",
        source: { kind: "local-file", selectedAt: new Date().toISOString() } });
      await tiny.store.bindToTurn(`turn-${index}`, [selected.id]);
      expect((await tiny.service.readForTurn(tiny.brainId, `turn-${index}`, selected.id, 0, 7)).content).toBe("other s");
      expect(tiny.verifier.statistics.proofBytes).toBeLessThanOrEqual(1024);
    }
  });
});
