import { afterEach, beforeEach, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { isUtf8 } from "node:buffer";
import { cp, lstat, mkdir, mkdtemp, readFile, readdir, realpath, rename, rm, symlink, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FileToolEffectError, FileToolService, type FileToolContext } from "../src/main/fileToolService";
import { mergeFileHandoffCustody, prepareFileHandoffCustody, rekeyFileHandoffCustody, validateFileHandoffCustody, withFileHandoffSnapshot } from "../src/main/fileToolHandoffCustody";
import { declaredFileCapabilityPack } from "../src/main/fileCapabilityPack";
import type { ChatTaskAttachmentHandle } from "../src/shared/taskAttachments";
import type { FileToolPage, FileToolSearchPage, TaskFileHandoff } from "../src/shared/fileTools";

let root: string, service: FileToolService, current = true, taskAllowed = true, taskScope = true;
let resourceHook: ((purpose: string) => void) | undefined;
const hash = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
const signal = () => new AbortController().signal;
const context = (extra: Partial<FileToolContext> = {}): FileToolContext => ({ policyRevision: "stable-policy-1", chatTurnId: "successor-turn",
  assertCurrent: async () => { if (!current) throw new Error("current origin/policy revoked"); }, ...extra });
const payload = Buffer.from("é€hi");
const handle: ChatTaskAttachmentHandle = { format: "omni-chat-task-attachment", version: 1, id: "attached-file", brainId: "original-owner", displayName: "Original imported file",
  kind: "file", mimeType: "text/plain", contentSha256: hash(payload), manifestSha256: "a".repeat(64), bytes: payload.length, state: "bound", turnId: "original-predecessor",
  createdAt: "2026-10-15T00:00:00Z", updatedAt: "2026-10-15T00:00:00Z", ingestionStarted: false, parametersChanged: false };
beforeEach(async () => {
  root = await realpath(await mkdtemp(join(tmpdir(), "omni-file-tools-"))); current = taskAllowed = taskScope = true; resourceHook = undefined;
  service = new FileToolService({ root: async id => { if (id !== "brain") throw new Error("wrong brain owner"); return root; },
    attachmentPermission: async () => { if (!taskAllowed) throw new Error("task attachment Off"); return "stable-task-policy-1"; },
    assertTaskSource: async (_brain, turn, source) => { if (!taskScope || turn !== "successor-turn" || source.attachmentId !== handle.id || source.manifestSha256 !== handle.manifestSha256) throw new Error("task scope retired"); },
    readTask: async (_brain, turn, id, offset, maximum) => {
      if (!taskScope || turn !== "successor-turn" || id !== handle.id) throw new Error("outside actual continued task");
      const selected = payload.subarray(offset, Math.min(payload.length, offset + maximum)); let bytes = selected;
      for (let trim = 0; trim <= Math.min(3, Math.max(0, selected.length - 1)); trim++) { const candidate = selected.subarray(0, selected.length - trim); if (isUtf8(candidate)) { bytes = candidate; break; } }
      return { format: "omni-chat-task-attachment-read", version: 1, attachment: handle, offset, returnedBytes: bytes.length, totalBytes: payload.length,
        encoding: isUtf8(bytes) ? "utf8" : "base64", content: bytes.toString(isUtf8(bytes) ? "utf8" : "base64"), ...(offset + bytes.length < payload.length ? { nextOffset: offset + bytes.length } : {}) };
    }, checkResources: async (_brain, bytes, purpose) => { expect(bytes).toBeGreaterThan(0); resourceHook?.(purpose); }, reserveWrite: async () => undefined });
});
afterEach(async () => { await service.close(); await rm(root, { recursive: true, force: true }); });
const execute = (action: string, args: Record<string, unknown>, requestId = action, callContext = context(), callSignal = signal()) => service.execute("brain", action, args, requestId, callSignal, callContext);
const bytes = (page: FileToolPage) => Buffer.from(page.content, page.encoding === "base64" ? "base64" : "utf8");

it("streams a complete file with exact segment/page hashes and no whole-file result", async () => {
  const path = join(root, "large-owned-file.bin"), source = Buffer.alloc(150_003, 0xff); source.write("owned", 65_535); await writeFile(path, source);
  let page = await execute("read-all", { path, maxBytes: 65536 }) as FileToolPage;
  const pieces = [bytes(page)]; expect(page.bytes).toBe(65536); expect(page.sha256).toBe(hash(pieces[0]!)); expect(page.wholeSha256).toBeUndefined();
  while (page.nextCursor) { page = await execute("read-next", { cursor: page.nextCursor, path }, "page-" + pieces.length) as FileToolPage; pieces.push(bytes(page)); }
  expect(Buffer.concat(pieces)).toEqual(source); expect(page.complete).toBe(true); expect(page.wholeSha256).toBe(hash(source));
  const segment = await execute("read-segment", { path, offset: 65_535, maxBytes: 5, expectedFileIdentity: page.source.identity }) as FileToolPage;
  expect(segment.content).toBe("owned"); expect(segment.wholeSha256).toBeUndefined(); expect(segment.nextCursor).toBeUndefined();
});

it("continues under a NEW current action when the first origin retired, but not under changed policy/task", async () => {
  const path = join(root, "owned.txt"); await writeFile(path, "abcdef"); let retired = false, firstCalls = 0, nextCalls = 0;
  const first = context({ assertCurrent: async () => { firstCalls++; if (retired) throw new Error("first action retired"); } });
  const page = await execute("read-all", { path, maxBytes: 2 }, "first", first) as FileToolPage;
  retired = true; const originalCalls = firstCalls;
  const next = await execute("read-next", { cursor: page.nextCursor, path }, "next", context({ assertCurrent: async () => { nextCalls++; } })) as FileToolPage;
  expect(next.content).toBe("cd"); expect(nextCalls).toBeGreaterThan(0); expect(firstCalls).toBe(originalCalls);
  await expect(execute("read-next", { cursor: next.nextCursor }, "wrong-task", context({ chatTurnId: "unrelated-turn" }))).rejects.toThrow("owner/task");
  await expect(execute("read-next", { cursor: next.nextCursor }, "revoked", context({ policyRevision: "new-policy" }))).rejects.toThrow("policy");
  await expect(execute("read-next", { cursor: next.nextCursor }, "old-policy")).rejects.toThrow("expired");
});

it("finds exact cross-window and overlapping content matches in bounded result/scan pages", async () => {
  const path = join(root, "content.txt"), source = Buffer.concat([Buffer.alloc(65534, 120), Buffer.from("ababab"), Buffer.alloc(100, 120)]); await writeFile(path, source);
  let page = await execute("search", { path, mode: "content", query: "abab", pageSize: 1, scanBytes: 65536 }) as FileToolSearchPage;
  const matches = [...page.matches]; let calls = 1;
  while (page.nextCursor) { expect(calls++).toBeLessThan(10); page = await execute("search-next", { cursor: page.nextCursor, path }, "search-page-" + calls) as FileToolSearchPage; matches.push(...page.matches); }
  expect(matches.map(row => row.byteOffset)).toEqual([65534, 65536]); expect(page.complete).toBe(true);
  await mkdir(join(root, "directory")); await writeFile(join(root, "directory", "REPORT.txt"), "name only"); await writeFile(join(root, "directory", "other.txt"), "other");
  let names = await execute("search", { path: join(root, "directory"), mode: "name", query: "report", pageSize: 1 }) as FileToolSearchPage;
  const found = [...names.matches]; while (names.nextCursor) { names = await execute("search-next", { cursor: names.nextCursor, path: join(root, "directory") }, "name-tail") as FileToolSearchPage; found.push(...names.matches); }
  expect(found.map(row => row.name)).toEqual(["REPORT.txt"]); expect(found[0]?.byteOffset).toBeUndefined();
});

it("rejects symlinks, replaced sources, forged cursors, path swapping and revoke-during-resource awaits", async () => {
  const path = join(root, "owned.txt"), linkPath = join(root, "link.txt"); await writeFile(path, "abcd"); await symlink(path, linkPath);
  await expect(execute("read", { path: linkPath })).rejects.toThrow("symbolic");
  const page = await execute("read-all", { path, maxBytes: 1 }) as FileToolPage;
  await expect(execute("read-next", { cursor: page.nextCursor, path: join(root, "different") })).rejects.toThrow("differs");
  await expect(execute("read-next", { cursor: page.nextCursor!.slice(0, -1) + "z" })).rejects.toThrow("host-owned");
  // Replace the actual named file, rather than depending on a same-size write
  // advancing the filesystem clock on a fast Windows runner.
  const replacement = join(root, "replacement.txt"), original = await lstat(path, { bigint: true });
  await writeFile(replacement, "wxyz"); await rename(replacement, path);
  expect((await lstat(path, { bigint: true })).ino).not.toBe(original.ino);
  await expect(execute("read-next", { cursor: page.nextCursor })).rejects.toThrow("changed");
  resourceHook = purpose => { if (purpose === "bounded file page and encoding") current = false; };
  await expect(execute("read-all", { path })).rejects.toThrow("revoked");
});

it("rejects in-place metadata changes without depending on an elapsed wall-clock tick", async () => {
  const path = join(root, "in-place.txt"); await writeFile(path, "abcd");
  const page = await execute("read-all", { path, maxBytes: 1 }) as FileToolPage;
  const before = await lstat(path, { bigint: true });
  await writeFile(path, "wxyz");
  await utimes(path, Number(before.atimeMs) / 1000, Number(before.mtimeMs) / 1000 + 10);
  const after = await lstat(path, { bigint: true });
  expect(after.ino).toBe(before.ino); expect(after.size).toBe(before.size);
  expect(after.mtimeNs).not.toBe(before.mtimeNs);
  await expect(execute("read-next", { cursor: page.nextCursor })).rejects.toThrow("changed");
});

it("reuses exact continued task reads including UTF-8-short pages, and materializes once without executing", async () => {
  let page = await execute("read-all", { attachmentId: handle.id, maxBytes: 4 }) as FileToolPage;
  const pieces = [bytes(page)]; expect(page.bytes).toBe(2);
  while (page.nextCursor) { page = await execute("read-next", { cursor: page.nextCursor }, "task-page-" + pieces.length) as FileToolPage; pieces.push(bytes(page)); }
  expect(Buffer.concat(pieces)).toEqual(payload); expect(page.wholeSha256).toBe(hash(payload));
  const first = await execute("materialize-task", { attachmentId: handle.id }, "handoff") as TaskFileHandoff & { path: string; executionStarted: false };
  expect(await readFile(first.path)).toEqual(payload); expect(first.executionStarted).toBe(false); expect(first.authority).toBe("historical-not-live");
  const inode = (await lstat(first.path, { bigint: true })).ino;
  const duplicate = await execute("materialize-task", { attachmentId: handle.id }, "handoff") as typeof first;
  expect(duplicate.sha256).toBe(first.sha256); expect((await lstat(first.path, { bigint: true })).ino).toBe(inode);
  taskAllowed = false; await expect(execute("read", { attachmentId: handle.id })).rejects.toThrow("Off");
});

it("preserves published-copy evidence when cancellation arrives after publication", async () => {
  const controller = new AbortController();
  resourceHook = purpose => { if (purpose === "bounded file handle/read scratch") controller.abort(new Error("late Stop")); };
  let failure: unknown;
  try { await execute("materialize-task", { attachmentId: handle.id }, "late-copy", context(), controller.signal); } catch (error) { failure = error; }
  expect(failure).toBeInstanceOf(FileToolEffectError);
  expect((failure as FileToolEffectError).receipt).toMatchObject({ copyPublished: true, executionStarted: false, authority: "historical-not-live" });
  const copied = (await readdir(join(root, "attachments", "handoffs"))).find(name => name.endsWith(".data"))!;
  expect(await readFile(join(root, "attachments", "handoffs", copied))).toEqual(payload);
});

it("preserves exact receipts/data through dormant rekey/full merge and rejects orphan custody without mutating parent source", async () => {
  const first = await execute("materialize-task", { attachmentId: handle.id }, "portable-handoff") as TaskFileHandoff & { path: string };
  const base = join(root, "attachments", "handoffs"), receiptName = (await readdir(base)).find(name => name.endsWith(".json"))!, raw = await readFile(join(base, receiptName));
  const sourceNames = (await readdir(base)).sort();
  const clone = join(root, "clone"), target = join(root, "merged"); await mkdir(clone); await mkdir(target);
  await cp(join(root, "attachments"), join(clone, "attachments"), { recursive: true });
  const options = { reserve: async () => undefined, admit: async () => undefined };
  await rekeyFileHandoffCustody(clone, "new-owner", options);
  expect(await readFile(join(clone, "attachments", "handoffs", receiptName))).toEqual(raw);
  expect(await readFile(join(clone, ...first.relativePath.split("/")))).toEqual(payload);
  await mergeFileHandoffCustody(target, "merged-owner", root, "brain", options);
  expect((await readdir(base)).sort()).toEqual(sourceNames); // No parent alias/history writes.
  expect(await readFile(join(base, receiptName))).toEqual(raw);
  expect(await readFile(join(target, ...first.relativePath.split("/")))).toEqual(payload);
  const aliasName = (await readdir(join(target, "attachments", "handoffs", "custody")))[0]!;
  expect(JSON.parse(await readFile(join(target, "attachments", "handoffs", "custody", aliasName), "utf8"))).toMatchObject({ ownerBrainId: "merged-owner", sourceBrainId: "brain", state: "completed-historical", authority: "historical-not-live" });
  await mkdir(join(base, "custody"), { recursive: true }); await writeFile(join(base, "custody", "f".repeat(64) + ".json"), JSON.stringify({ format: "omni-task-file-handoff-custody" }));
  await expect(validateFileHandoffCustody(root, "brain", options)).rejects.toThrow();
});

it("explicitly retains interrupted staging, never calls it completed, and admits only declared file capabilities", async () => {
  await execute("materialize-task", { attachmentId: handle.id }, "interrupted-source");
  const base = join(root, "attachments", "handoffs"), name = "e".repeat(64) + ".data.11111111-1111-4111-8111-111111111111.next";
  await writeFile(join(base, name), "partial exact bytes");
  await withFileHandoffSnapshot(root, () => prepareFileHandoffCustody(root, "brain", { reserve: async () => undefined, admit: async () => undefined }));
  const note = JSON.parse(await readFile(join(base, "interrupted", hash(name) + ".json"), "utf8"));
  expect(note).toMatchObject({ state: "interrupted-unattributed", usableAsCompletedHandoff: false, sha256: hash("partial exact bytes"), authority: "historical-not-live" });
  expect(await readFile(join(base, name), "utf8")).toBe("partial exact bytes");
  const pack = declaredFileCapabilityPack(); expect(pack.executedHostActions).toBe(false);
  expect(pack.schemas[0]?.actions).toEqual(expect.arrayContaining(["read-segment", "read-all", "search", "materialize-task"]));
});

it("rejects orphan interrupted custody and binds every declared continuation using codepoint canonical hashes", async () => {
  const base = join(root, "attachments", "handoffs"), staging = "f".repeat(64) + ".data.11111111-1111-4111-8111-111111111111.next";
  await mkdir(join(base, "interrupted"), { recursive: true });
  await writeFile(join(base, "interrupted", hash(staging) + ".json"), JSON.stringify({ format: "omni-interrupted-file-handoff", version: 1,
    ownerBrainId: "brain", originalRelativePath: "attachments/handoffs/" + staging, bytes: 1, sha256: hash("x"),
    state: "interrupted-unattributed", usableAsCompletedHandoff: false, authority: "historical-not-live" }));
  const names = (await readdir(base)).sort();
  await expect(validateFileHandoffCustody(root, "brain", { reserve: async () => undefined, admit: async () => undefined })).rejects.toThrow();
  expect((await readdir(base)).sort()).toEqual(names);
  const pack = declaredFileCapabilityPack(), { sha256, ...body } = pack;
  const sorted = (value: unknown): unknown => Array.isArray(value) ? value.map(sorted) : value && typeof value === "object"
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => [key, sorted(item)])) : value;
  expect(sha256).toBe(hash(JSON.stringify(sorted(body))));
  for (const action of ["list", "list-next", "read", "read-segment", "read-all", "read-next", "search", "search-next", "close", "materialize-task"])
    expect(pack.records.some(record => record.action === action)).toBe(true);
  expect(pack.records.every(record => !("result" in record) && !("output" in record))).toBe(true);
  expect(pack.executedHostActions).toBe(false);
});
