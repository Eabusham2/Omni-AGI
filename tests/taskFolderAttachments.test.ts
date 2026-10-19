/** Real tiny owned files/SQLite; fake repository metadata only. No brain or model. */
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { cp, mkdir, mkdtemp, readdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import type { BrainRepository } from "../src/main/brainRepository";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { TaskAttachmentReadVerifier } from "../src/main/taskAttachmentReadVerifier";
import { TaskContinuationAttachments } from "../src/main/taskContinuationAttachments";
import { rekeyTaskAttachmentWorkspace, validateTaskAttachmentWorkspace } from "../src/main/taskAttachmentPortability";
import { TASK_FOLDER_MIME } from "../src/main/taskFolderContainer";
import type { TaskFolderEntry } from "../src/shared/taskAttachments";

const roots: string[] = [];
afterEach(async () => { vi.restoreAllMocks(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture() {
  const temporary = await mkdtemp(join(tmpdir(), "omni-saved-task-folder-")); roots.push(temporary);
  const root = join(temporary, "owner"), source = join(temporary, "selected-folder"), brainId = "folder-owner";
  await mkdir(source); await mkdir(root);
  const learn = vi.fn();
  const repository = { get: async (id: string) => { if (id !== brainId) throw new Error("Wrong owner"); return { id }; },
    brainDirectory: () => root, ingest: learn } as unknown as BrainRepository;
  const verifier = new TaskAttachmentReadVerifier({ strongLocalChangeTimes: async () => true });
  const service = new TaskAttachmentService(repository, verifier);
  return { temporary, root, source, brainId, repository, service, learn };
}

it("stages every folder record into one immutable container and pages exact retained members without learning", async () => {
  const f = await fixture(); await mkdir(join(f.source, "nested")); await mkdir(join(f.source, "empty"));
  const binary = Buffer.alloc(130_000); for (let index = 0; index < binary.length; index++) binary[index] = index % 251;
  await writeFile(join(f.source, "nested", "original.txt"), "original retained words");
  await writeFile(join(f.source, "binary.bin"), binary); await writeFile(join(f.source, "zero.txt"), "");
  await Promise.all(Array.from({ length: 130 }, (_, index) => writeFile(join(f.source, `item-${index}.txt`), `item ${index}`)));
  const [attachment] = await f.service.stagePaths(f.brainId, "folder-stage", [{ path: f.source }], "folder");
  expect(attachment).toMatchObject({ kind: "file", mimeType: TASK_FOLDER_MIME, ingestionStarted: false, parametersChanged: false, state: "staged" });
  expect(JSON.stringify(attachment)).not.toContain(f.source); expect(f.learn).not.toHaveBeenCalled();
  expect(await f.service.requestHandles(f.brainId, "folder-stage")).toEqual([attachment]);
  await f.service.bindToTurn(f.brainId, "folder-turn", [attachment!.id]);
  await rm(f.source, { recursive: true });
  const entries: TaskFolderEntry[] = [];
  let offset: number | undefined = 0, pages = 0;
  while (offset !== undefined) {
    const page = await f.service.readForTurn(f.brainId, "folder-turn", attachment!.id, offset, 64 * 1024);
    expect(page.folder).toMatchObject({ cursorUnit: "entries", traversal: { complete: true, discoveredFiles: 133,
      processedFiles: 133, discoveredDirectories: 3, processedDirectories: 3, rejectedEntries: 0 } });
    entries.push(...(JSON.parse(page.content) as { entries: TaskFolderEntry[] }).entries);
    offset = page.nextOffset; pages++;
  }
  expect(pages).toBe(2); expect(entries).toHaveLength(135);
  const exact = await f.service.readForTurn(f.brainId, "folder-turn", attachment!.id, 65_000, 1000, undefined, "binary.bin");
  expect(Buffer.from(exact.content, "base64")).toEqual(binary.subarray(65_000, 66_000));
  expect(exact.folder?.entrySha256).toBe(createHash("sha256").update(binary).digest("hex"));
  expect(await f.service.readForTurn(f.brainId, "folder-turn", attachment!.id, 0, 8, undefined, "nested/original.txt"))
    .toMatchObject({ encoding: "utf8", content: "original", nextOffset: 8, folder: { cursorUnit: "bytes" } });
  await expect(f.service.readForTurn(f.brainId, "other-turn", attachment!.id, 0, 64, undefined, "nested/original.txt")).rejects.toThrow(/outside its exact chat task/);
  await expect(f.service.readForTurn(f.brainId, "folder-turn", attachment!.id, 0, 64, undefined, "../outside")).rejects.toThrow(/exact relative/);
  expect(await validateTaskAttachmentWorkspace(f.root, f.brainId)).toBe(true);
  const destination = join(f.temporary, "imported"); await cp(f.root, destination, { recursive: true });
  await rekeyTaskAttachmentWorkspace(destination, f.brainId, "imported-folder-owner");
  expect(await validateTaskAttachmentWorkspace(destination, "imported-folder-owner")).toBe(true);
  expect(await readFile(join(destination, "attachments", "blobs", `${attachment!.contentSha256}.data`)))
    .toEqual(await readFile(join(f.root, "attachments", "blobs", `${attachment!.contentSha256}.data`)));
});

it("carries explicit saved-member reads through warm task scopes without rebinding or original host access", async () => {
  const f = await fixture(); await writeFile(join(f.source, "note.txt"), "saved folder member");
  const [attachment] = await f.service.stagePaths(f.brainId, "warm-folder", [{ path: f.source }], "folder");
  await f.service.bindToTurn(f.brainId, "first", [attachment!.id]);
  const before = await f.service.resolveForTurn(f.brainId, "first", attachment!.id);
  const facade = new TaskContinuationAttachments(f.repository, f.service, async () => 7);
  const sha = (value: string) => createHash("sha256").update(value).digest("hex");
  await facade.transfer({ brainId: f.brainId, predecessorTurnId: "first", successorTurnId: "second",
    predecessorInputSha256: sha("first"), successorInputSha256: sha("second"), attentionEpoch: 7 }, new AbortController().signal);
  await rm(f.source, { recursive: true });
  expect(await facade.execute(f.brainId, "second", "read", { attachmentId: attachment!.id, entryPath: "note.txt" }, new AbortController().signal))
    .toMatchObject({ content: "saved folder member", attachment: { id: attachment!.id, turnId: "first" } });
  expect(await f.service.resolveForTurn(f.brainId, "first", attachment!.id)).toEqual(before);
  await expect(facade.execute(f.brainId, "unrelated", "read", { attachmentId: attachment!.id, entryPath: "note.txt" }, new AbortController().signal)).rejects.toThrow(/outside the actual continued task/);
});

it("cancels a real streaming copy and rejects symlinks without publishing partial attachments", async () => {
  const f = await fixture(); await writeFile(join(f.source, "many-bytes.bin"), Buffer.alloc(200_000, 3));
  for (const phase of ["copying", "sealing"]) {
    const requestId = `cancel-folder-${phase}`;
    let copied = 0;
    await expect(f.service.stagePaths(f.brainId, requestId, [{ path: f.source }], "folder", undefined, report => {
      copied = report.copiedBytes;
      if (phase === "copying" && copied >= 64 * 1024 || phase === "sealing" && report.complete) {
        expect(f.service.cancelStage(f.brainId, requestId)).toBe(true);
      }
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(copied).toBe(phase === "copying" ? 64 * 1024 : 200_000);
    expect(await f.service.requestHandles(f.brainId, requestId)).toBeUndefined();
    expect(await readdir(join(f.root, "attachments", "blobs"))).toEqual([]);
  }
  const outside = join(f.temporary, "outside"); await mkdir(outside); await writeFile(join(outside, "outside.txt"), "must not be read");
  await symlink(outside, join(f.source, "escape"), process.platform === "win32" ? "junction" : "dir");
  await expect(f.service.stagePaths(f.brainId, "unsafe-folder", [{ path: f.source }], "folder")).rejects.toThrow(/symlink or special/);
  expect(await f.service.list(f.brainId)).toEqual([]);
  expect(await readdir(join(f.root, "attachments", "blobs"))).toEqual([]);
});

it("rejects source mutation observed through the held file descriptor instead of attaching a mixed snapshot", async () => {
  const f = await fixture(), path = join(f.source, "changing.bin");
  await writeFile(path, Buffer.alloc(200_000, 1));
  let changed = false;
  await expect(f.service.stagePaths(f.brainId, "mutating-folder", [{ path: f.source }], "folder", undefined, report => {
    if (!changed && report.copiedBytes >= 64 * 1024) { changed = true; writeFileSync(path, Buffer.alloc(200_000, 9)); }
  })).rejects.toThrow(/file changed during staging/);
  expect(changed).toBe(true);
  expect(await f.service.list(f.brainId)).toEqual([]);
  expect(await readdir(join(f.root, "attachments", "blobs"))).toEqual([]);
});
