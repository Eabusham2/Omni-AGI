import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { expect, it, vi } from "vitest";
import type { BrainRepository } from "../src/main/brainRepository";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";

it("queries exact turns/IDs/read cards without scanning unrelated attachment history", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-task-query-")), brainId = "query-fixture";
  const store = new TaskAttachmentStore(root, brainId);
  let restoreScan: (() => void) | undefined;
  try {
    const first = await store.stageBytes("first", { data: Buffer.from("first task"), displayName: "first.txt", mimeType: "text/plain", kind: "file",
      source: { kind: "local-file", selectedAt: new Date().toISOString() } });
    const other = await store.stageBytes("other", { data: Buffer.from("other task"), displayName: "other.txt", mimeType: "text/plain", kind: "file",
      source: { kind: "local-file", selectedAt: new Date().toISOString() } });
    await store.bindToTurn("first-turn", [first.id]);
    expect((await store.list(false, "staged")).map(value => value.id)).toEqual([other.id]);
    await expect(store.list(false, "bound" as "staged")).rejects.toThrow(/state filter/);
    const database = new DatabaseSync(TaskAttachmentStore.databasePath(root));
    try {
      const plan = database.prepare("EXPLAIN QUERY PLAN SELECT * FROM attachments WHERE turn_id=? AND state='bound' ORDER BY request_id,request_index").all("first-turn");
      expect(JSON.stringify(plan)).toContain("attachments_turn");
      // Corrupt unrelated history must not be loaded as part of one exact read.
      database.prepare("UPDATE attachments SET manifest_sha256='unrelated-corrupt' WHERE id=?").run(other.id);
    } finally { database.close(); }
    const scan = vi.spyOn(TaskAttachmentStore.prototype, "list").mockRejectedValue(new Error("Full history scan forbidden in this case"));
    restoreScan = () => { scan.mockRestore(); };
    const repository = { get: async () => ({ id: brainId }), brainDirectory: () => root } as unknown as BrainRepository;
    const service = new TaskAttachmentService(repository);
    expect((await service.handlesForTurn(brainId, "first-turn")).map(value => value.id)).toEqual([first.id]);
    expect((await service.handlesByIds(brainId, [first.id]))[0]?.state).toBe("bound");
    expect((await service.readForTurn(brainId, "first-turn", first.id, 0, 5)).content).toBe("first");
    await expect(service.handlesByIds(brainId, [first.id, first.id])).rejects.toThrow(/duplicated/);
    await expect(service.handlesByIds(brainId, ["missing"])).rejects.toThrow(/unavailable/);
    expect(scan).not.toHaveBeenCalled();
  } finally { restoreScan?.(); await rm(root, { recursive: true, force: true }); }
});
