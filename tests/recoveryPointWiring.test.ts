import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

describe("Recovery Point IPC wiring", () => {
  it("uses one fail-hard checkpoint plus the storage-operation path", async () => {
    const [ipc, preload, worker] = await Promise.all([
      readFile(join(process.cwd(), "src", "main", "ipc.ts"), "utf8"),
      readFile(join(process.cwd(), "src", "preload", "index.ts"), "utf8"),
      readFile(join(process.cwd(), "engine", "worker.py"), "utf8")
    ]);

    const snapshotHandler = ipc.slice(
      ipc.indexOf("handle(IPC.brain.snapshot"),
      ipc.indexOf("handle(IPC.brain.listSnapshots")
    );
    expect(snapshotHandler).toContain(
      'storageOperations.create(operationId, "snapshot", brainId)'
    );
    expect(snapshotHandler).toContain("service.createRecoveryPoint(");
    expect(snapshotHandler).not.toContain("engine.tryRequest");

    expect(preload).toContain(
      "invoke(IPC.brain.snapshot, id, label, operationId)"
    );
    expect(worker).toContain('"checkpoint": self.checkpoint');
    expect(worker).toContain("return self._get(params).checkpoint(operation_id)");
  });

  it("requires restore acknowledgement instead of swallowing reload failure", async () => {
    const ipc = await readFile(
      join(process.cwd(), "src", "main", "ipc.ts"),
      "utf8"
    );
    const restoreHandler = ipc.slice(
      ipc.indexOf("handle(IPC.brain.restoreSnapshot"),
      ipc.indexOf("handle(IPC.brain.export")
    );
    expect(restoreHandler).toContain(
      'storageOperations.create(operationId, "restore", brainId)'
    );
    expect(restoreHandler).toContain("service.restoreRecoveryPoint(");
    expect(restoreHandler).not.toContain("engine.tryRequest");
  });
});
