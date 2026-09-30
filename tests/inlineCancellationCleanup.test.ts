import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { EventEmitter } from "node:events";
import { BrainService } from "../src/main/brainService";
import type { BrainRepository } from "../src/main/brainRepository";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { cleanupCancelledInlineStage } from "../src/main/inlineCancellationCleanup";

describe("post-PID-exit inline cleanup", () => {
  it("settles pending artifact cleanup only for its confirmed exited PID without opening a brain", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-inline-pid-"));
    try {
      const stage = join(directory, "engine", ".inline-imagination", "a".repeat(32));
      await mkdir(stage, { recursive: true });
      const engine = Object.assign(new EventEmitter(), {
        pid: 73,
        cancelInlineGeneration: async () => ({ requested: true, acknowledged: false })
      });
      const repository = { brainDirectory: () => directory } as unknown as BrainRepository;
      const service = new BrainService(repository, engine as unknown as EngineSupervisor);
      const acks: unknown[] = [];
      const remove = service.onInlineImaginationCancelled((ack) => acks.push(ack));
      await service.cancelInlineImagination("brain", "turn", "a".repeat(32));
      engine.emit("worker-closed", { pid: 74 });
      expect(acks).toEqual([]);
      const acknowledged = new Promise<void>((resolve) => {
        const listener = service.onInlineImaginationCancelled(() => { listener(); resolve(); });
      });
      engine.emit("worker-closed", { pid: 73 });
      await acknowledged;
      expect(acks).toEqual([{ brainId: "brain", turnId: "turn", actionId: "a".repeat(32), pid: 73 }]);
      remove();
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
  it("cleans only the exact unpromoted action and preserves the saved turn and other artifact", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-inline-cleanup-"));
    try {
      const engine = join(directory, "engine");
      const target = join(engine, ".inline-imagination", "a".repeat(32));
      const sibling = join(engine, ".inline-imagination", "b".repeat(32));
      await mkdir(target, { recursive: true });
      await mkdir(sibling, { recursive: true });
      await writeFile(join(engine, "brain.json"), "exact committed turn");
      await writeFile(join(sibling, "artifact.bin"), "unrelated artifact");
      expect(await cleanupCancelledInlineStage(directory, "a".repeat(32))).toBe(true);
      expect(await readFile(join(engine, "brain.json"), "utf8")).toBe("exact committed turn");
      expect(await readFile(join(sibling, "artifact.bin"), "utf8")).toBe("unrelated artifact");
      expect(await cleanupCancelledInlineStage(directory, "../engine")).toBe(false);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("refuses a staging parent symlink and does not delete an outside directory", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-inline-symlink-"));
    try {
      const engine = join(directory, "engine");
      const outside = join(directory, "outside");
      await mkdir(engine);
      await mkdir(join(outside, "a".repeat(32)), { recursive: true });
      await writeFile(join(outside, "a".repeat(32), "keep.bin"), "keep");
      await symlink(outside, join(engine, ".inline-imagination"), "dir");
      expect(await cleanupCancelledInlineStage(directory, "a".repeat(32))).toBe(false);
      expect(await readFile(join(outside, "a".repeat(32), "keep.bin"), "utf8")).toBe("keep");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
