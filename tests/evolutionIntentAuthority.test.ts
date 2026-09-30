import { mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { EvolutionController } from "../src/main/evolutionController";
import type { BrainRepository } from "../src/main/brainRepository";

describe("native evolution intent authority method fixtures", () => {
  it("rejects Off before creating an experiment or calling source execution", async () => {
    const repository = { get: vi.fn(async () => ({ toolPermissions: [{ toolId: "source.self-modify", level: "off" }] })) } as unknown as BrainRepository;
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) };
    const controller = new EvolutionController(repository, tools);
    const archive = vi.spyOn(controller as unknown as { mutateArchive(): Promise<unknown> }, "mutateArchive");
    await expect(controller.start({
      brainId: "fixture", objective: "typed source operation", candidateKind: "source",
      sourceEdits: [{ path: "src/fixture.ts", content: "fixture only", expectedSha256: null }]
    })).rejects.toThrow("disabled");
    expect(archive).not.toHaveBeenCalled();
    expect(tools.execute).not.toHaveBeenCalled();
  });

  it("binds a durable authorized native action intent before even a mocked worker request", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-evolve-intent-"));
    const owner = join(directory, "fixture");
    await mkdir(owner);
    const repository = {
      get: async () => ({ toolPermissions: [{ toolId: "source.self-modify", level: "ask", updatedAt: "revision" }] }),
      brainDirectory: () => owner
    } as unknown as BrainRepository;
    const controller = new EvolutionController(repository, { execute: vi.fn(), cancel: vi.fn(() => 0) });
    const archive = { runs: [] as unknown[], candidates: [] as unknown[] };
    vi.spyOn(controller as unknown as { detectLimitations(): Promise<unknown[]> }, "detectLimitations").mockResolvedValue([]);
    vi.spyOn(controller as unknown as { mutateArchive(id: string, operation: (value: typeof archive) => unknown): Promise<unknown> }, "mutateArchive")
      .mockImplementation(async (_id, operation) => operation(archive));
    const request = vi.spyOn(controller as unknown as { workerRequest(): Promise<unknown> }, "workerRequest").mockImplementation(async () => {
      const files = await readdir(join(owner, "engine", "operational-tool-intents"));
      expect(files).toHaveLength(1);
      const intent = JSON.parse(await readFile(join(owner, "engine", "operational-tool-intents", files[0]!), "utf8"));
      expect(intent).toMatchObject({ state: "authorized-before-side-effects", permission: "ask", brainId: "fixture", neuralActionId: "abcdef0123456789abcdef0123456789", chatTurnId: "turn-fixture" });
      throw new Error("fixture ends here; no model/worker operation executes");
    });
    try {
      const result = await controller.start({
        brainId: "fixture", objective: "native method fixture", candidateKind: "neural", latentReplay: true,
        provenance: { neuralActionId: "abcdef0123456789abcdef0123456789", chatTurnId: "turn-fixture" }
      });
      expect(request).toHaveBeenCalledOnce();
      expect(result.state).toBe("failed");
      expect(result.error).toContain("fixture ends here");
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
});
