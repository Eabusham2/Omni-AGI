import { describe, expect, it, vi } from "vitest";
import { BrainService } from "../src/main/brainService";
describe("read-only current tool action metadata", () => {
  it("uses the permission metadata reader without loading a brain, querying a provider or granting Off tools", async () => {
    const listToolPermissions = vi.fn(async () => [{ toolId: "runtime.vm", label: "VM", level: "off" as const, updatedAt: "2026-10-13T00:00:00Z" }]);
    const service = { listToolPermissions, externalToolSchemas: new Map(), engine: { request() { throw new Error("No worker request allowed"); } } };
    const page = await BrainService.prototype.describeTools.call(service as unknown as BrainService, "brain-a");
    expect(listToolPermissions).toHaveBeenCalledExactlyOnceWith("brain-a");
    expect(page).toMatchObject({ brainId: "brain-a", tools: [{ id: "runtime.vm", source: "builtin", actions: expect.arrayContaining(["capabilities"]) }] });
    expect(page.tools[0]).not.toHaveProperty("grant");
  });
});
