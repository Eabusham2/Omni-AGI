import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");

function read(relativePath: string): string {
  return readFileSync(resolve(root, relativePath), "utf8");
}

describe("native OmniCortex Build boundary", () => {
  it("does not expose foundation catalog or legacy creation selectors", () => {
    const types = read("src/shared/types.ts");
    const channels = read("src/shared/ipc.ts");
    const preload = read("src/preload/index.ts");
    const ipc = read("src/main/ipc.ts");
    const createRequest = types
      .split("export interface CreateBrainRequest {")[1]
      ?.split("\n}")[0] ?? "";
    for (const value of [types, channels, preload, ipc]) {
      expect(value).not.toContain("listFoundationModels");
    }
    expect(createRequest).not.toMatch(/starterUrl|foundationModelId|foundationRiskAcknowledged|origin\?/);
  });

  it("creates only a locally initialized native core", () => {
    const renderer = read("src/renderer/src/App.tsx");
    const service = read("src/main/brainService.ts");
    expect(renderer).toContain("Every new mind initializes its native core locally");
    expect(renderer).not.toContain('origin: "starter"');
    expect(renderer).not.toContain('foundationModelId: "none"');
    expect(service).not.toContain('const foundationModelId = "none" as const');
    expect(service).toContain('origin: "ground-up" as const');
    expect(service).not.toContain('"foundation.list"');
  });
});
