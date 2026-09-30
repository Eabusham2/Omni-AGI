import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const source = (name: string) => readFileSync(new URL(name, import.meta.url), "utf8");
describe("BrainMap cortical viewer source contracts (no UI/brain run)", () => {
  it("keeps committed inspection load-free and on the inspection worker", () => {
    const backend = source("../engine/omni_core/cortical_inspection.py");
    expect(backend).not.toMatch(/AdaptiveBrain\(|OmniDecoder\(/);
    expect(backend).toContain('"wholeRolePayloadScanned": False');
    expect(source("../src/main/engineSupervisor.ts")).toContain('method === "query_cortex"');
  });
  it("virtualizes viewport rows and separates numeric trits from observed activity", () => {
    const viewer = source("../src/renderer/src/BrainMapCortex.tsx");
    expect(viewer).toContain("records.slice(first, first + 16)");
    expect(viewer).toContain("activation from confidence or weight sign");
    expect(viewer).toContain('data-trit={element.value}');
    expect(viewer).toContain("firing map");
    const app = source("../src/renderer/src/App.tsx");
    expect(app).toContain("Packed cortical model");
    expect(app).not.toContain("activation: assembly.confidence");
    expect(app).toContain("assembly.activationObserved === true");
  });
});
