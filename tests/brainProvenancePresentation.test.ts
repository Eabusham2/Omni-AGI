import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { brainProvenancePresentation } from "../src/renderer/src/brainProvenancePresentation";

describe("native brain provenance presentation", () => {
  const nativeRuntime = {
    origin_kind: "ground-up",
    pretrained: false,
    baseFrozen: false,
    randomInitialization: { algorithm: "torch-seeded-random-v1", seed: 42 }
  };

  it("shows locally initialized OmniCortex provenance", () => {
    expect(brainProvenancePresentation({ runtimeCard: nativeRuntime })).toEqual({
      originLabel: "Locally initialized",
      compactLabel: "OmniCortex · locally initialized native core",
      ariaLabel:
        "OmniCortex · locally initialized native core. Its capabilities and adaptations are learned locally."
    });
  });

  it("never presents imported or contradictory foundation metadata as native", () => {
    expect(brainProvenancePresentation({
      provenance: { originKind: "legacy" }
    })).toBeUndefined();
    expect(brainProvenancePresentation({
      provenance: { originKind: "ground-up" },
      runtimeCard: { ...nativeRuntime, pretrained: true, baseFrozen: true }
    })).toBeUndefined();
    expect(brainProvenancePresentation({
      provenance: { originKind: "ground-up" },
      runtimeCard: {
        ...nativeRuntime,
        pretrained_text_cortex: { id: "external-base" }
      }
    })).toBeUndefined();
    expect(brainProvenancePresentation({})).toBeUndefined();
  });

  it("wires native provenance into Library, Data, and Runtime", () => {
    const app = readFileSync(resolve(import.meta.dirname, "../src/renderer/src/App.tsx"), "utf8");
    expect(app).toContain('className="brain-card__provenance"');
    expect(app).toContain('className="data-provenance-note"');
    expect(app).toContain('className="runtime-card__provenance"');
    expect(app).toContain("brainProvenancePresentation({");
  });
});
