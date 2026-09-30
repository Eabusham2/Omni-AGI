import { describe, expect, it } from "vitest";
import { normalizeCompatibleArchitectureMutation } from "../src/shared/architectureMutation";

describe("compatible native architecture operation protocol", () => {
  it("admits typed depth/router/region growth without silently using expert growth", () => {
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-depth", addLayers: 2 })).toEqual({ mutation: "grow-depth", addLayers: 2 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-router", addNeurons: 64 })).toEqual({ mutation: "grow-router", addNeurons: 64 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-regions", addRegions: 3, neuronsPerRegion: 16 })).toEqual({ mutation: "grow-regions", addRegions: 3, neuronsPerRegion: 16 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-experts" })).toEqual({ mutation: "grow-experts", addExperts: 1 });
  });
  it("rejects blind width changes, unexpected axes and invalid counts", () => {
    for (const mutation of [
      { mutation: "grow-width", width: 512 },
      { mutation: "grow-depth", addLayers: 0 },
      { mutation: "grow-router", addNeurons: true },
      { mutation: "grow-regions", addRegions: 2 },
      { mutation: "grow-depth", addLayers: 2, newHeads: 8 }
    ]) expect(() => normalizeCompatibleArchitectureMutation(mutation)).toThrow();
  });
});
