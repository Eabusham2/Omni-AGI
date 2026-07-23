import { describe, expect, it } from "vitest";
import {
  decodeFingerprint,
  encodeConcept,
  encodeFingerprint,
  seededRandom,
  similarity,
  textSeed
} from "../src/main/core/vectorSymbolic";
import { classifyIdea, extractConcepts, splitIntoIdeas } from "../src/main/core/language";

describe("vector-symbolic memory", () => {
  it("is deterministic and replayable", () => {
    expect(encodeFingerprint(["liquid", "synapse", "memory"])).toBe(
      encodeFingerprint(["liquid", "synapse", "memory"])
    );
    expect(textSeed("the same event", 3)).toBe(textSeed("the same event", 3));
    expect(Array.from({ length: 5 }, seededRandom(42))).toEqual(
      Array.from({ length: 5 }, seededRandom(42))
    );
  });

  it("keeps morphologically related concepts closer", () => {
    const learning = encodeConcept("learning");
    const learned = encodeConcept("learned");
    const unrelated = encodeConcept("volcano");

    expect(similarity(learning, learned)).toBeGreaterThan(similarity(learning, unrelated));
  });

  it("round-trips serialized fingerprints", () => {
    const fingerprint = encodeFingerprint(["working", "memory"]);
    expect(similarity(decodeFingerprint(fingerprint), decodeFingerprint(fingerprint))).toBe(1);
  });
});

describe("idea extraction", () => {
  it("extracts salient concepts without losing order", () => {
    const concepts = extractConcepts("OmniCortex connects liquid neurons with adaptive synapses.");
    expect(concepts.map((concept) => concept.key)).toEqual([
      "omnicortex",
      "connects",
      "liquid",
      "neurons",
      "adaptive",
      "synapses"
    ]);
  });

  it("splits and classifies atomic ideas", () => {
    const ideas = splitIntoIdeas(
      "My name is Ada. I prefer fuzzy answers. What did the network remember?"
    );
    expect(ideas).toHaveLength(3);
    expect(classifyIdea(ideas[0] ?? "")).toBe("knowledge");
    expect(classifyIdea(ideas[1] ?? "")).toBe("preference");
    expect(classifyIdea(ideas[2] ?? "")).toBe("question");
  });
});
