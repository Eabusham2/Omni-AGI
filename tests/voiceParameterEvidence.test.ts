import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { readCommittedVoiceParameterEvidence } from "../src/main/brainRepository";
import { canonicalVoiceFixture, nativeVoiceMetadata } from "./fixtures/nativeVoiceMetadata";

const roots: string[] = [];
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-voice-evidence-")); roots.push(root); return root; }
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

describe("native speech threshold count provenance", () => {
  it("reads the native measured logical count, not advertised config size, without tensors/model loading", async () => {
    const fixture = await nativeVoiceMetadata(await directory(), 200_000_000_001);
    const before = await readFile(join(fixture.engine, "brain.json"));
    const value = await readCommittedVoiceParameterEvidence(fixture.engine, "voice-brain");
    expect(value.totalNeuralParameters).toBe(200_000_000_001);
    expect(value.mutableGenerationSha256).toBe(fixture.mutable.contentSha256);
    expect(value.substrateGenerationSha256).toBe(fixture.substrate.contentSha256);
    expect(value.parameterChecksum).toBe("b".repeat(64)); expect(value.metadataSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(await readFile(join(fixture.engine, "brain.json"))).toEqual(before);
  });
  it("uses measured whole connected count without equating it to the outer graph alone", async () => {
    const fixture = await nativeVoiceMetadata(await directory(), 300_000_000_000, { connected: true });
    expect((await readCommittedVoiceParameterEvidence(fixture.engine, "voice-brain")).totalNeuralParameters).toBe(300_000_000_000);
  });
  it("rejects missing or inconsistent native accounting rather than guessing a size", async () => {
    const fixture = await nativeVoiceMetadata(await directory(), 100);
    fixture.metadata.runtime_card.parameterAccounting.totalNeuralParameters++;
    await writeFile(join(fixture.engine, "brain.json"), canonicalVoiceFixture(fixture.metadata));
    await expect(readCommittedVoiceParameterEvidence(fixture.engine, "voice-brain")).rejects.toThrow(/unavailable or inconsistent/);
    await expect(readCommittedVoiceParameterEvidence(fixture.engine, "another-brain")).rejects.toThrow(/supported committed instance/);
  });
  it("rejects a changed manifest and an ahead-of-metadata publication pointer", async () => {
    const fixture = await nativeVoiceMetadata(await directory(), 100);
    await writeFile(join(fixture.engine, "state", "manifest.json"), canonicalVoiceFixture({ ...fixture.mutablePointer, activeGeneration: "d".repeat(64) }));
    await expect(readCommittedVoiceParameterEvidence(fixture.engine, "voice-brain")).rejects.toThrow(/still publishing/);
    await writeFile(join(fixture.engine, "state", "manifest.json"), canonicalVoiceFixture(fixture.mutablePointer));
    await writeFile(join(fixture.engine, "substrate", fixture.substratePointer.generationManifest), "{}");
    await expect(readCommittedVoiceParameterEvidence(fixture.engine, "voice-brain")).rejects.toThrow(/manifest checksum/);
  });
});
