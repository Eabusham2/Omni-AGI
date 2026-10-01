import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { safeEvolutionContinuationPath, savedEvolutionArchive, savedEvolutionContinuationFiles,
  validateEvolutionContinuationFiles } from "../src/main/savedEvolutionContinuation";

vi.mock("node:os", async importActual => ({ ...await importActual<typeof import("node:os")>(), freemem: () => 8 * 1024 ** 3 }));

const roots: string[] = [], id = "a".repeat(32);
const sha = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-evolution-continuation-")); roots.push(root);
  const engine = join(root, "engine"); await mkdir(engine);
  const put = async (relative: string, contents: string | Buffer) => { const path = join(engine, ...relative.split("/")); await mkdir(dirname(path), { recursive: true }); await writeFile(path, contents); return path; };
  const anchor = Buffer.from("fixture immutable native anchor bytes"), scores = '{"category":"token","key":"actual-native-label-id","loss":2,"weight":1}\n';
  await put(`evolution-baselines/${id}.safetensors`, anchor);
  await put(`evolution-baselines/${id}.paired-baseline.jsonl`, scores);
  await put(`evolution-baselines/${id}.json`, JSON.stringify({ format: "omni-neural-evolution-baseline", candidateId: id, anchorTensorSha256: sha(anchor),
    geometryHoldouts: { pairedScores: { path: `${id}.paired-baseline.jsonl`, sha256: sha(scores) } } }));
  await put(`evolution-baselines/${id}.trained.json`, JSON.stringify({ candidateId: id, actualElapsed: 1 }));
  await put(`evolution-baselines/${id}.paired-candidate.jsonl`, scores.replace('"loss":2', '"loss":1'));
  await put(`evolution-baselines/${id}.evaluation.json`, JSON.stringify({ candidateId: id, geometryHoldouts: {
    pairedScores: { path: `${id}.paired-candidate.jsonl`, sha256: sha(scores.replace('"loss":2', '"loss":1')) } } }));
  await put(`candidates/${id}/candidate.json`, JSON.stringify({ id, kind: "neural-evolution", recursiveParentCandidateId: "parent", pendingContinuation: true }));
  await put(`candidates/${id}/stable/brain.json`, '{"ancestral":true}');
  await put(`candidates/${id}/model/engine/brain.json`, '{"actualCandidate":true}');
  await put(`candidates/${id}/geometry-insertion/brain.json`, '{"actualInsertion":true}');
  await put(`candidates/${id}/geometry-migration.json`, '{"actualMigration":true}');
  return { root, engine, put };
}
describe("inert exact evolution lineage portability", () => {
  it("inventories baseline, trained/evaluated and paired observations plus exact candidate trees without reinterpreting provenance", async () => {
    const value = await fixture(), files = await savedEvolutionContinuationFiles(value.engine);
    expect(files.size).toBe(11);
    expect(JSON.parse(await readFile(files.get(`candidates/${id}/candidate.json`)!, "utf8"))).toMatchObject({ recursiveParentCandidateId: "parent", pendingContinuation: true });
    expect([...files.keys()]).toContain(`evolution-baselines/${id}.paired-candidate.jsonl`);
  });
  it("rejects changed/missing paired observations and changed anchors rather than treating them as a promotion success", async () => {
    const value = await fixture(), files = await savedEvolutionContinuationFiles(value.engine);
    const missing = new Map(files); missing.delete(`evolution-baselines/${id}.paired-candidate.jsonl`);
    await expect(validateEvolutionContinuationFiles(missing)).rejects.toThrow("paired observations");
    await value.put(`evolution-baselines/${id}.safetensors`, "changed anchor");
    await expect(savedEvolutionContinuationFiles(value.engine)).rejects.toThrow("tensor binding");
  });
  it("keeps the unsanitized main continuation archive as inert data", async () => {
    const value = await fixture(), directory = join(value.root, "evolution"); await mkdir(directory);
    const text = JSON.stringify({ schemaVersion: 1, runs: [{ continuationRequest: { objective: "actual stored objective", raw: "do not modify" }, processMeasurement: { observed: true } }], candidates: [] });
    await writeFile(join(directory, "archive.json"), text);
    const archive = await savedEvolutionArchive(value.root); expect(await readFile(archive!, "utf8")).toBe(text);
  });
  it("rejects links and live SQLite sidecars, with no setup or evaluation execution", async () => {
    const value = await fixture(), candidate = join(value.engine, "candidates", id);
    await writeFile(join(candidate, "stable", "state.sqlite3-wal"), "live state");
    await expect(savedEvolutionContinuationFiles(value.engine)).rejects.toThrow("quiescent");
    await rm(join(candidate, "stable", "state.sqlite3-wal"));
    await symlink(join(candidate, "stable", "brain.json"), join(candidate, "stable", "redirect.json"));
    await expect(savedEvolutionContinuationFiles(value.engine)).rejects.toThrow("filesystem link");
  });
  it("accepts exact documented continuation paths but never executable/traversal/source setup payloads", () => {
    expect(safeEvolutionContinuationPath(`candidates/${id}/model/engine/packed-ternary/shards/a.raw`)).toBe(true);
    expect(safeEvolutionContinuationPath(`evolution-baselines/${id}.paired-baseline.jsonl`)).toBe(true);
    for (const path of [`candidates/${id}/model/engine/../outside`, `candidates/${id}/model/engine/setup.py`, "candidates/not-an-id/candidate.json", "arbitrary.py"]) {
      expect(safeEvolutionContinuationPath(path)).toBe(false);
    }
  });
  it("source wires the same hashed trees into fork/export/import/recovery, and import remains dormant", async () => {
    const source = await readFile(new URL("../src/main/brainRepository.ts", import.meta.url), "utf8");
    expect(source.match(/savedEvolutionContinuationFiles\(/g)!.length).toBeGreaterThanOrEqual(4);
    expect(source).toContain('`evolution/${scope}/${relative}`'); expect(source).toContain('"evolution/host/archive.json"');
    expect(source).toContain('validateEvolutionContinuationFiles(evolutionPaths[scope])');
    expect(source).toContain('imported.config.idleCognition = false');
    expect(source).toContain('await rekeyActionEvidenceJobs(targetEngine, targetBrainId)');
    expect(source).toContain('await rekeyActionEvidenceJobs(join(directory, "engine"), imported.id)');
  });
});
