import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { nativeExecutionFiles, snapshotNativeExecution } from "../src/main/executionCheckpointPortability";
import { rewriteNativeJsonValues } from "../src/main/nativeJsonValues";
import { collectConnectedExecutionInputs, validateConnectedExecutionArchive } from "../src/main/connectedExecutionPortability";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const hash = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
const canonical = (value: any): string => Array.isArray(value) ? `[${value.map(canonical).join(",")}]` : value && typeof value === "object"
  ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}` : JSON.stringify(value);
const map = (value: Record<string, any>): any => ({ kind: "map", entries: Object.entries(value).map(([key, item]) => [key, item && typeof item === "object" ? map(item) : item]) });
const byteTensorFile = (name: string, data: Uint8Array): Buffer => {
  const json = JSON.stringify({ [name]: { dtype: "U8", shape: [data.length], data_offsets: [0, data.length] } });
  const header = Buffer.from(json.padEnd(Math.ceil(json.length / 8) * 8, " ")), prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length)); return Buffer.concat([prefix, header, data]);
};

async function fixture(currentSeal = false) {
  const folder = await mkdtemp(join(tmpdir(), "omni-execution-portable-")); roots.push(folder);
  const engine = join(folder, "source"), target = join(folder, "target"), checkpointId = "c".repeat(32);
  await mkdir(join(engine, ".execution", checkpointId, "diagnostic"), { recursive: true }); await mkdir(target);
  const activityClock = { format: "omni-native-activity-clock", version: 1, elapsed: 12.25, paused: true };
  const metadata = { brain_id: "source", mutable_state: { activeGeneration: "d".repeat(64) },
    substrate: { persistence: { activeGeneration: "e".repeat(64) } }, fresh_attention_boundary: null,
    ...(currentSeal ? { activity_clock: activityClock } : {}) };
  await writeFile(join(engine, "brain.json"), canonical(metadata));
  await writeFile(join(target, "brain.json"), canonical({ ...metadata, brain_id: "target" }));
  const neuralFiles: Record<string, unknown> = {};
  for (const name of ["core.safetensors", "plasticity.safetensors"]) {
    const bytes = byteTensorFile("fixture", Buffer.from(name));
    await writeFile(join(engine, name), bytes); await writeFile(join(target, name), bytes);
    neuralFiles[name] = { sha256: hash(bytes), bytes: bytes.length };
  }
  const binding = { brainId: "source", turnId: "turn", mutable: metadata.mutable_state,
    substrate: metadata.substrate.persistence, coreSha256: "f".repeat(64), attentionEpoch: 0 };
  const bindingId = hash(canonical(binding)), directory = join(engine, ".execution", checkpointId);
  const database = new DatabaseSync(join(directory, "diagnostic/changed.sqlite"));
  database.exec("CREATE TABLE original (owner TEXT, block INTEGER, positions BLOB, codes BLOB, PRIMARY KEY(owner,block)) WITHOUT ROWID"); database.close();
  await writeFile(join(directory, "diagnostic/controls.safetensors"), byteTensorFile("fixture", Buffer.from("controls")));
  const diagnosticFiles: Record<string, unknown> = {};
  for (const name of ["changed.sqlite", "controls.safetensors"]) {
    const data = await readFile(join(directory, "diagnostic", name)); diagnosticFiles[name] = { sha256: hash(data), bytes: data.length };
  }
  const journal = canonical({ format: "omni-core-delta-resume-1", bindingId, includeState: false,
    owners: [], parallelTopology: [], sparseContainers: [], files: diagnosticFiles });
  await writeFile(join(directory, "diagnostic/manifest.json"), journal);
  const receipt: Record<string, string> = { format: "omni-chat-tool-observation-v1", brainId: "source", turnId: "turn", neuralActionId: "a".repeat(32),
    actionEventId: "event", executionId: "execution", toolId: "web.search", action: "search", completedAt: "2026-10-11T00:00:00Z", payloadSha256: hash('{"outputPresent":true,"output":"source is private content"}') };
  const names = ["format", "brainId", "turnId", "neuralActionId", "actionEventId", "executionId", "toolId", "action", "completedAt", "payloadSha256"];
  const observationId = hash(names.map(key => receipt[key]).join("\0")); receipt.observationId = observationId;
  const inbox = { format: "omni-chat-observation-frontier", version: 2, brainId: "source", turnId: "turn", expected: { [receipt.neuralActionId!]: ["web.search", "search"] },
    seen: [observationId], awaiting: [], pending: [], receipts: { [observationId]: receipt }, waitSeconds: 30 };
  const frontier = { checkpointId, turnId: "turn", inputSha256: hash("source is a user word"), lastStreamSequence: 2,
    sampledTokens: 2, publishedTokens: 2, phase: "native-text-token-boundary", exactSupportedResume: true };
  const worker = { brainId: "source", turnId: "turn", lastStreamSequence: 2, publishedText: "Hi", actions: [], inbox };
  const rngBytes = byteTensorFile("activity", Buffer.from([1, 2, 3, 4]));
  await writeFile(join(directory, "t-0.safetensors"), rngBytes);
  const payload = { kind: "map", entries: [["context", map({ clean: "source is a user word", observed: { observationId }, noise: .5 })],
    ["cursor", { kind: "generation", fields: map({}) }], ["continuation", map({})], ["globalRng", { kind: "map", entries: [
      ["cpu", { kind: "tensor", file: "t-0.safetensors" }], ["device", "cpu"], ["accelerator", null]] }]] };
  const value = { format: "omni-native-execution-checkpoint", version: 2, brainId: "source", checkpointId, status: "suspended", binding, bindingId,
    frontier, phase: "single-pass", minimumContextTokens: 8, worker, presentation: { input: "source is a user word", publishedText: "Hi", actions: [] },
    files: { "t-0.safetensors": { sha256: hash(rngBytes), bytes: rngBytes.length, device: "cpu", dtype: "torch.uint8", shape: [4] } }, payload,
    journal: { bindingId, manifestSha256: hash(journal) }, neuralFiles, backend: "cpu" };
  if (currentSeal) {
    // Structural file fixture only: the native runner will independently
    // require the actual machine/build declaration, never this dummy body.
    const declaration = { format: "omni-exact-runtime-compatibility", version: 1, backend: "cpu", fixtureOnly: true };
    Object.assign(value, { runtimeCompatibility: { declaration, sha256: hash(canonical(declaration)) } });
    payload.entries.push(["activityClock", map(activityClock)]);
  }
  const text = canonical(value).replace('"noise",0.5', '"noise",0.50');
  // Canonical native Python spelling, including 30.0; these small format
  // fixtures are not a trained model or a claim of native RNG validity.
  const native = text.replace('"waitSeconds":30', '"waitSeconds":30.0').replace('"noise",0.50', '"noise",0.5');
  const manifestSha256 = hash(native);
  await writeFile(join(directory, `m-${manifestSha256}.json`), native);
  await writeFile(join(engine, ".execution/active.json"), canonical({ brainId: "source", checkpointId, manifestSha256 }));
  await writeFile(join(engine, ".execution/unloaded.json"), canonical({ brainId: "source", explicitLoadRequired: true }));
  return { folder, engine, target, checkpointId, manifestSha256, observationId, receipt, names };
}

describe("authenticated native execution portability", () => {
  it("retains the new runtime seal and exact paused clock and refuses a mismatched selected clock", async () => {
    const f = await fixture(true), before = (await nativeExecutionFiles(f.engine, "source")).manifests[0]!.value;
    await snapshotNativeExecution(f.engine, f.target, { sourceBrainId: "source", targetBrainId: "target" });
    const after = (await nativeExecutionFiles(f.target, "target")).manifests[0]!.value;
    expect(after.runtimeCompatibility).toEqual(before.runtimeCompatibility);
    expect(after.payload.entries.find((entry: any) => entry[0] === "activityClock")).toEqual(before.payload.entries.find((entry: any) => entry[0] === "activityClock"));
    const metadata = JSON.parse(await readFile(join(f.target, "brain.json"), "utf8")); metadata.activity_clock.elapsed += 1;
    await writeFile(join(f.target, "brain.json"), canonical(metadata));
    await expect(nativeExecutionFiles(f.target, "target")).rejects.toThrow(/paused activity clock/);
  });
  it("preserves numeric lexemes and user words when replacing one typed identity", () => {
    const text = '{"brainId":"source","control":1.0,"nested":{"name":"source"}}';
    expect(rewriteNativeJsonValues(text, new Map([[JSON.stringify(["brainId"]), '"target"']]))).toBe('{"brainId":"target","control":1.0,"nested":{"name":"source"}}');
    expect(() => rewriteNativeJsonValues('{"x":1,"x":2}', new Map())).toThrow(/duplicate/);
  });
  it("validates and rebinds a sealed cursor/evidence baseline without copying source authority", async () => {
    const f = await fixture();
    const before = await readFile(join(f.engine, ".execution/active.json"), "utf8");
    expect((await nativeExecutionFiles(f.engine, "source")).manifests).toHaveLength(1);
    await snapshotNativeExecution(f.engine, f.target, { sourceBrainId: "source", targetBrainId: "target" });
    const copied = await nativeExecutionFiles(f.target, "target");
    const manifest = copied.manifests[0]!;
    expect(manifest.value.binding.brainId).toBe("target");
    expect(manifest.value.worker.inbox.seen).not.toContain(f.observationId);
    expect(manifest.value.payload.entries.find((pair: any) => pair[0] === "context")[1].entries.find((pair: any) => pair[0] === "clean")[1]).toBe("source is a user word");
    expect(await readFile(join(f.engine, ".execution/active.json"), "utf8")).toBe(before);
    expect(manifest.value.worker.inbox.receipts[manifest.value.worker.inbox.seen[0]].executionId).toBe("execution");
    expect(manifest.text).toContain('"noise",0.5');
  });
  it("validates an opaque archive inventory without creating a filesystem model owner", async () => {
    const f = await fixture(), source = await nativeExecutionFiles(f.engine, "source");
    const archive = await nativeExecutionFiles(join(f.folder, "never-created"), "source", { executionFiles: source.files,
      metadataPath: join(f.engine, "brain.json"), neuralPaths: { "core.safetensors": join(f.engine, "core.safetensors"),
        "plasticity.safetensors": join(f.engine, "plasticity.safetensors") } });
    expect(archive.active).toEqual(source.active); expect(archive.files.size).toBe(source.files.size);
  });
  it("preserves and authenticates original plus re-suspended regional runtime closure", async () => {
    const f = await fixture();
    for (const name of ["conversation.sqlite3", "state/replay.sqlite3", "state/working-memory.sqlite3"]) {
      await mkdir(join(f.engine, name, ".."), { recursive: true });
      const database = new DatabaseSync(join(f.engine, name)); database.exec("CREATE TABLE fixture(value TEXT)"); database.close();
    }
    const runtime = join(f.folder, "regional-runtime"), engine = join(f.folder, "engine");
    await cp(f.engine, runtime, { recursive: true }); await mkdir(engine);
    const original = (await nativeExecutionFiles(f.engine, "source")).manifests[0]!.value;
    const entry = { id: "a".repeat(64), rootTurnId: "ancestral-turn", parentId: "source", originalTurnId: "turn", regionPath: ["region"],
      sourceEnginePath: "source", sourceCheckpointId: f.checkpointId, sourceManifestSha256: f.manifestSha256,
      sourceBindingId: original.bindingId, resumeEnginePath: "regional-runtime", resumeCheckpointId: f.checkpointId,
      resumeManifestSha256: f.manifestSha256, status: "suspended", lineage: [], historyParentId: "source",
      historyHumanMessageId: "human", rootHumanMessageId: "merged-human", backend: "cpu" };
    const metadata = { brain_id: "merged", connected_executions: { format: "omni-connected-execution-queue", formatVersion: 1, entries: [entry] } };
    const closure = await collectConnectedExecutionInputs(engine, metadata);
    expect([...closure.keys()].some(name => name.startsWith("connected-execution-inputs/source/.execution/"))).toBe(true);
    expect([...closure.keys()].some(name => name.startsWith("connected-execution-inputs/regional-runtime/.execution/"))).toBe(true);
    const lookup = (name: string) => { const sourcePath = closure.get(name.slice("engine/".length)); return sourcePath ? { name, sourcePath } : undefined; };
    await expect(validateConnectedExecutionArchive(metadata, "engine", lookup)).resolves.toBeUndefined();
    closure.delete(`connected-execution-inputs/regional-runtime/.execution/${f.checkpointId}/t-0.safetensors`);
    await expect(validateConnectedExecutionArchive(metadata, "engine", lookup)).rejects.toThrow(/missing/);
  });
  it("rejects payload corruption, undeclared activity and wrong neural generation before publication", async () => {
    const f = await fixture();
    await writeFile(join(f.engine, ".execution", f.checkpointId, "t-999.safetensors"), "unexpected");
    await expect(nativeExecutionFiles(f.engine, "source")).rejects.toThrow(/unmanifested/);
    await rm(join(f.engine, ".execution", f.checkpointId, "t-999.safetensors"));
    await writeFile(join(f.engine, "core.safetensors"), "corrupt");
    await expect(snapshotNativeExecution(f.engine, f.target, { sourceBrainId: "source", targetBrainId: "target" })).rejects.toThrow(/checksum/);
  });
  it("rejects a hash-valid safetensor whose activity geometry disagrees with its descriptor", async () => {
    const f = await fixture(), directory = join(f.engine, ".execution", f.checkpointId);
    const pointer = JSON.parse(await readFile(join(f.engine, ".execution/active.json"), "utf8"));
    const priorName = `m-${pointer.manifestSha256}.json`, value = JSON.parse(await readFile(join(directory, priorName), "utf8"));
    value.files["t-0.safetensors"].shape = [5];
    const text = canonical(value), digest = hash(text);
    await rm(join(directory, priorName)); await writeFile(join(directory, `m-${digest}.json`), text);
    await writeFile(join(f.engine, ".execution/active.json"), canonical({ ...pointer, manifestSha256: digest }));
    await expect(nativeExecutionFiles(f.engine, "source")).rejects.toThrow(/geometry/);
    await expect(snapshotNativeExecution(f.engine, f.target, { sourceBrainId: "source", targetBrainId: "target" })).rejects.toThrow(/geometry/);
    await expect(readFile(join(f.target, ".execution/active.json"))).rejects.toMatchObject({ code: "ENOENT" });
  });
});
