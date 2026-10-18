/** Opaque artifact/receipt portability; no assembler, native process or model run. */
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Ajv from "ajv";
import { afterEach, describe, expect, it } from "vitest";
import { assemblyArtifactPage, mergeAssemblyWorkspace, validateAssemblyWorkspace } from "../src/main/assemblyPortability";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { declaredAssemblyCapabilityPack } from "../src/main/assemblyCapabilityPack";
const roots: string[] = [], sha = (value: string) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-assembly-copy-")); roots.push(root); return root; }
async function seed(root: string, marker: string, id = "a".repeat(32)) {
  const path = join(root, "assembly", id); await mkdir(path, { recursive: true }); const source = `.text\n// ${marker}`, binary = "SQLite format 3\0opaque machine bytes " + marker;
  const receipt = { format: "omni-assembly-receipt", version: 1, artifactId: id, operation: "compile", state: "compiled", kind: "flat-binary", architecture: "x64", platform: "linux", directory: path, receiptPath: join(path, "compile.json"), sourceSha256: sha(source), partialFiles: [{ name: "source.asm", bytes: Buffer.byteLength(source), sha256: sha(source) }, { name: "program.bin", bytes: Buffer.byteLength(binary), sha256: sha(binary) }], artifact: { name: "program.bin", bytes: Buffer.byteLength(binary), sha256: sha(binary) }, processes: [], output: { brainId: "opaque-user-field", private: marker } };
  await writeFile(join(path, "source.asm"), source); await writeFile(join(path, "program.bin"), binary); await writeFile(join(path, "compile.json"), JSON.stringify(receipt)); await writeFile(join(path, "useful-result-wal"), marker);
  return { id, path, receipt, binary };
}
describe("native-code files and declared teaching", () => {
  it("preserves raw machine bytes, private receipts and journal-shaped output names through complete export/import", async () => {
    const source = await directory(), packaged = await directory(), target = await directory(), saved = await seed(source, "private-raw-output");
    const archive = await snapshotLearningWorkspace(source, packaged, "source-brain"); await adoptLearningWorkspace(archive!, target, "copied-brain"); await validateAssemblyWorkspace(target);
    expect(await readFile(join(target, "assembly", saved.id, "program.bin"), "utf8")).toBe(saved.binary);
    expect(await readFile(join(target, "assembly", saved.id, "useful-result-wal"), "utf8")).toBe("private-raw-output");
    expect(await readFile(join(target, "assembly", saved.id, "compile.json"), "utf8")).toBe(await readFile(join(saved.path, "compile.json"), "utf8"));
  });
  it("deduplicates proven same-ID bytes, preserves divergent raw receipts under a bound local alias, and does not replay code", async () => {
    const first = await directory(), second = await directory(), target = await directory(); const a = await seed(first, "first"), b = await seed(second, "second");
    await mergeAssemblyWorkspace(first, target, "parent-a"); await mergeAssemblyWorkspace(first, target, "parent-a"); await mergeAssemblyWorkspace(second, target, "parent-b"); await mergeAssemblyWorkspace(second, target, "parent-b");
    const page = await assemblyArtifactPage(target); expect(page.total).toBe(2);
    const alias = page.entries.find(row => row.artifactId !== a.id)!.artifactId as string; expect(page.entries.find(row => row.artifactId === alias)?.originalArtifactId).toBe(b.id);
    expect(await readFile(join(target, "assembly", alias, "compile.json"), "utf8")).toBe(await readFile(join(b.path, "compile.json"), "utf8"));
    expect(JSON.parse(await readFile(join(target, "assembly", alias, "binding.json"), "utf8"))).toMatchObject({ artifactId: alias, originalArtifactId: b.id, parentId: "parent-b" });
  });
  it("uses bounded pages without limiting stored artifact count and preserves interrupted compiler input as data only", async () => {
    const root = await directory(); for (let index = 1; index <= 65; index++) { const id = index.toString(16).padStart(32, "0"); await mkdir(join(root, "assembly", id), { recursive: true }); await writeFile(join(root, "assembly", id, "source.s"), "unfinished input"); }
    await validateAssemblyWorkspace(root); const first = await assemblyArtifactPage(root, "", 50); expect(first.total).toBe(65); expect(first.entries).toHaveLength(50); expect(first.entries[0]).toMatchObject({ state: "incomplete", nativeProcessResumable: false }); expect((await assemblyArtifactPage(root, first.nextAfter, 50)).entries).toHaveLength(15);
  });
  it("validates a separately versioned honest declared pack without touching frozen origin or executing placeholders", () => {
    const pack = declaredAssemblyCapabilityPack(), ajv = new Ajv({ strict: false }); expect(pack.id).toBe("native-assembly-tools-1"); expect(pack.executedHostActions).toBe(false); expect(pack.rewardModel).toBe(false); expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/);
    for (const row of pack.records) { const schema = pack.schemas.find(schema => schema.id === row.toolId)!; expect(ajv.compile(schema.actionInputSchemas[row.action!]!)(row.arguments)).toBe(true); }
    for (const schema of pack.schemas) for (const action of schema.actions) expect(pack.records.some(row => row.action === action)).toBe(true);
  });
});
