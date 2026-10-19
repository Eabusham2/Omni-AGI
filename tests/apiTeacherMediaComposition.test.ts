/** Exact tiny custody composition, not native assembly, learning or provider proof. */
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, expect, it } from "vitest";
import { pausedTeacherMediaFixture } from "./fixtures/apiTeacherMedia";
import { snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { validateApiTeacherMediaArchiveSources } from "../src/main/apiTeacherMediaArchive";
import { captureApiTeacherWorkspace, rekeyApiTeacherWorkspace, validateApiTeacherWorkspace } from "../src/main/apiTeacherPortability";
import { ApiTeacherStore } from "../src/main/apiTeacherStore";
import { FullBrainMergeCoordinator, type FullMergeHooks } from "../src/main/fullBrainMerge";
import { adoptFullMergeLearningWorkspace } from "../src/main/fullMergeLearningWorkspace";
import { sealFullMergeParent } from "../src/main/fullMergeHost";
import { DEFAULT_CONFIG } from "../src/shared/types";
import type { StreamingZipSource } from "../src/main/streamingZip";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function temporary() { const root = await mkdtemp(join(tmpdir(), "omni-teacher-composition-")); roots.push(root); return root; }

it("validates typed teacher origin/history/opaque connected inventories and rejects missing parent assets", async () => {
  const root = await temporary(), source = join(root, "source"); await mkdir(source);
  const f = await pausedTeacherMediaFixture(source, "source", "opaque");
  const workspace = await snapshotLearningWorkspace(source, join(root, "workspace"), "source");
  const files: Array<[string, StreamingZipSource]> = [];
  for (const prefix of ["origin/learning-workspace", "snapshots/point/learning-workspace", "merge/parents/parent/portable-learning-workspace"])
    for (const [name, path] of workspace!.files) files.push([`${prefix}/${name}`, { name, sourcePath: path }]);
  const stage = join(root, "validate"); await mkdir(stage);
  const admissions: string[] = [];
  expect(await validateApiTeacherMediaArchiveSources(files, stage, { admitMetadata: async (_bytes, purpose) => { admissions.push(purpose); } })).toBe(true);
  expect(admissions).toContain("teacher archive SQLite inventory cache"); expect(admissions).toContain("teacher authenticated JSON-row parsing");
  const brokenStage = join(root, "broken"); await mkdir(brokenStage);
  await expect(validateApiTeacherMediaArchiveSources(files.filter(([name]) => name !== `merge/parents/parent/portable-learning-workspace/blobs/${f.imageSha256}.data`), brokenStage))
    .rejects.toThrow(/missing.*teacher custody blob/);
});

it("reads, copies, and adopts exact historical teacher custody through deep managed paths without reviving source authority", async () => {
  const root = await temporary(), sourceRoot = join(root, "source"); await mkdir(sourceRoot);
  await pausedTeacherMediaFixture(sourceRoot, "original", "deep");
  await rekeyApiTeacherWorkspace(sourceRoot, "original", "source");
  const original = await ApiTeacherStore.open(sourceRoot, "source", async () => undefined, { create: false });
  const archive = original!.lineages()[0]!; original!.close();
  const exactArchive = await readFile(join(sourceRoot, archive.relativePath));
  let targetRoot = root;
  const minimumLength = process.platform === "win32" ? 1200 : 300;
  for (let i = 0; targetRoot.length < minimumLength; i++) targetRoot = join(targetRoot, `owned-${i}-${"x".repeat(48)}`);
  await mkdir(targetRoot, { recursive: true });
  const admissions: string[] = [];
  const admitMetadata = async (_bytes: number, purpose: string) => { admissions.push(purpose); };
  const copied = await captureApiTeacherWorkspace(sourceRoot, targetRoot, "source", { admitMetadata });
  expect(copied.has(archive.relativePath)).toBe(true);
  expect(await readFile(join(targetRoot, archive.relativePath))).toEqual(exactArchive);
  await rekeyApiTeacherWorkspace(targetRoot, "source", "copy", { admitMetadata, retireCopiedTeacherAuthority: true });
  await validateApiTeacherWorkspace(targetRoot, "copy", undefined, admitMetadata);
  const current = await ApiTeacherStore.open(targetRoot, "copy", async () => { throw new Error("Read-only check cannot admit writes."); }, { create: false });
  try {
    expect(current!.jobs().length).toBeGreaterThan(0);
    expect(current!.jobs().every(job => job.state === "dormant")).toBe(true);
  } finally { current?.close(); }
  expect(admissions).toContain("teacher custody SQLite cache");
  expect(admissions).toContain("teacher authenticated JSON-row parsing");
  expect(await readFile(join(targetRoot, archive.relativePath))).toEqual(exactArchive);
  expect(await readFile(join(sourceRoot, archive.relativePath))).toEqual(exactArchive);
});

it("threads selected metadata admission through actual full-merge snapshot/adopt/media merge and keeps sealed teacher bytes exact", async () => {
  const root = await temporary(), originals: Array<{ root: string; owner: string; imagePath: string; image: Buffer }> = [];
  for (const id of ["a", "b"]) {
    const directory = join(root, id); await mkdir(join(directory, "engine"), { recursive: true });
    await writeFile(join(directory, "brain.json"), JSON.stringify({ id })); await writeFile(join(directory, "origin.json"), JSON.stringify({ id }));
    await writeFile(join(directory, "engine", "brain.json"), JSON.stringify({ brain_id: id }));
    const f = await pausedTeacherMediaFixture(directory, id === "a" ? "original-a" : id, id);
    if (id === "a") await rekeyApiTeacherWorkspace(directory, "original-a", "a");
    originals.push({ root: directory, owner: id, imagePath: f.imagePath, image: f.image });
  }
  const source = await ApiTeacherStore.open(join(root, "a"), "a", async () => undefined, { create: false });
  const oldArchive = source!.lineages()[0]!; source!.close();
  const originalBytes = await readFile(join(root, "a", oldArchive.relativePath));
  const admissions: Array<{ phase: string; purpose: string }> = []; let phase = "snapshot";
  const hooks: FullMergeHooks = {
    withParentsFrozen: async (_ids, operation) => operation(),
    sealParent: async (id, signal) => sealFullMergeParent(join(root, id), { id, name: id } as never, signal,
      async () => undefined, async (_bytes, purpose) => { admissions.push({ phase, purpose }); }),
    validateSettings: async () => undefined, reserve: async () => undefined,
    admit: async (_bytes, purpose) => { admissions.push({ phase, purpose }); },
    assemble: async input => {
      phase = "adopt-merge";
      await mkdir(join(input.targetDirectory, "engine"), { recursive: true });
      await writeFile(join(input.targetDirectory, "engine", "brain.json"), JSON.stringify({ brain_id: input.targetBrainId, connected_executions: { entries: [] } }));
      await adoptFullMergeLearningWorkspace(input);
      await writeFile(join(input.targetDirectory, "brain.json"), JSON.stringify({ id: input.targetBrainId }));
    },
    verify: async input => { await validateApiTeacherWorkspace(input.targetDirectory, input.targetBrainId); }
  };
  const coordinator = new FullBrainMergeCoordinator(root, hooks), done = await coordinator.start({ operationId: "teacher-media-merge", parentIds: ["a", "b"],
    name: "Joined", config: { ...DEFAULT_CONFIG, name: "Joined" } });
  expect(done.state).toBe("complete");
  for (const scope of ["snapshot", "adopt-merge"]) expect(admissions.some(row => row.phase === scope && row.purpose.includes("teacher"))).toBe(true);
  const target = join(root, done.targetBrainId), store = await ApiTeacherStore.open(target, done.targetBrainId, async () => undefined, { create: false });
  try { expect(store!.jobs()).toHaveLength(2); expect(store!.jobs().every(job => job.state === "dormant")).toBe(true); }
  finally { store?.close(); }
  expect(await readFile(join(target, "merge", "parents", "a", oldArchive.relativePath))).toEqual(originalBytes);
  expect(await readFile(join(root, "a", oldArchive.relativePath))).toEqual(originalBytes);
  for (const f of originals) expect(await readFile(join(target, f.imagePath))).toEqual(f.image);
});
