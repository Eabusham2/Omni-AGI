import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { clearStagedConnectedWorkingPublication } from "../src/main/connectedWorkingPublicationStaging";

const sha256 = (value: string | Buffer): string =>
  createHash("sha256").update(value).digest("hex");

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function seedPublication(directory: string, brainId: string) {
  const executionId = sha256(`execution:${brainId}`);
  const backupRelative = `connected-executions/${executionId}/working-before/${"1".repeat(32)}.sqlite3`;
  const backup = join(directory, ...backupRelative.split("/"));
  const metadata = JSON.stringify({ brain_id: brainId, selected: "current-live-root" });
  const bytes = Buffer.from(`private working beforeimage for ${brainId}`);
  await mkdir(dirname(backup), { recursive: true });
  await writeFile(join(directory, "brain.json"), metadata);
  await writeFile(backup, bytes);
  await writeFile(
    join(directory, ".connected-working-publication.json"),
    JSON.stringify({
      format: "omni-inherited-working-publication",
      version: 1,
      brainId,
      executionId,
      beforeRootSha256: sha256(metadata),
      backup: backupRelative,
      sha256: sha256(bytes)
    })
  );
  return {
    executionId,
    backup,
    marker: join(directory, ".connected-working-publication.json"),
    metadata: join(directory, "brain.json")
  };
}

describe("private recovery staging cleanup", () => {
  const roots: string[] = [];
  afterEach(async () => {
    await Promise.all(
      roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
    );
  });

  it("clears only a validated copied marker and exact backup", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-working-stage-"));
    roots.push(root);
    const staged = join(root, `.engine-${randomUUID()}.restore`);
    const live = join(root, "engine");
    const selected = join(root, "snapshots", "chosen", "engine");
    await Promise.all([
      mkdir(staged, { recursive: true }),
      mkdir(live, { recursive: true }),
      mkdir(selected, { recursive: true })
    ]);
    const stagedState = await seedPublication(staged, "brain-a");
    const liveState = await seedPublication(live, "brain-a");
    await writeFile(join(selected, "brain.json"), '{"brain_id":"brain-a","selected":"snapshot"}');
    await writeFile(join(selected, "snapshot-only.bin"), "chosen snapshot bytes");
    const liveMarker = await readFile(liveState.marker);
    const liveBackup = await readFile(liveState.backup);
    const selectedMetadata = await readFile(join(selected, "brain.json"));
    const selectedBytes = await readFile(join(selected, "snapshot-only.bin"));

    await expect(
      clearStagedConnectedWorkingPublication(staged)
    ).resolves.toEqual({
      cleared: true,
      executionId: stagedState.executionId
    });
    expect(await exists(stagedState.marker)).toBe(false);
    expect(await exists(stagedState.backup)).toBe(false);
    expect(await readFile(stagedState.metadata, "utf8")).toContain(
      '"selected":"current-live-root"'
    );
    expect(await readFile(liveState.marker)).toEqual(liveMarker);
    expect(await readFile(liveState.backup)).toEqual(liveBackup);
    expect(await readFile(join(selected, "brain.json"))).toEqual(selectedMetadata);
    expect(await readFile(join(selected, "snapshot-only.bin"))).toEqual(selectedBytes);
  });

  it("fails closed and preserves private evidence when validation fails", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-working-invalid-"));
    roots.push(root);
    const staged = join(root, `.engine-${randomUUID()}.restore`);
    await mkdir(staged, { recursive: true });
    const state = await seedPublication(staged, "brain-b");
    const marker = JSON.parse(await readFile(state.marker, "utf8"));
    marker.sha256 = "f".repeat(64);
    await writeFile(state.marker, JSON.stringify(marker));

    await expect(
      clearStagedConnectedWorkingPublication(staged)
    ).rejects.toThrow(/checksum/i);
    expect(await exists(state.marker)).toBe(true);
    expect(await exists(state.backup)).toBe(true);
  });

  it("refuses to operate on a live engine path", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-working-live-"));
    roots.push(root);
    const live = join(root, "engine");
    await mkdir(live, { recursive: true });
    const state = await seedPublication(live, "brain-c");

    await expect(
      clearStagedConnectedWorkingPublication(live)
    ).rejects.toThrow(/private recovery staging/i);
    expect(await exists(state.marker)).toBe(true);
    expect(await exists(state.backup)).toBe(true);
  });
});
