import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { EnvironmentStore } from "../src/main/environmentStore";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { ENVIRONMENT_RECORD_FORMAT } from "../src/shared/environments";

const roots: string[] = [], sourceId = "66666666-6666-4666-8666-666666666666", targetId = "77777777-7777-4777-8777-777777777777";
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });

describe("managed environment complete workspace archive", () => {
  it("streams exact sandbox files and adopts them dormant without a live adapter identity", async () => {
    const source = await mkdtemp(join(tmpdir(), "omni-environment-archive-source-")), archiveRoot = await mkdtemp(join(tmpdir(), "omni-environment-archive-package-")),
      target = await mkdtemp(join(tmpdir(), "omni-environment-archive-target-")); roots.push(source, archiveRoot, target);
    const store = await EnvironmentStore.open(source, sourceId), environmentId = "1".repeat(32), at = new Date().toISOString(), directory = store.instanceDirectory(environmentId);
    await mkdir(join(directory, "workspace"), { recursive: true }); await writeFile(join(directory, "workspace", "result.bin"), Buffer.from([0, 1, 2, 3]));
    store.putEnvironment({ format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: environmentId, brainId: sourceId, kind: "sandbox", name: "Archive fixture",
      adapter: "darwin-sandbox-exec", state: "stopped", revision: 2, network: "none", createdAt: at, updatedAt: at }); store.close();
    const archive = await snapshotLearningWorkspace(source, archiveRoot, sourceId); expect(archive).toBeDefined();
    await adoptLearningWorkspace(archive!, target, targetId); const adopted = await EnvironmentStore.open(target, targetId, true);
    const record = adopted.getEnvironment(environmentId)!; expect(record).toMatchObject({ brainId: targetId, state: "dormant" }); expect(record.externalId).toBeUndefined(); adopted.close();
    await expect(import("node:fs/promises").then(fs => fs.readFile(join(target, "environments", "instances", environmentId, "workspace", "result.bin"))))
      .resolves.toEqual(Buffer.from([0, 1, 2, 3]));
  });
});
