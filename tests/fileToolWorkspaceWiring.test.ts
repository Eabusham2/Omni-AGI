import { afterEach, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import {
  adoptLearningWorkspace,
  snapshotLearningWorkspace
} from "../src/main/learningWorkspacePortability";
import {
  mergeFileHandoffCustody,
  validateFileHandoffCustody
} from "../src/main/fileToolHandoffCustody";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";

const sha256 = (value: string | Uint8Array): string =>
  createHash("sha256").update(value).digest("hex");

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

interface SeededHandoff {
  requestKey: string;
  requestSha256: string;
  intent: Buffer;
  receipt: Buffer;
  data: Buffer;
}

async function seedCompletedHandoff(root: string, brainId: string): Promise<SeededHandoff> {
  const data = Buffer.from("tiny exact imported task bytes\n", "utf8");
  const turnId = "turn-original", requestId = "request-original";
  const attachments = new TaskAttachmentStore(root, brainId);
  const staged = await attachments.stageBytes("attachment-request", {
    data,
    displayName: "tiny-source.txt",
    mimeType: "text/plain",
    kind: "file",
    source: { kind: "local-file", originalPath: "/reviewed/tiny-source.txt", selectedAt: "2026-10-15T00:00:00.000Z" }
  });
  const [bound] = await attachments.bindToTurn(turnId, [staged.id]);
  const source = {
    kind: "task-attachment",
    attachmentId: bound!.attachmentId,
    manifestSha256: bound!.manifestSha256,
    contentSha256: bound!.contentSha256,
    displayName: bound!.displayName,
    bytes: bound!.bytes,
    identity: sha256(JSON.stringify([brainId, bound!.attachmentId, bound!.contentSha256,
      bound!.manifestSha256, bound!.bytes, turnId]))
  };
  const requestKey = sha256(JSON.stringify([brainId, turnId, requestId]));
  const requestSha256 = sha256(JSON.stringify([brainId, turnId, requestId, source]));
  const relativePath = `attachments/handoffs/${requestSha256}.data`;
  const intent = Buffer.from(JSON.stringify({
    format: "omni-task-file-handoff-intent",
    version: 1,
    brainId,
    turnId,
    requestId,
    requestSha256,
    source,
    relativePath
  }));
  const receipt = Buffer.from(JSON.stringify({
    format: "omni-task-file-handoff",
    version: 1,
    brainId,
    turnId,
    requestId,
    requestSha256,
    source,
    relativePath,
    bytes: data.length,
    sha256: sha256(data),
    createdAt: "2026-10-15T00:00:00.000Z",
    authority: "historical-not-live"
  }));
  const directory = join(root, "attachments", "handoffs");
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, `${requestKey}.json.intent`), intent);
  await writeFile(join(directory, `${requestKey}.json`), receipt);
  await writeFile(join(directory, `${requestSha256}.data`), data);
  return { requestKey, requestSha256, intent, receipt, data };
}

async function exactTree(root: string, prefix = ""): Promise<Map<string, Buffer>> {
  const result = new Map<string, Buffer>();
  const entries = await readdir(root, { withFileTypes: true });
  for (const entry of entries.sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0)) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      for (const [name, bytes] of await exactTree(join(root, entry.name), relative)) result.set(name, bytes);
    } else if (entry.isFile()) result.set(relative, await readFile(join(root, entry.name)));
    else throw new Error("Fixture workspace contains a non-file entry.");
  }
  return result;
}

async function archivedFile(archiveRoot: string, logicalPath: string): Promise<Buffer> {
  const inventory = new DatabaseSync(join(archiveRoot, "inventory.sqlite3"), { readOnly: true });
  try {
    const row = inventory.prepare("SELECT sha256 FROM files WHERE path=?").get(logicalPath) as { sha256?: string } | undefined;
    if (!row?.sha256) throw new Error(`Archive is missing ${logicalPath}.`);
    return readFile(join(archiveRoot, "blobs", `${row.sha256}.data`));
  } finally {
    inventory.close();
  }
}

it("snapshots, adopts, recaptures, and merges complete file handoff custody without rewriting its source", async () => {
  const sandbox = await realpath(await mkdtemp(join(tmpdir(), "omni-file-workspace-wiring-")));
  roots.push(sandbox);
  const sourceRoot = join(sandbox, "source"), archiveRoot = join(sandbox, "archive"),
    adoptedRoot = join(sandbox, "adopted"), recapturedRoot = join(sandbox, "recaptured"),
    mergedRoot = join(sandbox, "merged");
  await Promise.all([mkdir(sourceRoot), mkdir(adoptedRoot), mkdir(mergedRoot)]);
  const seeded = await seedCompletedHandoff(sourceRoot, "source-brain");
  const hooks = { checkDisk: async (bytes: number) => { expect(bytes).toBeGreaterThanOrEqual(0); } };

  const archive = await snapshotLearningWorkspace(sourceRoot, archiveRoot, "source-brain", hooks);
  expect(archive?.brainId).toBe("source-brain");
  await adoptLearningWorkspace(archive!, adoptedRoot, "adopted-brain", hooks);

  const handoffRoot = join(adoptedRoot, "attachments", "handoffs");
  expect(await readFile(join(handoffRoot, `${seeded.requestKey}.json.intent`))).toEqual(seeded.intent);
  expect(await readFile(join(handoffRoot, `${seeded.requestKey}.json`))).toEqual(seeded.receipt);
  expect(await readFile(join(handoffRoot, `${seeded.requestSha256}.data`))).toEqual(seeded.data);
  const adoptedAlias = JSON.parse(await readFile(join(handoffRoot, "custody", `${seeded.requestKey}.json`), "utf8"));
  expect(adoptedAlias).toMatchObject({
    ownerBrainId: "adopted-brain",
    sourceBrainId: "source-brain",
    requestKey: seeded.requestKey,
    state: "completed-historical",
    authority: "historical-not-live"
  });
  expect(await validateFileHandoffCustody(adoptedRoot, "adopted-brain", {
    reserve: async () => undefined,
    admit: async () => undefined
  })).toBe(true);

  const recaptured = await snapshotLearningWorkspace(adoptedRoot, recapturedRoot, "adopted-brain", hooks);
  expect(recaptured?.brainId).toBe("adopted-brain");
  expect(await archivedFile(recapturedRoot, `attachments/handoffs/${seeded.requestKey}.json.intent`)).toEqual(seeded.intent);
  expect(await archivedFile(recapturedRoot, `attachments/handoffs/${seeded.requestKey}.json`)).toEqual(seeded.receipt);
  expect(await archivedFile(recapturedRoot, `attachments/handoffs/${seeded.requestSha256}.data`)).toEqual(seeded.data);

  const beforeMerge = await exactTree(handoffRoot);
  await mergeFileHandoffCustody(mergedRoot, "merged-brain", adoptedRoot, "adopted-brain", {
    reserve: async () => undefined,
    admit: async () => undefined
  });
  expect(await exactTree(handoffRoot)).toEqual(beforeMerge);
  expect(await readFile(join(mergedRoot, "attachments", "handoffs", `${seeded.requestKey}.json.intent`))).toEqual(seeded.intent);
  expect(await readFile(join(mergedRoot, "attachments", "handoffs", `${seeded.requestKey}.json`))).toEqual(seeded.receipt);
  expect(await readFile(join(mergedRoot, "attachments", "handoffs", `${seeded.requestSha256}.data`))).toEqual(seeded.data);
  expect(JSON.parse(await readFile(join(mergedRoot, "attachments", "handoffs", "custody", `${seeded.requestKey}.json`), "utf8")))
    .toMatchObject({ ownerBrainId: "merged-brain", sourceBrainId: "source-brain", state: "completed-historical", authority: "historical-not-live" });
});
