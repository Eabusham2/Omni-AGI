import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { ExternalAdviceRequest, ExternalAdviceSelectedContext } from "../src/shared/externalAdvice";
import type { ActionReviewerApprovalReview } from "../src/shared/actionReviewer";
import { externalAdviceCanonical as canonical, externalAdviceSha256 as sha, EXTERNAL_ADVICE_FORMAT,
  EXTERNAL_ADVICE_RELATIVE_PATH, validateExternalAdviceWorkspace } from "../src/main/externalAdviceService";
import { mergeExternalAdviceWorkspace, rekeyExternalAdviceWorkspace, validateExternalAdvicePortabilityWorkspace } from "../src/main/externalAdvicePortability";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { copyMutableFileIsolated } from "../src/main/mutableFileIsolation";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function root(): Promise<string> { const path = await mkdtemp(join(tmpdir(), "omni-advice-portability-")); roots.push(path); return path; }
const configuration = (model = "fixture-model") => ({ enabled: true, provider: "openai", model,
  maxOutputTokens: 2048, context: { includeTask: true, includeActivity: false, maxCharacters: 32000 } });
const time = "2026-10-13T21:00:00.000Z";
function parsedReview(owner: string): ActionReviewerApprovalReview {
  return { kind: "action-review", formatVersion: 1, reviewId: "standalone-host-review", brainId: owner,
    requestId: "tool-request", configurationRevision: "configuration-1", ownerRevision: "old-device-revision",
    reviewerId: "optional-reviewer", permissionRevision: "permission-1", proposalSha256: sha(`proposal-${owner}`),
    proposedInvocation: { brainId: owner, toolId: "runtime.shell", action: "run", arguments: { command: "exact proposed command; never executed" } },
    disposition: "timeout", verdict: "ask", summary: "The optional adviser timed out; no permission was granted.", evidence: [],
    startedAt: time, finishedAt: "2026-10-13T21:00:30.000Z", elapsedMs: 30000,
    observedContext: [{ id: "task", kind: "task", text: "Original observed host context🙂" }], requestBytes: 256, responseBytes: 0 };
}

function request(owner: string, id: string, state: ExternalAdviceRequest["state"] = "complete",
  response = "Exact private reply🙂\nSecond line", selected: "assistant" | "reviewer" = "assistant"): ExternalAdviceRequest {
  const text = "Private user details: quoted \"text\" and source material🙂\nNo sanitizing.",
    context: ExternalAdviceSelectedContext = { format: "omni-external-advice-selected-context", version: 1, brainId: owner,
      channel: selected, selection: configuration().context, items: [{ id: "task", kind: "task", label: "Visible task", text, contentSha256: sha(text) }],
      includedCharacters: text.length, omittedItems: 0, truncated: false };
  const base = { brainId: owner, channel: selected, provider: "openai" as const, model: "fixture-model",
    prompt: "Help with my exact task; approvalToken here is ordinary quoted text, not a field.", requestText: `Prompt\n${text}`,
    context, settingsRevision: 1, authorityRevision: 3, authorityOwnerEpoch: "historical-device-epoch-not-a-grant",
    contextSha256: sha(canonical(context)), maxOutputTokens: 2048 };
  const preview = { format: "omni-external-advice-preview", version: 1, ...base, advisoryOnly: true, automaticLearning: false };
  const intent = { id, ...base, previewSha256: sha(canonical(preview)) };
  return { format: "omni-external-advice-request", version: 1, ...intent, state, response,
    createdAt: time, updatedAt: time, requestSha256: sha(canonical(intent)), providerRequestId: "historical-provider-audit-id",
    advisoryOnly: true, automaticLearning: false, hostActionAuthority: false, credentialsStoredInBrain: false };
}

async function fixture(path: string, owner: string, records: ExternalAdviceRequest[], model = "fixture-model",
  reviews: ActionReviewerApprovalReview[] = []): Promise<void> {
  await mkdir(join(path, "external-advice"), { recursive: true });
  const database = new DatabaseSync(join(path, EXTERNAL_ADVICE_RELATIVE_PATH));
  try {
    database.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
      CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE settings(channel TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
      CREATE TABLE requests(id TEXT PRIMARY KEY,channel TEXT NOT NULL,state TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
      CREATE INDEX requests_updated ON requests(updated_at,id);
      CREATE TABLE inherited_requests(source_brain_id TEXT NOT NULL,source_request_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,source_request_id,source_sha256));
      CREATE TABLE inherited_settings(source_brain_id TEXT NOT NULL,channel TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,channel,source_sha256));
      CREATE TABLE inherited_meta(source_brain_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,source_sha256));
      CREATE TABLE reviews(id TEXT PRIMARY KEY,owner_brain_id TEXT NOT NULL,source_brain_id TEXT NOT NULL,review_id TEXT NOT NULL,finished_at TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);`);
    for (const [key, value] of Object.entries({ format: EXTERNAL_ADVICE_FORMAT, brainId: owner, settingsRevision: "1", settingsUpdatedAt: time }))
      database.prepare("INSERT INTO meta VALUES(?,?)").run(key, value);
    for (const channel of ["assistant", "reviewer"]) { const payload = canonical(configuration(model));
      database.prepare("INSERT INTO settings VALUES(?,?,?)").run(channel, payload, sha(payload)); }
    for (const value of records) { const payload = canonical(value);
      database.prepare("INSERT INTO requests VALUES(?,?,?,?,?,?,?)").run(value.id, value.channel, value.state, value.createdAt, value.updatedAt, payload, sha(payload)); }
    for (const value of reviews) { const payload = canonical(value);
      database.prepare("INSERT INTO reviews VALUES(?,?,?,?,?,?,?)").run(value.reviewId, owner, value.brainId, value.reviewId, value.finishedAt, payload, sha(payload)); }
  } finally { database.close(); }
  await validateExternalAdviceWorkspace(path, owner);
}

async function hash(path: string): Promise<string> { return createHash("sha256").update(await readFile(path)).digest("hex"); }
function rows(path: string): { requests: ExternalAdviceRequest[]; originals: string[]; settings: string[]; metadata: string[]; reviews: string[] } {
  const database = new DatabaseSync(join(path, EXTERNAL_ADVICE_RELATIVE_PATH), { readOnly: true });
  try { return { requests: database.prepare("SELECT payload FROM requests ORDER BY id").all().map(row => JSON.parse(String(row.payload))),
    originals: database.prepare("SELECT payload FROM inherited_requests ORDER BY source_brain_id,source_request_id,source_sha256").all().map(row => String(row.payload)),
    settings: database.prepare("SELECT payload FROM inherited_settings ORDER BY source_brain_id,channel,source_sha256").all().map(row => String(row.payload)),
    metadata: database.prepare("SELECT payload FROM inherited_meta ORDER BY source_brain_id,source_sha256").all().map(row => String(row.payload)),
    reviews: database.prepare("SELECT payload FROM reviews ORDER BY id").all().map(row => String(row.payload)) }; }
  finally { database.close(); }
}
async function staged(parent: string, sourceOwner: string, targetOwner: string): Promise<string> {
  const stage = await root(); await copyMutableFileIsolated(join(parent, EXTERNAL_ADVICE_RELATIVE_PATH), join(stage, EXTERNAL_ADVICE_RELATIVE_PATH));
  await rekeyExternalAdviceWorkspace(stage, sourceOwner, targetOwner); return stage;
}

describe("complete advisory/reviewer history portability without copied device authority", () => {
  it("streams the full workspace and preserves unsanitized source bytes while retiring an active prefix", async () => {
    const source = await root(), target = await root(), packaged = await root(),
      assistant = request("source", "in-flight", "streaming", "Exact visible unfinished prefix🙂"),
      reviewer = request("source", "review-result", "complete", '{"verdict":"ask","evidence":"exact observed source"}', "reviewer");
    const review = parsedReview("source"); await fixture(source, "source", [assistant, reviewer], "fixture-model", [review]);
    const path = join(source, EXTERNAL_ADVICE_RELATIVE_PATH), before = await hash(path);
    const archive = await snapshotLearningWorkspace(source, packaged, "source");
    expect(archive).toBeDefined(); await adoptLearningWorkspace(archive!, target, "target");
    await validateExternalAdvicePortabilityWorkspace(target, "target");
    const copied = rows(target), inventory = await validateExternalAdviceWorkspace(target, "target");
    expect(inventory.settings.assistant).toEqual(configuration()); expect(inventory.settings.reviewer.enabled).toBe(true);
    expect(copied.requests).toHaveLength(2); expect(copied.requests.find(value => value.channel === "assistant")).toMatchObject({ brainId: "target", state: "cancelled", response: assistant.response });
    expect(copied.originals).toContain(canonical(assistant)); expect(copied.originals).toContain(canonical(reviewer));
    expect(copied.requests.every(value => value.hostActionAuthority === false && value.credentialsStoredInBrain === false)).toBe(true);
    expect(copied.metadata.some(value => JSON.parse(value).brainId === "source")).toBe(true);
    expect(copied.reviews).toEqual([canonical(review)]); expect(inventory.reviews).toBe(1);
    expect(await hash(path)).toBe(before);
    expect((await lstat(path)).ino).not.toBe((await lstat(join(target, EXTERNAL_ADVICE_RELATIVE_PATH))).ino);
  });

  it("rekeys idempotently and preserves original ancestry through a second trusted copy", async () => {
    const source = await root(), original = request("parent", "original", "complete"); await fixture(source, "parent", [original]);
    const copy = await staged(source, "parent", "copy"), initial = rows(copy);
    await rekeyExternalAdviceWorkspace(copy, "parent", "copy"); expect(rows(copy).requests).toEqual(initial.requests);
    const third = await staged(copy, "copy", "third"), later = rows(third);
    expect(later.requests).toHaveLength(1); expect(later.requests[0]?.inheritedFrom).toEqual({ brainId: "parent", requestId: original.id, sha256: sha(canonical(original)) });
    expect(later.originals).toEqual([canonical(original)]); expect(later.requests[0]?.context).toEqual(original.context);
    expect((await validateExternalAdviceWorkspace(third, "third")).settings.assistant).toEqual(configuration());
  });

  it("merges different source responses/settings without prompt-based dedup or parent mutation", async () => {
    const a = await root(), b = await root(), merged = await root();
    const first = request("a", "same-id", "complete", "first exact response"), second = request("b", "same-id", "complete", "second exact response");
    await fixture(a, "a", [first], "first-config", [parsedReview("a")]); await fixture(b, "b", [second], "second-config", [parsedReview("b")]);
    const beforeA = await hash(join(a, EXTERNAL_ADVICE_RELATIVE_PATH)), beforeB = await hash(join(b, EXTERNAL_ADVICE_RELATIVE_PATH));
    const stageA = await staged(a, "a", "merged"), stageB = await staged(b, "b", "merged");
    await mergeExternalAdviceWorkspace(stageA, merged, "merged", "a", async () => undefined);
    await mergeExternalAdviceWorkspace(stageB, merged, "merged", "b", async () => undefined);
    const beforeRetry = rows(merged); await mergeExternalAdviceWorkspace(stageB, merged, "merged", "b", async () => undefined);
    expect(rows(merged)).toEqual(beforeRetry); expect(beforeRetry.requests.map(value => value.response).sort()).toEqual([first.response, second.response].sort());
    expect(beforeRetry.originals).toContain(canonical(first)); expect(beforeRetry.originals).toContain(canonical(second));
    expect(new Set(beforeRetry.reviews)).toEqual(new Set([canonical(parsedReview("a")), canonical(parsedReview("b"))]));
    expect(new Set(beforeRetry.settings.map(value => JSON.parse(value).model))).toEqual(new Set(["first-config", "second-config"]));
    expect((await validateExternalAdviceWorkspace(merged, "merged")).settings.assistant.model).toBe("first-config");
    expect(await hash(join(a, EXTERNAL_ADVICE_RELATIVE_PATH))).toBe(beforeA); expect(await hash(join(b, EXTERNAL_ADVICE_RELATIVE_PATH))).toBe(beforeB);
  });

  it("deduplicates shared verified ancestry, not independent identical-looking text", async () => {
    const source = await root(), target = await root(), original = request("ancestral", "shared"); await fixture(source, "ancestral", [original], "fixture-model", [parsedReview("ancestral")]);
    const one = await staged(source, "ancestral", "one"), two = await staged(source, "ancestral", "two"),
      stageOne = await staged(one, "one", "target"), stageTwo = await staged(two, "two", "target");
    await mergeExternalAdviceWorkspace(stageOne, target, "target", "one", async () => undefined);
    await mergeExternalAdviceWorkspace(stageTwo, target, "target", "two", async () => undefined);
    expect(rows(target).requests).toHaveLength(1); expect(rows(target).originals).toEqual([canonical(original)]);
    expect(rows(target).reviews).toEqual([canonical(parsedReview("ancestral"))]);
    const independent = await root(); await fixture(independent, "independent", [request("independent", "shared")]);
    await mergeExternalAdviceWorkspace(await staged(independent, "independent", "target"), target, "target", "independent", async () => undefined);
    expect(rows(target).requests).toHaveLength(2);
  });

  it("rejects missing/tampered ancestry and hidden credential fields without stripping human text", async () => {
    const source = await root(), original = request("parent", "request"); await fixture(source, "parent", [original]);
    const copy = await staged(source, "parent", "copy"), database = new DatabaseSync(join(copy, EXTERNAL_ADVICE_RELATIVE_PATH));
    try { database.exec("DELETE FROM inherited_requests"); } finally { database.close(); }
    await expect(validateExternalAdvicePortabilityWorkspace(copy, "copy")).rejects.toThrow(/source|archive/i);
    const raw = new DatabaseSync(join(source, EXTERNAL_ADVICE_RELATIVE_PATH));
    try { const payload = canonical({ ...original, credentialId: "must-not-be-portable" }); raw.prepare("UPDATE requests SET payload=?,sha256=?").run(payload, sha(payload)); }
    finally { raw.close(); }
    await expect(validateExternalAdvicePortabilityWorkspace(source, "parent")).rejects.toThrow(/unsupported|unknown|field/i);
  });

  it("rolls back a cancelled merge without changing existing history or either source", async () => {
    const first = await root(), second = await root(), target = await root();
    await fixture(first, "first", [request("first", "first")]); const next = request("second", "second"); await fixture(second, "second", [next]);
    await mergeExternalAdviceWorkspace(await staged(first, "first", "target"), target, "target", "first", async () => undefined);
    const incoming = await staged(second, "second", "target"), before = rows(target), controller = new AbortController();
    await expect(mergeExternalAdviceWorkspace(incoming, target, "target", "second", async bytes => {
      if (bytes === Buffer.byteLength(canonical(next)) * 2 + 8192) controller.abort(new Error("fixture cancellation"));
    }, controller.signal)).rejects.toThrow(/cancellation/);
    expect(rows(target)).toEqual(before); expect(rows(incoming).originals).toContain(canonical(next));
  });

  it("retains later same-ID recovery history and parsed review receipts without reviving a pending source", async () => {
    const checkpoint = await root(), current = await root(), restored = await root(),
      earlier = request("same-owner", "same-request", "streaming", "Exact earlier prefix"),
      later = request("same-owner", "same-request", "complete", "Exact later complete response"), review = parsedReview("same-owner");
    await fixture(checkpoint, "same-owner", [earlier], "checkpoint-config");
    await fixture(current, "same-owner", [later], "current-config", [review]);
    await copyMutableFileIsolated(join(checkpoint, EXTERNAL_ADVICE_RELATIVE_PATH), join(restored, EXTERNAL_ADVICE_RELATIVE_PATH));
    await rekeyExternalAdviceWorkspace(restored, "same-owner", "same-owner");
    const currentHash = await hash(join(current, EXTERNAL_ADVICE_RELATIVE_PATH));
    await mergeExternalAdviceWorkspace(current, restored, "same-owner", "recovery-snapshot", async () => undefined);
    const result = rows(restored);
    expect(result.requests.map(value => value.state).sort()).toEqual(["cancelled", "complete"]);
    expect(result.originals).toContain(canonical(earlier)); expect(result.originals).toContain(canonical(later));
    expect(result.reviews).toEqual([canonical(review)]);
    expect((await validateExternalAdviceWorkspace(restored, "same-owner")).settings.assistant.model).toBe("checkpoint-config");
    expect(new Set(result.settings.map(value => JSON.parse(value).model))).toEqual(new Set(["checkpoint-config", "current-config"]));
    expect(await hash(join(current, EXTERNAL_ADVICE_RELATIVE_PATH))).toBe(currentHash);
    const repositorySource = await readFile(new URL("../src/main/brainRepository.ts", import.meta.url), "utf8");
    const hook = repositorySource.indexOf("await mergeExternalAdviceWorkspace(brainDirectory, stagedHost");
    expect(hook).toBeGreaterThan(0); expect(hook).toBeLessThan(repositorySource.indexOf("await rename(stagedEngine, targetEngine)", hook));
    expect(repositorySource.slice(hook).split("for (const name of retainedHostRoots)")[0]).toContain('"external-advice", "workspace.sqlite3"');
  });
});
