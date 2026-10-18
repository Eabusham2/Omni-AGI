import { cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  rekeySavedUserQuestions,
  rekeyUserQuestionRecord,
  safeUserQuestionPath,
  savedUserQuestionFiles,
  validateSavedUserQuestionFiles
} from "../src/main/userQuestionPortability";
import { UserQuestionService, validateUserQuestionRecord } from "../src/main/userQuestions";

describe("question portability", () => {
  let root: string;
  let now: number;
  let service: UserQuestionService;
  const owner = (brainId: string) => join(root, brainId);
  const request = (requestId: string) => service.request({ brainId: "parent", requestId, mode: "single", prompt: "Which topic?",
    options: [{ id: "first", label: "First", recommended: true }, { id: "second", label: "Second" }] });
  const copy = async () => { await cp(join(owner("parent"), "questions"), join(owner("copy"), "questions"), { recursive: true }); };

  beforeEach(async () => {
    now = Date.parse("2026-10-02T12:00:00.000Z");
    root = await mkdtemp(join(tmpdir(), "omni-question-copy-"));
    await Promise.all([mkdir(owner("parent")), mkdir(owner("copy"))]);
    service = new UserQuestionService({ brainDirectory: owner, now: () => now });
  });
  afterEach(async () => { service.close(); await rm(root, { recursive: true, force: true }); });

  it("enumerates exact portable state and rejects escaped paths and changed ownership", async () => {
    expect(await savedUserQuestionFiles(owner("parent"), "parent")).toEqual(new Map());
    await service.setPreferences({ brainId: "parent", timeoutEnabled: true, timeoutSeconds: 1200 });
    const record = await request("one");
    const files = await savedUserQuestionFiles(owner("parent"), "parent");
    expect([...files.keys()].sort()).toEqual(["questions/preferences.json", `questions/records/${record.id}.json`].sort());
    await expect(validateSavedUserQuestionFiles(files, "copy")).rejects.toThrow("ownership");
    expect(safeUserQuestionPath("questions/../../secrets.json")).toBe(false);
    expect(safeUserQuestionPath("questions/records/program.js")).toBe(false);
    await expect(validateSavedUserQuestionFiles(new Map([["../preferences.json", files.get("questions/preferences.json")!]]), "parent")).rejects.toThrow("Unsafe");
  });

  it("keeps copied requests and resolved answers inert through startup until explicit resume", async () => {
    await service.setPreferences({ brainId: "parent", timeoutEnabled: true, timeoutSeconds: 10 });
    const pending = await request("pending");
    const asked = await request("answered");
    const answered = await service.respond({ brainId: "parent", questionId: asked.id, response: { optionIds: ["second"] } });
    const originalBytes = await readFile(join(owner("parent"), "questions", "records", `${pending.id}.json`), "utf8");
    await copy();
    expect(await rekeySavedUserQuestions(owner("copy"), "parent", "copy", { copiedAt: new Date(now).toISOString() })).toBe(2);
    now += 60_000;
    await service.recover(["copy"]);
    const dormant = await service.get("copy", pending.id);
    expect(dormant).toMatchObject({ brainId: "copy", status: "pending", suspended: { reason: "copied-instance", fromBrainId: "parent" }, lineage: [{ brainId: "parent", questionId: pending.id }] });
    expect(dormant.requestSha256).not.toBe(pending.requestSha256);
    expect(await service.undelivered("copy")).toEqual([]);
    await expect(service.respond({ brainId: "copy", questionId: pending.id, response: { optionIds: ["first"] } })).rejects.toThrow("Resume");
    const resumed = await service.resume("copy");
    expect(resumed.every((record) => !record.suspended)).toBe(true);
    expect(resumed.find((record) => record.id === pending.id)?.timeout.deadlineAt).toBe(new Date(now + 10_000).toISOString());
    const copiedAnswer = (await service.undelivered("copy"))[0]!;
    expect(copiedAnswer.resolution).toEqual(answered.resolution);
    expect(copiedAnswer.lineage?.[0]?.brainId).toBe("parent");
    expect(await readFile(join(owner("parent"), "questions", "records", `${pending.id}.json`), "utf8")).toBe(originalBytes);
    now += 10_000;
    expect((await service.get("copy", pending.id)).status).toBe("assumed");
  });

  it("preserves timeout provenance and acknowledged results when rekeying is retried", async () => {
    await service.setPreferences({ brainId: "parent", timeoutEnabled: true, timeoutSeconds: 1 });
    const created = await request("timeout");
    now += 1_000;
    const assumed = await service.get("parent", created.id);
    const delivered = await service.acknowledge("parent", assumed.id, assumed.resolution!.id);
    const copiedAt = new Date(now).toISOString();
    const rebound = rekeyUserQuestionRecord(delivered, "parent", "copy", copiedAt);
    expect(rebound.resolution).toEqual(delivered.resolution);
    expect(rebound.deliveredAt).toBe(delivered.deliveredAt);
    expect(validateUserQuestionRecord(rebound, "copy")).toEqual(rebound);
    expect(rekeyUserQuestionRecord(rebound, "parent", "copy", new Date(now + 200).toISOString())).toEqual(rebound);
    const third = rekeyUserQuestionRecord(rebound, "copy", "third", new Date(now + 400).toISOString());
    expect(third.lineage?.map(({ brainId }) => brainId)).toEqual(["parent", "copy"]);
    expect(third.resolution?.permissionsGranted).toBe(false);
    const restore = rekeyUserQuestionRecord(delivered, "parent", "parent", copiedAt);
    expect(rekeyUserQuestionRecord(restore, "parent", "parent", copiedAt)).toEqual(restore);
  });

  it("supports interrupted staging retries and leaves the original untouched on cancellation", async () => {
    const created = await request("one");
    await copy();
    const controller = new AbortController();
    controller.abort(new Error("Cancelled copy."));
    await expect(rekeySavedUserQuestions(owner("copy"), "parent", "copy", { signal: controller.signal })).rejects.toThrow("Cancelled copy");
    expect((await savedUserQuestionFiles(owner("copy"), "parent")).size).toBe(1);
    await rekeySavedUserQuestions(owner("copy"), "parent", "copy");
    await rekeySavedUserQuestions(owner("copy"), "parent", "copy");
    expect((await service.get("copy", created.id)).lineage).toHaveLength(1);
    expect((await service.get("parent", created.id)).lineage).toBeUndefined();
  });

  it("rejects corrupt metadata and symlinked roots without materializing external state", async () => {
    await mkdir(join(owner("parent"), "questions"));
    await writeFile(join(owner("parent"), "questions", "preferences.json"), JSON.stringify({ format: "wrong" }));
    await expect(savedUserQuestionFiles(owner("parent"), "parent")).rejects.toThrow("Unsupported");
    await symlink(join(owner("parent"), "questions"), join(owner("copy"), "questions"), process.platform === "win32" ? "junction" : "dir");
    await expect(savedUserQuestionFiles(owner("copy"), "copy")).rejects.toThrow("unsafe");
    await expect(rekeySavedUserQuestions(owner("copy"), "parent", "copy")).rejects.toThrow("unsafe");
  });
});
