import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UserQuestionService, validateUserQuestionRecord } from "../src/main/userQuestions";
import { UserQuestionDelivery } from "../src/main/userQuestionDelivery";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation } from "../src/shared/types";
import {
  DEFAULT_USER_QUESTION_PREFERENCES,
  type UserQuestionEvent,
  type UserQuestionRequest
} from "../src/shared/userQuestions";

describe("durable structured task questions", () => {
  let directory: string;
  let service: UserQuestionService;
  let now: number;
  const services: UserQuestionService[] = [];
  const owner = (brainId = "first") => join(directory, brainId);
  const createService = (extra: Partial<ConstructorParameters<typeof UserQuestionService>[0]> = {}) => {
    const next = new UserQuestionService({ brainDirectory: owner, now: () => now, ...extra });
    services.push(next);
    return next;
  };
  const choice = (overrides: Partial<UserQuestionRequest> = {}): UserQuestionRequest => ({
    brainId: "first", requestId: "task-choice-1", prompt: "Which chapter should I work on?", mode: "single",
    options: [
      { id: "later", label: "Later chapter", rank: 4 },
      { id: "current", label: "Current chapter", description: "Finish the section already started.", recommended: true, rank: 3 },
      { id: "intro", label: "Introduction", rank: 1 }
    ], source: { kind: "brain", turnId: "turn1", actionId: "action1" }, ...overrides
  });

  beforeEach(async () => {
    now = Date.parse("2026-10-02T12:00:00.000Z");
    directory = await mkdtemp(join(tmpdir(), "omni-questions-"));
    await Promise.all([mkdir(owner()), mkdir(owner("second"))]);
    service = createService();
  });

  afterEach(async () => {
    for (const current of services.splice(0)) current.close();
    vi.useRealTimers();
    await rm(directory, { recursive: true, force: true });
  });

  it("routes an ordinary chat question with trusted turn ownership and delivers its answer only as retained action evidence", async () => {
    const executor = new ToolExecutor({} as BrainService, {} as RuntimeJobManager, undefined, { execute: vi.fn() });
    executor.setQuestionService(service);
    const invocation: ToolInvocation = { brainId: "first", toolId: "studio.question", action: "ask", arguments: {
      prompt: "Which section should I discuss?", mode: "single",
      options: [{ id: "intro", label: "Introduction" }, { id: "details", label: "Details" }],
      brainId: "second", requestId: "generated-request-id", source: { kind: "host", turnId: "generated-turn", actionId: "generated-action" }
    } };
    const dispatch = executor as unknown as { dispatch(invocation: ToolInvocation, signal: AbortSignal, requestId: string,
      progress: undefined, permission: "auto", actionId: string, learning: undefined, chatTurnId: string): Promise<unknown> };
    const created = await dispatch.dispatch(invocation, new AbortController().signal, "effect-fixture", undefined, "auto", "main-action", undefined, "main-turn");
    expect(created).toMatchObject({ brainId: "first", requestId: "main-action", status: "pending",
      source: { kind: "brain", actionId: "main-action", turnId: "main-turn" } });
    const current = validateUserQuestionRecord(created, "first");
    const sink = { queueCompletedToolExecution: vi.fn(async () => undefined) }, errors = vi.fn();
    const delivery = new UserQuestionDelivery(service, sink, errors);
    try {
      expect(sink.queueCompletedToolExecution).not.toHaveBeenCalled();
      await service.respond({ brainId: "first", questionId: current.id, response: { optionIds: ["intro"] } });
      await delivery.settle("first");
      expect(sink.queueCompletedToolExecution).toHaveBeenCalledOnce();
      expect(sink.queueCompletedToolExecution).toHaveBeenCalledWith(expect.objectContaining({ brainId: "first", toolId: "studio.question" }),
        expect.objectContaining({ state: "complete", output: expect.objectContaining({ questionId: current.id,
          resolution: expect.objectContaining({ source: "user", authority: "task-choice-only", permissionsGranted: false }) }) }), "main-turn");
      expect((await service.get("first", current.id)).deliveredAt).toBeDefined();
      expect(errors).not.toHaveBeenCalled();
      // The sink deliberately exposes no native inbox, chat, model or tool API.
      // This proves retained-delivery wiring, not live decoder use or learning.
    } finally { delivery.close(); }
  });

  it("persists exact prompts, ranks recommendations, isolates instances, and idempotently retries", async () => {
    const events: UserQuestionEvent[] = [];
    service.subscribe((event) => events.push(event));
    const prompt = "Choose the next chapter.\nKeep this exact detail, including its punctuation.";
    const created = await service.request(choice({ prompt }));
    expect(created).toMatchObject({ brainId: "first", prompt, status: "pending", allowOther: true,
      timeout: { enabled: false, seconds: 1200 }, source: { kind: "brain", turnId: "turn1" } });
    expect(created.options.map(({ id }) => id)).toEqual(["current", "intro", "later"]);
    expect(events.map(({ type }) => type)).toEqual(["created"]);
    const saved = JSON.parse(await readFile(join(owner(), "questions", "records", `${created.id}.json`), "utf8"));
    expect(saved).toEqual(created);
    expect(await service.request(choice({ prompt }))).toEqual(created);
    await expect(service.request(choice({ prompt: "A different question" }))).rejects.toThrow("different content");
    expect(await service.list("second")).toEqual([]);
    await expect(service.respond({ brainId: "second", questionId: created.id, response: { optionIds: ["current"] } })).rejects.toThrow("does not exist");
    service.close();
    service = createService();
    expect(await service.list("first")).toEqual([created]);
  });

  it("validates multiple-choice minimums and retains a user answer and delivery receipt across restart", async () => {
    const created = await service.request(choice({ mode: "multiple", minSelections: 2, maxSelections: 3, allowOther: false }));
    await expect(service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["current"] } })).rejects.toThrow("Select 2–3");
    await expect(service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["current", "current"] } })).rejects.toThrow("only once");
    await expect(service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["current", "unknown"] } })).rejects.toThrow("unavailable");
    const answered = await service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["later", "current"] } });
    expect(answered.resolution).toMatchObject({ source: "user", response: { optionIds: ["current", "later"] }, authority: "task-choice-only", permissionsGranted: false });
    expect(await service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["current", "later"] } })).toEqual(answered);
    await expect(service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["current", "intro"] } })).rejects.toThrow("already resolved");
    service.close();
    service = createService();
    expect(await service.undelivered("first")).toEqual([answered]);
    await expect(service.acknowledge("first", created.id, "00000000-0000-0000-0000-000000000000")).rejects.toThrow("does not match");
    const delivered = await service.acknowledge("first", created.id, answered.resolution!.id);
    expect(delivered.deliveredAt).toBe(new Date(now).toISOString());
    expect(await service.undelivered("first")).toEqual([]);
    service.close();
    service = createService();
    expect(await service.acknowledge("first", created.id, answered.resolution!.id)).toEqual(delivered);
  });

  it("supports explicit Other and written answers without treating arbitrary fields as consent", async () => {
    const other = await service.request(choice());
    const written = await service.request({ brainId: "first", requestId: "written", mode: "text", prompt: "Describe your preferred ending." });
    await expect(service.respond({ brainId: "first", questionId: written.id, response: { optionIds: ["current"] } })).rejects.toThrow("written answer");
    const exactText = "  Finish with a quiet conversation.\nLeave the mystery open.  ";
    expect((await service.respond({ brainId: "first", questionId: written.id, response: { text: exactText } })).resolution?.response).toEqual({ text: exactText });
    await expect(service.respond({ brainId: "first", questionId: other.id, response: { text: "Yes", approvalToken: "untrusted" } as never })).rejects.toThrow("cannot grant permissions");
    expect((await service.respond({ brainId: "first", questionId: other.id, response: { text: "Start with the appendix." } })).resolution?.response).toEqual({ text: "Start with the appendix." });
    await expect(service.request({ ...choice(), requestId: "permission", kind: "permission" } as never)).rejects.toThrow("not permission");
  });

  it("starts disabled, uses a user-configured twenty-minute default, and records expired assumptions on restart", async () => {
    expect(await service.preferences("first")).toEqual(DEFAULT_USER_QUESTION_PREFERENCES);
    const beforeSetting = await service.request(choice());
    now += 60 * 60 * 1_000;
    expect((await service.get("first", beforeSetting.id)).status).toBe("pending");
    await service.setPreferences({ brainId: "first", timeoutEnabled: true, timeoutSeconds: 1200 });
    const pending = await service.get("first", beforeSetting.id);
    expect(pending.timeout.deadlineAt).toBe(new Date(now + 20 * 60 * 1_000).toISOString());
    service.close();
    now += 25 * 60 * 1_000;
    service = createService();
    const events: UserQuestionEvent[] = [];
    service.subscribe((event) => events.push(event));
    expect(await service.recover(["first", "second"])).toEqual({ recovered: ["first", "second"], errors: [] });
    const assumed = await service.get("first", pending.id);
    expect(assumed).toMatchObject({ status: "assumed", resolution: {
      source: "timeout", response: { optionIds: ["current"] }, deadlineAt: pending.timeout.deadlineAt,
      overdueMilliseconds: 5 * 60 * 1_000, authority: "task-choice-only", permissionsGranted: false
    } });
    expect(events.filter(({ type }) => type === "resolved")).toHaveLength(1);
    await service.recover(["first"]);
    expect(events.filter(({ type }) => type === "resolved")).toHaveLength(1);
    await expect(service.respond({ brainId: "first", questionId: pending.id, response: { optionIds: ["intro"] } })).rejects.toThrow("already resolved");
  });

  it("runs a live timer, respects checkbox bounds, and never invents a free-text assumption", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    await service.setPreferences({ brainId: "first", timeoutEnabled: true, timeoutSeconds: 1 });
    const multiple = await service.request(choice({ mode: "multiple", minSelections: 2, maxSelections: 2 }));
    const written = await service.request({ brainId: "first", requestId: "text", mode: "text", prompt: "Explain the change." });
    const suggested = await service.request({ brainId: "first", requestId: "suggested", mode: "text", prompt: "Choose a voice.", suggestedResponse: { text: "Use a neutral narrator." } });
    expect(written.timeout).toMatchObject({ enabled: true, unavailableReason: "no-suggested-response" });
    expect(written.timeout.deadlineAt).toBeUndefined();
    const waiting = service.waitForResolution("first", multiple.id);
    now += 1_000;
    await vi.advanceTimersByTimeAsync(1_000);
    expect((await waiting).resolution?.response).toEqual({ optionIds: ["current", "intro"] });
    expect((await service.get("first", written.id)).status).toBe("pending");
    expect((await service.get("first", suggested.id)).resolution?.response).toEqual({ text: "Use a neutral narrator." });
  });

  it("updates pending timeout policies durably, and disables an existing deadline", async () => {
    await service.setPreferences({ brainId: "first", timeoutEnabled: true, timeoutSeconds: 5 });
    const created = await service.request(choice());
    now += 1_000;
    await service.setPreferences({ brainId: "first", timeoutEnabled: true, timeoutSeconds: 10 });
    expect((await service.get("first", created.id)).timeout.deadlineAt).toBe(new Date(now + 10_000).toISOString());
    await service.setPreferences({ brainId: "first", timeoutEnabled: false, timeoutSeconds: 10 });
    now += 60_000;
    service.close();
    service = createService();
    expect(await service.preferences("first")).toEqual({ timeoutEnabled: false, timeoutSeconds: 10 });
    expect(await service.get("first", created.id)).toMatchObject({ status: "pending", timeout: { enabled: false } });
  });

  it("serializes competing answers and cancellation without overwriting the winning durable result", async () => {
    const created = await service.request(choice());
    const outcomes = await Promise.allSettled([
      service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["current"] } }),
      service.respond({ brainId: "first", questionId: created.id, response: { optionIds: ["later"] } }),
      service.cancel("first", created.id, "Moved to a different task.")
    ]);
    expect(outcomes.map(({ status }) => status)).toEqual(["fulfilled", "rejected", "rejected"]);
    expect((await service.get("first", created.id)).resolution?.response).toEqual({ optionIds: ["current"] });
    const cancelled = await service.request(choice({ requestId: "cancelled" }));
    const result = await service.cancel("first", cancelled.id, "Skip this choice.");
    expect(result).toMatchObject({ status: "cancelled", resolution: { source: "cancellation", reason: "Skip this choice.", permissionsGranted: false } });
    expect(await service.cancel("first", cancelled.id)).toEqual(result);
  });

  it("keeps requests saved when a wait is aborted, unsubscribed, or the host closes", async () => {
    const request = await service.request(choice());
    const controller = new AbortController();
    const waiting = service.waitForResolution("first", request.id, controller.signal);
    controller.abort(new Error("Turn stopped."));
    await expect(waiting).rejects.toThrow("Turn stopped");
    expect((await service.get("first", request.id)).status).toBe("pending");
    const closed = service.waitForResolution("first", request.id);
    service.close();
    await expect(closed).rejects.toThrow("remains saved");
    service = createService();
    expect((await service.get("first", request.id)).status).toBe("pending");
  });

  it("rejects malformed requests before writing and never silently repairs corrupt saved state", async () => {
    const invalid: Partial<UserQuestionRequest>[] = [
      { prompt: " " }, { minSelections: 2 }, { maxSelections: 9 },
      { options: [{ id: "one", label: "Only one" }] },
      { options: [{ id: "one", label: "First" }, { id: "one", label: "Duplicate" }] },
      { suggestedResponse: { optionIds: ["missing"] } }
    ];
    for (const value of invalid) await expect(service.request(choice(value))).rejects.toThrow();
    await expect(service.list("../escape")).rejects.toThrow("Invalid question instance");
    await expect(service.setPreferences({ brainId: "first", timeoutEnabled: true, timeoutSeconds: 0 })).rejects.toThrow("timeout");
    await expect(service.setPreferences({ brainId: "first", timeoutEnabled: true, timeoutSeconds: 8_900_000_000_000 })).rejects.toThrow("date range");
    expect(await service.preferences("first")).toEqual(DEFAULT_USER_QUESTION_PREFERENCES);
    const created = await service.request(choice());
    const path = join(owner(), "questions", "records", `${created.id}.json`);
    const saved = JSON.parse(await readFile(path, "utf8"));
    saved.prompt = "Tampered question";
    await writeFile(path, JSON.stringify(saved));
    await expect(service.list("first")).rejects.toThrow("failed validation");
    expect((await service.recover(["first", "second"])).errors).toEqual([{ brainId: "first", message: "Saved question contents failed validation." }]);
    expect(JSON.parse(await readFile(path, "utf8")).prompt).toBe("Tampered question");
    expect(await service.list("second")).toEqual([]);
  });

  it("rejects redirected storage and corrupt authority while preserving uncommitted temporary files", async () => {
    const outside = join(directory, "outside");
    await mkdir(outside);
    await symlink(outside, join(owner(), "questions"), process.platform === "win32" ? "junction" : "dir");
    await expect(service.request(choice())).rejects.toThrow("owned real directory");
    expect(await readdir(outside)).toEqual([]);
    const created = await service.request(choice({ brainId: "second" }));
    const answered = await service.respond({ brainId: "second", questionId: created.id, response: { optionIds: ["current"] } });
    expect(() => validateUserQuestionRecord({ ...answered, resolution: { ...answered.resolution, permissionsGranted: true } }, "second")).toThrow("cannot grant permissions");
    const tempName = `.${created.id}.json.00000000-0000-0000-0000-000000000000.tmp`;
    await writeFile(join(owner("second"), "questions", "records", tempName), "partial");
    expect(await service.list("second")).toEqual([answered]);
    expect(await readFile(join(owner("second"), "questions", "records", tempName), "utf8")).toBe("partial");
  });

  it("publishes events only after persistence and prevents a UI listener from mutating saved records", async () => {
    const events: UserQuestionEvent[] = [];
    const unsubscribe = service.subscribe((event) => {
      events.push(event);
      if ("question" in event) event.question.prompt = "UI mutation";
    });
    const created = await service.request(choice());
    expect(created.prompt).not.toBe("UI mutation");
    expect((await service.get("first", created.id)).prompt).toBe(choice().prompt);
    unsubscribe();
    await service.cancel("first", created.id);
    expect(events).toHaveLength(1);
  });
});
