/** Real tiny durable teacher metadata with inert transport/learner. No API/model/learning is run. */
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiTeacherTrainingService } from "../src/main/teacherTraining";
import { API_TEACHER_DATABASE, apiTeacherSha256 } from "../src/main/apiTeacherStore";
import type { HostActionOriginAuthority } from "../src/shared/hostActionAuthority";
import type { RuntimeJob, RuntimeJobEvent } from "../src/shared/types";

const roots: string[] = [], services: ApiTeacherTrainingService[] = [];
afterEach(async () => {
  await Promise.all(services.splice(0).map(service => service.close()));
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })));
});
const request = { brainId: "teacher-owner", requestId: "owned-teacher-request", provider: "openai" as const,
  model: "explicit-model", prompts: ["An explicitly selected test lesson"], maxOutputTokens: 32 };
const response = () => new Response(JSON.stringify({ status: "completed", output_text: "Synthetic provider bytes for custody only." }),
  { status: 200, headers: { "content-type": "application/json" } });
async function fixture(fetcher: typeof fetch = vi.fn(async () => response()) as never) {
  const root = await mkdtemp(join(tmpdir(), "omni-teacher-authority-")); roots.push(root);
  const preflight = vi.fn(async () => undefined), admitWrite = vi.fn(async () => undefined), learn = vi.fn(async (brainId: string, input: { content: string }) => ({
    brain: { id: brainId }, source: { id: "inert-source", contentHash: apiTeacherSha256(input.content), learnedIdeas: 0,
      learnedConcepts: 0, learnedSynapses: 0, learnedParameterSteps: 0, parametersChanged: false }
  }));
  const key = "synthetic-device-credential-not-user-key";
  const service = new ApiTeacherTrainingService({ preflightStart: preflight, learnStructuredExperience: learn } as never,
    { get: async (id: string) => id === "teacher:openai" ? key : undefined, persistence: () => "session" } as never,
    { brainRoot: async id => { expect(id).toBe(request.brainId); return root; }, admitWrite }, fetcher);
  services.push(service);
  const controller = new AbortController(); let allowed = true;
  const authority: HostActionOriginAuthority = { revision: "private-live-tool-owner", signal: controller.signal,
    assertCurrent: vi.fn(async () => { if (!allowed) throw new Error("Current tool permission refused."); }) };
  return { service, root, key, preflight, admitWrite, learn, authority, controller, revoke: () => { allowed = false; } };
}
/** Subscribe before Start; await the real committed terminal event, not an IO spin count. */
function terminal(service: ApiTeacherTrainingService, expected: string): Promise<RuntimeJob> {
  return new Promise((resolve, reject) => {
    const listener = ({ job }: RuntimeJobEvent): void => {
      const state = (job.output as { durableState?: string } | undefined)?.durableState;
      if (job.brainId !== request.brainId || !state || !["complete", "failed", "paused", "review-required", "stopped"].includes(state)) return;
      // The ordinary runtime projection uses cancelled for several paused/held
      // dispositions; the teacher report retains their exact durable state.
      service.off("event", listener);
      if (state === expected) resolve(job); else reject(new Error(`Expected ${expected}, got ${state}: ${job.error ?? job.label}`));
    };
    service.on("event", listener);
  });
}

describe("API teacher detached tool authority and current resource boundaries", () => {
  it("refuses stale authority before creating or dispatching a job", async () => {
    const fetcher = vi.fn(async () => response()), value = await fixture(fetcher as never); value.revoke();
    await expect(value.service.start(request, value.authority)).rejects.toThrow(/permission refused/);
    expect(fetcher).not.toHaveBeenCalled(); expect(value.admitWrite).not.toHaveBeenCalled(); expect(value.learn).not.toHaveBeenCalled();
  });

  it("keeps a pre-provider resource refusal pending and paused, not a fictional unknown remote effect", async () => {
    const fetcher = vi.fn(async () => response()), value = await fixture(fetcher as never);
    let checks = 0;
    value.preflight.mockImplementation(async () => { if (++checks === 3) throw new Error("Current RAM budget refused."); });
    const paused = terminal(value.service, "paused"), job = await value.service.start(request, value.authority);
    await paused; await value.service.close();
    const details = await value.service.details(request.brainId, job.id);
    expect(details).toMatchObject({ state: "paused", nextIndex: 0, learned: 0, items: [{ state: "pending" }] });
    expect(details.items[0]?.attemptId).toBeUndefined(); expect(details.items[0]?.requestSha256).toBeUndefined();
    expect(fetcher).not.toHaveBeenCalled(); expect(value.learn).not.toHaveBeenCalled();
    expect(JSON.stringify(details)).not.toContain("private-live-tool-owner");
  });

  it("rechecks after provider ACK, retains received bytes, and resumes learning without another provider request", async () => {
    let revoke: () => void = () => undefined;
    const fetcher = vi.fn(async () => { revoke(); return response(); }), value = await fixture(fetcher as never);
    revoke = value.revoke;
    const paused = terminal(value.service, "paused"), job = await value.service.start(request, value.authority);
    await paused; await value.service.close();
    const held = await value.service.details(request.brainId, job.id);
    expect(held).toMatchObject({ state: "paused", learned: 0, items: [{ state: "response-committed", response: "Synthetic provider bytes for custody only." }] });
    expect(fetcher).toHaveBeenCalledOnce(); expect(value.learn).not.toHaveBeenCalled();
    const complete = terminal(value.service, "complete");
    await value.service.resume(request.brainId, job.id, { revision: "fresh-host-review", signal: new AbortController().signal, assertCurrent: async () => undefined });
    await complete; await value.service.close();
    expect(fetcher).toHaveBeenCalledOnce(); expect(value.learn).toHaveBeenCalledOnce();
    const details = await value.service.details(request.brainId, job.id);
    expect(details.items[0]?.receipt).toMatchObject({ brainId: request.brainId, parametersChanged: false });
    const bytes = await readFile(join(value.root, API_TEACHER_DATABASE));
    for (const secret of [value.key, "private-live-tool-owner", "fresh-host-review"]) expect(bytes.includes(Buffer.from(secret))).toBe(false);
  });

  it("marks a revoked in-flight request uncertain and never resumes/retries it without exact review", async () => {
    let entered!: () => void; const started = new Promise<void>(resolve => { entered = resolve; });
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      entered(); return new Promise<Response>((_resolve, reject) => {
        const stop = () => reject(init?.signal?.reason ?? new Error("Fixture request aborted"));
        init?.signal?.addEventListener("abort", stop, { once: true }); if (init?.signal?.aborted) stop();
      });
    });
    const value = await fixture(fetcher as never), paused = terminal(value.service, "review-required");
    const job = await value.service.start(request, value.authority); await started;
    value.service.invalidateToolAuthority(request.brainId); await paused; await value.service.close();
    const details = await value.service.details(request.brainId, job.id);
    expect(details).toMatchObject({ state: "review-required", learned: 0, items: [{ state: "uncertain-review" }] });
    expect(fetcher).toHaveBeenCalledOnce(); expect(value.learn).not.toHaveBeenCalled();
    await expect(value.service.resume(request.brainId, job.id, { ...value.authority, assertCurrent: async () => undefined })).rejects.toThrow(/uncertain/);
    expect(fetcher).toHaveBeenCalledOnce();
  });
});
