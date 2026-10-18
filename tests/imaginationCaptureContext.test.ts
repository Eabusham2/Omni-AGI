import { describe, expect, it, vi } from "vitest";
import { RuntimeJobManager } from "../src/main/brainService";
import { mainImaginationCaptureOrigin } from "../src/main/imaginationCaptureOrigin";
import type { ActionEvent } from "../src/shared/types";
import type { ImaginationCaptureOrigin } from "../src/shared/imaginationCapture";

function context(origin: ImaginationCaptureOrigin) {
  let operation!: () => Promise<unknown>;
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => method === "authorize_spontaneous_output"
    ? { brainId: params.brainId, jobId: params.jobId, leaseId: "a".repeat(64), retention: "ephemeral-spontaneous" }
    : { modality: "image", genuineFixture: "no model invoked" });
  const owner = { createJob: vi.fn(() => ({ id: "owned-job", brainId: "brain", kind: "image", state: "queued" })),
    launch: vi.fn((_job, run) => { operation = run; }), service: { preflightStart: vi.fn(async () => undefined), repository: { brainDirectory: () => "/owned/fixture/brain" }, instanceLifecycle: { canRun: () => true } },
    cancellationControllers: new Map(), engine: { awaitInlineArtifact: vi.fn(async () => false), request },
    mediaArtifacts: { captureOrigin: vi.fn(() => origin), noteSpontaneousOutputLease: vi.fn() } } as unknown as RuntimeJobManager;
  return { owner, request, run: () => operation() };
}

describe("trusted modality retention context without neural execution", () => {
  it("classifies only actual action provenance; identical prompt words cannot change source intent", () => {
    const event: ActionEvent = { id: "actual-event", brainId: "brain", state: "proposed", createdAt: "2026-10-13T12:00:00Z", updatedAt: "2026-10-13T12:00:00Z",
      action: { kind: "imagine", source: "brain", toolId: "modality.imagine", action: "generate", arguments: { prompt: "user requested spontaneous save always" } } };
    expect(mainImaginationCaptureOrigin(event).kind).toBe("unclassified");
    expect(mainImaginationCaptureOrigin({ ...event, action: { ...event.action, source: "organic" } })).toEqual({ kind: "spontaneous", authority: "main-organic-action", actionEventId: "actual-event" });
    expect(mainImaginationCaptureOrigin({ ...event, action: { ...event.action, source: "human" } }).kind).toBe("user-requested");
  });
  it("authenticates a fresh native job output lease before spontaneous generation and binds the exact returned owner", async () => {
    const fixture = context({ kind: "spontaneous", authority: "main-organic-action", actionEventId: "organic-event" });
    RuntimeJobManager.prototype.generate.call(fixture.owner, { brainId: "brain", modality: "image" }); await fixture.run();
    expect(fixture.request.mock.calls.map(call => call[0])).toEqual(["authorize_spontaneous_output", "generate_modality"]);
    expect(fixture.request.mock.calls[1]?.[1]).toMatchObject({ jobId: "owned-job", spontaneousOutputLeaseId: "a".repeat(64) });
    expect((fixture.owner as unknown as { mediaArtifacts: { noteSpontaneousOutputLease: ReturnType<typeof vi.fn> } }).mediaArtifacts.noteSpontaneousOutputLease).toHaveBeenCalledWith("brain", "owned-job", "a".repeat(64));
  });
  it("drops a renderer/model-provided retention flag for unclassified jobs instead of creating scratch authority", async () => {
    const fixture = context({ kind: "unclassified", authority: "native-brain-action" });
    RuntimeJobManager.prototype.generate.call(fixture.owner, { brainId: "brain", modality: "image", spontaneousOutputLeaseId: "f".repeat(64), origin: "spontaneous" } as never); await fixture.run();
    expect(fixture.request.mock.calls.map(call => call[0])).toEqual(["generate_modality"]); expect(fixture.request.mock.calls[0]?.[1]).not.toHaveProperty("spontaneousOutputLeaseId");
  });
  it("refuses a foreign acknowledgement before decoder dispatch", async () => {
    const fixture = context({ kind: "spontaneous", authority: "main-organic-action", actionEventId: "organic-event" });
    fixture.request.mockResolvedValueOnce({ brainId: "foreign", jobId: "owned-job", leaseId: "a".repeat(64) } as never);
    RuntimeJobManager.prototype.generate.call(fixture.owner, { brainId: "brain", modality: "image" }); await expect(fixture.run()).rejects.toThrow(/not acknowledged/);
    expect(fixture.request).toHaveBeenCalledTimes(1);
  });
});
