import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExternalAdviceClientRequest, ExternalAdviceDeltaHandler } from "../src/main/externalAdviceClient";
import { ExternalAdviceService } from "../src/main/externalAdviceService";
import type { ExternalAdviceChannel, ExternalAdviceEvent } from "../src/shared/externalAdvice";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function base(client: { stream(request: ExternalAdviceClientRequest, onDelta: ExternalAdviceDeltaHandler, signal: AbortSignal): Promise<{ text: string }> },
  admitWrite: (brainId: string, bytes: number, signal: AbortSignal) => Promise<void>, monotonicNow?: () => number) {
  const root = await mkdtemp(join(tmpdir(), "omni-advice-resource-")); roots.push(root); const brainRoot = join(root, "brain"); await mkdir(brainRoot);
  const grants: Record<ExternalAdviceChannel, { brainId: string; channel: ExternalAdviceChannel; enabled: boolean; ownerEpoch: string; revision: number }> = {
    assistant: { brainId: "brain-one", channel: "assistant", enabled: true, ownerEpoch: "assistant-owner", revision: 1 },
    reviewer: { brainId: "brain-one", channel: "reviewer", enabled: false, ownerEpoch: "reviewer-owner", revision: 1 }
  };
  const secrets = { get: async () => undefined, has: async () => false, set: async () => "session" as const, remove: async () => false,
    persistence: () => "session" as const };
  const options = { deviceDirectory: join(root, "device"), brainRoot: () => brainRoot, admitWrite,
    authority: async (_brainId: string, channel: ExternalAdviceChannel) => ({ ...grants[channel] }),
    selectContext: async ({ brainId, channel, selection }: { brainId: string; channel: ExternalAdviceChannel; selection: { includeTask: boolean; includeActivity: boolean; maxCharacters: number } }) => ({
      format: "omni-external-advice-selected-context" as const, version: 1 as const, brainId, channel, selection,
      items: [], includedCharacters: 0, omittedItems: 0, truncated: false }), ...(monotonicNow ? { monotonicNow } : {}) };
  const service = new ExternalAdviceService(client, secrets, options);
  const settings = await service.settings("brain-one");
  await service.configure("brain-one", { expectedRevision: settings.revision, channel: "assistant", configuration: { enabled: true,
    provider: "openai-compatible-local", model: "local-manual", endpoint: "http://127.0.0.1:9988/v1", maxOutputTokens: 65_536,
    context: { includeTask: true, includeActivity: false, maxCharacters: 8_000 } } });
  return { root, brainRoot, service, options };
}

describe("external advice resource-admitted batching and recovery", () => {
  it("checkpoints a long redacted stream in bounded batches instead of one FULL-sync row per delta", async () => {
    let clock = 0; const text = "x".repeat(1_024), calls = vi.fn(async (_brainId: string, _bytes: number, signal: AbortSignal) => signal.throwIfAborted());
    const client = { stream: vi.fn(async (_request: ExternalAdviceClientRequest, onDelta: ExternalAdviceDeltaHandler) => {
      let output = ""; for (let index = 0; index < 200; index += 1) { clock += 10; output += text; await onDelta(text); } return { text: output };
    }) };
    const f = await base(client, calls, () => clock), baseline = calls.mock.calls.length, events: ExternalAdviceEvent[] = [];
    f.service.onEvent(event => events.push(event));
    const preview = await f.service.preview("brain-one", { channel: "assistant", prompt: "stream" });
    const done = new Promise<void>((resolve) => f.service.onEvent(event => { if (event.request.id === "batch-one" && event.request.state === "complete") resolve(); }));
    await f.service.start("brain-one", { channel: "assistant", prompt: "stream", requestId: "batch-one",
      expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 }); await done;
    const startAdmissions = calls.mock.calls.length - baseline;
    expect(startAdmissions).toBeGreaterThanOrEqual(4); expect(startAdmissions).toBeLessThan(12); expect(startAdmissions).toBeLessThan(200);
    expect(calls.mock.calls.every(([, bytes]) => bytes >= 512 * 1024)).toBe(true);
    expect(events.some(event => event.persistence && event.persistence.receivedCharacters > event.persistence.persistedCharacters)).toBe(true);
    const history = await f.service.history("brain-one");
    expect(history.entries[0]).toMatchObject({ id: "batch-one", state: "complete" });
    expect(history.entries[0]!.response).toHaveLength(200 * 1_024); expect(history.activeRequestIds).toEqual([]);
  });

  it("refuses request persistence before provider dispatch when disk admission denies the write", async () => {
    let deny = false;
    const client = { stream: vi.fn(async () => ({ text: "must not run" })) };
    const f = await base(client, async (_brainId, _bytes, signal) => { signal.throwIfAborted(); if (deny) throw new Error("20 GiB reserve would be crossed"); });
    const preview = await f.service.preview("brain-one", { channel: "assistant", prompt: "do not dispatch" }); deny = true;
    await expect(f.service.start("brain-one", { channel: "assistant", prompt: "do not dispatch", requestId: "denied-one",
      expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 })).rejects.toThrow(/20 GiB reserve/);
    expect(client.stream).not.toHaveBeenCalled(); deny = false;
    expect((await f.service.history("brain-one")).entries).toEqual([]);
  });

  it("retires an orphaned prior-process prefix on first history read without replaying the provider", async () => {
    const firstClient = { stream: vi.fn(async () => new Promise<{ text: string }>(() => undefined)) };
    const admit = async (_brainId: string, _bytes: number, signal: AbortSignal) => signal.throwIfAborted();
    const f = await base(firstClient, admit), preview = await f.service.preview("brain-one", { channel: "assistant", prompt: "crash fixture" });
    const streaming = new Promise<void>((resolve) => f.service.onEvent(event => { if (event.request.id === "orphan-one" && event.request.state === "streaming") resolve(); }));
    await f.service.start("brain-one", { channel: "assistant", prompt: "crash fixture", requestId: "orphan-one",
      expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 }); await streaming;

    const secondClient = { stream: vi.fn(async () => ({ text: "must not replay" })) };
    const recovered = new ExternalAdviceService(secondClient, { get: async () => undefined, has: async () => false, set: async () => "session" as const,
      remove: async () => false, persistence: () => "session" as const }, { ...f.options, admitWrite: admit });
    const history = await recovered.history("brain-one");
    expect(history.entries[0]).toMatchObject({ id: "orphan-one", state: "cancelled", response: "" });
    expect(history.entries[0]!.error).toMatch(/previous external-advice process ended/i);
    expect(history.activeRequestIds).toEqual([]); expect(history.retirementBlocked).toBeUndefined();
    expect(firstClient.stream).toHaveBeenCalledTimes(1); expect(secondClient.stream).not.toHaveBeenCalled();
  });
});
