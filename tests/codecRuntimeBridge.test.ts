import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";
import { CodecRuntimeSetupBridge, codecRuntimeChallenge, type CodecRuntimeChallenge, type CodecRuntimeReceipt } from "../src/main/codecRuntimeBridge";
import type { PreparedVideoRuntime } from "../src/main/videoRuntimeProvisioner";

// Stub protocol/file setup only: no app, neural worker or codec process starts.
const prepared = { state: "ready", executablePath: "/fixture/cache/pinned/ffmpeg", artifactSha256: "a".repeat(64),
  binarySha256: "b".repeat(64), binarySizeBytes: 128, target: "linux-x64" } as const satisfies PreparedVideoRuntime;
const challenge = (requestId: string, overrides: Partial<CodecRuntimeChallenge> = {}): CodecRuntimeChallenge => ({
  challengeId: "c".repeat(32), requestId, brainId: "brain", jobId: "job", streamId: "", actionId: "", purpose: "decode-video", ...overrides
});
function fixture(prepare = vi.fn(async (_signal: AbortSignal, progress: (value: {state: "checking"; target: string; message: string}) => void) => {
  progress({ state: "checking", target: "linux-x64", message: "Checking exact codec runtime" });
  return prepared;
})) {
  const signal = vi.fn();
  const engine = new EngineSupervisor({ appPath: "/fixture/no-app", videoRuntimeCacheRoot: "/fixture/cache", prepareVideoRuntime: prepare, sendSignal: signal });
  const write = vi.fn((_line: string, callback?: (error?: Error) => void) => { callback?.(); return true; });
  const child = Object.assign(new EventEmitter(), { pid: 4242, killed: false, exitCode: null,
    stdin: { writable: true, write }, kill: vi.fn() }) as unknown as ChildProcessWithoutNullStreams;
  const internals = engine as unknown as { child: ChildProcessWithoutNullStreams; consumeLine(line: string): void; terminateChild(): Promise<void> };
  internals.child = child;
  vi.spyOn(engine, "start").mockResolvedValue(true);
  const terminate = vi.spyOn(internals, "terminateChild").mockResolvedValue(undefined);
  const frames = () => write.mock.calls.map(([line]) => JSON.parse(line) as { id: string; method: string; params: Record<string, unknown> });
  const frame = (method: string) => frames().reverse().find((value) => value.method === method)!;
  const response = (method: string, result: unknown) => internals.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: frame(method).id, result }));
  const event = (type: string, data: Record<string, unknown>, owner?: Partial<CodecRuntimeChallenge>) => {
    internals.consumeLine(JSON.stringify({ jsonrpc: "2.0", method: "event", params: { type,
      brainId: owner?.brainId ?? data.brainId, jobId: owner?.jobId ?? data.jobId,
      streamId: owner?.streamId ?? data.streamId, actionId: owner?.actionId ?? data.actionId, data } }));
  };
  return { engine, internals, child, write, frames, frame, response, event, terminate, prepare, signal };
}

describe("first organic and mixed-ingest codec bridge", () => {
  it("rejects malformed ownership and arbitrary setter fields", () => {
    expect(codecRuntimeChallenge(challenge("rpc"))).toBeDefined();
    expect(codecRuntimeChallenge({ ...challenge("rpc"), executablePath: "/arbitrary/ffmpeg" })).toBeUndefined();
    expect(codecRuntimeChallenge(challenge("rpc", { brainId: "../other" }))).toBeUndefined();
    expect(codecRuntimeChallenge(challenge("rpc", { purpose: "chat-prompt" }))).toBeUndefined();
  });

  it("services a busy mixed-ingest request by an out-of-band receipt, never queued configure", async () => {
    const f = fixture();
    const operation = f.engine.request("ingest", { brainId: "brain", jobId: "job", path: "/fixture/mixed.zip", kind: "archive" }, 0);
    await vi.waitFor(() => expect(f.frames()).toHaveLength(1));
    expect(f.prepare).not.toHaveBeenCalled(); // mixed archive is not an explicit media preflight
    const need = challenge(f.frame("ingest").id);
    const setupEvents: Array<Record<string, unknown>> = [];
    f.engine.on("event", (value: Record<string, unknown>) => setupEvents.push(value));
    f.event("codec-runtime-needed", need as unknown as Record<string, unknown>);
    await vi.waitFor(() => expect(f.frame("resolve_codec_runtime")).toBeDefined());
    expect(f.frames().map((value) => value.method)).toEqual(["ingest", "resolve_codec_runtime"]);
    const { purpose: _purpose, ...ownedNeed } = need;
    expect(f.frame("resolve_codec_runtime").params).toMatchObject({ ...ownedNeed, outcome: "ready",
      configuration: { artifactSha256: "a".repeat(64), binarySha256: "b".repeat(64) } });
    expect(setupEvents).toContainEqual(expect.objectContaining({ type: "video-runtime-setup", jobId: "job",
      data: expect.objectContaining({ setupOnly: true, state: "checking" }) }));
    expect(setupEvents.find((value) => value.type === "video-runtime-setup")).not.toHaveProperty("progress");
    f.response("resolve_codec_runtime", { acknowledged: true, challengeId: need.challengeId });
    const { purpose: _releasedPurpose, ...released } = need;
    f.event("codec-runtime-released", released);
    f.response("ingest", { complete: true });
    await expect(operation).resolves.toEqual({ complete: true });
    expect(f.terminate).not.toHaveBeenCalled();
  });

  it("does not provision a forged challenge or enter private control via public request", async () => {
    const f = fixture();
    const operation = f.engine.request("ingest", { brainId: "brain", jobId: "job", path: "/fixture/mixed.zip" }, 0);
    await vi.waitFor(() => expect(f.frames()).toHaveLength(1));
    f.event("codec-runtime-needed", challenge(f.frame("ingest").id, { brainId: "other" }) as unknown as Record<string, unknown>);
    f.event("codec-runtime-needed", challenge("stale-rpc") as unknown as Record<string, unknown>);
    await Promise.resolve();
    expect(f.prepare).not.toHaveBeenCalled();
    await expect(f.engine.request("resolve_codec_runtime", {})).rejects.toThrow("internal verified-main");
    f.response("ingest", { complete: true });
    await operation;
  });

  it("preserves exact inline codec ownership after the original text RPC ends", async () => {
    const f = fixture();
    const operation = f.engine.request("chat", { brainId: "brain", streamId: "turn" }, 0);
    await vi.waitFor(() => expect(f.frames()).toHaveLength(1));
    const requestId = f.frame("chat").id, actionId = "a".repeat(32);
    f.event("inline-imagination-started", { requestId }, { brainId: "brain", streamId: "turn", actionId });
    f.response("chat", { text: "saved reply" });
    await operation;
    const need = challenge(requestId, { jobId: "", streamId: "turn", actionId });
    f.event("codec-runtime-needed", need as unknown as Record<string, unknown>);
    await vi.waitFor(() => expect(f.prepare).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(f.frame("resolve_codec_runtime")).toBeDefined());
    f.response("resolve_codec_runtime", { acknowledged: true, challengeId: need.challengeId });
    const { purpose: _purpose, ...released } = need;
    f.event("codec-runtime-released", released);
    f.event("inline-imagination-finished", { requestId }, { brainId: "brain", streamId: "turn", actionId });
    f.event("codec-runtime-needed", { ...need, challengeId: "d".repeat(32) });
    await Promise.resolve();
    expect(f.prepare).toHaveBeenCalledOnce();
    expect(f.terminate).not.toHaveBeenCalled();
  });

  it("cancels blocked setup through exact control and accepts actual typed cleanup without killing worker", async () => {
    const prepare = vi.fn((_signal: AbortSignal, _progress: unknown) => new Promise<typeof prepared>(() => undefined));
    const f = fixture(prepare);
    const operation = f.engine.request("ingest", { brainId: "brain", jobId: "job", path: "/fixture/mixed.zip" }, 0);
    const rejected = operation.catch((error: unknown) => error);
    await vi.waitFor(() => expect(f.frames()).toHaveLength(1));
    const need = challenge(f.frame("ingest").id);
    f.event("codec-runtime-needed", need as unknown as Record<string, unknown>);
    await vi.waitFor(() => expect(prepare).toHaveBeenCalledOnce());
    const cancellation = f.engine.cancelRequest("job");
    await vi.waitFor(() => expect(f.frame("resolve_codec_runtime")).toBeDefined());
    expect(f.frame("resolve_codec_runtime").params).toMatchObject({ outcome: "cancelled", requestId: need.requestId, challengeId: need.challengeId });
    f.response("resolve_codec_runtime", { acknowledged: true, challengeId: need.challengeId });
    f.internals.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: need.requestId, error: { code: -32800,
      message: "owned codec setup cancelled", data: { codecRuntimeCancelled: true, safeBoundary: true } } }));
    await rejected;
    await expect(cancellation).resolves.toMatchObject({ acknowledged: true, phase: "running", workerTerminationAcknowledged: false, codecSetupCancellationAcknowledged: true });
    expect(f.terminate).not.toHaveBeenCalled();
    expect(f.child.kill).not.toHaveBeenCalled();
  });

  it("cancels a claimed inline artifact through its retired chat scope, not by killing the current worker", async () => {
    const f = fixture();
    const chat = f.engine.request("chat", { brainId: "brain", streamId: "turn" }, 0);
    await vi.waitFor(() => expect(f.frames()).toHaveLength(1));
    const requestId = f.frame("chat").id, actionId = "a".repeat(32);
    f.event("inline-imagination-started", { requestId }, { brainId: "brain", streamId: "turn", actionId });
    f.response("chat", { text: "saved reply" });
    await chat;
    const job = f.engine.request("generate_modality", { brainId: "brain", jobId: "job", modality: "image", neuralActionId: actionId }, 0);
    const rejected = job.catch((error: unknown) => error);
    await vi.waitFor(() => expect(f.frame("generate_modality")).toBeDefined());
    const cancellation = f.engine.cancelRequest("job");
    await vi.waitFor(() => expect(f.frame("cancel_inline_generation")).toBeDefined());
    expect(f.frame("cancel_inline_generation").params).toEqual({ brainId: "brain", streamId: "turn", neuralActionId: actionId });
    f.response("cancel_inline_generation", { requested: true, acknowledged: false });
    f.internals.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: f.frame("generate_modality").id,
      error: { code: -32800, message: "owned inline artifact cancelled after handler cleanup" } }));
    await rejected;
    await expect(cancellation).resolves.toMatchObject({ acknowledged: true, workerTerminationAcknowledged: false, artifactCancellationAcknowledged: true });
    expect(f.signal).not.toHaveBeenCalled();
    expect(f.terminate).not.toHaveBeenCalled();
    f.event("inline-imagination-finished", { requestId }, { brainId: "brain", streamId: "turn", actionId });
  });

  it("a directly generated artifact cooperatively cancels at its real safe boundary", async () => {
    const f = fixture();
    const job = f.engine.request("generate_modality", { brainId: "brain", jobId: "job", modality: "image" }, 0);
    const rejected = job.catch((error: unknown) => error);
    await vi.waitFor(() => expect(f.frame("generate_modality")).toBeDefined());
    const cancellation = f.engine.cancelRequest("job");
    await vi.waitFor(() => expect(f.frame("cancel_artifact_request")).toBeDefined());
    expect(f.frame("cancel_artifact_request").params).toEqual({ requestId: f.frame("generate_modality").id,
      brainId: "brain", jobId: "job", streamId: "", actionId: "" });
    f.response("cancel_artifact_request", { requested: true, requestId: f.frame("generate_modality").id });
    expect(f.signal).not.toHaveBeenCalled();
    f.internals.consumeLine(JSON.stringify({ jsonrpc: "2.0", id: f.frame("generate_modality").id, error: { code: -32800,
      message: "owned modality cancelled", data: { modalityCancelled: true, safeBoundary: true } } }));
    await rejected;
    await expect(cancellation).resolves.toMatchObject({ acknowledged: true, workerTerminationAcknowledged: false, artifactCancellationAcknowledged: true });
    expect(f.terminate).not.toHaveBeenCalled();
    expect(f.child.kill).not.toHaveBeenCalled();
  });
});

describe("shared pinned preparation ownership", () => {
  it("one artifact Stop does not cancel a sibling's shared setup download", async () => {
    let finish!: (value: PreparedVideoRuntime) => void;
    let setupSignal!: AbortSignal;
    const prepare = vi.fn((signal: AbortSignal) => { setupSignal = signal; return new Promise<PreparedVideoRuntime>((resolve) => { finish = resolve; }); });
    const bridge = new CodecRuntimeSetupBridge(prepare);
    const first = new AbortController(), second = new AbortController();
    const receipts: CodecRuntimeReceipt[] = [];
    const send = vi.fn(async (receipt: CodecRuntimeReceipt) => { receipts.push(receipt); });
    bridge.accept(challenge("first", { actionId: "a".repeat(32) }), first.signal, send, vi.fn());
    bridge.accept(challenge("second", { challengeId: "d".repeat(32), actionId: "b".repeat(32) }), second.signal, send, vi.fn());
    await vi.waitFor(() => expect(prepare).toHaveBeenCalledOnce());
    first.abort();
    expect(setupSignal.aborted).toBe(false);
    finish(prepared);
    await vi.waitFor(() => expect(receipts).toHaveLength(2));
    expect(receipts).toEqual(expect.arrayContaining([expect.objectContaining({ requestId: "first", outcome: "cancelled" }),
      expect.objectContaining({ requestId: "second", outcome: "ready" })]));
    second.abort(); // selected owner may still cancel worker-side verification/selection
  });
});
