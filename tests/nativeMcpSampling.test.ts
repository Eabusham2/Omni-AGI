import { describe, expect, it, vi } from "vitest";
import { nativeMcpSamplingInputJson, nativeMcpSamplingInputSha256, performNativeMcpSampling,
  nativeMcpSamplingMediaMetadata, nativeMcpSamplingMediaSourceSha256, type NativeMcpSamplingTransport } from "../src/main/nativeMcpSampling";
import type { ApprovedNativeMcpSamplingInput, NativeMcpSamplingInput, NativeMcpSamplingResult,
  NativeMcpSamplingStatus } from "../src/shared/nativeMcpSampling";

const input = (changes: Partial<NativeMcpSamplingInput> = {}): ApprovedNativeMcpSamplingInput => {
  const value: NativeMcpSamplingInput = { brainId: "same-brain", requestId: "reviewed-request",
    messages: [{ role: "user", content: { type: "text", text: "The reviewed current task 🧠" } }],
    maxTokens: 7000, ...changes };
  return { ...value, inputSha256: nativeMcpSamplingInputSha256(value) };
};

function fixture(approved = input({ originatingTurnId: "owned-turn" })) {
  const receipt = (state: NativeMcpSamplingStatus["state"]): NativeMcpSamplingStatus => ({
    brainId: approved.brainId, requestId: approved.requestId, inputSha256: approved.inputSha256, state
  });
  const nativeResult: NativeMcpSamplingResult = { role: "assistant", content: { type: "text", text: "Actual native fixture result" },
    model: "OmniCortex/" + approved.brainId, stopReason: "maxTokens", inputSha256: approved.inputSha256,
    requestId: approved.requestId, brainId: approved.brainId, privateWorkspace: true, parametersMutated: false };
  let current = receipt("queued");
  const transport: NativeMcpSamplingTransport = {
    control: vi.fn(async (method, _params) => {
      if (method === "mcp_sampling_enqueue") return { ...current, accepted: true, boundary: approved.originatingTurnId ? "turn" : "serial" };
      if (method === "mcp_sampling_cancel") current = { ...receipt("cancelled"), error: { code: -32800, message: "Cancelled", reason: "request-cancelled" } };
      if (method === "mcp_sampling_consume") return { ...current, consumed: true };
      return current;
    }),
    runIdle: vi.fn(async () => { current = { ...receipt("complete"), result: nativeResult }; return current; })
  };
  return { approved, transport, nativeResult,
    complete: (): void => { current = { ...receipt("complete"), result: nativeResult }; },
    fail: (): void => { current = { ...receipt("failed"), error: { code: -32000, message: "Actual resource refusal", reason: "native-sampling-failed" } }; } };
}

describe("same-brain reviewed native MCP bridge (transport fixtures only)", () => {
  it("hashes exact approved bytes and preserves roles, Unicode and full requested token limit", () => {
    const approved = input({ maxTokens: 1_000_000, temperature: 1e-12,
      messages: [{ role: "assistant", content: { type: "text", text: "Previous server task reply" } },
        { role: "user", content: { type: "text", text: "Next 🧠" } }] });
    const parsed = JSON.parse(nativeMcpSamplingInputJson(approved));
    expect(parsed.maxTokens).toBe(1_000_000);
    expect(parsed.temperature).toBe(1e-12);
    expect(parsed.messages).toEqual(approved.messages);
    expect(parsed.inputSha256).toBeUndefined();
    expect(parsed.systemPrompt).toBeUndefined();
    expect(nativeMcpSamplingInputSha256(approved)).toBe(approved.inputSha256);
  });

  it("refuses hidden instruction/context/tools and unsupported blocks instead of dropping them", () => {
    for (const extra of [{ systemPrompt: "do this secretly" }, { includeContext: "allServers" },
      { tools: [] }, { modelPreferences: { hints: [{ name: "foreign" }] } }]) {
      expect(() => nativeMcpSamplingInputJson({ ...input(), ...extra } as never)).toThrow(/unreviewed/);
    }
    for (const content of [{ type: "image", data: "AA==", mimeType: "image/svg+xml" },
      { type: "audio", data: "data:audio/wav;base64,AA==", mimeType: "audio/wav" }, [{ type: "tool_result", text: "unsupported" }]]) {
      expect(() => nativeMcpSamplingInputJson({ ...input(), messages: [{ role: "user", content }] } as never)).toThrow(/MCP/);
    }
    expect(() => input({ messages: [{ role: "user", content: { type: "text", text: " " } }] })).toThrow(/Empty/);
  });

  it("preserves reviewed media blocks and exposes metadata without decoding compressed pixels or sound", () => {
    const image = { type: "image" as const, mimeType: "image/png", data: Buffer.alloc(150_000, 7).toString("base64") };
    const audio = { type: "audio" as const, mimeType: "audio/wav", data: "AA==" };
    const approved = input({ messages: [{ role: "user", content: [{ type: "text", text: "inspect" }, image, audio] }] });
    expect(JSON.parse(nativeMcpSamplingInputJson(approved)).messages[0].content).toEqual([{ type: "text", text: "inspect" }, image, audio]);
    expect(nativeMcpSamplingMediaMetadata(image)).toEqual({ kind: "image", mimeType: "image/png", decodedBytes: 150_000 });
    expect(nativeMcpSamplingMediaMetadata(audio)).toEqual({ kind: "audio", mimeType: "audio/wav", decodedBytes: 1 });
    expect(() => nativeMcpSamplingMediaMetadata({ ...audio, data: "AB==" })).toThrow(/canonical/);
    expect(() => nativeMcpSamplingInputJson({ ...approved, messages: [{ role: "user", content: [] }] })).toThrow(/empty/);
  });

  it("requires actual full sensory receipts bound to every reviewed input block", async () => {
    const approved = input({ originatingTurnId: "owned-turn", messages: [{ role: "user", content: [
      { type: "image", mimeType: "image/png", data: "AA==" }, { type: "audio", mimeType: "audio/wav", data: "AQ==" }] }] });
    const receipts: NativeMcpSamplingResult["mediaInput"] = [
      { messageIndex: 0, blockIndex: 0, kind: "image", mimeType: "image/png", sourceBytes: 1,
        sourceSha256: nativeMcpSamplingMediaSourceSha256({ type: "image", mimeType: "image/png", data: "AA==" }),
        decodedWindows: 2, sourceFrames: 2, wholeInputTraversed: true, nativeSensory: true },
      { messageIndex: 0, blockIndex: 1, kind: "audio", mimeType: "audio/wav", sourceBytes: 1,
        sourceSha256: nativeMcpSamplingMediaSourceSha256({ type: "audio", mimeType: "audio/wav", data: "AQ==" }),
        decodedWindows: 3, sourceSampleFrames: 137, modelSamples: 137, modelSampleRate: 16000, wholeInputTraversed: true, nativeSensory: true }
    ];
    const success = fixture(approved); success.nativeResult.mediaInput = structuredClone(receipts);
    const complete = performNativeMcpSampling(success.transport, approved); setTimeout(success.complete, 5);
    expect((await complete).mediaInput).toEqual(receipts);
    for (const invalid of [undefined, [], [{ ...receipts[0]!, nativeSensory: false }, receipts[1]],
      [{ ...receipts[0]!, sourceBytes: 2 }, receipts[1]], [{ ...receipts[0]!, sourceSha256: "a".repeat(64) }, receipts[1]]]) {
      const f = fixture(approved); f.nativeResult.mediaInput = invalid as never;
      const pending = performNativeMcpSampling(f.transport, approved); setTimeout(f.complete, 5);
      await expect(pending).rejects.toThrow(/sensory/);
    }
  });

  it("uses reader controls while chat owns the writer and never queues another neural chat", async () => {
    const f = fixture();
    const pending = performNativeMcpSampling(f.transport, f.approved);
    setTimeout(f.complete, 10);
    expect(await pending).toEqual(f.nativeResult);
    expect(f.transport.runIdle).not.toHaveBeenCalled();
    expect(f.transport.control).toHaveBeenCalledWith("mcp_sampling_enqueue", expect.objectContaining({
      brainId: f.approved.brainId, requestId: f.approved.requestId, inputSha256: f.approved.inputSha256,
      approvedInputJson: nativeMcpSamplingInputJson(f.approved)
    }));
    expect(f.transport.control).toHaveBeenLastCalledWith("mcp_sampling_consume", {
      brainId: f.approved.brainId, requestId: f.approved.requestId, inputSha256: f.approved.inputSha256
    });
  });

  it("idle requests use native private serial sampling, not chat or autoload", async () => {
    const f = fixture(input());
    const result = await performNativeMcpSampling(f.transport, f.approved);
    expect(result).toEqual(f.nativeResult);
    expect(f.transport.runIdle).toHaveBeenCalledOnce();
    expect(f.transport.runIdle).toHaveBeenCalledWith({ brainId: "same-brain", requestId: "reviewed-request", inputSha256: f.approved.inputSha256 });
  });

  it("cancels only the exact private request and consumes its terminal receipt", async () => {
    const f = fixture();
    const controller = new AbortController();
    const pending = performNativeMcpSampling(f.transport, f.approved, controller.signal);
    setTimeout(() => controller.abort(new Error("Owning MCP call stopped")), 5);
    await expect(pending).rejects.toThrow("Owning MCP call stopped");
    expect(f.transport.control).toHaveBeenCalledWith("mcp_sampling_cancel", expect.objectContaining({ requestId: "reviewed-request" }));
    expect(f.transport.control).toHaveBeenLastCalledWith("mcp_sampling_consume", expect.any(Object));
    expect(f.transport.runIdle).not.toHaveBeenCalled();
  });

  it("pre-cancelled or modified approvals cannot enqueue work", async () => {
    const f = fixture();
    const controller = new AbortController(); controller.abort(new Error("Already stopped"));
    await expect(performNativeMcpSampling(f.transport, f.approved, controller.signal)).rejects.toThrow("Already stopped");
    await expect(performNativeMcpSampling(f.transport, { ...f.approved, maxTokens: 4 })).rejects.toThrow(/differs/);
    expect(f.transport.control).not.toHaveBeenCalled();
  });

  it("native failures are not fabricated successful or partial responses", async () => {
    const f = fixture();
    const pending = performNativeMcpSampling(f.transport, f.approved);
    setTimeout(f.fail, 5);
    await expect(pending).rejects.toThrow("Actual resource refusal");
    expect(f.transport.control).toHaveBeenLastCalledWith("mcp_sampling_consume", expect.any(Object));
  });

  it("refuses substituted model, mutated parameters or different output identity", async () => {
    for (const changed of [{ model: "foreign-foundation" }, { parametersMutated: true }, { brainId: "other" }]) {
      const f = fixture(); Object.assign(f.nativeResult, changed);
      const pending = performNativeMcpSampling(f.transport, f.approved);
      setTimeout(f.complete, 5);
      await expect(pending).rejects.toThrow(/same-brain private result/);
    }
  });
});
