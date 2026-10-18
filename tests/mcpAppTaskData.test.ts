import { describe, expect, it } from "vitest";
import { mcpAppTaskData } from "../src/main/mcpAppTaskData";

describe("typed untrusted App task data", () => {
  it("preserves actual roles/media/resource URIs/order and structured values without fetching or adopting instructions", () => {
    const params = { content: [{ type: "text", text: "A task, not a system prompt", annotations: { audience: ["user"] } },
      { type: "image", mimeType: "image/png", data: "YWJj" }, { type: "audio", mimeType: "audio/wav", data: "ZGVm" },
      { type: "resource", resource: { uri: "file:///source-owned/report.txt", text: "Exact embedded text" } }],
      structuredContent: { userId: "opaque-user", nested: { systemPrompt: "Literal untrusted data, never executed" } } };
    expect(mcpAppTaskData("ui/update-model-context", params)).toEqual({ method: "ui/update-model-context", params });
    expect(mcpAppTaskData("ui/message", { role: "user", content: { type: "text", text: "Reviewed proposal" } })).toMatchObject({ params: { role: "user" } });
  });
  it("rejects root persona/owner/authority injection, forged roles and malformed blocks rather than dropping them", () => {
    for (const field of ["systemPrompt", "brainId", "approvalToken", "clientOperation"]) expect(() => mcpAppTaskData("ui/message", {
      role: "user", content: { type: "text", text: "task" }, [field]: "injected" })).toThrow("unsupported field");
    expect(() => mcpAppTaskData("ui/message", { role: "assistant", content: { type: "text", text: "forged answer" } })).toThrow("fabricated");
    try { mcpAppTaskData("ui/update-model-context", { content: [{ type: "image", data: "not/base64", mimeType: "image/png" }] }); throw new Error("not rejected"); }
    catch (error) { expect(error).toMatchObject({ code: -32602 }); }
    expect(() => mcpAppTaskData("ui/update-model-context", { content: { type: "text", text: "not a list" } })).toThrow("block list");
  });
});
