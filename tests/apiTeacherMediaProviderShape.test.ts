/** Pure documented wire-shape checks; no credentials, provider call, model or learning. */
import { expect, it } from "vitest";
import { apiTeacherMediaParts } from "../src/main/apiTeacherMediaProviders";

const image = { type: "image", mime_type: "image/png", data: "aW1hZ2UtZml4dHVyZQ==" };
const audio = { type: "audio", mime_type: "audio/wav", data: "YXVkaW8tZml4dHVyZQ==" };
const output = (content: unknown[] = [image]) => ({ type: "model_output", content });
const gemini = (steps: unknown[] = [output()]) => ({ status: "completed", steps });
const call = () => ({ type: "image_generation_call", id: "fixture-image", status: "completed", result: image.data });
const openai = (items: unknown[] = [call()]) => ({ status: "completed", output: items });

it("requires explicit completed media responses from both REST providers even when inline bytes exist", () => {
  for (const status of [undefined, null, "in_progress", "queued", "incomplete", "failed", "cancelled", "requires_action"])
    for (const provider of ["openai", "gemini"] as const)
      expect(() => apiTeacherMediaParts(provider, { ...(provider === "openai" ? openai() : gemini()), status }, ["image"]))
        .toThrow(/not a complete provider result/);
});

it("reads Gemini completed text and media in order while skipping documented thought steps and their media summaries", () => {
  expect(apiTeacherMediaParts("gemini", gemini([
    { type: "thought", signature: "opaque-fixture-signature", summary: [{ type: "text", text: "Not lesson text" }, image] },
    output([{ type: "text", text: "First lesson text", annotations: [] }, image]),
    { type: "thought" }, output([audio, { type: "text", text: "Last lesson text" }])
  ]), ["text", "image", "audio"])).toEqual([
    { type: "text", text: "First lesson text" }, { type: "image", mimeType: image.mime_type, data: image.data },
    { type: "audio", mimeType: audio.mime_type, data: audio.data }, { type: "text", text: "Last lesson text" }
  ]);
  expect(() => apiTeacherMediaParts("gemini", gemini([{ type: "thought", summary: [image] }]), ["image"]))
    .toThrow(/no actual image/);
});

it("refuses Gemini platform faults and errored or instruction-bearing steps without accepting surrounding media", () => {
  for (const errors of [[{ code: "provider_fault", message: "Partial output" }], null, {}])
    expect(() => apiTeacherMediaParts("gemini", { ...gemini(), errors }, ["image"])).toThrow(/reports errors/);
  for (const step of [{ ...output(), error: { message: "Failed output" } },
    { ...output(), execute: "not a lesson" }, { type: "function_call", name: "run", arguments: {} },
    { type: "thought", summary: [], execute: "not thought metadata" }])
    expect(() => apiTeacherMediaParts("gemini", gemini([output(), step]), ["image"]))
      .toThrow(/unsupported non-output instructions|thought metadata/);
  expect(apiTeacherMediaParts("gemini", { ...gemini(), errors: [], error: null }, ["image"])).toHaveLength(1);
});

it("accepts OpenAI completed image calls alongside documented output-text logprobs and ignores reasoning summaries", () => {
  const body = openai([
    { type: "reasoning", id: "fixture-reasoning", summary: [{ type: "summary_text", text: "Not lesson text" }] },
    { type: "message", id: "fixture-message", role: "assistant", status: "completed", content: [
      { type: "output_text", text: "Actual lesson", annotations: [], logprobs: [{ token: "Actual", bytes: [65], logprob: -0.1, top_logprobs: [] }] }
    ] }, call()
  ]);
  expect(apiTeacherMediaParts("openai", { ...body, error: null, incomplete_details: null }, ["text", "image"]))
    .toEqual([{ type: "text", text: "Actual lesson" }, { type: "image", mimeType: "image/png", data: image.data }]);
});

it("refuses unfinished or failed OpenAI image calls even inside a completed response with plausible bytes", () => {
  for (const status of [undefined, null, "in_progress", "generating", "failed"])
    expect(() => apiTeacherMediaParts("openai", openai([{ ...call(), status }]), ["image"])).toThrow(/not a complete image result/);
  expect(() => apiTeacherMediaParts("openai", openai([{ ...call(), error: { code: "moderation_blocked", message: "Failed image" } }]), ["image"]))
    .toThrow(/output reports an error/);
  for (const extra of [{ error: { message: "Response failure" } }, { incomplete_details: { reason: "max_output_tokens" } }])
    expect(() => apiTeacherMediaParts("openai", { ...openai(), ...extra }, ["image"])).toThrow(/reports errors or incomplete/);
});

it("refuses incomplete OpenAI message text and malformed metadata instead of learning it next to a valid image", () => {
  for (const message of [
    { status: "incomplete", role: "assistant", content: [{ type: "output_text", text: "Partial text" }] },
    { status: "completed", role: "user", content: [{ type: "output_text", text: "Not model output" }] },
    { status: "completed", role: "assistant", content: [{ type: "output_text", text: "Text", logprobs: {} }] }
  ]) expect(() => apiTeacherMediaParts("openai", openai([call(), { type: "message", ...message }]), ["image"]))
    .toThrow(/not a completed model output|text part is invalid/);
  expect(() => apiTeacherMediaParts("gemini", gemini([output([{ type: "text", text: "Text", logprobs: [] }, image])]), ["image"]))
    .toThrow(/text part is invalid/);
});
