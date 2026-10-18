import { expect, it } from "vitest";
import { apiTeacherHttpRequest, apiTeacherResponseText } from "../src/main/apiTeacherProviders";

const input = { model: "gemini-synthetic-shape", prompt: "Synthetic lesson request", maxOutputTokens: 321, credential: "synthetic-not-a-live-key" };
const step = (text = "Actual typed fixture lesson") => ({ type: "model_output", content: [{ type: "text", text }] });

it("sends the documented Gemini Interactions nested output-token budget without adding tools, history or instructions", () => {
  expect(apiTeacherHttpRequest("gemini", input)).toEqual({ url: "https://generativelanguage.googleapis.com/v1beta/interactions",
    headers: { "content-type": "application/json", "x-goog-api-key": input.credential },
    body: { model: input.model, input: input.prompt, store: false, generation_config: { max_output_tokens: 321 } } });
});
it("reads completed documented model-output steps in order and does not substitute thought summaries or an aggregate lookalike", () => {
  const body = { status: "completed", output_text: "Must not replace actual REST steps", steps: [
    { type: "thought", signature: "opaque-provider-thought", summary: [{ type: "text", text: "Not lesson output" }] },
    { type: "model_output", content: [{ type: "text", text: "First" }, { type: "text", text: "Second", annotations: [] }] }, step("Third") ] };
  expect(apiTeacherResponseText("gemini", body)).toBe("First\nSecond\nThird");
});
it("refuses missing, partial, queued, failed and action-required completion status even when text fields exist", () => {
  for (const status of [undefined, "in_progress", "requires_action", "incomplete", "failed", "cancelled", "queued"]) {
    expect(() => apiTeacherResponseText("gemini", { status, output_text: "Not a completed result", steps: [step()] })).toThrow(/not a completed/);
  }
  expect(() => apiTeacherResponseText("gemini", { status: "completed", errors: [{ message: "partial failure" }], steps: [step()] })).toThrow(/reports errors/);
  expect(() => apiTeacherResponseText("gemini", { status: "completed", steps: [{ ...step(), error: { message: "failed output step" } }] })).toThrow(/errored steps/);
});
it("rejects tool/user-input steps and nontext content rather than extracting plausible text around instructions", () => {
  for (const type of ["function_call", "function_result", "mcp_server_tool_call", "google_search_call", "user_input"]) {
    expect(() => apiTeacherResponseText("gemini", { status: "completed", output_text: "Bypass text", steps: [step(), { type, text: "Not lesson data", content: [{ type: "text", text: "Not model output" }] }] })).toThrow(/not tool instructions/);
  }
  for (const block of [{ type: "function_call", text: "run code" }, { type: "image", text: "not image learning", data: "AA==" }, { type: "text", text: "tool disguise", tool_calls: [{}] }]) {
    expect(() => apiTeacherResponseText("gemini", { status: "completed", steps: [{ type: "model_output", content: [{ type: "text", text: "Visible prefix" }, block] }] })).toThrow(/non-text data or tool instructions/);
  }
});
it("does not silently accept legacy output arrays, missing steps, malformed blocks or empty thought-only responses", () => {
  for (const body of [{ output: [{ text: "Wrong wire shape" }] }, { output_text: "SDK helper is not a REST step" }, { steps: [] },
    { steps: [{ type: "thought", summary: [{ type: "text", text: "Only thinking" }] }] }, { steps: [{ type: "model_output" }] },
    { steps: [{ type: "model_output", content: [{ type: "text", text: 3 }] }] }, { steps: [step("   ")] }]) {
    expect(() => apiTeacherResponseText("gemini", { status: "completed", ...body })).toThrow(/Gemini/);
  }
});
it("leaves the other four providers' existing text extraction callbacks unchanged", () => {
  expect(apiTeacherResponseText("openai", { output_text: " OpenAI fixture " })).toBe("OpenAI fixture");
  expect(apiTeacherResponseText("anthropic", { content: [{ type: "text", text: "Claude fixture" }] })).toBe("Claude fixture");
  for (const provider of ["openrouter", "ollama"] as const) expect(apiTeacherResponseText(provider, { choices: [{ message: { content: " Chat fixture ", tool_calls: [], function_call: null } }] })).toBe("Chat fixture");
});
