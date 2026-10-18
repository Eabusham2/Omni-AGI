import type { ApiTeacherProvider } from "../shared/types";
import { apiTeacherLessonModalities, type ApiTeacherLessonModality, type ApiTeacherMediaKind } from "../shared/apiTeacherMedia";
import { API_TEACHER_ENDPOINTS, apiTeacherHttpRequest, type ApiTeacherHttpInput, type ApiTeacherHttpRequest } from "./apiTeacherProviders";

export type ApiTeacherProviderPart = { type: "text"; text: string } | { type: ApiTeacherMediaKind; mimeType: string; data: string };
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);

export function apiTeacherMediaEndpoint(provider: ApiTeacherProvider, requested: readonly ApiTeacherLessonModality[]): string {
  const modalities = apiTeacherLessonModalities(requested);
  if (!modalities.some(kind => kind !== "text")) return API_TEACHER_ENDPOINTS[provider];
  if (provider !== "gemini" && !(provider === "openai" || provider === "openrouter") ||
      provider !== "gemini" && modalities.some(kind => kind === "audio" || kind === "video"))
    throw new Error(`${provider} has no documented adapter for the requested teacher media; no request was sent.`);
  return provider === "openrouter" ? "https://openrouter.ai/api/v1/images" : API_TEACHER_ENDPOINTS[provider];
}
/** Explicit documented adapters only; no provider/model substitution or arbitrary URLs. */
export function apiTeacherMediaRequest(provider: ApiTeacherProvider, input: ApiTeacherHttpInput,
  requested: readonly ApiTeacherLessonModality[]): ApiTeacherHttpRequest {
  const modalities = apiTeacherLessonModalities(requested), endpoint = apiTeacherMediaEndpoint(provider, modalities);
  if (!modalities.some(kind => kind !== "text")) return apiTeacherHttpRequest(provider, input);
  const request = apiTeacherHttpRequest(provider, input);
  if (provider === "openai") request.body.tools = [{ type: "image_generation", output_format: "png" }];
  else if (provider === "gemini") {
    request.body.response_format = modalities.map(type => type === "text" ? { type } :
      type === "image" ? { type, delivery: "inline", mime_type: "image/jpeg" } :
      type === "audio" ? { type, mime_type: "audio/wav" } : { type, delivery: "inline" });
    request.body.generation_config = { max_output_tokens: input.maxOutputTokens };
  } else {
    request.url = endpoint;
    request.body = { model: input.model, prompt: input.prompt, stream: false, provider: { allow_fallbacks: false } };
  }
  return request;
}

/** Text labels, tool results and remote references never become media assets. */
export function apiTeacherMediaParts(provider: ApiTeacherProvider, body: Record<string, unknown>,
  requested: readonly ApiTeacherLessonModality[]): ApiTeacherProviderPart[] {
  const parts: ApiTeacherProviderPart[] = [];
  const add = (value: unknown): void => {
    if (!record(value) || typeof value.type !== "string") throw new Error("Teacher returned an untyped lesson part.");
    if (value.type === "text" || value.type === "output_text") {
      const fields = provider === "openai" && value.type === "output_text"
        ? ["type", "text", "annotations", "logprobs"] : ["type", "text", "annotations"];
      if (typeof value.text !== "string" || value.annotations !== undefined && !Array.isArray(value.annotations) ||
          value.logprobs !== undefined && !Array.isArray(value.logprobs) || Object.keys(value).some(key => !fields.includes(key)))
        throw new Error("Teacher text part is invalid.");
      if (value.text.trim()) parts.push({ type: "text", text: value.text });
      return;
    }
    if (!["image", "audio", "video"].includes(value.type) || typeof value.data !== "string" ||
        typeof value.mime_type !== "string" || value.uri !== undefined ||
        Object.keys(value).some(key => !(value.type === "audio" ? ["type", "data", "mime_type", "sample_rate", "channels"] : ["type", "data", "mime_type", "resolution"]).includes(key)))
      throw new Error("Teacher media needs actual inline typed bytes; references or captions are not media learning.");
    parts.push({ type: value.type as ApiTeacherMediaKind, mimeType: value.mime_type, data: value.data });
  };
  if (provider === "openai" || provider === "gemini") {
    if (body.status !== "completed") throw new Error("Teacher media response is not a complete provider result.");
    if (body.error != null || provider === "gemini" && body.errors !== undefined && (!Array.isArray(body.errors) || body.errors.length) ||
        provider === "openai" && body.incomplete_details != null)
      throw new Error("Teacher media response reports errors or incomplete results.");
  }
  if (provider === "gemini") {
    if (!Array.isArray(body.steps)) throw new Error("Gemini teacher response has no typed model-output steps.");
    for (const step of body.steps) {
      if (!record(step)) throw new Error("Gemini teacher step is invalid.");
      if (step.type === "thought") {
        if (Object.keys(step).some(key => !["type", "signature", "summary"].includes(key)) ||
            step.signature !== undefined && typeof step.signature !== "string" || step.summary !== undefined && !Array.isArray(step.summary))
          throw new Error("Gemini teacher thought metadata is invalid.");
        continue; // Provider thought summaries are not model-output lesson parts.
      }
      if (step.type !== "model_output" || !Array.isArray(step.content) || step.error != null ||
          Object.keys(step).some(key => !["type", "content", "error"].includes(key)))
        throw new Error("Gemini teacher response contains unsupported non-output instructions.");
      for (const block of step.content) add(block);
    }
  } else if (provider === "openai") {
    if (!Array.isArray(body.output)) throw new Error("OpenAI teacher response has no typed output.");
    for (const output of body.output) {
      if (!record(output)) throw new Error("OpenAI teacher output is invalid.");
      if (output.error != null) throw new Error("OpenAI teacher output reports an error.");
      if (output.type === "image_generation_call") {
        if (output.status !== "completed" || typeof output.result !== "string")
          throw new Error("OpenAI teacher image generation is not a complete image result.");
        parts.push({ type: "image", mimeType: "image/png", data: output.result });
      } else if (output.type === "message" && Array.isArray(output.content)) {
        if (output.status !== "completed" || output.role !== "assistant")
          throw new Error("OpenAI teacher message is not a completed model output.");
        for (const block of output.content) add(block);
      }
      else if (output.type !== "reasoning") throw new Error("OpenAI teacher returned unsupported output instructions.");
    }
  } else if (provider === "openrouter") {
    if (!Array.isArray(body.data)) throw new Error("OpenRouter image teacher has no image data.");
    for (const image of body.data) {
      if (!record(image) || typeof image.b64_json !== "string" || typeof image.media_type !== "string")
        throw new Error("OpenRouter image teacher needs identifiable inline image bytes.");
      parts.push({ type: "image", mimeType: image.media_type, data: image.b64_json });
    }
  } else throw new Error(`${provider} has no documented teacher media response adapter.`);
  const actual = new Set(parts.map(part => part.type));
  for (const kind of apiTeacherLessonModalities(requested).filter(kind => kind !== "text"))
    if (!actual.has(kind)) throw new Error(`Teacher returned no actual ${kind} bytes; text was not counted as media.`);
  if (parts.some(part => part.type !== "text" && !requested.includes(part.type)))
    throw new Error("Teacher returned unrequested media; it was not silently learned.");
  return parts;
}
