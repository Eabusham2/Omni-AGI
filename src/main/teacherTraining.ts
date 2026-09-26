import { createHash, randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import type {
  ApiTeacherCredentialRequest,
  ApiTeacherProvider,
  ApiTeacherProviderStatus,
  ApiTeacherTrainingReport,
  ApiTeacherTrainingRequest,
  RuntimeJob,
  RuntimeJobEvent
} from "../shared/types";
import type { BrainService } from "./brainService";
import type { SecureSecretStore } from "./secureSecretStore";

type FetchLike = typeof fetch;

const PROVIDERS: ApiTeacherProvider[] = ["openai", "anthropic", "gemini"];
const PROVIDER_URLS: Record<ApiTeacherProvider, string> = {
  openai: "https://api.openai.com/v1/responses",
  anthropic: "https://api.anthropic.com/v1/messages",
  gemini: "https://generativelanguage.googleapis.com/v1beta/interactions"
};

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function requireProvider(value: unknown): ApiTeacherProvider {
  if (!PROVIDERS.includes(value as ApiTeacherProvider)) {
    throw new Error("Unsupported API teacher provider.");
  }
  return value as ApiTeacherProvider;
}

function requireModel(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$/.test(value)
  ) {
    throw new Error("Invalid API teacher model name.");
  }
  return value;
}

function credentialId(provider: ApiTeacherProvider): string {
  return `teacher:${provider}`;
}

function cleanRemoteError(value: string): string {
  return value
    .replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]")
    .replace(/\bAIza[A-Za-z0-9_-]{16,}\b/g, "[REDACTED]")
    .replace(/[\r\n\0]+/g, " ")
    .slice(0, 1_000);
}

function redactKnownSecret(value: string, secret?: string): string {
  const clean = cleanRemoteError(value);
  return secret ? clean.split(secret).join("[REDACTED]") : clean;
}

async function readBoundedBody(response: Response): Promise<Uint8Array> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > 16 * 1024 * 1024) {
    throw new Error("API teacher response exceeded 16 MiB.");
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > 16 * 1024 * 1024) {
      await reader.cancel().catch(() => undefined);
      throw new Error("API teacher response exceeded 16 MiB.");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function responseJson(response: Response): Promise<Record<string, unknown>> {
  const bytes = await readBoundedBody(response);
  const text = new TextDecoder().decode(bytes);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(`API teacher returned invalid JSON (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const record = typeof parsed === "object" && parsed !== null ? parsed as Record<string, unknown> : {};
    const error = typeof record.error === "object" && record.error !== null
      ? record.error as Record<string, unknown>
      : {};
    const message = String(error.message ?? record.message ?? `HTTP ${response.status}`);
    throw new Error(`API teacher request failed: ${cleanRemoteError(message)}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("API teacher returned an invalid response object.");
  }
  return parsed as Record<string, unknown>;
}

function textBlocks(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item !== "object" || item === null || Array.isArray(item)) return [];
    const record = item as Record<string, unknown>;
    if (typeof record.text === "string") return [record.text];
    if (Array.isArray(record.content)) return textBlocks(record.content);
    return [];
  });
}

function extractTeacherText(provider: ApiTeacherProvider, body: Record<string, unknown>): string {
  if (typeof body.output_text === "string") return body.output_text.trim();
  if (provider === "anthropic") return textBlocks(body.content).join("\n").trim();
  return textBlocks(body.output).join("\n").trim();
}

export class ApiTeacherTrainingService extends EventEmitter {
  private readonly jobs = new Map<string, RuntimeJob>();
  private readonly controllers = new Map<string, AbortController>();

  constructor(
    private readonly brain: Pick<BrainService, "learnStructuredExperience" | "preflightStart">,
    private readonly secrets: SecureSecretStore,
    private readonly fetcher: FetchLike = fetch
  ) {
    super();
  }

  private publish(job: RuntimeJob): void {
    job.updatedAt = new Date().toISOString();
    this.emit("event", { job: { ...job } } satisfies RuntimeJobEvent);
  }

  async status(): Promise<ApiTeacherProviderStatus[]> {
    return Promise.all(PROVIDERS.map(async (provider) => {
      const key = await this.secrets.get(credentialId(provider));
      return {
        provider,
        configured: Boolean(key),
        persistence: key ? this.secrets.persistence() : "none",
        ...(key ? { keyHint: `••••${key.slice(-4)}` } : {})
      } satisfies ApiTeacherProviderStatus;
    }));
  }

  async saveCredential(request: ApiTeacherCredentialRequest): Promise<ApiTeacherProviderStatus[]> {
    const provider = requireProvider(request?.provider);
    if (typeof request.apiKey !== "string") throw new Error("API credential is required.");
    await this.secrets.set(credentialId(provider), request.apiKey);
    return this.status();
  }

  async removeCredential(provider: ApiTeacherProvider): Promise<ApiTeacherProviderStatus[]> {
    await this.secrets.remove(credentialId(requireProvider(provider)));
    return this.status();
  }

  list(brainId?: string): RuntimeJob[] {
    return [...this.jobs.values()]
      .filter((job) => !brainId || job.brainId === brainId)
      .map((job) => ({ ...job }))
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
  }

  cancel(jobId: string): RuntimeJob {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error("API teacher job was not found.");
    this.controllers.get(jobId)?.abort(new Error("API teacher training was cancelled."));
    if (job.state === "queued" || job.state === "running") {
      job.state = "cancelled";
      job.label = "API teacher training cancelled";
      this.publish(job);
    }
    return { ...job };
  }

  start(request: ApiTeacherTrainingRequest): RuntimeJob {
    const provider = requireProvider(request?.provider);
    const model = requireModel(request?.model);
    if (typeof request.brainId !== "string" || !request.brainId) {
      throw new Error("A brain is required for API teacher training.");
    }
    if (!Array.isArray(request.prompts) || request.prompts.length === 0) {
      throw new Error("Add at least one explicit learning question.");
    }
    const prompts = request.prompts.map((prompt) => {
      if (typeof prompt !== "string") throw new Error("Teacher prompts must be text.");
      const clean = prompt.replace(/\0/g, "").trim();
      if (!clean || Buffer.byteLength(clean) > 1_000_000) {
        throw new Error("A teacher prompt is empty or exceeds 1 MB.");
      }
      return clean;
    });
    const maxOutputTokens = request.maxOutputTokens ?? 2_048;
    if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens < 1 || maxOutputTokens > 65_536) {
      throw new Error("API teacher output tokens must be between 1 and 65,536.");
    }
    if (
      [...this.jobs.values()].some(
        (job) => job.brainId === request.brainId && ["queued", "running"].includes(job.state)
      )
    ) {
      throw new Error("This brain already has an active API teacher job.");
    }
    const now = new Date().toISOString();
    const job: RuntimeJob = {
      id: randomUUID(),
      brainId: request.brainId,
      kind: "teacher",
      state: "queued",
      progress: 0,
      label: `Preparing ${provider} teacher training`,
      createdAt: now,
      updatedAt: now
    };
    this.jobs.set(job.id, job);
    const controller = new AbortController();
    this.controllers.set(job.id, controller);
    this.publish(job);
    void this.run(job, { ...request, provider, model, prompts, maxOutputTokens }, controller);
    return { ...job };
  }

  private async requestTeacher(
    provider: ApiTeacherProvider,
    model: string,
    prompt: string,
    maxOutputTokens: number,
    apiKey: string,
    signal: AbortSignal
  ): Promise<string> {
    const headers: Record<string, string> = { "content-type": "application/json" };
    let body: Record<string, unknown>;
    if (provider === "openai") {
      headers.authorization = `Bearer ${apiKey}`;
      body = { model, input: prompt, store: false, max_output_tokens: maxOutputTokens };
    } else if (provider === "anthropic") {
      headers["x-api-key"] = apiKey;
      headers["anthropic-version"] = "2023-06-01";
      body = {
        model,
        max_tokens: maxOutputTokens,
        messages: [{ role: "user", content: prompt }]
      };
    } else {
      headers["x-goog-api-key"] = apiKey;
      body = { model, input: prompt, store: false };
    }
    const response = await this.fetcher(PROVIDER_URLS[provider], {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal
    });
    const text = extractTeacherText(provider, await responseJson(response));
    if (!text) throw new Error("API teacher returned no learnable text.");
    return text;
  }

  private async run(
    job: RuntimeJob,
    request: ApiTeacherTrainingRequest & { maxOutputTokens: number },
    controller: AbortController
  ): Promise<void> {
    const report: ApiTeacherTrainingReport = {
      provider: request.provider,
      model: request.model,
      requested: request.prompts.length,
      learned: 0,
      failed: 0,
      requestHashes: [],
      responseHashes: [],
      rawCredentialsStoredInBrain: false,
      hiddenBehavioralPromptUsed: false
    };
    let apiKey = "";
    try {
      await this.brain.preflightStart(request.brainId);
      apiKey = await this.secrets.get(credentialId(request.provider)) ?? "";
      if (!apiKey) throw new Error(`No ${request.provider} API credential is configured.`);
      job.state = "running";
      job.label = `Learning from ${request.provider} · ${request.model}`;
      this.publish(job);
      for (let index = 0; index < request.prompts.length; index += 1) {
        controller.signal.throwIfAborted();
        const prompt = request.prompts[index]!;
        const requestHash = sha256(prompt);
        report.requestHashes.push(requestHash);
        job.label = `Teacher response ${index + 1}/${request.prompts.length}`;
        job.progress = index / request.prompts.length;
        this.publish(job);
        try {
          const answer = await this.requestTeacher(
            request.provider,
            request.model,
            prompt,
            request.maxOutputTokens,
            apiKey,
            controller.signal
          );
          const responseHash = sha256(answer);
          report.responseHashes.push(responseHash);
          const trajectory = JSON.stringify({
            format: "omni-api-teacher-trajectory",
            version: 1,
            teacher: { provider: request.provider, model: request.model },
            objective: "corpus response imitation without preference optimization",
            input: prompt,
            output: answer,
            hiddenBehavioralPrompt: false,
            rewardModel: false,
            rlhf: false
          });
          await this.brain.learnStructuredExperience(
            request.brainId,
            {
              content: trajectory,
              name: `${request.provider}:${request.model}:${requestHash.slice(0, 12)}`,
              sourceLabel: `${request.provider} API teacher trajectory`,
              provenanceUrl: PROVIDER_URLS[request.provider],
              license: "User-authorized API response training",
              licenseUrl: PROVIDER_URLS[request.provider]
            },
            controller.signal
          );
          report.learned += 1;
        } catch (error) {
          if (controller.signal.aborted) throw error;
          report.failed += 1;
          job.label = `Teacher item ${index + 1} failed; continuing`;
          job.output = {
            ...report,
            lastError: redactKnownSecret(error instanceof Error ? error.message : String(error), apiKey)
          };
          this.publish(job);
        }
        job.progress = (index + 1) / request.prompts.length;
        job.output = { ...report };
        this.publish(job);
      }
      if (report.learned === 0) throw new Error("No API teacher trajectories were learned.");
      job.state = "complete";
      job.progress = 1;
      job.label = `Learned ${report.learned}/${report.requested} API teacher trajectories`;
      job.output = report;
      this.publish(job);
    } catch (error) {
      if (controller.signal.aborted || job.state === "cancelled") {
        job.state = "cancelled";
        job.label = "API teacher training cancelled";
      } else {
        job.state = "failed";
        job.error = redactKnownSecret(error instanceof Error ? error.message : String(error), apiKey);
        job.label = "API teacher training failed";
      }
      job.output = report;
      this.publish(job);
    } finally {
      this.controllers.delete(job.id);
    }
  }
}
