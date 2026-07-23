import { createHash, randomUUID } from "node:crypto";
import { lookup } from "node:dns/promises";
import { lstat, readFile, readdir, realpath, stat } from "node:fs/promises";
import { isIP } from "node:net";
import { basename, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { EventEmitter } from "node:events";
import type {
  AgentMergeFilePreview,
  AgentMergePreview,
  BrainConfig,
  BrainDocument,
  BuildRecipe,
  ChatResult,
  CreateBrainRequest,
  DataIngestionPolicy,
  FeedbackRequest,
  InstalledModalityPack,
  ImportUrlRequest,
  IngestWebRequest,
  IngestResult,
  ModalityGenerateRequest,
  RuntimeHealth,
  RuntimeJob,
  RuntimeJobEvent,
  StartTrainingRequest,
  ToolPermissionLevel,
  ToolPermissionRecord,
  TrainingSource,
  WebCrawlRequest,
  WebCrawlResult
} from "../shared/types";
import { consolidateBrain, learnText, runFallbackChat } from "./adaptiveCore";
import {
  listInstalledPacks,
  recordInstalledPack,
  removeStagedPack,
  stageModalityPack,
  validateBuildRecipe
} from "./catalogInstaller";
import { extractConcepts, normalizeConcept } from "./core/language";
import { BrainRepository } from "./brainRepository";
import { EngineSupervisor, type EngineEvent } from "./engineSupervisor";

const MAX_INGEST_FILE_BYTES = 128 * 1024 * 1024;
const MAX_EXTRACTED_TEXT_CHARS = 16_000_000;
const MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024;
const MAX_FOLDER_FILES = 2_000;
const MAX_MERGE_FILES = 512;
const MAX_MERGE_FILE_BYTES = 128 * 1024 * 1024;
const MAX_MERGE_TOTAL_BYTES = 512 * 1024 * 1024;
const MAX_MERGE_CONFLICTS = 100;
const TOOL_ACTIONS: Readonly<Record<string, readonly string[]>> = {
  "windows.files": ["list", "read", "write"],
  "windows.powershell": ["run"],
  "code.execute": ["run"],
  "web.fetch": ["fetch"],
  "web.search": ["search"],
  "browser.automation": ["task"],
  "modality.imagine": ["generate"],
  "agent.fork": ["start"],
  "source.self-modify": ["propose", "diff", "test", "promote"]
};

interface WorkerChatResult {
  text?: string;
  response?: string;
  content?: string;
  runtime?: string;
  trace?: {
    id?: string;
    created_at?: string;
    seed?: number;
    parameter_checksum_before?: string;
    parameter_checksum_after?: string;
    parameter_delta_norm?: number;
    stdp_update?: number;
    spike_rate?: number;
    train_loss?: number;
    generation_entropy?: number;
    ponder_steps?: number;
    steps?: Array<{ stage?: string; detail?: string; value?: string }>;
    note?: string;
  };
  metrics?: Record<string, unknown>;
  runtimeCard?: Record<string, unknown>;
}

interface JobRecord extends RuntimeJob {
  cancelled?: boolean;
}

interface MergeFileCandidate extends AgentMergeFilePreview {
  absoluteSourcePath?: string;
  blobHash?: string;
  evidenceFingerprint?: string;
}

interface MergePlan {
  source: BrainDocument;
  target: BrainDocument;
  evidence: Array<{ fingerprint: string; source: TrainingSource }>;
  files: MergeFileCandidate[];
  preview: AgentMergePreview;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function safeMergeName(value: string): string {
  return (
    value
      .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "-")
      .replace(/[. ]+$/g, "")
      .slice(0, 100) || "artifact"
  );
}

function portableRelative(path: string): string {
  return path.split(sep).join("/");
}

function pathWithin(root: string, candidate: string): boolean {
  const value = relative(resolve(root), resolve(candidate));
  return value === "" || (!isAbsolute(value) && value !== ".." && !value.startsWith(`..${sep}`));
}

async function containedRegularFile(
  root: string,
  candidate: string
): Promise<string | undefined> {
  try {
    const info = await lstat(candidate);
    if (info.isSymbolicLink() || !info.isFile()) return undefined;
    const [realRoot, realCandidate] = await Promise.all([realpath(root), realpath(candidate)]);
    return pathWithin(realRoot, realCandidate) ? realCandidate : undefined;
  } catch {
    return undefined;
  }
}

async function assertSafeDestinationParents(root: string, candidate: string): Promise<void> {
  if (!pathWithin(root, candidate) || resolve(root) === resolve(candidate)) {
    throw new Error("Merge destination escapes the target brain directory.");
  }
  const relativeDestination = relative(resolve(root), resolve(candidate));
  const parts = relativeDestination.split(sep).slice(0, -1);
  let current = resolve(root);
  for (const part of parts) {
    current = join(current, part);
    try {
      const info = await lstat(current);
      if (info.isSymbolicLink() || !info.isDirectory()) {
        throw new Error("Merge destination contains a symbolic link or non-directory parent.");
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
  }
}

function evidenceFingerprint(source: TrainingSource): string {
  if (source.contentHash && /^[a-f0-9]{64}$/i.test(source.contentHash)) {
    return `content:${source.contentHash.toLocaleLowerCase()}`;
  }
  if (source.blobHash && /^[a-f0-9]{64}$/i.test(source.blobHash)) {
    return `blob:${source.blobHash.toLocaleLowerCase()}`;
  }
  return `metadata:${sha256(
    JSON.stringify({
      name: source.name,
      kind: source.kind,
      bytes: source.bytes,
      provenanceUrl: source.provenanceUrl ?? "",
      importedAt: source.importedAt
    })
  )}`;
}

function retainsMergedBlob(target: BrainDocument, source: TrainingSource): boolean {
  const recipe = target.config.memoryRecipe ?? "human-consolidation";
  if (recipe === "synapses-only") return false;
  return (
    recipe === "total-recall" ||
    source.policy === "archive" ||
    source.kind === "image" ||
    source.kind === "audio" ||
    source.kind === "video"
  );
}

function cleanMessage(value: string): string {
  const clean = value.replace(/\0/g, "").trim();
  if (!clean) throw new Error("A chat message cannot be empty.");
  if (clean.length > 100_000) throw new Error("A chat message cannot exceed 100,000 characters.");
  return clean;
}

function normalizeToolId(toolId: string): string {
  const normalized = toolId.trim().toLocaleLowerCase();
  if (!/^[a-z][a-z0-9.-]{1,79}$/.test(normalized)) throw new Error("Invalid tool id.");
  return normalized;
}

function trainingKind(path: string): TrainingSource["kind"] {
  const extension = extname(path).toLocaleLowerCase();
  if (extension === ".pdf") return "pdf";
  if (extension === ".txt") return "text";
  if ([".md", ".mdx", ".rst"].includes(extension)) return "markdown";
  if ([".json", ".jsonl"].includes(extension)) return "json";
  if (
    [
      ".ts",
      ".tsx",
      ".js",
      ".jsx",
      ".py",
      ".rs",
      ".go",
      ".java",
      ".cs",
      ".cpp",
      ".cc",
      ".c",
      ".h",
      ".hpp",
      ".swift",
      ".kt",
      ".sql",
      ".html",
      ".css",
      ".scss",
      ".yaml",
      ".yml",
      ".toml"
    ].includes(extension)
  ) {
    return "code";
  }
  if ([".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"].includes(extension)) {
    return "image";
  }
  if ([".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"].includes(extension)) return "audio";
  if ([".mp4", ".webm", ".mov", ".mkv", ".avi"].includes(extension)) return "video";
  return "unknown";
}

function workerText(result: WorkerChatResult | undefined): string | undefined {
  if (!result) return undefined;
  for (const candidate of [result.text, result.response, result.content]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  }
  return undefined;
}

function displayToolLabel(toolId: string): string {
  return toolId
    .split(/[.-]/)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function privateAddress(address: string): boolean {
  const normalized = address.replace(/^\[|\]$/g, "").toLocaleLowerCase();
  if (normalized === "::1" || normalized === "0:0:0:0:0:0:0:1") return true;
  if (normalized.startsWith("fc") || normalized.startsWith("fd") || /^fe[89ab]/.test(normalized)) {
    return true;
  }
  if (isIP(normalized) === 4) {
    const parts = normalized.split(".").map(Number);
    const first = parts[0] ?? 0;
    const second = parts[1] ?? 0;
    return (
      first === 0 ||
      first === 10 ||
      first === 127 ||
      (first === 169 && second === 254) ||
      (first === 172 && second >= 16 && second <= 31) ||
      (first === 192 && second === 168) ||
      first >= 224
    );
  }
  return false;
}

async function assertSafeRemoteUrl(url: URL): Promise<void> {
  if (url.username || url.password) throw new Error("URLs containing credentials are not allowed.");
  const hostname = url.hostname.replace(/^\[|\]$/g, "").toLocaleLowerCase();
  const localhost = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
  const allowLocal = process.env.OMNI_ALLOW_LOCAL_URLS === "1";
  if (url.protocol !== "https:" && !(allowLocal && url.protocol === "http:" && localhost)) {
    throw new Error("Remote URLs require HTTPS.");
  }
  if (localhost) {
    if (!allowLocal) throw new Error("Private or loopback network URLs are not allowed.");
    return;
  }
  const addresses = isIP(hostname)
    ? [{ address: hostname }]
    : await lookup(hostname, { all: true, verbatim: true });
  if (addresses.length === 0 || addresses.some((entry) => privateAddress(entry.address))) {
    throw new Error("Remote URL resolves to a private or reserved network address.");
  }
}

export async function safeFetch(
  initialUrl: URL,
  init: RequestInit,
  maximumRedirects = 5
): Promise<Response> {
  let current = initialUrl;
  for (let redirect = 0; redirect <= maximumRedirects; redirect += 1) {
    init.signal?.throwIfAborted();
    await assertSafeRemoteUrl(current);
    init.signal?.throwIfAborted();
    const response = await fetch(current, { ...init, redirect: "manual" });
    if (![301, 302, 303, 307, 308].includes(response.status)) return response;
    const location = response.headers.get("location");
    if (!location) throw new Error("Remote server returned a redirect without a location.");
    if (redirect === maximumRedirects) throw new Error("Remote URL redirected too many times.");
    await response.body?.cancel("following validated redirect").catch(() => undefined);
    current = new URL(location, current);
  }
  throw new Error("Remote URL redirected too many times.");
}

export async function readResponseBounded(
  response: Response,
  maximumBytes: number
): Promise<Buffer> {
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (Number.isFinite(declared) && declared > maximumBytes) {
    throw new Error("Remote content exceeds the allowed size.");
  }
  if (!response.body) return Buffer.alloc(0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > maximumBytes) {
        await reader.cancel("response size limit exceeded");
        throw new Error("Remote content exceeds the allowed size.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)), total);
}

async function expandInputPaths(inputPaths: string[]): Promise<string[]> {
  const files: string[] = [];
  const queue = [...new Set(inputPaths)].map((path) => ({ path, depth: 0 }));
  while (queue.length > 0 && files.length < MAX_FOLDER_FILES) {
    const next = queue.shift();
    if (!next) break;
    const info = await lstat(next.path);
    if (info.isSymbolicLink()) continue;
    if (info.isFile()) {
      files.push(next.path);
      continue;
    }
    if (!info.isDirectory() || next.depth >= 16) continue;
    const children = await readdir(next.path);
    for (const name of children.sort()) {
      if (files.length + queue.length >= MAX_FOLDER_FILES) break;
      queue.push({ path: join(next.path, name), depth: next.depth + 1 });
    }
  }
  return files;
}

function htmlToText(value: string): string {
  return value
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function linksFromHtml(value: string, base: URL): URL[] {
  const links: URL[] = [];
  const pattern = /<a\b[^>]*\bhref\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))/gi;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value)) && links.length < 2_000) {
    const raw = match[1] ?? match[2] ?? match[3];
    if (!raw || raw.startsWith("#")) continue;
    try {
      const url = new URL(raw, base);
      url.hash = "";
      if (url.protocol === "https:" || url.protocol === "http:") links.push(url);
    } catch {
      // Ignore malformed links from untrusted pages.
    }
  }
  return links;
}

function robotsDisallows(value: string): string[] {
  const disallowed: string[] = [];
  let applies = false;
  for (const rawLine of value.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, "").trim();
    const separator = line.indexOf(":");
    if (separator < 0) continue;
    const key = line.slice(0, separator).trim().toLocaleLowerCase();
    const entry = line.slice(separator + 1).trim();
    if (key === "user-agent") applies = entry === "*";
    else if (key === "disallow" && applies && entry) disallowed.push(entry);
  }
  return disallowed;
}

export class BrainService {
  constructor(
    readonly repository: BrainRepository,
    readonly engine: EngineSupervisor
  ) {}

  async create(request: CreateBrainRequest): Promise<BrainDocument> {
    if (request.origin && !["blank", "starter"].includes(request.origin)) {
      throw new Error("Invalid brain origin.");
    }
    if (
      request.hardwareTier &&
      !["micro", "personal", "gpu", "workstation"].includes(request.hardwareTier)
    ) {
      throw new Error("Invalid hardware tier.");
    }
    const modalities = [...new Set(request.modalities ?? [])];
    if (modalities.some((value) => !["vision", "image", "audio", "video"].includes(value))) {
      throw new Error("Invalid modality selection.");
    }
    if ((request.initialToolPermissions?.length ?? 0) > 100) {
      throw new Error("Too many initial tool permission records.");
    }
    for (const permission of request.initialToolPermissions ?? []) {
      normalizeToolId(permission.toolId);
      if (!["off", "ask", "auto", "full"].includes(permission.level)) {
        throw new Error("Invalid initial tool permission level.");
      }
    }
    let brain: BrainDocument;
    const starter = request.origin === "starter";
    if (starter) {
      if (!request.starterUrl) throw new Error("A starter brain requires a verified .omni URL.");
      brain = await this.importUrl({ url: request.starterUrl });
      try {
        const materialized = await stat(
          join(this.repository.brainDirectory(brain.id), "engine", "brain.json")
        );
        if (!materialized.isFile()) throw new Error("not a file");
      } catch {
        await this.repository.remove(brain.id).catch(() => false);
        throw new Error(
          "The selected starter contains no materialized OmniCortex checkpoint."
        );
      }
      brain.config = { ...brain.config, ...request.config, runtime: "adaptive-core" };
      brain.name = brain.config.name;
    } else {
      brain = await this.repository.create(request.config);
    }
    if (request.initialToolPermissions) {
      const selected = new Map(
        request.initialToolPermissions.map((permission) => [
          normalizeToolId(permission.toolId),
          permission.level
        ])
      );
      brain.toolPermissions = (brain.toolPermissions ?? []).map((permission) => ({
        ...permission,
        level: selected.get(permission.toolId) ?? permission.level,
        updatedAt: new Date().toISOString()
      }));
      for (const [toolId, level] of selected) {
        if (!brain.toolPermissions.some((permission) => permission.toolId === toolId)) {
          brain.toolPermissions.push({
            toolId,
            label: displayToolLabel(toolId),
            level,
            updatedAt: new Date().toISOString()
          });
        }
      }
    }
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: new Date().toISOString(),
        kind: "system",
        summary: `Build profile: ${request.hardwareTier ?? "automatic"}; modalities: ${(request.modalities ?? []).join(", ") || "text"}.`
      }
    ];
    brain = await this.repository.save(brain);
    const storagePath = this.repository.brainDirectory(brain.id);
    if (starter) {
      // A starter is already a complete safe-tensor checkpoint. Loading it and
      // applying only shape-preserving builder controls keeps its pretrained
      // parameters intact; calling "create" here would silently randomize it.
      await this.engine.tryRequest("unload", { brainId: brain.id }, 30_000);
      await this.engine.tryRequest(
        "load",
        {
          brainId: brain.id,
          config: brain.config,
          storagePath
        },
        300_000
      );
      await this.engine.tryRequest(
        "update_config",
        {
          brainId: brain.id,
          config: brain.config,
          storagePath
        },
        300_000
      );
    } else {
      await this.engine.tryRequest(
        "create",
        {
          brainId: brain.id,
          config: brain.config,
          hardwareTier: request.hardwareTier,
          modalities,
          origin: request.origin ?? "blank",
          storagePath
        },
        60_000
      );
    }
    return brain;
  }

  async chat(id: string, input: string, signal?: AbortSignal): Promise<ChatResult> {
    signal?.throwIfAborted();
    const brain = await this.repository.get(id);
    const message = cleanMessage(input);
    const toolSchemas = (brain.toolPermissions ?? [])
      .filter((permission) => permission.level !== "off")
      .flatMap((permission) => {
        const actions = TOOL_ACTIONS[permission.toolId];
        return actions
          ? [
              {
                id: permission.toolId,
                actions,
                grant: permission.level
              }
            ]
          : [];
      })
      .slice(0, 100);
    await this.engine.tryRequest(
      "load",
      {
        brainId: id,
        config: brain.config,
        storagePath: this.repository.brainDirectory(id)
      },
      30_000,
      signal
    );
    signal?.throwIfAborted();
    const workerResult = await this.engine.tryRequest<WorkerChatResult>(
      "chat",
      {
        brainId: id,
        input: message,
        toolSchemas,
        config: brain.config,
        storagePath: this.repository.brainDirectory(id)
      },
      300_000,
      signal
    );
    signal?.throwIfAborted();
    const result = runFallbackChat(brain, message, workerText(workerResult));
    if (workerText(workerResult)) {
      const runtime = "adaptive-core";
      result.humanMessage.runtime = runtime;
      result.brainMessage.runtime = runtime;
      result.trace.runtime = runtime;
      if (workerResult?.trace?.id) {
        result.trace.id = workerResult.trace.id;
        result.brainMessage.traceId = workerResult.trace.id;
      }
      if (workerResult?.trace?.created_at) {
        result.trace.createdAt = workerResult.trace.created_at;
      }
      if (typeof workerResult?.trace?.seed === "number") {
        result.trace.seed = workerResult.trace.seed;
      }
      if (workerResult?.trace?.steps) {
        result.trace.steps = [
          ...workerResult.trace.steps
            .filter(
              (step): step is { stage: string; detail: string; value?: string } =>
                typeof step.stage === "string" && typeof step.detail === "string"
            )
            .slice(0, 100),
          {
            stage: "desktop-association-index",
            detail:
              "Mirrored the learned turn into the inspectable concept graph used by the Windows interface."
          }
        ];
      }
      const trace = workerResult?.trace;
      if (trace) {
        const mutations = [
          typeof trace.parameter_delta_norm === "number"
            ? `parameter delta ${trace.parameter_delta_norm.toExponential(4)}`
            : undefined,
          typeof trace.stdp_update === "number"
            ? `STDP update ${trace.stdp_update.toExponential(4)}`
            : undefined,
          typeof trace.train_loss === "number" ? `loss ${trace.train_loss.toFixed(6)}` : undefined,
          trace.parameter_checksum_before && trace.parameter_checksum_after
            ? `${trace.parameter_checksum_before.slice(0, 12)} → ${trace.parameter_checksum_after.slice(0, 12)}`
            : undefined
        ].filter(Boolean);
        if (mutations.length > 0) {
          result.trace.steps.push({
            stage: "verified-neural-mutation",
            detail: mutations.join("; ")
          });
        }
        if (typeof trace.ponder_steps === "number") {
          result.trace.branches = Math.max(1, Math.round(trace.ponder_steps));
          result.trace.selectedBranch = result.trace.branches - 1;
        }
      }
      if (workerResult?.trace?.note) result.trace.note = workerResult.trace.note;
      const metrics = workerResult?.metrics;
      if (metrics) {
        if (typeof metrics.plasticityEvents === "number") {
          result.brain.counters.plasticityEvents = metrics.plasticityEvents;
        }
        const counters =
          typeof metrics.counters === "object" && metrics.counters !== null
            ? (metrics.counters as Record<string, unknown>)
            : {};
        if (typeof counters.inference_count === "number") {
          result.brain.counters.inferenceCount = counters.inference_count;
        }
        if (typeof counters.consolidation_cycles === "number") {
          result.brain.counters.consolidationCycles = counters.consolidation_cycles;
        }
      }
    }
    result.brain = await this.repository.save(result.brain);
    return result;
  }

  async feedback(request: FeedbackRequest): Promise<BrainDocument> {
    const brain = await this.repository.get(request.brainId);
    const message = brain.messages.find((entry) => entry.id === request.messageId);
    if (!message) throw new Error("The message was not found.");
    const keys = new Set(extractConcepts(message.content, 64).map((entry) => normalizeConcept(entry.key)));
    const conceptIds = new Set(
      Object.values(brain.concepts)
        .filter((concept) => keys.has(normalizeConcept(concept.label)))
        .map((concept) => concept.id)
    );
    const direction = request.direction === "up" ? 1 : -1;
    for (const synapse of Object.values(brain.synapses)) {
      if (!conceptIds.has(synapse.sourceId) && !conceptIds.has(synapse.targetId)) continue;
      synapse.latentWeight = Math.max(
        -1,
        Math.min(1, synapse.latentWeight + direction * brain.config.learningRate * 0.1)
      );
      synapse.effectiveWeight =
        synapse.latentWeight >= 0.2 ? 1 : synapse.latentWeight <= -0.2 ? -1 : 0;
      synapse.stability = Math.max(0, Math.min(1, synapse.stability + direction * 0.01));
      synapse.lastUpdatedAt = new Date().toISOString();
    }
    return this.repository.save(brain);
  }

  async consolidate(id: string): Promise<BrainDocument> {
    const brain = consolidateBrain(await this.repository.get(id));
    await this.engine.tryRequest(
      "consolidate",
      {
        brainId: id,
        config: brain.config,
        storagePath: this.repository.brainDirectory(id)
      },
      300_000
    );
    return this.repository.save(brain);
  }

  async ingestPaths(
    brainId: string,
    paths: string[],
    policy: DataIngestionPolicy = "encode"
  ): Promise<IngestResult[]> {
    let brain = await this.repository.get(brainId);
    const results: IngestResult[] = [];
    const expandedPaths = await expandInputPaths(paths);
    for (const path of expandedPaths) {
      const fileInfo = await stat(path);
      if (!fileInfo.isFile()) continue;
      if (fileInfo.size > MAX_INGEST_FILE_BYTES) {
        throw new Error(`${basename(path)} exceeds the 128 MB per-file ingestion limit.`);
      }
      const bytes = await readFile(path);
      const contentHash = sha256(bytes);
      const duplicate = brain.trainingSources.find((source) => source.contentHash === contentHash);
      if (duplicate) {
        results.push({
          brain,
          source: duplicate,
          warnings: [`${basename(path)} was already encoded; no duplicate synapses were created.`]
        });
        continue;
      }
      const kind = trainingKind(path);
      const warnings: string[] = [];
      let text = "";
      if (kind === "pdf") {
        const parserModule = await import("pdf-parse");
        const parser = parserModule.default;
        const parsed = await parser(bytes);
        text = parsed.text;
        if (!text.trim()) warnings.push("The PDF contained no extractable text; OCR is not enabled.");
      } else if (["text", "markdown", "code", "json"].includes(kind)) {
        text = bytes.toString("utf8");
      } else if (["image", "audio", "video"].includes(kind)) {
        warnings.push(
          `The ${kind} file was registered for the neural modality worker; the local text fallback cannot decode it.`
        );
      } else {
        const decoded = bytes.toString("utf8");
        const replacementRatio =
          decoded.length === 0 ? 1 : (decoded.match(/\uFFFD/g)?.length ?? 0) / decoded.length;
        if (replacementRatio < 0.01) text = decoded;
        else warnings.push("The file appears binary and has no recognized modality.");
      }
      text = text.replace(/\0/g, "").slice(0, MAX_EXTRACTED_TEXT_CHARS);
      const beforeConcepts = Object.keys(brain.concepts).length;
      const beforeSynapses = Object.keys(brain.synapses).length;
      const beforeIdeas = brain.ideas.length;
      if (text && policy !== "archive") {
        learnText(brain, text, "document", basename(path));
      }
      await this.engine.tryRequest(
        "ingest",
        {
          brainId,
          path,
          kind,
          policy,
          contentHash,
          config: brain.config,
          storagePath: this.repository.brainDirectory(brainId)
        },
        600_000
      );
      if (policy === "consolidate") consolidateBrain(brain);
      const retainRaw =
        (brain.config.memoryRecipe ?? "human-consolidation") === "total-recall" &&
        brain.config.retainSourceText &&
        Boolean(text);
      const memoryRecipe = brain.config.memoryRecipe ?? "human-consolidation";
      const preserveBlob =
        memoryRecipe !== "synapses-only" &&
        (policy === "archive" ||
          memoryRecipe === "total-recall" ||
          kind === "image" ||
          kind === "audio" ||
          kind === "video");
      const source: TrainingSource = {
        id: randomUUID(),
        name: basename(path),
        path,
        kind,
        bytes: bytes.byteLength,
        learnedIdeas: brain.ideas.length - beforeIdeas,
        learnedConcepts: Object.keys(brain.concepts).length - beforeConcepts,
        learnedSynapses: Object.keys(brain.synapses).length - beforeSynapses,
        importedAt: new Date().toISOString(),
        rawTextRetained: retainRaw,
        rawText: retainRaw ? text : undefined,
        contentHash,
        blobHash: preserveBlob ? await this.repository.storeBlob(bytes) : undefined,
        policy,
        license: "User-provided source; license not declared"
      };
      brain.trainingSources.push(source);
      brain.journal = [
        ...(brain.journal ?? []),
        {
          id: randomUUID(),
          createdAt: source.importedAt,
          kind: "learning",
          summary: `${policy === "archive" ? "Archived" : "Learned from"} ${source.name}.`,
          detail: `${source.learnedIdeas} ideas, ${source.learnedConcepts} concepts, ${source.learnedSynapses} synapses`
        }
      ];
      brain = await this.repository.save(brain);
      results.push({ brain, source, warnings });
    }
    return results;
  }

  async ingestWeb(request: IngestWebRequest): Promise<IngestResult> {
    const url = new URL(request.url);
    const response = await safeFetch(url, {
      signal: AbortSignal.timeout(120_000),
      headers: { Accept: "text/html, text/plain, application/json;q=0.9" }
    });
    if (!response.ok) throw new Error(`Web ingestion failed with HTTP ${response.status}.`);
    const finalUrl = new URL(response.url);
    await assertSafeRemoteUrl(finalUrl);
    const raw = (
      await readResponseBounded(response, MAX_EXTRACTED_TEXT_CHARS)
    ).toString("utf8");
    const contentType = response.headers.get("content-type") ?? "";
    const text = contentType.includes("html") ? htmlToText(raw) : raw.replace(/\0/g, "");
    const contentHash = sha256(text);
    let brain = await this.repository.get(request.brainId);
    const duplicate = brain.trainingSources.find((source) => source.contentHash === contentHash);
    if (duplicate) {
      return {
        brain,
        source: duplicate,
        warnings: ["This web content was already ingested; no duplicate synapses were created."]
      };
    }
    const policy = request.policy ?? "encode";
    const quarantined = request.quarantine ?? true;
    const before = {
      ideas: brain.ideas.length,
      concepts: Object.keys(brain.concepts).length,
      synapses: Object.keys(brain.synapses).length
    };
    if (!quarantined && policy !== "archive") {
      learnText(brain, text, "document", finalUrl.toString());
    }
    await this.engine.tryRequest(
      "ingest",
      {
        brainId: brain.id,
        url: finalUrl.toString(),
        text: quarantined ? undefined : text,
        kind: "text",
        policy: quarantined ? "archive" : policy,
        quarantine: quarantined,
        contentHash,
        storagePath: this.repository.brainDirectory(brain.id)
      },
      600_000
    );
    if (!quarantined && policy === "consolidate") consolidateBrain(brain);
    const retainRaw =
      !quarantined &&
      (brain.config.memoryRecipe ?? "human-consolidation") === "total-recall" &&
      brain.config.retainSourceText;
    const source: TrainingSource = {
      id: randomUUID(),
      name: finalUrl.hostname + finalUrl.pathname,
      kind: "text",
      bytes: Buffer.byteLength(raw),
      learnedIdeas: brain.ideas.length - before.ideas,
      learnedConcepts: Object.keys(brain.concepts).length - before.concepts,
      learnedSynapses: Object.keys(brain.synapses).length - before.synapses,
      importedAt: new Date().toISOString(),
      rawTextRetained: retainRaw,
      rawText: retainRaw ? text : undefined,
      contentHash,
      blobHash:
        (brain.config.memoryRecipe ?? "human-consolidation") !== "synapses-only" &&
        (quarantined || (brain.config.memoryRecipe ?? "human-consolidation") === "total-recall")
          ? await this.repository.storeBlob(Buffer.from(raw, "utf8"))
          : undefined,
      policy: quarantined ? "archive" : policy,
      provenanceUrl: finalUrl.toString(),
      license: "Web source; verify the publisher's terms",
      licenseUrl: finalUrl.toString()
    };
    brain.trainingSources.push(source);
    brain = await this.repository.save(brain);
    return {
      brain,
      source,
      warnings: quarantined
        ? ["The downloaded source is quarantined and has not changed neural parameters."]
        : []
    };
  }

  async crawlWeb(
    request: WebCrawlRequest,
    cancelled: () => boolean = () => false,
    progress: (value: number, message: string) => void = () => undefined
  ): Promise<WebCrawlResult> {
    const start = new URL(request.url);
    await assertSafeRemoteUrl(start);
    const maximumPages = Math.max(1, Math.min(50, Math.round(request.maxPages ?? 8)));
    const maximumDepth = Math.max(0, Math.min(4, Math.round(request.maxDepth ?? 1)));
    const sameOrigin = request.sameOrigin ?? true;
    const respectRobots = request.respectRobots ?? true;
    const warnings: string[] = [];
    const robotsByOrigin = new Map<string, string[]>();
    const rulesFor = async (url: URL): Promise<string[]> => {
      if (!respectRobots) return [];
      const existing = robotsByOrigin.get(url.origin);
      if (existing) return existing;
      let rules: string[] = [];
      try {
        const robotsUrl = new URL("/robots.txt", url.origin);
        const robotsResponse = await safeFetch(robotsUrl, {
          signal: AbortSignal.timeout(30_000),
          headers: { Accept: "text/plain" }
        });
        if (robotsResponse.ok) {
          rules = robotsDisallows(
            (await readResponseBounded(robotsResponse, 1_000_000)).toString("utf8")
          );
        }
      } catch (error) {
        warnings.push(
          `${url.origin}/robots.txt could not be read: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }
      robotsByOrigin.set(url.origin, rules);
      return rules;
    };
    const queue: Array<{ url: URL; depth: number }> = [{ url: start, depth: 0 }];
    const queued = new Set([start.toString()]);
    const visited = new Set<string>();
    const results: IngestResult[] = [];
    let skipped = 0;
    while (queue.length > 0 && visited.size < maximumPages) {
      if (cancelled()) break;
      const next = queue.shift();
      if (!next) break;
      const disallowed = await rulesFor(next.url);
      if (
        disallowed.some(
          (prefix) => prefix === "/" || (prefix.length > 1 && next.url.pathname.startsWith(prefix))
        )
      ) {
        skipped += 1;
        continue;
      }
      try {
        await assertSafeRemoteUrl(next.url);
        const response = await safeFetch(next.url, {
          signal: AbortSignal.timeout(120_000),
          headers: { Accept: "text/html, text/plain, application/json;q=0.9" }
        });
        if (!response.ok) {
          warnings.push(`${next.url.toString()} returned HTTP ${response.status}.`);
          skipped += 1;
          continue;
        }
        const contentType = response.headers.get("content-type") ?? "";
        const raw = (
          await readResponseBounded(response, MAX_EXTRACTED_TEXT_CHARS)
        ).toString("utf8");
        visited.add(next.url.toString());
        const result = await this.ingestWeb({
          brainId: request.brainId,
          url: next.url.toString(),
          policy: request.policy,
          quarantine: request.quarantine ?? true
        });
        results.push(result);
        if (next.depth < maximumDepth && contentType.includes("html")) {
          for (const link of linksFromHtml(raw, next.url)) {
            if (queue.length >= 5_000) break;
            if (sameOrigin && link.origin !== start.origin) continue;
            const key = link.toString();
            if (queued.has(key)) continue;
            queued.add(key);
            queue.push({ url: link, depth: next.depth + 1 });
          }
        }
      } catch (error) {
        warnings.push(
          `${next.url.toString()}: ${error instanceof Error ? error.message : String(error)}`
        );
        skipped += 1;
      }
      progress(
        Math.min(0.99, visited.size / maximumPages),
        `Crawled ${visited.size} of ${maximumPages} pages`
      );
    }
    return {
      startUrl: start.toString(),
      visited: visited.size,
      skipped,
      results,
      warnings
    };
  }

  async importUrl(request: ImportUrlRequest): Promise<BrainDocument> {
    const { data, finalUrl } = await this.downloadCatalogArtifact(request);
    return this.repository.importBundleBuffer(data, basename(finalUrl.pathname) || "catalog.omni");
  }

  private async downloadCatalogArtifact(
    request: ImportUrlRequest
  ): Promise<{ data: Buffer; finalUrl: URL }> {
    const url = new URL(request.url);
    const expected = request.expectedSha256?.trim().toLocaleLowerCase();
    if (expected && !/^[a-f0-9]{64}$/.test(expected)) {
      throw new Error("Expected SHA-256 must be 64 hexadecimal characters.");
    }
    const response = await safeFetch(url, {
      signal: AbortSignal.timeout(120_000),
      headers: { Accept: "application/json, application/octet-stream;q=0.9" }
    });
    if (!response.ok) throw new Error(`Catalog download failed with HTTP ${response.status}.`);
    const finalUrl = new URL(response.url);
    await assertSafeRemoteUrl(finalUrl);
    const data = await readResponseBounded(response, MAX_DOWNLOAD_BYTES);
    const actual = sha256(data);
    if (expected && actual !== expected) throw new Error("Catalog bundle checksum verification failed.");
    return { data, finalUrl };
  }

  async loadRecipeBuffer(contents: Buffer, sourceLabel: string): Promise<BuildRecipe> {
    return validateBuildRecipe(contents, sourceLabel);
  }

  async loadRecipeUrl(request: ImportUrlRequest): Promise<BuildRecipe> {
    const { data, finalUrl } = await this.downloadCatalogArtifact(request);
    return this.loadRecipeBuffer(data, finalUrl.toString());
  }

  async installModalityPackBuffer(
    brainId: string,
    contents: Buffer,
    sourceLabel: string
  ): Promise<InstalledModalityPack> {
    await this.repository.get(brainId);
    const directory = this.repository.brainDirectory(brainId);
    const staged = await stageModalityPack(directory, contents, sourceLabel);
    try {
      await this.engine.request<Record<string, unknown>>(
        "install_modality_pack",
        {
          brainId,
          storagePath: directory,
          packPath: staged.packPath,
          manifest: staged.manifest
        },
        300_000
      );
      await recordInstalledPack(directory, staged.result);
      return staged.result;
    } catch (error) {
      await removeStagedPack(staged.stagingDirectory);
      throw error;
    }
  }

  async installModalityPackUrl(
    brainId: string,
    request: ImportUrlRequest
  ): Promise<InstalledModalityPack> {
    const { data, finalUrl } = await this.downloadCatalogArtifact(request);
    return this.installModalityPackBuffer(
      brainId,
      data,
      basename(finalUrl.pathname) || "modality.omnipack"
    );
  }

  async listModalityPacks(brainId: string): Promise<InstalledModalityPack[]> {
    await this.repository.get(brainId);
    return listInstalledPacks(this.repository.brainDirectory(brainId));
  }

  async listToolPermissions(brainId: string): Promise<ToolPermissionRecord[]> {
    const brain = await this.repository.get(brainId);
    return [...(brain.toolPermissions ?? [])];
  }

  async setToolPermission(
    brainId: string,
    toolId: string,
    level: ToolPermissionLevel
  ): Promise<ToolPermissionRecord[]> {
    if (!["off", "ask", "auto", "full"].includes(level)) throw new Error("Invalid permission level.");
    const brain = await this.repository.get(brainId);
    const id = normalizeToolId(toolId);
    const records = brain.toolPermissions ?? [];
    const record = records.find((entry) => entry.toolId === id);
    const updatedAt = new Date().toISOString();
    if (record) {
      record.level = level;
      record.updatedAt = updatedAt;
    } else {
      records.push({ toolId: id, label: displayToolLabel(id), level, updatedAt });
    }
    brain.toolPermissions = records.sort((left, right) => left.label.localeCompare(right.label));
    brain.journal = [
      ...(brain.journal ?? []),
      {
        id: randomUUID(),
        createdAt: updatedAt,
        kind: "tool",
        summary: `${displayToolLabel(id)} permission changed to ${level}.`
      }
    ];
    const saved = await this.repository.save(brain);
    return saved.toolPermissions ?? [];
  }

  private async planMergeFile(
    target: BrainDocument,
    candidate: Omit<MergeFileCandidate, "destinationPath" | "disposition">,
    conflicts: string[]
  ): Promise<MergeFileCandidate | undefined> {
    const destinationRoot =
      candidate.kind === "evidence" ? "evidence" : join("artifacts", "merged");
    const sourceKey = sha256(candidate.sourcePath).slice(0, 12);
    const destinationPath = portableRelative(
      join(destinationRoot, candidate.sha256, `${sourceKey}-${safeMergeName(basename(candidate.sourcePath))}`)
    );
    const absoluteDestination = join(
      this.repository.brainDirectory(target.id),
      ...destinationPath.split("/")
    );
    let disposition: MergeFileCandidate["disposition"] = "copy";
    try {
      const existing = await lstat(absoluteDestination);
      if (
        existing.isSymbolicLink() ||
        !existing.isFile() ||
        existing.size !== candidate.bytes ||
        sha256(await readFile(absoluteDestination)) !== candidate.sha256
      ) {
        conflicts.push(
          `Target file ${destinationPath} does not match its content address and was not overwritten.`
        );
        return undefined;
      }
      disposition = "duplicate";
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    return { ...candidate, destinationPath, disposition };
  }

  private async buildMergePlan(
    sourceBrainId: string,
    targetBrainId: string
  ): Promise<MergePlan> {
    if (sourceBrainId === targetBrainId) throw new Error("A brain cannot merge into itself.");
    const [source, target] = await Promise.all([
      this.repository.get(sourceBrainId),
      this.repository.get(targetBrainId)
    ]);
    const conflicts: string[] = [];
    if (source.lineage.rootId !== target.lineage.rootId) {
      conflicts.push(
        "The brains have different immutable origins; only reviewed sparse overlays will merge."
      );
    }
    const changedConcepts = Object.entries(source.concepts).filter(
      ([id, concept]) =>
        target.concepts[id] !== undefined &&
        sha256(JSON.stringify(target.concepts[id])) !== sha256(JSON.stringify(concept))
    ).length;
    const changedSynapses = Object.entries(source.synapses).filter(
      ([id, synapse]) =>
        target.synapses[id] !== undefined &&
        sha256(JSON.stringify(target.synapses[id])) !== sha256(JSON.stringify(synapse))
    ).length;
    if (changedConcepts > 0 || changedSynapses > 0) {
      conflicts.push(
        `${changedConcepts} existing concepts and ${changedSynapses} existing synapses diverged; the target versions will be preserved.`
      );
    }

    const targetEvidence = new Set(target.trainingSources.map(evidenceFingerprint));
    const evidence: MergePlan["evidence"] = [];
    for (const sourceEvidence of source.trainingSources) {
      const fingerprint = evidenceFingerprint(sourceEvidence);
      if (targetEvidence.has(fingerprint)) continue;
      targetEvidence.add(fingerprint);
      evidence.push({ fingerprint, source: sourceEvidence });
    }

    const files: MergeFileCandidate[] = [];
    let skippedFiles = 0;
    let examinedBytes = 0;
    let entryCount = 0;
    const addFile = async (
      candidate: Omit<MergeFileCandidate, "destinationPath" | "disposition">
    ): Promise<void> => {
      if (
        files.length >= MAX_MERGE_FILES ||
        candidate.bytes > MAX_MERGE_FILE_BYTES ||
        examinedBytes + candidate.bytes > MAX_MERGE_TOTAL_BYTES
      ) {
        skippedFiles += 1;
        conflicts.push(
          `Skipped ${candidate.sourcePath}: the reviewed file limit is 512 files, 128 MB each, and 512 MB total.`
        );
        return;
      }
      examinedBytes += candidate.bytes;
      const planned = await this.planMergeFile(target, candidate, conflicts);
      if (planned) files.push(planned);
      else skippedFiles += 1;
    };

    const sourceDirectory = this.repository.brainDirectory(source.id);
    const artifactRoots = [
      { directory: join(sourceDirectory, "artifacts"), label: "artifacts" },
      { directory: join(sourceDirectory, "engine", "artifacts"), label: "engine/artifacts" }
    ];
    for (const root of artifactRoots) {
      const queue: Array<{ path: string; relativePath: string; depth: number }> = [
        { path: root.directory, relativePath: "", depth: 0 }
      ];
      while (queue.length > 0) {
        const next = queue.shift();
        if (!next) break;
        let info;
        try {
          info = await lstat(next.path);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
          throw error;
        }
        if (info.isSymbolicLink()) {
          skippedFiles += 1;
          conflicts.push(
            `Skipped symbolic link ${portableRelative(join(root.label, next.relativePath))}.`
          );
          continue;
        }
        if (info.isDirectory()) {
          if (next.depth >= 24) {
            skippedFiles += 1;
            conflicts.push(
              `Skipped deep artifact directory ${portableRelative(join(root.label, next.relativePath))}.`
            );
            continue;
          }
          const children = (await readdir(next.path)).sort();
          for (const name of children) {
            entryCount += 1;
            if (entryCount > MAX_MERGE_FILES * 8) {
              skippedFiles += 1;
              conflicts.push("Stopped artifact discovery after 4,096 branch-local entries.");
              queue.length = 0;
              break;
            }
            queue.push({
              path: join(next.path, name),
              relativePath: join(next.relativePath, name),
              depth: next.depth + 1
            });
          }
          continue;
        }
        const sourcePath = portableRelative(join(root.label, next.relativePath));
        if (!info.isFile()) {
          skippedFiles += 1;
          conflicts.push(`Skipped non-regular artifact ${sourcePath}.`);
          continue;
        }
        if (info.size > MAX_MERGE_FILE_BYTES) {
          await addFile({
            kind: "artifact",
            sourcePath,
            sha256: "0".repeat(64),
            bytes: info.size,
            absoluteSourcePath: next.path
          });
          continue;
        }
        const contents = await readFile(next.path);
        await addFile({
          kind: "artifact",
          sourcePath,
          sha256: sha256(contents),
          bytes: contents.byteLength,
          absoluteSourcePath: next.path
        });
      }
    }

    for (const record of evidence) {
      if (!retainsMergedBlob(target, record.source)) continue;
      let contents: Buffer | undefined;
      let blobHash: string | undefined;
      let absoluteSourcePath: string | undefined;
      if (record.source.blobHash) {
        try {
          contents = await this.repository.getBlob(record.source.blobHash);
          blobHash = record.source.blobHash;
        } catch {
          skippedFiles += 1;
          conflicts.push(
            `Evidence ${record.source.name} references a missing or corrupt blob; metadata only will merge.`
          );
          continue;
        }
      } else if (
        record.source.path &&
        pathWithin(sourceDirectory, record.source.path)
      ) {
        try {
          const info = await lstat(record.source.path);
          if (info.isFile() && !info.isSymbolicLink()) {
            contents = await readFile(record.source.path);
            absoluteSourcePath = record.source.path;
          }
        } catch {
          // Branch-local evidence can be absent after a user removes it. The
          // metadata remains useful and the missing bytes are reported below.
        }
      }
      if (!contents) {
        if (record.source.path || record.source.blobHash) {
          skippedFiles += 1;
          conflicts.push(
            `Evidence ${record.source.name} has no safe retained branch-local file; metadata only will merge.`
          );
        }
        continue;
      }
      await addFile({
        kind: "evidence",
        sourcePath: `evidence/${safeMergeName(record.source.name)}`,
        sha256: sha256(contents),
        bytes: contents.byteLength,
        absoluteSourcePath,
        blobHash,
        evidenceFingerprint: record.fingerprint
      });
    }

    const targetIdeaFingerprints = new Set(target.ideas.map((idea) => idea.fingerprint));
    const newConceptEntries = Object.entries(source.concepts)
      .filter(([id]) => target.concepts[id] === undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    const newSynapseEntries = Object.entries(source.synapses)
      .filter(([id]) => target.synapses[id] === undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    const newIdeaEntries = source.ideas.filter(
      (idea) => !targetIdeaFingerprints.has(idea.fingerprint)
    );
    const duplicateEvidence = source.trainingSources.length - evidence.length;
    const publicFiles = files.map(
      ({ kind, sourcePath, destinationPath, sha256: checksum, bytes, disposition }) => ({
        kind,
        sourcePath,
        destinationPath,
        sha256: checksum,
        bytes,
        disposition
      })
    );
    const boundedConflicts = conflicts.slice(0, MAX_MERGE_CONFLICTS);
    if (conflicts.length > boundedConflicts.length) {
      boundedConflicts.push(
        `${conflicts.length - boundedConflicts.length} additional file warnings were omitted.`
      );
    }
    const reviewDescriptor = {
      sourceBrainId,
      targetBrainId,
      sourceUpdatedAt: source.updatedAt,
      targetUpdatedAt: target.updatedAt,
      concepts: newConceptEntries.map(([id, value]) => [id, sha256(JSON.stringify(value))]),
      synapses: newSynapseEntries.map(([id, value]) => [id, sha256(JSON.stringify(value))]),
      ideas: newIdeaEntries.map((idea) => [
        idea.fingerprint,
        sha256(JSON.stringify(idea))
      ]),
      evidence: evidence
        .map(
          (entry) =>
            [entry.fingerprint, sha256(JSON.stringify(entry.source))] as const
        )
        .sort(([left], [right]) => left.localeCompare(right)),
      files: publicFiles,
      skippedFiles,
      conflicts: boundedConflicts
    };
    const preview: AgentMergePreview = {
      sourceBrainId,
      targetBrainId,
      reviewToken: sha256(JSON.stringify(reviewDescriptor)),
      newConcepts: newConceptEntries.length,
      newIdeas: newIdeaEntries.length,
      newSynapses: newSynapseEntries.length,
      newEvidence: evidence.length,
      duplicateEvidence,
      newFiles: files.filter((file) => file.disposition === "copy").length,
      duplicateFiles: files.filter((file) => file.disposition === "duplicate").length,
      skippedFiles,
      fileBytes: files
        .filter((file) => file.disposition === "copy")
        .reduce((sum, file) => sum + file.bytes, 0),
      files: publicFiles,
      conflicts: boundedConflicts,
      note:
        "Merge copies reviewed ideas, evidence, content-addressed files, and neural replay overlays. It never averages complete model weights."
    };
    return { source, target, evidence, files, preview };
  }

  async previewMerge(sourceBrainId: string, targetBrainId: string): Promise<AgentMergePreview> {
    return (await this.buildMergePlan(sourceBrainId, targetBrainId)).preview;
  }

  async merge(
    sourceBrainId: string,
    targetBrainId: string,
    reviewToken: string
  ): Promise<BrainDocument> {
    if (!/^[a-f0-9]{64}$/.test(reviewToken)) {
      throw new Error("A valid merge review token is required.");
    }
    const plan = await this.buildMergePlan(sourceBrainId, targetBrainId);
    if (plan.preview.reviewToken !== reviewToken) {
      throw new Error("The merge preview is stale. Review the branch overlay again.");
    }

    for (const file of plan.files) {
      let hash: string;
      if (file.blobHash) {
        const contents = await this.repository.getBlob(file.blobHash);
        hash = sha256(contents);
      } else if (file.absoluteSourcePath) {
        const sourceDirectory = this.repository.brainDirectory(plan.source.id);
        if (
          !pathWithin(sourceDirectory, file.absoluteSourcePath) ||
          (await lstat(file.absoluteSourcePath)).isSymbolicLink()
        ) {
          throw new Error(`Merge source ${file.sourcePath} is no longer a safe regular file.`);
        }
        const contents = await readFile(file.absoluteSourcePath);
        hash = await this.repository.storeBlob(contents);
      } else {
        throw new Error(`Merge source ${file.sourcePath} is unavailable.`);
      }
      if (hash !== file.sha256) {
        throw new Error(`Merge source ${file.sourcePath} changed after it was reviewed.`);
      }
      await this.repository.linkBlobTo(
        hash,
        join(
          this.repository.brainDirectory(plan.target.id),
          ...file.destinationPath.split("/")
        )
      );
    }

    await this.engine.tryRequest(
      "merge_overlay",
      {
        targetBrainId,
        targetStoragePath: this.repository.brainDirectory(targetBrainId),
        sourceBrainId,
        sourceStoragePath: this.repository.brainDirectory(sourceBrainId)
      },
      600_000
    );

    const { source, target } = plan;
    for (const [id, concept] of Object.entries(source.concepts)) {
      if (!target.concepts[id]) {
        target.concepts[id] = JSON.parse(JSON.stringify(concept)) as typeof concept;
      }
    }
    for (const [id, synapse] of Object.entries(source.synapses)) {
      if (!target.synapses[id]) {
        target.synapses[id] = JSON.parse(JSON.stringify(synapse)) as typeof synapse;
      }
    }
    const ideaFingerprints = new Set(target.ideas.map((idea) => idea.fingerprint));
    for (const idea of source.ideas) {
      if (!ideaFingerprints.has(idea.fingerprint)) {
        target.ideas.push(JSON.parse(JSON.stringify(idea)) as typeof idea);
        ideaFingerprints.add(idea.fingerprint);
      }
    }

    const targetEvidenceFingerprints = new Set(target.trainingSources.map(evidenceFingerprint));
    const evidenceIds = new Set(target.trainingSources.map((entry) => entry.id));
    for (const record of plan.evidence) {
      if (targetEvidenceFingerprints.has(record.fingerprint)) continue;
      const copied = JSON.parse(JSON.stringify(record.source)) as TrainingSource;
      if (evidenceIds.has(copied.id)) copied.id = randomUUID();
      evidenceIds.add(copied.id);
      delete copied.path;
      const evidenceFile = plan.files.find(
        (file) => file.kind === "evidence" && file.evidenceFingerprint === record.fingerprint
      );
      if (evidenceFile) {
        copied.path = join(
          this.repository.brainDirectory(target.id),
          ...evidenceFile.destinationPath.split("/")
        );
        copied.blobHash = evidenceFile.sha256;
      } else {
        delete copied.blobHash;
      }
      const retainRawText =
        (target.config.memoryRecipe ?? "human-consolidation") === "total-recall" &&
        target.config.retainSourceText;
      if (!retainRawText) {
        copied.rawTextRetained = false;
        delete copied.rawText;
      }
      target.trainingSources.push(copied);
      targetEvidenceFingerprints.add(record.fingerprint);
    }

    const mergedAt = new Date().toISOString();
    const manifest = Buffer.from(
      JSON.stringify(
        {
          schemaVersion: 1,
          sourceBrainId,
          targetBrainId,
          reviewToken,
          mergedAt,
          strategy: "ideas-evidence-files-replay-overlay",
          wholeModelWeightsAveraged: false,
          evidence: plan.evidence.map((entry) => entry.fingerprint),
          files: plan.preview.files
        },
        null,
        2
      ),
      "utf8"
    );
    const manifestHash = await this.repository.storeBlob(manifest);
    await this.repository.linkBlobTo(
      manifestHash,
      join(
        this.repository.brainDirectory(target.id),
        "artifacts",
        "merge-manifests",
        `${reviewToken}.json`
      )
    );
    target.journal = [
      ...(target.journal ?? []),
      {
        id: randomUUID(),
        createdAt: mergedAt,
        kind: "fork",
        summary: `Merged reviewed overlays from ${source.name}.`,
        detail: JSON.stringify({
          sourceBrainId: source.id,
          reviewToken,
          ideas: plan.preview.newIdeas,
          evidence: plan.preview.newEvidence,
          files: plan.preview.newFiles,
          replay: "neural-overlay",
          weightsAveraged: false
        })
      }
    ];
    return this.repository.save(target);
  }

  async runtimeHealth(id: string): Promise<RuntimeHealth> {
    const brain = await this.repository.get(id);
    const health = await this.engine.health();
    return {
      runtime: brain.config.runtime,
      ready: health.ready,
      label: health.worker === "python" ? "Omni neural worker" : "Adaptive core fallback",
      detail: health.detail
    };
  }
}

export class RuntimeJobManager extends EventEmitter {
  private readonly jobs = new Map<string, JobRecord>();

  constructor(
    private readonly service: BrainService,
    private readonly engine: EngineSupervisor
  ) {
    super();
    engine.on("event", (event: EngineEvent) => this.consumeEngineEvent(event));
  }

  list(brainId?: string): RuntimeJob[] {
    return [...this.jobs.values()]
      .filter((job) => !brainId || job.brainId === brainId)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .map((job) => ({ ...job }));
  }

  startTraining(request: StartTrainingRequest): RuntimeJob {
    const job = this.createJob(request.brainId, "training", "Training slow neural parameters");
    void this.run(job, async () => {
      const result = await this.engine.tryRequest<unknown>(
        "train",
        {
          jobId: job.id,
          brainId: request.brainId,
          epochs: Math.max(1, Math.min(10_000, Math.round(request.epochs ?? 1))),
          learningRate: request.learningRate,
          sourceIds: request.sourceIds,
          storagePath: this.service.repository.brainDirectory(request.brainId)
        },
        3_600_000
      );
      if (result === undefined) {
        if (job.cancelled) return { cancelled: true };
        const brain = await this.service.consolidate(request.brainId);
        return {
          fallback: true,
          detail: "Worker unavailable; consolidated local fast weights instead.",
          cycles: brain.counters.consolidationCycles
        };
      }
      return result;
    });
    return { ...job };
  }

  startCrawl(request: WebCrawlRequest): RuntimeJob {
    const job = this.createJob(request.brainId, "crawl", "Crawling quarantined web sources");
    void this.run(job, () =>
      this.service.crawlWeb(
        request,
        () => Boolean(job.cancelled),
        (progress, message) => {
          if (job.cancelled) return;
          job.progress = Math.max(job.progress, Math.min(0.99, progress));
          job.label = message;
          job.updatedAt = new Date().toISOString();
          this.publish(job);
        }
      )
    );
    return { ...job };
  }

  generate(request: ModalityGenerateRequest): RuntimeJob {
    const job = this.createJob(
      request.brainId,
      request.modality,
      `Generating ${request.modality}`
    );
    void this.run(job, async () => {
      const output = await this.engine.request<unknown>(
        "generate_modality",
        {
          jobId: job.id,
          ...request,
          storagePath: this.service.repository.brainDirectory(request.brainId)
        },
        3_600_000
      );
      return output;
    });
    return { ...job };
  }

  async cancel(jobId: string): Promise<RuntimeJob> {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error("The runtime job was not found.");
    if (["complete", "failed", "cancelled"].includes(job.state)) return { ...job };
    job.cancelled = true;
    job.state = "cancelled";
    job.updatedAt = new Date().toISOString();
    if (job.kind === "crawl") {
      // The crawler cooperatively checks job.cancelled between bounded fetches.
    } else {
      for (const other of this.jobs.values()) {
        if (
          other.id !== job.id &&
          !other.cancelled &&
          ["queued", "running"].includes(other.state) &&
          other.kind !== "crawl"
        ) {
          other.cancelled = true;
          other.state = "failed";
          other.error = "Neural worker restarted because another running job was cancelled.";
          other.updatedAt = job.updatedAt;
          this.publish(other);
        }
      }
      await this.engine.interruptAndRestart();
    }
    this.publish(job);
    return { ...job };
  }

  private createJob(
    brainId: string,
    kind: RuntimeJob["kind"],
    label: string
  ): JobRecord {
    const now = new Date().toISOString();
    const job: JobRecord = {
      id: randomUUID(),
      brainId,
      kind,
      state: "queued",
      progress: 0,
      label,
      createdAt: now,
      updatedAt: now
    };
    this.jobs.set(job.id, job);
    this.publish(job);
    return job;
  }

  private async run(job: JobRecord, operation: () => Promise<unknown>): Promise<void> {
    if (job.cancelled) return;
    job.state = "running";
    job.progress = Math.max(job.progress, 0.01);
    job.updatedAt = new Date().toISOString();
    this.publish(job);
    try {
      const output = await operation();
      if (job.cancelled) return;
      job.state = "complete";
      job.progress = 1;
      job.output = output;
    } catch (error) {
      if (job.cancelled) return;
      job.state = "failed";
      job.error = error instanceof Error ? error.message : String(error);
    }
    job.updatedAt = new Date().toISOString();
    this.publish(job);
  }

  private consumeEngineEvent(event: EngineEvent): void {
    if (!event.jobId) return;
    const job = this.jobs.get(event.jobId);
    if (!job || job.cancelled) return;
    if (typeof event.progress === "number") {
      job.progress = Math.max(job.progress, Math.min(0.99, Math.max(0, event.progress)));
    }
    if (event.message) job.label = event.message.slice(0, 200);
    job.updatedAt = new Date().toISOString();
    this.publish(job);
  }

  private publish(job: JobRecord): void {
    const event: RuntimeJobEvent = { job: { ...job } };
    this.emit("event", event);
  }
}
