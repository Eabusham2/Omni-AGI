import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, readdir, realpath, rename, unlink } from "node:fs/promises";
import { basename, join } from "node:path";
import {
  DEFAULT_USER_QUESTION_PREFERENCES,
  type UserQuestionEvent,
  type UserQuestionListQuery,
  type UserQuestionOption,
  type UserQuestionPreferences,
  type UserQuestionRecord,
  type UserQuestionRequest,
  type UserQuestionResolution,
  type UserQuestionRespondRequest,
  type UserQuestionResponse,
  type UserQuestionSource,
  type UserQuestionStatus,
  type UserQuestionTimeout
} from "../shared/userQuestions";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const QUESTION_ID = /^[a-f0-9]{64}$/;
const UUID = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const MAX_TIMER_MILLISECONDS = 2_147_483_647;
const STATES: readonly UserQuestionStatus[] = ["pending", "answered", "assumed", "cancelled"];

type NormalizedRequest = Pick<UserQuestionRecord,
  "brainId" | "requestId" | "kind" | "prompt" | "detail" | "mode" | "options" |
  "minSelections" | "maxSelections" | "allowOther" | "suggestedResponse" | "source">;

export interface SavedUserQuestionPreferences extends UserQuestionPreferences {
  format: "omni-user-question-preferences";
  version: 1;
  revision: string;
  updatedAt: string;
}

export interface UserQuestionServiceOptions {
  /** Must resolve an existing app-managed instance; the service never creates an instance. */
  brainDirectory(brainId: string): string | Promise<string>;
  now?: () => number;
  onError?: (error: Error, brainId: string) => void;
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Invalid ${label}.`);
  return value as Record<string, unknown>;
}

function id(value: unknown, label: string): string {
  if (typeof value !== "string" || !SAFE_ID.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error(`Enter ${label}.`);
  return value;
}

function integer(value: unknown, minimum: number, label: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) throw new Error(`Invalid ${label}.`);
  return value;
}

function iso(value: unknown, label: string): string {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString() !== value) {
    throw new Error(`Invalid ${label}.`);
  }
  return value;
}

function bool(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`Invalid ${label}.`);
  return value;
}

function preferences(value: unknown): UserQuestionPreferences {
  const source = object(value, "question preferences");
  const timeoutSeconds = integer(source.timeoutSeconds, 1, "question timeout");
  // Prevent JS date/timer overflow; this is representation validation, not a product duration preset.
  if (timeoutSeconds * 1_000 > Number.MAX_SAFE_INTEGER) throw new Error("Question timeout exceeds supported date precision.");
  return { timeoutEnabled: bool(source.timeoutEnabled, "question timeout switch"), timeoutSeconds };
}

function responseFor(
  question: Pick<UserQuestionRecord, "mode" | "options" | "minSelections" | "maxSelections" | "allowOther">,
  value: unknown
): UserQuestionResponse {
  const source = object(value, "question response");
  if (Object.keys(source).some((key) => key !== "optionIds" && key !== "text")) {
    throw new Error("A question response contains only task choices or text; it cannot grant permissions.");
  }
  const optionIds = source.optionIds === undefined ? [] : source.optionIds;
  if (!Array.isArray(optionIds) || optionIds.some((value) => typeof value !== "string")) throw new Error("Invalid selected question options.");
  if (new Set(optionIds).size !== optionIds.length) throw new Error("Select each question option only once.");
  const answerText = source.text === undefined ? undefined : text(source.text, "an answer");
  if (question.mode === "text") {
    if (optionIds.length || answerText === undefined) throw new Error("This question requires a written answer.");
    return { text: answerText };
  }
  if (answerText !== undefined) {
    if (!question.allowOther || optionIds.length) throw new Error("Choose listed options or enter an allowed Other answer.");
    return { text: answerText };
  }
  const choices = new Set(question.options.map((option) => option.id));
  if (optionIds.some((optionId) => !choices.has(optionId))) throw new Error("The response names an unavailable question option.");
  if (optionIds.length < question.minSelections || optionIds.length > question.maxSelections) {
    throw new Error(`Select ${question.minSelections === question.maxSelections ? question.minSelections : `${question.minSelections}–${question.maxSelections}`} option(s).`);
  }
  // Canonical order makes equivalent retries idempotent without changing option identities.
  return { optionIds: question.options.filter((option) => optionIds.includes(option.id)).map((option) => option.id) };
}

function normalizeRequest(value: unknown, requestId: string): NormalizedRequest {
  const input = object(value, "question request");
  const brainId = id(input.brainId, "question instance");
  if (input.kind !== undefined && input.kind !== "task-choice") throw new Error("Structured questions are task choices, not permission requests.");
  if (!["single", "multiple", "text"].includes(String(input.mode))) throw new Error("Invalid question mode.");
  const mode = input.mode as NormalizedRequest["mode"];
  const rawOptions = input.options ?? [];
  if (!Array.isArray(rawOptions)) throw new Error("Question options must be a list.");
  const ids = new Set<string>();
  const options: UserQuestionOption[] = rawOptions.map((value, index) => {
    const option = object(value, "question option");
    const optionId = id(option.id, "question option id");
    if (ids.has(optionId)) throw new Error("Question option IDs must be unique.");
    ids.add(optionId);
    return {
      id: optionId, label: text(option.label, "an option label"),
      ...(option.description === undefined ? {} : { description: text(option.description, "an option description") }),
      rank: option.rank === undefined ? index + 1 : integer(option.rank, 0, "option rank"),
      recommended: option.recommended === undefined ? false : bool(option.recommended, "option recommendation")
    };
  }).sort((a, b) => Number(b.recommended) - Number(a.recommended) || (a.rank ?? 0) - (b.rank ?? 0));
  options.forEach((option, index) => { option.rank = index + 1; });
  if (mode === "text" ? options.length > 0 : options.length < 2) {
    throw new Error(mode === "text" ? "Written questions do not have choice options." : "Provide at least two choices.");
  }
  const minSelections = input.minSelections === undefined ? (mode === "text" ? 0 : 1) : integer(input.minSelections, mode === "text" ? 0 : 1, "minimum selections");
  const maxSelections = input.maxSelections === undefined ? (mode === "multiple" ? options.length : mode === "single" ? 1 : 0) : integer(input.maxSelections, 0, "maximum selections");
  if (minSelections > maxSelections || maxSelections > options.length || (mode === "single" && (minSelections !== 1 || maxSelections !== 1)) || (mode === "text" && (minSelections || maxSelections))) {
    throw new Error("Question selection bounds do not match its mode and options.");
  }
  const allowOther = input.allowOther === undefined ? mode !== "text" : bool(input.allowOther, "Other answer switch");
  let source: UserQuestionSource | undefined;
  if (input.source !== undefined) {
    const rawSource = object(input.source, "question source");
    if (!["brain", "fact-check", "learn", "host"].includes(String(rawSource.kind))) throw new Error("Invalid question source kind.");
    source = { kind: rawSource.kind as UserQuestionSource["kind"],
      ...(rawSource.turnId === undefined ? {} : { turnId: id(rawSource.turnId, "question turn id") }),
      ...(rawSource.actionId === undefined ? {} : { actionId: id(rawSource.actionId, "question action id") }) };
  }
  const result: NormalizedRequest = {
    brainId, requestId: id(requestId, "question request id"), kind: "task-choice",
    prompt: text(input.prompt, "a question"),
    ...(input.detail === undefined ? {} : { detail: text(input.detail, "question details") }),
    mode, options, minSelections, maxSelections, allowOther,
    ...(source ? { source } : {})
  };
  if (input.suggestedResponse !== undefined) result.suggestedResponse = responseFor(result, input.suggestedResponse);
  return result;
}

function hash(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function makeTimeout(question: NormalizedRequest, createdAt: string, policy: SavedUserQuestionPreferences): UserQuestionTimeout {
  const timeout: UserQuestionTimeout = { enabled: policy.timeoutEnabled, seconds: policy.timeoutSeconds, policyRevision: policy.revision };
  if (!timeout.enabled) return timeout;
  let assumption = question.suggestedResponse;
  if (!assumption && question.mode !== "text") {
    const recommended = question.options.filter((option) => option.recommended);
    const count = question.mode === "single" ? 1 : Math.max(question.minSelections, Math.min(recommended.length, question.maxSelections));
    assumption = { optionIds: question.options.slice(0, count).map((option) => option.id) };
  }
  if (!assumption) return { ...timeout, unavailableReason: "no-suggested-response" };
  const due = Math.max(Date.parse(createdAt), Date.parse(policy.updatedAt)) + policy.timeoutSeconds * 1_000;
  if (!Number.isFinite(due) || due > 8_640_000_000_000_000) throw new Error("Question timeout exceeds the supported date range.");
  return { ...timeout, deadlineAt: new Date(due).toISOString(), assumption: responseFor(question, assumption) };
}

export function userQuestionRequestSha256(request: UserQuestionRequest & { requestId: string }): string {
  return hash(normalizeRequest(request, request.requestId));
}

export function validateUserQuestionPreferences(value: unknown): SavedUserQuestionPreferences {
  const source = object(value, "saved question preferences");
  if (source.format !== "omni-user-question-preferences" || source.version !== 1) throw new Error("Unsupported question preferences format.");
  return { format: "omni-user-question-preferences", version: 1, ...preferences(source),
    revision: id(source.revision, "question policy revision"), updatedAt: iso(source.updatedAt, "question policy update time") };
}

/** Validates persisted content before it becomes a UI choice or native continuation. */
export function validateUserQuestionRecord(value: unknown, brainId: string, questionId?: string): UserQuestionRecord {
  const raw = object(value, "saved question");
  if (raw.format !== "omni-user-question" || raw.version !== 1) throw new Error("Unsupported saved question format.");
  const normalized = normalizeRequest(raw, id(raw.requestId, "saved question request id"));
  const expectedId = hash({ requestId: normalized.requestId });
  if (normalized.brainId !== brainId || raw.id !== expectedId || (questionId !== undefined && questionId !== expectedId)) throw new Error("Saved question ownership does not match this instance.");
  if (raw.requestSha256 !== hash(normalized)) throw new Error("Saved question contents failed validation.");
  if (!STATES.includes(raw.status as UserQuestionStatus)) throw new Error("Invalid saved question status.");
  const createdAt = iso(raw.createdAt, "question creation time");
  const updatedAt = iso(raw.updatedAt, "question update time");
  const rawTimeout = object(raw.timeout, "saved question timeout");
  const policy = preferences({ timeoutEnabled: rawTimeout.enabled, timeoutSeconds: rawTimeout.seconds });
  const policyRevision = id(rawTimeout.policyRevision, "question timeout revision");
  const timeout: UserQuestionTimeout = { enabled: policy.timeoutEnabled, seconds: policy.timeoutSeconds, policyRevision };
  if (rawTimeout.deadlineAt !== undefined) timeout.deadlineAt = iso(rawTimeout.deadlineAt, "question deadline");
  if (rawTimeout.assumption !== undefined) timeout.assumption = responseFor(normalized, rawTimeout.assumption);
  if (rawTimeout.unavailableReason !== undefined) {
    if (rawTimeout.unavailableReason !== "no-suggested-response" || normalized.mode !== "text") throw new Error("Invalid question timeout reason.");
    timeout.unavailableReason = rawTimeout.unavailableReason;
  }
  if ((!timeout.enabled && (timeout.deadlineAt || timeout.assumption || timeout.unavailableReason)) ||
      (timeout.enabled && (!timeout.deadlineAt || !timeout.assumption) && !timeout.unavailableReason) ||
      (timeout.unavailableReason && (timeout.deadlineAt || timeout.assumption))) throw new Error("Saved question timeout is incomplete.");
  const result: UserQuestionRecord = {
    format: "omni-user-question", version: 1, id: expectedId, ...normalized,
    requestSha256: raw.requestSha256 as string, status: raw.status as UserQuestionStatus,
    createdAt, updatedAt, timeout
  };
  if (raw.lineage !== undefined) {
    if (!Array.isArray(raw.lineage)) throw new Error("Invalid question lineage.");
    result.lineage = raw.lineage.map((value) => {
      const link = object(value, "question lineage link");
      if (typeof link.questionId !== "string" || !QUESTION_ID.test(link.questionId)) throw new Error("Invalid ancestral question ID.");
      return { brainId: id(link.brainId, "ancestral question instance"), questionId: link.questionId, copiedAt: iso(link.copiedAt, "question copy time") };
    });
  }
  if (raw.suspended !== undefined) {
    const suspended = object(raw.suspended, "question suspension");
    if (suspended.reason !== "copied-instance") throw new Error("Invalid question suspension reason.");
    result.suspended = { reason: "copied-instance", fromBrainId: id(suspended.fromBrainId, "question copy source"), copiedAt: iso(suspended.copiedAt, "question copy time") };
    const parent = result.lineage?.at(-1);
    if (!parent || parent.brainId !== result.suspended.fromBrainId || parent.copiedAt !== result.suspended.copiedAt) throw new Error("Suspended question has no matching parent provenance.");
  }
  if (raw.resumedAt !== undefined) result.resumedAt = iso(raw.resumedAt, "question resume time");
  if (result.status === "pending") {
    if (raw.resolution !== undefined || raw.deliveredAt !== undefined) throw new Error("A pending question cannot have a resolution.");
    return result;
  }
  const resolution = object(raw.resolution, "saved question resolution");
  if (typeof resolution.id !== "string" || !UUID.test(resolution.id)) throw new Error("Invalid question resolution ID.");
  const requiredSource = result.status === "answered" ? "user" : result.status === "assumed" ? "timeout" : "cancellation";
  if (resolution.source !== requiredSource || resolution.authority !== "task-choice-only" || resolution.permissionsGranted !== false) throw new Error("Question resolution cannot grant permissions.");
  result.resolution = {
    id: resolution.id, source: requiredSource, resolvedAt: iso(resolution.resolvedAt, "question resolution time"),
    authority: "task-choice-only", permissionsGranted: false
  };
  if (result.status === "cancelled") {
    if (resolution.response !== undefined) throw new Error("Cancelled questions cannot contain an answer.");
    if (resolution.reason !== undefined) result.resolution.reason = text(resolution.reason, "a cancellation reason");
  } else result.resolution.response = responseFor(normalized, resolution.response);
  if (result.status === "assumed") {
    if (!timeout.enabled || !timeout.deadlineAt || resolution.deadlineAt !== timeout.deadlineAt || hash(result.resolution.response) !== hash(timeout.assumption)) throw new Error("Assumed response does not match the saved timeout policy.");
    result.resolution.deadlineAt = timeout.deadlineAt;
    result.resolution.overdueMilliseconds = integer(resolution.overdueMilliseconds, 0, "question overdue duration");
  }
  if (raw.deliveredAt !== undefined) result.deliveredAt = iso(raw.deliveredAt, "question delivery time");
  return result;
}

export async function readUserQuestionJson(path: string): Promise<unknown | undefined> {
  let info;
  try { info = await lstat(path); } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Question state must be a regular file.");
  const handle = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  try {
    const actual = await handle.stat();
    if (!actual.isFile() || actual.dev !== info.dev || actual.ino !== info.ino) throw new Error("Question state changed while opening.");
    return JSON.parse(await handle.readFile("utf8"));
  } finally { await handle.close(); }
}

export async function writeUserQuestionJson(directory: string, name: string, value: unknown): Promise<void> {
  if (name !== "preferences.json" && !/^[a-f0-9]{64}\.json$/.test(name)) throw new Error("Invalid question state filename.");
  const parentInfo = await lstat(directory);
  if (!parentInfo.isDirectory() || parentInfo.isSymbolicLink()) throw new Error("Question state parent must be a real directory.");
  const destination = join(directory, name);
  try {
    const info = await lstat(destination);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Question state destination must be a regular file.");
  } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  const temporary = join(directory, `.${name}.${randomUUID()}.tmp`);
  const handle = await open(temporary, "wx", 0o600);
  try { await handle.writeFile(`${JSON.stringify(value)}\n`, "utf8"); await handle.sync(); }
  catch (error) { await handle.close(); await unlink(temporary).catch(() => undefined); throw error; }
  await handle.close();
  try {
    await rename(temporary, destination);
    if (process.platform !== "win32") {
      const parent = await open(directory, "r");
      try { await parent.sync(); } finally { await parent.close(); }
    }
  } finally { await unlink(temporary).catch(() => undefined); }
}

/** Single-host durable question coordinator. All operations for one instance are serialized. */
export class UserQuestionService {
  private readonly operations = new Map<string, Promise<unknown>>();
  private readonly timers = new Map<string, ReturnType<typeof setTimeout>>();
  private readonly listeners = new Set<(event: UserQuestionEvent) => void>();
  private readonly closeListeners = new Set<() => void>();
  private closed = false;

  constructor(private readonly options: UserQuestionServiceOptions) {}

  subscribe(listener: (event: UserQuestionEvent) => void): () => void {
    if (this.closed) throw new Error("The question service is closed.");
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  private emit(event: UserQuestionEvent): void {
    for (const listener of this.listeners) {
      try { listener(structuredClone(event)); }
      catch (error) { this.options.onError?.(error instanceof Error ? error : new Error(String(error)), event.brainId); }
    }
  }

  private now(): number {
    const value = this.options.now?.() ?? Date.now();
    if (!Number.isFinite(value) || !Number.isFinite(new Date(value).getTime())) throw new Error("Question clock is unavailable.");
    return Math.trunc(value);
  }

  private serial<T>(brainId: string, operation: () => Promise<T>): Promise<T> {
    id(brainId, "question instance");
    if (this.closed) return Promise.reject(new Error("The question service is closed."));
    const pending = (this.operations.get(brainId) ?? Promise.resolve()).catch(() => undefined).then(operation);
    this.operations.set(brainId, pending);
    void pending.then(() => { if (this.operations.get(brainId) === pending) this.operations.delete(brainId); },
      () => { if (this.operations.get(brainId) === pending) this.operations.delete(brainId); });
    return pending;
  }

  private async directories(brainId: string): Promise<{ root: string; records: string }> {
    const owner = await this.options.brainDirectory(brainId);
    const info = await lstat(owner);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Question instance must be a real directory.");
    const resolved = await realpath(owner);
    if (basename(resolved) !== brainId) throw new Error("Question directory does not match its instance.");
    let directory = resolved;
    for (const segment of ["questions", "records"]) {
      directory = join(directory, segment);
      await mkdir(directory, { mode: 0o700 }).catch((error: NodeJS.ErrnoException) => { if (error.code !== "EEXIST") throw error; });
      const entry = await lstat(directory);
      if (!entry.isDirectory() || entry.isSymbolicLink()) throw new Error("Question storage must be an owned real directory.");
    }
    return { root: join(resolved, "questions"), records: directory };
  }

  private async policy(root: string): Promise<SavedUserQuestionPreferences> {
    const raw = await readUserQuestionJson(join(root, "preferences.json"));
    if (raw === undefined) return { format: "omni-user-question-preferences", version: 1,
      ...DEFAULT_USER_QUESTION_PREFERENCES, revision: "default", updatedAt: new Date(0).toISOString() };
    return validateUserQuestionPreferences(raw);
  }

  private async load(brainId: string, paths: { root: string; records: string }): Promise<UserQuestionRecord[]> {
    const policy = await this.policy(paths.root);
    const result: UserQuestionRecord[] = [];
    for (const name of await readdir(paths.records)) {
      if (!/^[a-f0-9]{64}\.json$/.test(name)) {
        if (/^\.[a-f0-9]{64}\.json\.[a-f0-9-]{36}\.tmp$/.test(name)) continue; // Uncommitted, never a request.
        throw new Error("Unexpected file in question storage.");
      }
      const record = validateUserQuestionRecord(await readUserQuestionJson(join(paths.records, name)), brainId, name.slice(0, -5));
      if (record.status === "pending" && !record.suspended) {
        if (record.timeout.policyRevision !== policy.revision) {
          record.timeout = makeTimeout(record, record.resumedAt ?? record.createdAt, policy);
          record.updatedAt = new Date(Math.max(this.now(), Date.parse(record.updatedAt))).toISOString();
          await writeUserQuestionJson(paths.records, name, record);
          this.emit({ type: "updated", brainId, question: record });
        }
        if (record.timeout.deadlineAt && Date.parse(record.timeout.deadlineAt) <= this.now()) {
          const now = Math.max(this.now(), Date.parse(record.updatedAt));
          record.status = "assumed";
          record.updatedAt = new Date(now).toISOString();
          record.resolution = { id: randomUUID(), source: "timeout", resolvedAt: record.updatedAt,
            response: record.timeout.assumption, deadlineAt: record.timeout.deadlineAt,
            overdueMilliseconds: Math.max(0, now - Date.parse(record.timeout.deadlineAt)),
            authority: "task-choice-only", permissionsGranted: false };
          await writeUserQuestionJson(paths.records, name, record);
          this.emit({ type: "resolved", brainId, question: record });
        }
      }
      result.push(record);
    }
    this.schedule(brainId, result);
    return result.sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
  }

  private schedule(brainId: string, records: UserQuestionRecord[]): void {
    const current = this.timers.get(brainId);
    if (current) clearTimeout(current);
    this.timers.delete(brainId);
    if (this.closed) return;
    const earliest = records.reduce((deadline, record) => record.status === "pending" && !record.suspended && record.timeout.deadlineAt ? Math.min(deadline, Date.parse(record.timeout.deadlineAt)) : deadline, Number.POSITIVE_INFINITY);
    if (!Number.isFinite(earliest)) return;
    const delay = Math.min(MAX_TIMER_MILLISECONDS, Math.max(0, earliest - this.now()));
    const timer = setTimeout(() => {
      this.timers.delete(brainId);
      void this.list(brainId).catch((error: unknown) => {
        const failure = error instanceof Error ? error : new Error(String(error));
        this.emit({ type: "error", brainId, message: failure.message });
        this.options.onError?.(failure, brainId);
      });
    }, delay);
    timer.unref?.();
    this.timers.set(brainId, timer);
  }

  async request(input: UserQuestionRequest): Promise<UserQuestionRecord> {
    const normalized = normalizeRequest(input, input.requestId ?? randomUUID());
    return this.serial(normalized.brainId, async () => {
      const paths = await this.directories(normalized.brainId);
      const records = await this.load(normalized.brainId, paths);
      const questionId = hash({ requestId: normalized.requestId });
      const fingerprint = hash(normalized);
      const existing = records.find((record) => record.id === questionId);
      if (existing) {
        if (existing.requestSha256 !== fingerprint) throw new Error("This question request ID already belongs to different content.");
        return existing;
      }
      const createdAt = new Date(this.now()).toISOString();
      const record: UserQuestionRecord = { format: "omni-user-question", version: 1,
        id: questionId, ...normalized, requestSha256: fingerprint, status: "pending", createdAt, updatedAt: createdAt,
        timeout: makeTimeout(normalized, createdAt, await this.policy(paths.root)) };
      await writeUserQuestionJson(paths.records, `${record.id}.json`, record);
      this.schedule(record.brainId, [...records, record]);
      this.emit({ type: "created", brainId: record.brainId, question: record });
      return structuredClone(record);
    });
  }

  async list(brainId: string, query: UserQuestionListQuery = {}): Promise<UserQuestionRecord[]> {
    const statuses = query.status === undefined ? undefined : Array.isArray(query.status) ? query.status : [query.status];
    if (statuses?.some((status) => !STATES.includes(status))) throw new Error("Invalid question status filter.");
    return this.serial(brainId, async () => {
      const records = await this.load(brainId, await this.directories(brainId));
      return statuses ? records.filter((record) => statuses.includes(record.status)) : records;
    });
  }

  async get(brainId: string, questionId: string): Promise<UserQuestionRecord> {
    if (!QUESTION_ID.test(questionId)) throw new Error("Invalid question ID.");
    const record = (await this.list(brainId)).find((question) => question.id === questionId);
    if (!record) throw new Error("The question does not exist for this instance.");
    return record;
  }

  async respond(input: UserQuestionRespondRequest): Promise<UserQuestionRecord> {
    if (!QUESTION_ID.test(input.questionId)) throw new Error("Invalid question ID.");
    return this.serial(input.brainId, async () => {
      const paths = await this.directories(input.brainId);
      const records = await this.load(input.brainId, paths);
      const record = records.find((record) => record.id === input.questionId);
      if (!record) throw new Error("The question does not exist for this instance.");
      if (record.suspended) throw new Error("Resume this copied question before answering it.");
      const response = responseFor(record, input.response);
      if (record.status !== "pending") {
        if (record.status === "answered" && hash(record.resolution?.response) === hash(response)) return record;
        throw new Error("This question was already resolved. Its answer was preserved.");
      }
      record.status = "answered";
      record.updatedAt = new Date(Math.max(this.now(), Date.parse(record.updatedAt))).toISOString();
      record.resolution = { id: randomUUID(), source: "user", resolvedAt: record.updatedAt, response,
        authority: "task-choice-only", permissionsGranted: false };
      await writeUserQuestionJson(paths.records, `${record.id}.json`, record);
      this.schedule(input.brainId, records);
      this.emit({ type: "resolved", brainId: input.brainId, question: record });
      return structuredClone(record);
    });
  }

  async cancel(brainId: string, questionId: string, reason?: string): Promise<UserQuestionRecord> {
    if (!QUESTION_ID.test(questionId)) throw new Error("Invalid question ID.");
    if (reason !== undefined) text(reason, "a cancellation reason");
    return this.serial(brainId, async () => {
      const paths = await this.directories(brainId);
      const records = await this.load(brainId, paths);
      const record = records.find((record) => record.id === questionId);
      if (!record) throw new Error("The question does not exist for this instance.");
      if (record.status === "cancelled") return record;
      if (record.status !== "pending") throw new Error("This question was already resolved. Its answer was preserved.");
      record.status = "cancelled";
      record.updatedAt = new Date(Math.max(this.now(), Date.parse(record.updatedAt))).toISOString();
      record.resolution = { id: randomUUID(), source: "cancellation", resolvedAt: record.updatedAt,
        ...(reason ? { reason } : {}), authority: "task-choice-only", permissionsGranted: false };
      await writeUserQuestionJson(paths.records, `${record.id}.json`, record);
      this.schedule(brainId, records);
      this.emit({ type: "cancelled", brainId, question: record });
      return structuredClone(record);
    });
  }

  async resume(brainId: string, questionId?: string): Promise<UserQuestionRecord[]> {
    if (questionId !== undefined && !QUESTION_ID.test(questionId)) throw new Error("Invalid question ID.");
    return this.serial(brainId, async () => {
      const paths = await this.directories(brainId);
      const records = await this.load(brainId, paths);
      if (questionId && !records.some((record) => record.id === questionId)) throw new Error("The question does not exist for this instance.");
      const policy = await this.policy(paths.root);
      for (const record of records) {
        if (!record.suspended || (questionId && record.id !== questionId)) continue;
        delete record.suspended;
        record.resumedAt = new Date(Math.max(this.now(), Date.parse(record.updatedAt))).toISOString();
        record.updatedAt = record.resumedAt;
        if (record.status === "pending") record.timeout = makeTimeout(record, record.resumedAt, policy);
        await writeUserQuestionJson(paths.records, `${record.id}.json`, record);
        this.emit({ type: "updated", brainId, question: record });
      }
      this.schedule(brainId, records);
      return records;
    });
  }

  async preferences(brainId: string): Promise<UserQuestionPreferences> {
    return this.serial(brainId, async () => {
      const value = await this.policy((await this.directories(brainId)).root);
      return { timeoutEnabled: value.timeoutEnabled, timeoutSeconds: value.timeoutSeconds };
    });
  }

  async setPreferences(input: UserQuestionPreferences & { brainId: string }): Promise<UserQuestionPreferences> {
    const value = preferences(input);
    if (this.now() + value.timeoutSeconds * 1_000 > 8_640_000_000_000_000) throw new Error("Question timeout exceeds the supported date range.");
    return this.serial(input.brainId, async () => {
      const paths = await this.directories(input.brainId);
      const current = await this.policy(paths.root);
      if (current.timeoutEnabled === value.timeoutEnabled && current.timeoutSeconds === value.timeoutSeconds) return value;
      const saved: SavedUserQuestionPreferences = { format: "omni-user-question-preferences", version: 1, ...value,
        revision: randomUUID(), updatedAt: new Date(this.now()).toISOString() };
      // Publish policy first; interrupted per-question changes are reconciled on next access/restart.
      await writeUserQuestionJson(paths.root, "preferences.json", saved);
      await this.load(input.brainId, paths);
      this.emit({ type: "preferences", brainId: input.brainId, preferences: value });
      return value;
    });
  }

  async undelivered(brainId: string): Promise<UserQuestionRecord[]> {
    return (await this.list(brainId)).filter((record) => record.resolution && !record.deliveredAt && !record.suspended);
  }

  async acknowledge(brainId: string, questionId: string, resolutionId: string): Promise<UserQuestionRecord> {
    if (!QUESTION_ID.test(questionId) || !UUID.test(resolutionId)) throw new Error("Invalid question delivery receipt.");
    return this.serial(brainId, async () => {
      const paths = await this.directories(brainId);
      const record = (await this.load(brainId, paths)).find((entry) => entry.id === questionId);
      if (!record?.resolution || record.resolution.id !== resolutionId || record.suspended) throw new Error("Question resolution does not match an active delivery receipt.");
      if (record.deliveredAt) return record;
      record.deliveredAt = new Date(Math.max(this.now(), Date.parse(record.updatedAt))).toISOString();
      await writeUserQuestionJson(paths.records, `${record.id}.json`, record);
      this.emit({ type: "delivered", brainId, question: record });
      return structuredClone(record);
    });
  }

  /** Cancelling a waiter never discards the durable question or pretends the user answered. */
  waitForResolution(brainId: string, questionId: string, signal?: AbortSignal): Promise<UserQuestionRecord> {
    return new Promise((resolve, reject) => {
      let settled = false;
      let unsubscribe = () => {};
      const finish = (record?: UserQuestionRecord, error?: unknown) => {
        if (settled) return;
        settled = true;
        unsubscribe();
        this.closeListeners.delete(onClose);
        signal?.removeEventListener("abort", onAbort);
        if (error !== undefined) reject(error); else resolve(record!);
      };
      const onAbort = () => finish(undefined, signal?.reason ?? new Error("Question wait was interrupted."));
      const onClose = () => finish(undefined, new Error("The question service closed; the question remains saved."));
      if (this.closed) return onClose();
      if (signal?.aborted) return onAbort();
      unsubscribe = this.subscribe((event) => {
        if ("question" in event && event.brainId === brainId && event.question.id === questionId && event.question.resolution && !event.question.suspended) finish(event.question);
      });
      this.closeListeners.add(onClose);
      signal?.addEventListener("abort", onAbort, { once: true });
      void this.get(brainId, questionId).then((record) => { if (record.resolution && !record.suspended) finish(record); }, (error) => finish(undefined, error));
    });
  }

  async recover(brainIds: readonly string[]): Promise<{ recovered: string[]; errors: Array<{ brainId: string; message: string }> }> {
    const result: { recovered: string[]; errors: Array<{ brainId: string; message: string }> } = { recovered: [], errors: [] };
    for (const brainId of new Set(brainIds)) {
      try { await this.list(brainId); result.recovered.push(brainId); }
      catch (error) { result.errors.push({ brainId, message: error instanceof Error ? error.message : String(error) }); }
    }
    return result;
  }

  async pauseForRecovery(brainId: string): Promise<void> {
    await this.serial(brainId, async () => {
      const timer = this.timers.get(brainId); if (timer) clearTimeout(timer);
      this.timers.delete(brainId);
    });
  }
  close(): void {
    this.closed = true;
    for (const timer of this.timers.values()) clearTimeout(timer);
    this.timers.clear();
    for (const listener of [...this.closeListeners]) listener();
    this.listeners.clear();
  }
}
