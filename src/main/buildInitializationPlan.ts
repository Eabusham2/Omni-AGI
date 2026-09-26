import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type {
  HardwareTier,
  ModalityKind,
  TrainingTelemetry,
  WebCrawlRequest
} from "../shared/types";
import { normalizePersistedTrainingTelemetry } from "../shared/trainingTelemetry";
import { BrainRepository } from "./brainRepository";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const FILE_NAME = "initialization.json";

export type InitialResourceReference =
  | { kind: "selection"; selectionId: string }
  | { kind: "web"; url: string };

export interface InitialFoundationPlan {
  hardwareTier: HardwareTier;
  modalities: ModalityKind[];
  origin: "ground-up";
  /** Stable component budgets only; live total/free is always remeasured. */
  diskBudget?: {
    mandatoryReserveBytes: number;
    selectedDatasetBytes: number;
    modelBytes: number;
    checkpointBytes: number;
    maximumWorkingMemorySpillBytes: number;
    futureGrowthBytes: number;
    operationWriteBytes: number;
  };
}

export interface InitialDatasetTask {
  id: string;
  kind: "selection";
  selectionId: string;
  state: "pending" | "running" | "complete" | "failed";
  manifestId?: string;
  error?: string;
  telemetry?: TrainingTelemetry;
}

export interface InitialWebTask {
  id: string;
  kind: "web";
  request: WebCrawlRequest;
  state: "pending" | "running" | "complete" | "failed";
  error?: string;
  telemetry?: TrainingTelemetry;
}

export type InitialLearningTask = InitialDatasetTask | InitialWebTask;

export interface BuildInitializationPlan {
  schemaVersion: 1;
  brainId: string;
  phase: "foundation" | "initial-learning" | "failed";
  attempt: number;
  foundation: InitialFoundationPlan;
  tasks: InitialLearningTask[];
  createdAt: string;
  updatedAt: string;
  failure?: {
    phase: "foundation" | "initial-learning";
    message: string;
    failedAt: string;
    retryable: true;
  };
}

function normalizedHttpUrl(value: string): string {
  if (typeof value !== "string" || value.length > 8_192 || value.includes("\0")) {
    throw new Error("An initial web source URL is invalid.");
  }
  const parsed = new URL(value);
  if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new Error("Initial web learning requires an HTTP(S) URL without credentials.");
  }
  parsed.hash = "";
  return parsed.toString();
}

function cleanMessage(value: unknown): string {
  const message = String(value ?? "Initialization failed")
    .replace(/\0/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return (message || "Initialization failed").slice(0, 2_000);
}

function clonePlan(plan: BuildInitializationPlan): BuildInitializationPlan {
  return structuredClone(plan);
}

function normalizeFoundation(value: unknown): InitialFoundationPlan {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("The initialization foundation plan is invalid.");
  }
  const input = value as Record<string, unknown>;
  const hardwareTier = String(input.hardwareTier) as HardwareTier;
  const origin = String(input.origin) as InitialFoundationPlan["origin"];
  if (
    !["micro", "personal", "gpu", "workstation"].includes(hardwareTier) ||
    origin !== "ground-up" ||
    Object.hasOwn(input, "foundationModelId") ||
    !Array.isArray(input.modalities) ||
    input.modalities.some((item) => !["vision", "image", "audio", "video"].includes(String(item)))
  ) {
    throw new Error("The initialization foundation plan is invalid.");
  }
  const rawDiskBudget = (
    typeof input.diskBudget === "object" &&
    input.diskBudget !== null &&
    !Array.isArray(input.diskBudget)
  ) ? input.diskBudget as Record<string, unknown> : undefined;
  const diskFields = [
    "mandatoryReserveBytes",
    "selectedDatasetBytes",
    "modelBytes",
    "checkpointBytes",
    "maximumWorkingMemorySpillBytes",
    "futureGrowthBytes",
    "operationWriteBytes"
  ] as const;
  const diskBudget = rawDiskBudget
    ? Object.fromEntries(diskFields.map((field) => [field, rawDiskBudget[field]]))
    : undefined;
  if (
    input.diskBudget !== undefined &&
    (!rawDiskBudget || diskFields.some((field) =>
      !Number.isSafeInteger(rawDiskBudget[field]) ||
      Number(rawDiskBudget[field]) < 0
    ))
  ) {
    throw new Error("The initialization disk budget is invalid.");
  }
  return {
    hardwareTier,
    origin,
    modalities: [...new Set(input.modalities.map(String))] as ModalityKind[],
    ...(diskBudget
      ? { diskBudget: diskBudget as InitialFoundationPlan["diskBudget"] }
      : {})
  };
}

function normalizeTask(value: unknown, brainId: string): InitialLearningTask {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("An initial-learning task is invalid.");
  }
  const input = value as Record<string, unknown>;
  const id = String(input.id ?? "");
  const state = String(input.state) as InitialLearningTask["state"];
  if (!SAFE_ID.test(id) || !["pending", "running", "complete", "failed"].includes(state)) {
    throw new Error("An initial-learning task is invalid.");
  }
  const error = input.error === undefined ? undefined : cleanMessage(input.error);
  // Invalid operational telemetry is discarded. It can never prevent the
  // authoritative dataset cursor or crawl frontier from resuming.
  const telemetry = normalizePersistedTrainingTelemetry(input.telemetry);
  if (input.kind === "selection") {
    const selectionId = String(input.selectionId ?? "");
    const manifestId = input.manifestId === undefined ? undefined : String(input.manifestId);
    if (!SAFE_ID.test(selectionId) || (manifestId !== undefined && !SAFE_ID.test(manifestId))) {
      throw new Error("An initial dataset task is invalid.");
    }
    return {
      id,
      kind: "selection",
      selectionId,
      state,
      ...(manifestId ? { manifestId } : {}),
      ...(error ? { error } : {}),
      ...(telemetry ? { telemetry } : {})
    };
  }
  if (input.kind !== "web" || typeof input.request !== "object" || input.request === null) {
    throw new Error("An initial web task is invalid.");
  }
  const request = input.request as Record<string, unknown>;
  const crawlId = String(request.crawlId ?? "");
  if (!SAFE_ID.test(crawlId) || request.brainId !== brainId) {
    throw new Error("An initial web task is invalid.");
  }
  return {
    id,
    kind: "web",
    state,
    request: {
      brainId,
      url: normalizedHttpUrl(String(request.url ?? "")),
      crawlId,
      policy: "pretrain",
      quarantine: false,
      sameOrigin: true,
      followExternalLinks: false,
      respectRobots: true,
      resume: true
    },
    ...(error ? { error } : {}),
    ...(telemetry ? { telemetry } : {})
  };
}

function normalizePlan(value: unknown, expectedBrainId: string): BuildInitializationPlan {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("The initialization recovery plan is invalid.");
  }
  const input = value as Record<string, unknown>;
  if (
    input.schemaVersion !== 1 ||
    input.brainId !== expectedBrainId ||
    !["foundation", "initial-learning", "failed"].includes(String(input.phase)) ||
    !Number.isSafeInteger(input.attempt) ||
    Number(input.attempt) < 1 ||
    !Array.isArray(input.tasks) ||
    typeof input.createdAt !== "string" ||
    !Number.isFinite(Date.parse(input.createdAt)) ||
    typeof input.updatedAt !== "string" ||
    !Number.isFinite(Date.parse(input.updatedAt))
  ) {
    throw new Error("The initialization recovery plan is invalid.");
  }
  const tasks = input.tasks.map((task) => normalizeTask(task, expectedBrainId));
  if (new Set(tasks.map((task) => task.id)).size !== tasks.length) {
    throw new Error("The initialization recovery plan contains duplicate tasks.");
  }
  const failure = input.failure;
  const normalizedFailure =
    typeof failure === "object" && failure !== null && !Array.isArray(failure)
      ? {
          phase: (failure as Record<string, unknown>).phase,
          message: cleanMessage((failure as Record<string, unknown>).message),
          failedAt: String((failure as Record<string, unknown>).failedAt ?? ""),
          retryable: (failure as Record<string, unknown>).retryable
        }
      : undefined;
  if (
    input.phase === "failed" &&
    (!normalizedFailure ||
      !["foundation", "initial-learning"].includes(String(normalizedFailure.phase)) ||
      !Number.isFinite(Date.parse(normalizedFailure.failedAt)) ||
      normalizedFailure.retryable !== true)
  ) {
    throw new Error("The initialization recovery failure record is invalid.");
  }
  return {
    schemaVersion: 1,
    brainId: expectedBrainId,
    phase: input.phase as BuildInitializationPlan["phase"],
    attempt: Number(input.attempt),
    foundation: normalizeFoundation(input.foundation),
    tasks,
    createdAt: new Date(input.createdAt).toISOString(),
    updatedAt: new Date(input.updatedAt).toISOString(),
    ...(normalizedFailure
      ? {
          failure: {
            phase: normalizedFailure.phase as "foundation" | "initial-learning",
            message: normalizedFailure.message,
            failedAt: new Date(normalizedFailure.failedAt).toISOString(),
            retryable: true as const
          }
        }
      : {})
  };
}

/**
 * Main-process-only initialization journal. It is deliberately outside the
 * portable brain manifest: local selection IDs and unfinished crawl URLs are
 * operational recovery data, not learned memory or exportable identity.
 */
export class BuildInitializationPlanStore {
  private readonly writes = new Map<string, Promise<void>>();

  constructor(private readonly repository: BrainRepository) {}

  private path(brainId: string): string {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid initialization brain id.");
    return join(this.repository.brainDirectory(brainId), FILE_NAME);
  }

  async create(
    brainId: string,
    foundation: InitialFoundationPlan,
    resources: InitialResourceReference[]
  ): Promise<BuildInitializationPlan> {
    const now = new Date().toISOString();
    const seenSelections = new Set<string>();
    const seenUrls = new Set<string>();
    const tasks: InitialLearningTask[] = resources.map((resource) => {
      if (resource.kind === "selection") {
        if (!SAFE_ID.test(resource.selectionId) || seenSelections.has(resource.selectionId)) {
          throw new Error("An initial dataset selection is invalid or duplicated.");
        }
        seenSelections.add(resource.selectionId);
        return {
          id: resource.selectionId,
          kind: "selection",
          selectionId: resource.selectionId,
          state: "pending"
        };
      }
      const url = normalizedHttpUrl(resource.url);
      if (seenUrls.has(url)) throw new Error("An initial web source is duplicated.");
      seenUrls.add(url);
      return {
        id: randomUUID(),
        kind: "web",
        state: "pending",
        request: {
          brainId,
          url,
          crawlId: randomUUID(),
          policy: "pretrain",
          quarantine: false,
          sameOrigin: true,
          followExternalLinks: false,
          respectRobots: true,
          resume: true
        }
      };
    });
    const plan = normalizePlan(
      {
        schemaVersion: 1,
        brainId,
        phase: "foundation",
        attempt: 1,
        foundation,
        tasks,
        createdAt: now,
        updatedAt: now
      },
      brainId
    );
    await this.write(plan);
    return clonePlan(plan);
  }

  async get(brainId: string): Promise<BuildInitializationPlan | undefined> {
    try {
      return normalizePlan(JSON.parse(await readFile(this.path(brainId), "utf8")), brainId);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        try {
          return normalizePlan(
            JSON.parse(await readFile(`${this.path(brainId)}.bak`, "utf8")),
            brainId
          );
        } catch (backupError) {
          if ((backupError as NodeJS.ErrnoException).code === "ENOENT") return undefined;
          throw backupError;
        }
      }
      throw error;
    }
  }

  async update(
    brainId: string,
    mutate: (plan: BuildInitializationPlan) => BuildInitializationPlan
  ): Promise<BuildInitializationPlan> {
    const previous = this.writes.get(brainId) ?? Promise.resolve();
    let result: BuildInitializationPlan | undefined;
    const operation = previous.then(async () => {
      const current = await this.get(brainId);
      if (!current) throw new Error("This mind has no initialization recovery plan.");
      const candidate = mutate(clonePlan(current));
      candidate.updatedAt = new Date().toISOString();
      result = normalizePlan(candidate, brainId);
      await this.writeDirect(result);
    });
    this.writes.set(brainId, operation);
    try {
      await operation;
      return clonePlan(result!);
    } finally {
      if (this.writes.get(brainId) === operation) this.writes.delete(brainId);
    }
  }

  async foundationReady(brainId: string): Promise<BuildInitializationPlan> {
    return this.update(brainId, (plan) => ({
      ...plan,
      phase: "initial-learning",
      failure: undefined
    }));
  }

  async taskState(
    brainId: string,
    taskId: string,
    state: InitialLearningTask["state"],
    details: { manifestId?: string; error?: unknown } = {}
  ): Promise<BuildInitializationPlan> {
    return this.update(brainId, (plan) => ({
      ...plan,
      tasks: plan.tasks.map((task) =>
        task.id !== taskId
          ? task
          : {
              ...task,
              state,
              ...(task.kind === "selection" && details.manifestId
                ? { manifestId: details.manifestId }
                : {}),
              ...(details.error ? { error: cleanMessage(details.error) } : { error: undefined })
            }
      )
    }));
  }

  async taskTelemetry(
    brainId: string,
    taskId: string,
    telemetry: TrainingTelemetry
  ): Promise<BuildInitializationPlan> {
    const normalized = normalizePersistedTrainingTelemetry(telemetry);
    if (!normalized) {
      throw new Error("Initial-learning telemetry snapshot is invalid.");
    }
    return this.update(brainId, (plan) => ({
      ...plan,
      tasks: plan.tasks.map((task) =>
        task.id === taskId ? { ...task, telemetry: normalized } : task
      )
    }));
  }

  async fail(
    brainId: string,
    phase: "foundation" | "initial-learning",
    error: unknown
  ): Promise<BuildInitializationPlan> {
    const failedAt = new Date().toISOString();
    return this.update(brainId, (plan) => ({
      ...plan,
      phase: "failed",
      failure: {
        phase,
        message: cleanMessage(error instanceof Error ? error.message : error),
        failedAt,
        retryable: true
      }
    }));
  }

  async retry(brainId: string): Promise<BuildInitializationPlan> {
    return this.update(brainId, (plan) => ({
      ...plan,
      phase: plan.failure?.phase ?? (plan.tasks.some((task) => task.state !== "complete")
        ? "initial-learning"
        : "foundation"),
      attempt: plan.attempt + 1,
      failure: undefined,
      tasks: plan.tasks.map((task) =>
        task.state === "failed" || task.state === "running"
          ? { ...task, state: "pending", error: undefined }
          : task
      )
    }));
  }

  async remove(brainId: string): Promise<void> {
    const previous = this.writes.get(brainId) ?? Promise.resolve();
    await previous;
    await Promise.all([
      rm(this.path(brainId), { force: true }),
      rm(`${this.path(brainId)}.bak`, { force: true })
    ]);
  }

  private async write(plan: BuildInitializationPlan): Promise<void> {
    const previous = this.writes.get(plan.brainId) ?? Promise.resolve();
    const operation = previous.then(() => this.writeDirect(plan));
    this.writes.set(plan.brainId, operation);
    try {
      await operation;
    } finally {
      if (this.writes.get(plan.brainId) === operation) this.writes.delete(plan.brainId);
    }
  }

  private async writeDirect(plan: BuildInitializationPlan): Promise<void> {
    const path = this.path(plan.brainId);
    await mkdir(dirname(path), { recursive: true });
    const temporary = `${path}.${randomUUID()}.next`;
    await writeFile(temporary, JSON.stringify(plan, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    try {
      await rename(temporary, path);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST" && code !== "EPERM") {
        await rm(temporary, { force: true });
        throw error;
      }
      const backup = `${path}.bak`;
      await rm(backup, { force: true });
      await rename(path, backup);
      try {
        await rename(temporary, path);
        await rm(backup, { force: true });
      } catch (replacementError) {
        await rename(backup, path).catch(() => undefined);
        await rm(temporary, { force: true });
        throw replacementError;
      }
    }
  }
}
