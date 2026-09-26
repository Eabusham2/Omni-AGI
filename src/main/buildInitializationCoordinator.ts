import type {
  BrainDocument,
  CreateBrainRequest,
  RuntimeJob,
  RuntimeJobEvent
} from "../shared/types";
import type { EngineEvent } from "./engineSupervisor";
import { BrainRepository } from "./brainRepository";
import { BrainService, RuntimeJobManager } from "./brainService";
import {
  BuildInitializationPlanStore,
  type BuildInitializationPlan,
  type InitialLearningTask,
  type InitialResourceReference
} from "./buildInitializationPlan";
import { BuildResourceSelectionStore } from "./buildResourceSelections";

export interface InitialLearningProgress {
  progress: number;
  label: string;
  job: RuntimeJob;
}

const terminalStates = new Set<RuntimeJob["state"]>([
  "complete",
  "failed",
  "cancelled"
]);
const TELEMETRY_PERSIST_INTERVAL_MS = 5_000;

function taskError(job: RuntimeJob): Error {
  if (job.state === "cancelled") {
    return new Error("Initial learning was cancelled and can be retried.");
  }
  return new Error(job.error || "Initial learning did not complete.");
}

function webTaskCompletionError(job: RuntimeJob): Error | undefined {
  const output =
    typeof job.output === "object" && job.output !== null && !Array.isArray(job.output)
      ? job.output as Record<string, unknown>
      : undefined;
  const coverage =
    typeof output?.coverage === "object" &&
    output.coverage !== null &&
    !Array.isArray(output.coverage)
      ? output.coverage as Record<string, unknown>
      : undefined;
  if (output?.stopped === true) {
    return new Error(
      "Initial web learning paused before complete crawl coverage was committed."
    );
  }
  if (coverage?.complete !== true) {
    return new Error(
      "Initial web learning ended without complete crawl coverage."
    );
  }
  return undefined;
}

function initialReferences(request: CreateBrainRequest): InitialResourceReference[] {
  return (request.initialResources ?? []).map((resource) => {
    if (resource.kind === "selection") {
      return { kind: "selection", selectionId: resource.selectionId };
    }
    return { kind: "web", url: resource.url };
  });
}

/**
 * Owns the durable handoff from ground-up neural-core construction through every
 * selected first-learning task. Renderer reloads are irrelevant: all source
 * references, crawl IDs, dataset cursors, and retry state live in main-owned
 * stores before the newly created mind can be returned to the UI.
 */
export class BuildInitializationCoordinator {
  private readonly active = new Map<string, Promise<BrainDocument>>();
  private pendingCreates = 0;

  constructor(
    private readonly repository: BrainRepository,
    private readonly service: BrainService,
    private readonly jobs: RuntimeJobManager,
    private readonly selections: BuildResourceSelectionStore,
    private readonly plans = new BuildInitializationPlanStore(repository)
  ) {}

  /**
   * True from the first synchronous edge of a fresh Build until its durable
   * neural core and selected initial learning have completed or failed.
   *
   * This intentionally starts before a brain id exists: core construction and
   * verification also use the one serial neural worker and must not queue
   * behind prompt-free cognition for an older instance.
   */
  isBusy(): boolean {
    return this.pendingCreates > 0 || this.active.size > 0;
  }

  async create(
    request: CreateBrainRequest,
    onFoundationProgress?: (event: EngineEvent) => void,
    onLearningProgress?: (event: InitialLearningProgress) => void
  ): Promise<BrainDocument> {
    this.pendingCreates += 1;
    let brainId = "";
    try {
      const brain = await this.service.create(
        request,
        onFoundationProgress,
        async (persisted, foundation) => {
          brainId = persisted.id;
          await this.plans.create(
            persisted.id,
            foundation,
            initialReferences(request)
          );
          this.jobs.beginInitialization(persisted.id);
          const plan = await this.plans.get(persisted.id);
          for (const task of plan?.tasks ?? []) {
            if (task.kind !== "selection") continue;
            const selection = await this.selections.get(task.selectionId);
            if (!selection) {
              throw new Error(`Initial dataset selection ${task.selectionId} is unavailable.`);
            }
            await this.selections.claim(task.selectionId, persisted.id);
          }
        }
      );
      brainId ||= brain.id;
      await this.plans.foundationReady(brain.id);
      return await this.run(brain.id, onFoundationProgress, onLearningProgress);
    } catch (error) {
      if (brainId) await this.recordFailure(brainId, error);
      throw error;
    } finally {
      this.pendingCreates = Math.max(0, this.pendingCreates - 1);
    }
  }

  async complete(brainId: string): Promise<BrainDocument> {
    const brain = await this.repository.get(brainId);
    if (brain.readiness.state === "ready") return brain;
    const plan = await this.plans.get(brainId);
    if (!plan || plan.phase !== "initial-learning") {
      throw new Error("Initial neural learning has not reached a recoverable completion point.");
    }
    if (plan.tasks.some((task) => task.state !== "complete")) {
      throw new Error("Initial neural learning is still running or requires retry.");
    }
    const completed = await this.repository.completeInitialization(brainId);
    this.jobs.completeInitialization(brainId);
    await Promise.allSettled([
      ...plan.tasks
        .filter((task) => task.kind === "selection")
        .map((task) => this.selections.delete(task.selectionId)),
      this.plans.remove(brainId)
    ]);
    return completed;
  }

  async retry(
    brainId: string,
    onFoundationProgress?: (event: EngineEvent) => void,
    onLearningProgress?: (event: InitialLearningProgress) => void
  ): Promise<BrainDocument> {
    const current = await this.repository.get(brainId);
    if (current.readiness.state === "ready") return current;
    try {
      await this.ensurePlan(current);
      await this.plans.retry(brainId);
      await this.repository.retryInitialization(brainId);
      this.jobs.beginInitialization(brainId);
      return await this.run(
        brainId,
        onFoundationProgress,
        onLearningProgress
      );
    } catch (error) {
      await this.recordFailure(brainId, error);
      throw error;
    }
  }

  async recoverAll(
    onError: (brainId: string, error: unknown) => void = () => undefined,
    onFoundationProgress?: (brainId: string, event: EngineEvent) => void,
    onLearningProgress?: (brainId: string, event: InitialLearningProgress) => void
  ): Promise<void> {
    const initializing: string[] = [];
    for (const summary of await this.repository.list()) {
      const brain = await this.repository.get(summary.id);
      if (brain.readiness.state === "initializing") initializing.push(brain.id);
    }
    // A single neural worker serializes mutations, and foundations can each
    // consume most available unified memory. Deterministic one-at-a-time
    // recovery prevents several interrupted 1B/3B builds from loading in
    // parallel and exhausting RAM before the resource planner can react.
    for (const brainId of initializing) {
      try {
        await this.ensurePlan(await this.repository.get(brainId));
        this.jobs.beginInitialization(brainId);
        await this.plans.update(brainId, (current) => ({
          ...current,
          tasks: current.tasks.map((task) =>
            task.state === "running"
              ? { ...task, state: "pending", error: undefined }
              : task
          )
        }));
        await this.run(
          brainId,
          (event) => onFoundationProgress?.(brainId, event),
          (event) => onLearningProgress?.(brainId, event)
        );
      } catch (error) {
        await this.recordFailure(brainId, error);
        onError(brainId, error);
      }
    }
  }

  private async run(
    brainId: string,
    onFoundationProgress?: (event: EngineEvent) => void,
    onLearningProgress?: (event: InitialLearningProgress) => void
  ): Promise<BrainDocument> {
    const existing = this.active.get(brainId);
    if (existing) return existing;
    const operation = this.runUnlocked(
      brainId,
      onFoundationProgress,
      onLearningProgress
    );
    this.active.set(brainId, operation);
    try {
      return await operation;
    } finally {
      if (this.active.get(brainId) === operation) this.active.delete(brainId);
    }
  }

  private async runUnlocked(
    brainId: string,
    onFoundationProgress?: (event: EngineEvent) => void,
    onLearningProgress?: (event: InitialLearningProgress) => void
  ): Promise<BrainDocument> {
    let plan = await this.requiredPlan(brainId);
    if (plan.phase === "failed") {
      throw new Error(plan.failure?.message ?? "Initialization requires retry.");
    }
    if (plan.phase === "foundation") {
      await this.service.resumeFoundation(
        brainId,
        plan.foundation,
        onFoundationProgress
      );
      plan = await this.plans.foundationReady(brainId);
    }
    for (let index = 0; index < plan.tasks.length; index += 1) {
      plan = await this.requiredPlan(brainId);
      const task = plan.tasks[index]!;
      if (task.state === "complete") continue;
      await this.runTask(
        brainId,
        task,
        index,
        plan.tasks.length,
        onLearningProgress
      );
    }
    return this.complete(brainId);
  }

  private async runTask(
    brainId: string,
    task: InitialLearningTask,
    index: number,
    total: number,
    onProgress?: (event: InitialLearningProgress) => void
  ): Promise<void> {
    await this.plans.taskState(brainId, task.id, "running", {
      ...(task.kind === "selection" && task.manifestId
        ? { manifestId: task.manifestId }
        : {})
    });
    let job: RuntimeJob;
    if (task.kind === "web") {
      job = this.jobs.startCrawl({ ...task.request, resume: true });
    } else {
      const selection = await this.selections.get(task.selectionId);
      if (!selection || (selection.brainId && selection.brainId !== brainId)) {
        throw new Error("An initial dataset selection is unavailable for recovery.");
      }
      if (selection.state === "selected" || selection.state === "preparing") {
        await this.selections.claim(task.selectionId, brainId);
      }
      if (selection.state === "complete") {
        await this.plans.taskState(brainId, task.id, "complete", {
          manifestId: selection.manifestId
        });
        return;
      }
      if (selection.manifestId) {
        job = this.jobs.startIngestion({
          brainId,
          manifestId: selection.manifestId,
          policy: "pretrain",
          epochs: 1,
          resume: true
        }, task.telemetry);
      } else {
        job = this.jobs.startBuildResource(
          {
            brainId,
            selectionId: task.selectionId,
            policy: "pretrain",
            epochs: 1
          },
          selection.paths,
          async (manifestId) => {
            await this.selections.commitManifest(task.selectionId, manifestId);
            await this.plans.taskState(brainId, task.id, "running", { manifestId });
          },
          async () => {
            await this.selections.complete(task.selectionId);
            const current = await this.selections.get(task.selectionId);
            await this.plans.taskState(brainId, task.id, "complete", {
              manifestId: current?.manifestId
            });
          },
          task.telemetry
        );
      }
      await this.selections.attachRuntimeJob(task.selectionId, job.id);
    }

    let pendingTelemetry = job.telemetry;
    let lastTelemetryPersistedAt = task.telemetry
      ? Date.parse(task.telemetry.updatedAt)
      : 0;
    let telemetryWrites: Promise<void> = Promise.resolve();
    const persistTelemetry = (changed: RuntimeJob, force = false): void => {
      if (!changed.telemetry) return;
      pendingTelemetry = changed.telemetry;
      const updatedAt = Date.parse(changed.telemetry.updatedAt);
      const due =
        force ||
        changed.telemetry.stage === "checkpoint" ||
        !Number.isFinite(lastTelemetryPersistedAt) ||
        lastTelemetryPersistedAt <= 0 ||
        updatedAt - lastTelemetryPersistedAt >= TELEMETRY_PERSIST_INTERVAL_MS;
      if (!due) return;
      const snapshot = pendingTelemetry;
      pendingTelemetry = undefined;
      lastTelemetryPersistedAt = updatedAt;
      telemetryWrites = telemetryWrites
        .catch(() => undefined)
        .then(() => this.plans.taskTelemetry(brainId, task.id, snapshot))
        .then(() => undefined);
    };
    const flushTelemetry = async (): Promise<void> => {
      if (pendingTelemetry) {
        persistTelemetry({ ...job, telemetry: pendingTelemetry }, true);
      }
      // Operational telemetry must not make authoritative learning fail. The
      // next worker event or recovery run can safely replace a missed sample.
      await telemetryWrites.catch(() => undefined);
    };
    const reportProgress = (changed: RuntimeJob): void => {
      persistTelemetry(changed, terminalStates.has(changed.state));
      onProgress?.({
        progress: (index + changed.progress) / Math.max(1, total),
        label: changed.label,
        job: changed
      });
    };
    const relay = ({ job: changed }: RuntimeJobEvent): void => {
      if (changed.id === job.id) reportProgress(changed);
    };
    this.jobs.on("event", relay);
    try {
      // Job creation/running and very early record events can occur before the
      // coordinator subscribes. Publish the current snapshot immediately so a
      // retry overlay never disappears while the durable job is still active.
      reportProgress(job);
      const finished = await this.jobs.wait(job.id, undefined, 0);
      persistTelemetry(finished, true);
      await flushTelemetry();
      if (!terminalStates.has(finished.state) || finished.state !== "complete") {
        throw taskError(finished);
      }
      if (task.kind === "selection") {
        const selection = await this.selections.get(task.selectionId);
        if (!selection?.manifestId) {
          throw new Error("Initial dataset training completed without a committed manifest.");
        }
        if (selection.state !== "complete") await this.selections.complete(task.selectionId);
        await this.plans.taskState(brainId, task.id, "complete", {
          manifestId: selection.manifestId
        });
      } else {
        const incompleteWeb = webTaskCompletionError(finished);
        if (incompleteWeb) throw incompleteWeb;
        await this.plans.taskState(brainId, task.id, "complete");
      }
    } catch (error) {
      await flushTelemetry();
      await this.plans.taskState(brainId, task.id, "failed", { error });
      if (task.kind === "selection") {
        // RuntimeJob ids are process-local. Preserve the claimed paths and
        // manifest/cursor for retry, but never leave a failed job looking
        // active in the durable pending-resource index.
        await this.selections
          .releaseForRetry(task.selectionId, brainId)
          .catch(() => undefined);
      }
      throw error;
    } finally {
      this.jobs.off("event", relay);
      await flushTelemetry();
    }
  }

  private async requiredPlan(brainId: string): Promise<BuildInitializationPlan> {
    const plan = await this.plans.get(brainId);
    if (!plan) throw new Error("This mind has no initialization recovery plan.");
    return plan;
  }

  private async ensurePlan(brain: BrainDocument): Promise<BuildInitializationPlan> {
    const existing = await this.plans.get(brain.id);
    if (existing) return existing;
    const seed = brain.readiness.recovery;
    if (!seed) {
      throw new Error(
        "Initialization recovery metadata is missing. This incomplete instance can be safely deleted and rebuilt."
      );
    }
    return this.plans.create(brain.id, seed.foundation, seed.resources);
  }

  private async recordFailure(brainId: string, error: unknown): Promise<void> {
    const plan = await this.plans.get(brainId).catch(() => undefined);
    const phase = plan?.phase === "foundation" ? "foundation" : "initial-learning";
    if (plan) await this.plans.fail(brainId, phase, error).catch(() => undefined);
    if (plan) {
      await Promise.allSettled(
        plan.tasks
          .filter(
            (task): task is Extract<InitialLearningTask, { kind: "selection" }> =>
              task.kind === "selection" && task.state !== "complete"
          )
          .map((task) => this.selections.releaseForRetry(task.selectionId, brainId))
      );
    }
    await this.repository.failInitialization(brainId, phase, error).catch(() => undefined);
    this.jobs.completeInitialization(brainId);
  }
}
