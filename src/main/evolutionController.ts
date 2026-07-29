import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type {
  EvolutionApprovalRequest,
  EvolutionCandidate,
  EvolutionEvaluation,
  EvolutionRollbackRequest,
  EvolutionRun,
  EvolutionStartRequest,
  PromotionRecord,
  ToolPermissionLevel,
  ToolExecutionResult,
  ToolInvocation
} from "../shared/types";
import type { BrainRepository } from "./brainRepository";
import type { EngineSupervisor } from "./engineSupervisor";
import {
  EVOLUTION_BENCHMARK_DOMAINS,
  EVOLUTION_EVALUATOR_VERSION,
  EVOLUTION_POLICY_SHA256,
  EVOLUTION_TEST_NAMES
} from "./evolutionPolicy";

export type EvolutionLimitationKind =
  | "prediction-error"
  | "failed-action"
  | "regression"
  | "resource-pressure"
  | "held-out-evaluation";

export type EvolutionCandidateKind =
  | "source"
  | "neural"
  | "data"
  | "substrate"
  | "architecture";

export interface EvolutionCandidateRoute {
  kind: EvolutionCandidateKind;
  available: boolean;
  protocol?: string;
  reason: string;
}

export interface EvolutionLimitationEvidence {
  id: string;
  kind: EvolutionLimitationKind;
  severity: number;
  observedAt: string;
  source: "brain-message" | "trace" | "journal" | "substrate" | "evolution-archive";
  summary: string;
}

export interface EvolutionResourceMeasurement {
  baselineDurationMs?: number;
  candidateDurationMs?: number;
  durationDeltaMs?: number;
  changedBytes?: number;
  changedPaths?: number;
  untrackedBytes?: number;
}

export interface NeuralEvolutionPromotionRecord {
  id: string;
  candidateId: string;
  createdAt: string;
  parameterChecksumBefore?: string;
  stateChecksumBefore: string;
  stateChecksumAfter: string;
  candidateDiffSha256: string;
  benchmarkSha256: string;
  approvalRequired: boolean;
  reason: string;
  rollbackAvailable: boolean;
  rolledBackAt?: string;
  restoredStateChecksum?: string;
}

export interface EvolutionEvaluationRecord extends EvolutionEvaluation {
  evaluatorVersion: number;
  evaluatorSha256?: string;
  parentCommit?: string;
  boundaryPassed: boolean;
  benchmarkDomains: string[];
  baselineChecks: Array<{
    name: string;
    passed: boolean;
    exitCode?: number;
    durationMs?: number;
  }>;
  regressions: string[];
  resources: EvolutionResourceMeasurement;
  workerResources?: Record<string, unknown>;
  workerMetrics?: Record<string, unknown>;
  workerEvaluationSha256?: string;
  rejectionReason?: string;
}

export interface EvolutionPromotionRecord extends PromotionRecord {
  evaluatorSha256: string;
  reason: string;
  approvalRequired: boolean;
  resources: EvolutionResourceMeasurement;
}

export interface EvolutionCandidateRecord extends EvolutionCandidate {
  candidateKind: EvolutionCandidateKind;
  hypothesis: string;
  limitations: EvolutionLimitationEvidence[];
  proposalParentCommit?: string;
  evaluatorVersion: number;
  evaluatorSha256?: string;
  benchmarkDomains: string[];
  evaluations: EvolutionEvaluationRecord[];
  promotion?: EvolutionPromotionRecord;
  neuralPromotion?: NeuralEvolutionPromotionRecord;
  workerProtocol?: "evolution.*";
  workerCandidateId?: string;
  workerStatus?: string;
  workerSyncError?: string;
  workerBenchmarkSha256?: string;
  workerBaselineManifestSha256?: string;
  workerParentParameterChecksum?: string;
  workerParentStateChecksum?: string;
  workerCandidateParameterChecksum?: string;
  workerCandidateStateChecksum?: string;
  workerCandidateDiffSha256?: string;
  workerSourceIds?: string[];
  workerObjectives?: string[];
  workerSnapshot?: Record<string, unknown>;
  promotionApprovalRequired?: boolean;
  promotionPolicy?: ToolPermissionLevel;
  recursiveNextRunId?: string;
  recursiveError?: string;
}

export interface EvolutionRunRecord extends EvolutionRun {
  candidateKind?: EvolutionCandidateKind;
  hypothesis: string;
  limitations: EvolutionLimitationEvidence[];
  evaluatorVersion: number;
  evaluatorPolicySha256: string;
  recursiveAssessmentOf?: string;
}

interface EvolutionArchive {
  schemaVersion: 1;
  runs: EvolutionRunRecord[];
  candidates: EvolutionCandidateRecord[];
}

export interface EvolutionToolExecutor {
  execute(invocation: ToolInvocation): Promise<ToolExecutionResult>;
  cancel(brainId: string): number;
}

type EvolutionEngine = Pick<EngineSupervisor, "request">;

const WORKER_BENCHMARK_DOMAINS = [
  "neural-objective",
  "capability-retention",
  "neural-retention",
  "ternary-integrity",
  "architecture-compatibility",
  "resource-compatibility"
] as const;

const REQUIRED_WORKER_CHECKS = [
  "integrity",
  "architectureCompatible",
  "resources",
  "ternaryCoverage",
  "objectiveNonRegression",
  "capabilityRetention",
  "neuralRetention",
  "changed"
] as const;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function cleanObjective(value: string): string {
  const clean = value.replace(/\0/g, "").trim();
  if (!clean) throw new Error("An evolution objective is required.");
  if (clean.length > 20_000) throw new Error("The evolution objective is too long.");
  return clean;
}

function outputRecord(result: ToolExecutionResult): Record<string, unknown> {
  return typeof result.output === "object" && result.output !== null && !Array.isArray(result.output)
    ? (result.output as Record<string, unknown>)
    : {};
}

function outputString(output: Record<string, unknown>, key: string): string | undefined {
  const value = output[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function executionError(result: ToolExecutionResult): string {
  return result.error ?? `${result.toolId}.${result.action} did not complete.`;
}

function boundedSeverity(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isSha256(value: string | undefined): value is string {
  return Boolean(value && /^[a-f0-9]{64}$/i.test(value));
}

function isCommit(value: string | undefined): value is string {
  return Boolean(value && /^[a-f0-9]{40,64}$/i.test(value));
}

function outputBoolean(output: Record<string, unknown>, key: string): boolean {
  return output[key] === true;
}

function outputStringArray(output: Record<string, unknown>, key: string): string[] {
  const value = output[key];
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function outputChecks(
  output: Record<string, unknown>,
  key: string
): Array<{ name: string; passed: boolean; exitCode?: number; durationMs?: number }> {
  const value = output[key];
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (entry): entry is Record<string, unknown> =>
        typeof entry === "object" && entry !== null && !Array.isArray(entry)
    )
    .map((entry) => ({
      name: typeof entry.name === "string" ? entry.name : "unknown",
      passed: entry.passed === true,
      exitCode: typeof entry.exitCode === "number" ? entry.exitCode : undefined,
      durationMs: typeof entry.durationMs === "number" ? entry.durationMs : undefined
    }));
}

function outputResources(output: Record<string, unknown>): EvolutionResourceMeasurement {
  const value = output.resources;
  if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
  const record = value as Record<string, unknown>;
  const number = (key: string): number | undefined =>
    typeof record[key] === "number" && Number.isFinite(record[key])
      ? (record[key] as number)
      : undefined;
  return {
    baselineDurationMs: number("baselineDurationMs"),
    candidateDurationMs: number("candidateDurationMs"),
    durationDeltaMs: number("durationDeltaMs"),
    changedBytes: number("changedBytes"),
    changedPaths: number("changedPaths"),
    untrackedBytes: number("untrackedBytes")
  };
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function workerChecks(output: Record<string, unknown>): Array<{
  name: string;
  passed: boolean;
}> {
  const checks = recordValue(output.checks);
  return Object.entries(checks)
    .filter((entry): entry is [string, boolean] => typeof entry[1] === "boolean")
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, passed]) => ({ name, passed }));
}

function cleanStringArray(value: unknown, label: string): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === "string")) {
    throw new Error(`${label} must be a string array.`);
  }
  return value
    .map((entry) => entry.replace(/\0/g, "").trim())
    .filter(Boolean);
}

function workerState(status: string | undefined): EvolutionCandidateRecord["state"] | undefined {
  switch (status) {
    case "training":
    case "ready":
      return "experimenting";
    case "evaluated":
      return "awaiting-review";
    case "promoted":
      return "promoted";
    case "rolled-back":
      return "rolled-back";
    case "rejected":
    case "stale":
      return "rejected";
    case "interrupted":
      return "failed";
    default:
      return undefined;
  }
}

export class EvolutionController {
  private readonly archiveLocks = new Map<string, Promise<void>>();

  constructor(
    private readonly repository: BrainRepository,
    private readonly tools: EvolutionToolExecutor,
    private readonly engine?: EvolutionEngine
  ) {}

  /**
   * Source evolution keeps its external immutable Git evaluator. Neural and
   * data learning use the worker's transactional safe-tensor overlay manager.
   * Architecture mutation remains unavailable until stable shape migration
   * and side-by-side restart are implemented.
   */
  candidateRoutes(): EvolutionCandidateRoute[] {
    return [
      {
        kind: "source",
        available: true,
        protocol: "source.self-modify",
        reason: "Isolated Git worktree with an external immutable evaluator and exact rollback."
      },
      {
        kind: "neural",
        available: Boolean(this.engine),
        protocol: this.engine ? "evolution.*" : undefined,
        reason: this.engine
          ? "Worker-owned safe-tensor overlay with immutable capability, retention, integrity, and resource evaluation."
          : "The supervised neural worker is unavailable."
      },
      {
        kind: "data",
        available: Boolean(this.engine),
        protocol: this.engine ? "evolution.*" : undefined,
        reason: this.engine
          ? "Selected retained sources or explicit text train only an isolated neural overlay with provenance-preserving promotion."
          : "The supervised neural worker is unavailable."
      },
      {
        kind: "substrate",
        available: Boolean(this.engine),
        protocol: this.engine ? "evolution.*" : undefined,
        reason: this.engine
          ? "Isolated latent replay updates assemblies, fast synapses, and slow substrate tensors under the immutable neural evaluator."
          : "The supervised neural worker is unavailable."
      },
      {
        kind: "architecture",
        available: Boolean(this.engine),
        protocol: this.engine ? "evolution.*" : undefined,
        reason: this.engine
          ? "Supports only resource-checked growth of load-compatible ternary residual experts with exact checkpoint rollback."
          : "The supervised neural worker is unavailable. Arbitrary width, depth, and tensor-shape migration remains unsupported."
      }
    ];
  }

  private async permission(brainId: string): Promise<ToolPermissionLevel> {
    const brain = await this.repository.get(brainId);
    return (
      brain.toolPermissions?.find((entry) => entry.toolId === "source.self-modify")
        ?.level ?? "off"
    );
  }

  private async assertWorkerPermission(
    brainId: string,
    action: "experiment" | "promote" | "reject" | "rollback"
  ): Promise<ToolPermissionLevel> {
    const permission = await this.permission(brainId);
    if (permission === "off") {
      throw new Error("Recursive improvement is disabled for the current brain.");
    }
    // Ask permits autonomous isolated experiments. Promotion and rollback enter
    // here only through the explicit approve/rollback IPC operations; Auto and
    // Full may invoke the same operations without an additional token.
    void action;
    return permission;
  }

  private async workerRequest<T extends Record<string, unknown>>(
    method:
      | "evolution.propose"
      | "evolution.evaluate"
      | "evolution.list"
      | "evolution.promote"
      | "evolution.reject"
      | "evolution.rollback",
    brainId: string,
    params: Record<string, unknown> = {},
    timeoutMs = 120_000
  ): Promise<T> {
    if (!this.engine) {
      throw new Error("The supervised neural evolution worker is unavailable.");
    }
    return this.engine.request<T>(
      method,
      {
        brainId,
        storagePath: this.repository.brainDirectory(brainId),
        ...params
      },
      timeoutMs
    );
  }

  private archivePath(brainId: string): string {
    return join(this.repository.brainDirectory(brainId), "evolution", "archive.json");
  }

  private async loadArchive(brainId: string): Promise<EvolutionArchive> {
    await this.repository.get(brainId);
    try {
      const value = JSON.parse(await readFile(this.archivePath(brainId), "utf8")) as EvolutionArchive;
      if (
        value.schemaVersion !== 1 ||
        !Array.isArray(value.runs) ||
        !Array.isArray(value.candidates)
      ) {
        throw new Error("The evolution archive is invalid.");
      }
      for (const run of value.runs) {
        run.candidateKind ??= "source";
        run.hypothesis ??= run.objective;
        run.limitations ??= [];
        run.evaluatorVersion ??= EVOLUTION_EVALUATOR_VERSION;
        run.evaluatorPolicySha256 ??= EVOLUTION_POLICY_SHA256;
      }
      for (const candidate of value.candidates) {
        candidate.candidateKind ??= "source";
        candidate.hypothesis ??= candidate.objective;
        candidate.limitations ??= [];
        candidate.evaluatorVersion ??= EVOLUTION_EVALUATOR_VERSION;
        candidate.benchmarkDomains ??= [...EVOLUTION_BENCHMARK_DOMAINS];
        candidate.evaluations = candidate.evaluations.map((evaluation) => {
          const migrated = { ...evaluation } as EvolutionEvaluationRecord;
          migrated.evaluatorVersion ??= EVOLUTION_EVALUATOR_VERSION;
          migrated.boundaryPassed ??= false;
          migrated.benchmarkDomains ??= [...EVOLUTION_BENCHMARK_DOMAINS];
          migrated.baselineChecks ??= [];
          migrated.regressions ??= [];
          migrated.resources ??= {};
          return migrated;
        });
      }
      return value;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return { schemaVersion: 1, runs: [], candidates: [] };
      }
      throw error;
    }
  }

  private async saveArchive(brainId: string, archive: EvolutionArchive): Promise<void> {
    const path = this.archivePath(brainId);
    await mkdir(dirname(path), { recursive: true });
    const temporary = `${path}.${randomUUID()}.omni-next`;
    await writeFile(temporary, JSON.stringify(archive, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    await rename(temporary, path);
  }

  private async mutateArchive<T>(
    brainId: string,
    mutate: (archive: EvolutionArchive) => T | Promise<T>
  ): Promise<T> {
    const previous = this.archiveLocks.get(brainId) ?? Promise.resolve();
    let release = (): void => undefined;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    const queued = previous.then(() => current);
    this.archiveLocks.set(brainId, queued);
    await previous;
    try {
      const archive = await this.loadArchive(brainId);
      const result = await mutate(archive);
      await this.saveArchive(brainId, archive);
      return clone(result);
    } finally {
      release();
      if (this.archiveLocks.get(brainId) === queued) this.archiveLocks.delete(brainId);
    }
  }

  async detectLimitations(brainId: string): Promise<EvolutionLimitationEvidence[]> {
    const brain = await this.repository.get(brainId);
    const evidence: EvolutionLimitationEvidence[] = [];
    const seen = new Set<string>();
    const add = (
      kind: EvolutionLimitationKind,
      severity: number,
      observedAt: string,
      source: EvolutionLimitationEvidence["source"],
      summary: string
    ): void => {
      const clean = summary.replace(/\s+/g, " ").trim().slice(0, 500);
      const identity = `${kind}\0${source}\0${clean.toLocaleLowerCase()}`;
      if (!clean || seen.has(identity)) return;
      seen.add(identity);
      evidence.push({
        id: randomUUID(),
        kind,
        severity: boundedSeverity(severity),
        observedAt,
        source,
        summary: clean
      });
    };
    const classify = (
      text: string,
      observedAt: string,
      source: EvolutionLimitationEvidence["source"]
    ): void => {
      const lower = text.toLocaleLowerCase();
      if (/(prediction error|uncertaint|loss\b|misunderst|incorrect)/.test(lower)) {
        add("prediction-error", 0.65, observedAt, source, text);
      }
      if (/(regress|retention loss|forgot|capability loss)/.test(lower)) {
        add("regression", 0.85, observedAt, source, text);
      }
      if (/(out of memory|disk|resource|latency|timeout|too slow|ram\b|vram\b)/.test(lower)) {
        add("resource-pressure", 0.75, observedAt, source, text);
      }
      if (/(held-out|benchmark|evaluation|test).*(fail|error|regress)/.test(lower)) {
        add("held-out-evaluation", 0.9, observedAt, source, text);
      }
      if (/(tool|action|command|process).*(fail|error|cancel|timeout)/.test(lower)) {
        add("failed-action", 0.8, observedAt, source, text);
      }
    };

    for (const message of brain.messages.slice(-500)) {
      if (message.status === "error") {
        add(
          "failed-action",
          0.8,
          message.createdAt,
          "brain-message",
          `Failed model turn: ${message.content}`
        );
      } else {
        classify(message.content, message.createdAt, "brain-message");
      }
    }
    for (const trace of brain.traces.slice(-500)) {
      for (const step of trace.steps) {
        classify(`${step.stage}: ${step.detail} ${step.value ?? ""}`, trace.createdAt, "trace");
      }
    }
    for (const journal of (brain.journal ?? []).slice(-500)) {
      classify(
        `${journal.summary} ${journal.detail ?? ""}`,
        journal.createdAt,
        "journal"
      );
    }
    const concepts = Object.values(brain.concepts);
    if (concepts.length) {
      const meanUncertainty =
        concepts.reduce((total, concept) => total + concept.uncertainty, 0) / concepts.length;
      if (meanUncertainty >= 0.65) {
        add(
          "prediction-error",
          meanUncertainty,
          brain.updatedAt,
          "substrate",
          `Mean learned-assembly uncertainty is ${meanUncertainty.toFixed(3)} across ${concepts.length} inspected assemblies.`
        );
      }
    }

    const archive = await this.loadArchive(brainId);
    for (const candidate of archive.candidates.slice(-100)) {
      const latest = candidate.evaluations.at(-1);
      if (latest && !latest.passed) {
        add(
          "held-out-evaluation",
          0.9,
          latest.createdAt,
          "evolution-archive",
          `Candidate ${candidate.id} failed immutable evaluation${
            latest.rejectionReason ? `: ${latest.rejectionReason}` : "."
          }`
        );
      }
      if (
        latest?.resources.durationDeltaMs !== undefined &&
        latest.resources.durationDeltaMs > 0
      ) {
        add(
          "resource-pressure",
          Math.min(1, 0.5 + latest.resources.durationDeltaMs / 60_000),
          latest.createdAt,
          "evolution-archive",
          `Candidate ${candidate.id} increased measured evaluation time by ${latest.resources.durationDeltaMs} ms.`
        );
      }
    }
    return clone(
      evidence.sort(
        (left, right) =>
          right.severity - left.severity || right.observedAt.localeCompare(left.observedAt)
      )
    );
  }

  private async explicitExecution(
    invocation: ToolInvocation
  ): Promise<{ result: ToolExecutionResult; approvalRequired: boolean }> {
    const first = await this.tools.execute(invocation);
    if (first.state !== "approval-required" || !first.approvalToken) {
      return { result: first, approvalRequired: false };
    }
    return {
      result: await this.tools.execute({ ...invocation, approvalToken: first.approvalToken }),
      approvalRequired: true
    };
  }

  async start(request: EvolutionStartRequest): Promise<EvolutionRunRecord> {
    return this.startWithPolicy(request, true);
  }

  private async startWithPolicy(
    request: EvolutionStartRequest,
    allowAutomaticPromotion: boolean
  ): Promise<EvolutionRunRecord> {
    const kind = request.candidateKind ?? "source";
    return kind === "source"
      ? this.startSource(request, allowAutomaticPromotion)
      : this.startWorker(request, kind, allowAutomaticPromotion);
  }

  private async startSource(
    request: EvolutionStartRequest,
    allowAutomaticPromotion: boolean
  ): Promise<EvolutionRunRecord> {
    const objective = cleanObjective(request.objective);
    const detectedLimitations = await this.detectLimitations(request.brainId);
    const ids = { run: randomUUID(), candidate: randomUUID() };
    const now = new Date().toISOString();
    const initial = await this.mutateArchive(request.brainId, (archive) => {
      const parent = request.parentCandidateId
        ? archive.candidates.find((candidate) => candidate.id === request.parentCandidateId)
        : undefined;
      if (request.parentCandidateId && !parent) {
        throw new Error("The parent evolution candidate was not found.");
      }
      if (
        parent &&
        !["promoted", "awaiting-review"].includes(parent.state)
      ) {
        throw new Error("Only a reviewed or promoted candidate may parent another generation.");
      }
      const generation = parent ? parent.generation + 1 : 0;
      const limitations = [...detectedLimitations];
      if (parent?.promotion) {
        limitations.push({
          id: randomUUID(),
          kind: "held-out-evaluation",
          severity: 0.6,
          observedAt: parent.promotion.createdAt,
          source: "evolution-archive",
          summary: `Reassess promoted candidate ${parent.id} against the immutable evaluator before extending its improvement process.`
        });
      }
      const strongest = limitations[0];
      const hypothesis = strongest
        ? `If an isolated source candidate addresses "${objective}" while preserving the immutable evaluator, it should reduce this measured limitation: ${strongest.summary}`
        : `If an isolated source candidate addresses "${objective}", it should pass the immutable baseline comparison without capability or integrity regressions.`;
      const run: EvolutionRunRecord = {
        id: ids.run,
        brainId: request.brainId,
        objective,
        hypothesis,
        limitations,
        state: "experimenting",
        recursive: request.recursive === true,
        generation,
        candidateIds: [ids.candidate],
        createdAt: now,
        updatedAt: now,
        evaluatorVersion: EVOLUTION_EVALUATOR_VERSION,
        evaluatorPolicySha256: EVOLUTION_POLICY_SHA256,
        candidateKind: "source",
        recursiveAssessmentOf: parent?.id
      };
      const candidate: EvolutionCandidateRecord = {
        id: ids.candidate,
        runId: ids.run,
        brainId: request.brainId,
        parentCandidateId: parent?.id,
        generation,
        objective,
        candidateKind: "source",
        hypothesis,
        limitations,
        state: "experimenting",
        createdAt: now,
        updatedAt: now,
        evaluations: [],
        evaluatorVersion: EVOLUTION_EVALUATOR_VERSION,
        benchmarkDomains: [...EVOLUTION_BENCHMARK_DOMAINS]
      };
      archive.runs.push(run);
      archive.candidates.push(candidate);
      return run;
    });

    let proposal: ToolExecutionResult;
    try {
      proposal = await this.tools.execute({
        brainId: request.brainId,
        toolId: "source.self-modify",
        action: "propose",
        arguments: {
          objective,
          hypothesis: initial.hypothesis,
          limitationEvidence: initial.limitations,
          evaluatorPolicySha256: EVOLUTION_POLICY_SHA256
        }
      });
    } catch (error) {
      const message = errorMessage(error);
      return this.mutateArchive(request.brainId, (archive) => {
        const run = archive.runs.find((entry) => entry.id === initial.id);
        const candidate = archive.candidates.find((entry) => entry.id === ids.candidate);
        if (!run || !candidate) throw new Error("The evolution archive changed unexpectedly.");
        const updatedAt = new Date().toISOString();
        run.state = "failed";
        run.error = message;
        run.updatedAt = updatedAt;
        candidate.state = "failed";
        candidate.error = message;
        candidate.updatedAt = updatedAt;
        return run;
      });
    }
    const sourcePolicy = await this.permission(request.brainId);
    const proposedRun = await this.mutateArchive(request.brainId, (archive) => {
      const run = archive.runs.find((entry) => entry.id === initial.id);
      const candidate = archive.candidates.find((entry) => entry.id === ids.candidate);
      if (!run || !candidate) throw new Error("The evolution archive changed unexpectedly.");
      const updatedAt = new Date().toISOString();
      run.updatedAt = updatedAt;
      candidate.updatedAt = updatedAt;
      if (proposal.state !== "complete") {
        const error =
          proposal.state === "approval-required"
            ? "Evolution experiments require permission before they can start."
            : executionError(proposal);
        run.state = "failed";
        run.error = error;
        candidate.state = "failed";
        candidate.error = error;
        return run;
      }
      const output = outputRecord(proposal);
      candidate.worktree = outputString(output, "worktree");
      candidate.branch = outputString(output, "branch");
      candidate.proposalParentCommit = outputString(output, "parentCommit");
      candidate.evaluatorSha256 = outputString(output, "evaluatorSha256");
      const evaluatorVersion = output.evaluatorVersion;
      if (typeof evaluatorVersion === "number" && Number.isInteger(evaluatorVersion)) {
        candidate.evaluatorVersion = evaluatorVersion;
      }
      candidate.benchmarkDomains = outputStringArray(output, "benchmarkDomains");
      candidate.promotionPolicy = sourcePolicy;
      candidate.promotionApprovalRequired = sourcePolicy !== "full";
      if (
        !candidate.worktree ||
        !candidate.branch ||
        !isCommit(candidate.proposalParentCommit) ||
        !isSha256(candidate.evaluatorSha256) ||
        candidate.evaluatorVersion !== EVOLUTION_EVALUATOR_VERSION ||
        candidate.benchmarkDomains.length === 0
      ) {
        const error =
          "The isolated evolution proposal did not return a verifiable worktree, parent, and immutable evaluator identity.";
        run.state = "failed";
        run.error = error;
        candidate.state = "failed";
        candidate.error = error;
      }
      return run;
    });
    if (
      allowAutomaticPromotion &&
      proposedRun.state === "experimenting" &&
      sourcePolicy === "full"
    ) {
      await this.approve({
        brainId: request.brainId,
        candidateId: ids.candidate
      });
      const archive = await this.loadArchive(request.brainId);
      return archive.runs.find((entry) => entry.id === proposedRun.id) ?? proposedRun;
    }
    return proposedRun;
  }

  private async startWorker(
    request: EvolutionStartRequest,
    kind: "neural" | "data" | "substrate" | "architecture",
    allowAutomaticPromotion: boolean
  ): Promise<EvolutionRunRecord> {
    const objective = cleanObjective(request.objective);
    const texts = cleanStringArray(request.texts, "texts") ?? [];
    const sourceIds = cleanStringArray(request.sourceIds, "sourceIds") ?? [];
    const objectives = cleanStringArray(request.objectives, "objectives");
    const architectureChange =
      kind === "architecture"
        ? request.architectureChange ?? {
            mutation: "grow-experts" as const,
            addExperts: 1
          }
        : undefined;
    if (
      architectureChange &&
      (
        architectureChange.mutation !== "grow-experts" ||
        (architectureChange.addExperts !== undefined &&
          (!Number.isSafeInteger(architectureChange.addExperts) ||
            architectureChange.addExperts < 1))
      )
    ) {
      throw new Error(
        "Stable v1 architecture evolution supports only a positive grow-experts mutation."
      );
    }
    const epochs = request.epochs ?? 1;
    if (!Number.isSafeInteger(epochs) || epochs < 1) {
      throw new Error("Evolution epochs must be a positive integer.");
    }
    if (
      request.learningRate !== undefined &&
      (!Number.isFinite(request.learningRate) || request.learningRate <= 0)
    ) {
      throw new Error("Evolution learningRate must be positive and finite.");
    }
    const latentReplay =
      request.latentReplay ??
      (["neural", "substrate"].includes(kind) &&
        texts.length === 0 &&
        sourceIds.length === 0);
    if (kind === "data" && texts.length === 0 && sourceIds.length === 0 && !latentReplay) {
      throw new Error(
        "A data evolution candidate requires texts, retained sourceIds, or latentReplay."
      );
    }
    let provenance: Record<string, unknown>;
    try {
      provenance = clone(request.provenance ?? {});
    } catch {
      throw new Error("Evolution provenance must contain JSON-safe values.");
    }

    const detectedLimitations = await this.detectLimitations(request.brainId);
    const ids = { run: randomUUID(), candidate: randomUUID() };
    const now = new Date().toISOString();
    const initial = await this.mutateArchive(request.brainId, (archive) => {
      const parent = request.parentCandidateId
        ? archive.candidates.find((candidate) => candidate.id === request.parentCandidateId)
        : undefined;
      if (request.parentCandidateId && !parent) {
        throw new Error("The parent evolution candidate was not found.");
      }
      if (parent && !["promoted", "awaiting-review"].includes(parent.state)) {
        throw new Error("Only a reviewed or promoted candidate may parent another generation.");
      }
      if (parent && parent.candidateKind !== kind) {
        throw new Error("A worker evolution lineage cannot change candidate kind.");
      }
      const generation = parent ? parent.generation + 1 : 0;
      const limitations = [...detectedLimitations];
      if (parent?.neuralPromotion) {
        limitations.push({
          id: randomUUID(),
          kind: "held-out-evaluation",
          severity: 0.6,
          observedAt: parent.neuralPromotion.createdAt,
          source: "evolution-archive",
          summary: `Reassess promoted ${kind} candidate ${parent.id} against its immutable neural benchmark before extending its improvement process.`
        });
      }
      const strongest = limitations[0];
      const hypothesis = strongest
        ? `If an isolated ${kind} overlay addresses "${objective}" while preserving neural retention and integrity, it should reduce this measured limitation: ${strongest.summary}`
        : `If an isolated ${kind} overlay addresses "${objective}", it should pass immutable capability, retention, integrity, ternary, and resource checks.`;
      const run: EvolutionRunRecord = {
        id: ids.run,
        brainId: request.brainId,
        objective,
        hypothesis,
        limitations,
        state: "experimenting",
        recursive: request.recursive === true,
        generation,
        candidateIds: [ids.candidate],
        createdAt: now,
        updatedAt: now,
        evaluatorVersion: EVOLUTION_EVALUATOR_VERSION,
        evaluatorPolicySha256: EVOLUTION_POLICY_SHA256,
        recursiveAssessmentOf: parent?.id,
        candidateKind: kind
      };
      const candidate: EvolutionCandidateRecord = {
        id: ids.candidate,
        runId: ids.run,
        brainId: request.brainId,
        parentCandidateId: parent?.id,
        generation,
        objective,
        candidateKind: kind,
        hypothesis,
        limitations,
        state: "experimenting",
        createdAt: now,
        updatedAt: now,
        evaluations: [],
        evaluatorVersion: EVOLUTION_EVALUATOR_VERSION,
        benchmarkDomains: [...WORKER_BENCHMARK_DOMAINS],
        workerProtocol: "evolution.*",
        workerSourceIds: sourceIds,
        workerObjectives: objectives
      };
      archive.runs.push(run);
      archive.candidates.push(candidate);
      return run;
    });

    let proposal: Record<string, unknown>;
    let policy: ToolPermissionLevel;
    try {
      policy = await this.assertWorkerPermission(request.brainId, "experiment");
      proposal = await this.workerRequest<Record<string, unknown>>(
        "evolution.propose",
        request.brainId,
        {
          jobId: ids.run,
          texts,
          sourceIds,
          epochs,
          learningRate: request.learningRate,
          latentReplay,
          objectives,
          architectureChange,
          provenance: {
            ...provenance,
            objective,
            hypothesis: initial.hypothesis,
            limitationEvidence: initial.limitations,
            route: kind
          }
        },
        30 * 60_000
      );
    } catch (error) {
      const message = errorMessage(error);
      return this.mutateArchive(request.brainId, (archive) => {
        const run = archive.runs.find((entry) => entry.id === initial.id);
        const candidate = archive.candidates.find((entry) => entry.id === ids.candidate);
        if (!run || !candidate) throw new Error("The evolution archive changed unexpectedly.");
        const updatedAt = new Date().toISOString();
        run.state = "failed";
        run.error = message;
        run.updatedAt = updatedAt;
        candidate.state = "failed";
        candidate.error = message;
        candidate.updatedAt = updatedAt;
        return run;
      });
    }

    const proposedRun = await this.mutateArchive(request.brainId, (archive) => {
      const run = archive.runs.find((entry) => entry.id === initial.id);
      const candidate = archive.candidates.find((entry) => entry.id === ids.candidate);
      if (!run || !candidate) throw new Error("The evolution archive changed unexpectedly.");
      const updatedAt = new Date().toISOString();
      run.updatedAt = updatedAt;
      candidate.updatedAt = updatedAt;
      const workerCandidateId = outputString(proposal, "id");
      const workerStatus = outputString(proposal, "status");
      const benchmarkSha256 = outputString(proposal, "benchmarkSha256");
      const baselineManifestSha256 = outputString(proposal, "baselineManifestSha256");
      const parentParameterChecksum = outputString(proposal, "parentParameterChecksum");
      const parentStateChecksum = outputString(proposal, "parentStateChecksum");
      const candidateParameterChecksum = outputString(proposal, "candidateParameterChecksum");
      const candidateStateChecksum = outputString(proposal, "candidateStateChecksum");
      const candidateDiffSha256 = outputString(proposal, "candidateDiffSha256");
      const candidateType = outputString(proposal, "candidateType");
      const validIdentity =
        Boolean(workerCandidateId && /^[a-f0-9]{32}$/i.test(workerCandidateId)) &&
        isSha256(benchmarkSha256) &&
        isSha256(baselineManifestSha256) &&
        isSha256(parentParameterChecksum) &&
        isSha256(parentStateChecksum) &&
        isSha256(candidateParameterChecksum) &&
        isSha256(candidateStateChecksum) &&
        isSha256(candidateDiffSha256) &&
        (kind !== "architecture" || candidateType === "architecture");
      if (!validIdentity || !["ready", "rejected"].includes(workerStatus ?? "")) {
        const error =
          "The worker evolution proposal did not return a complete candidate, baseline, benchmark, and state-checksum identity.";
        run.state = "failed";
        run.error = error;
        candidate.state = "failed";
        candidate.error = error;
        return run;
      }
      candidate.workerCandidateId = workerCandidateId;
      candidate.workerStatus = workerStatus;
      candidate.workerBenchmarkSha256 = benchmarkSha256;
      candidate.workerBaselineManifestSha256 = baselineManifestSha256;
      candidate.workerParentParameterChecksum = parentParameterChecksum;
      candidate.workerParentStateChecksum = parentStateChecksum;
      candidate.workerCandidateParameterChecksum = candidateParameterChecksum;
      candidate.workerCandidateStateChecksum = candidateStateChecksum;
      candidate.workerCandidateDiffSha256 = candidateDiffSha256;
      candidate.workerSourceIds = outputStringArray(proposal, "sourceIds");
      candidate.workerObjectives = outputStringArray(proposal, "objectives");
      candidate.workerSnapshot = clone(proposal);
      candidate.evaluatorSha256 = benchmarkSha256;
      candidate.promotionPolicy = policy;
      candidate.promotionApprovalRequired =
        policy === "ask" || (policy === "auto" && kind === "architecture");
      if (workerStatus === "rejected") {
        const reason = outputString(proposal, "reason") ?? "The worker rejected the candidate.";
        candidate.state = "rejected";
        candidate.error = reason;
        run.state = "rejected";
        run.error = reason;
      }
      return run;
    });
    const automaticPromotion =
      allowAutomaticPromotion &&
      proposedRun.state === "experimenting" &&
      (
        (["neural", "data", "substrate"].includes(kind) &&
          ["auto", "full"].includes(policy)) ||
        (kind === "architecture" && policy === "full")
      );
    if (automaticPromotion) {
      await this.approve({
        brainId: request.brainId,
        candidateId: ids.candidate
      });
      const archive = await this.loadArchive(request.brainId);
      return (
        archive.runs.find((entry) => entry.id === proposedRun.id) ??
        proposedRun
      );
    }
    return proposedRun;
  }

  private async syncWorkerArchive(brainId: string): Promise<void> {
    const archive = await this.loadArchive(brainId);
    const workerCandidates = archive.candidates.filter(
      (candidate) => candidate.candidateKind !== "source" && candidate.workerCandidateId
    );
    if (!this.engine || workerCandidates.length === 0) return;
    let listed: Record<string, unknown>;
    try {
      listed = await this.workerRequest<Record<string, unknown>>(
        "evolution.list",
        brainId
      );
    } catch (error) {
      await this.mutateArchive(brainId, (current) => {
        const message = errorMessage(error);
        for (const candidate of current.candidates) {
          if (candidate.candidateKind !== "source" && candidate.workerCandidateId) {
            candidate.workerSyncError = message;
          }
        }
        return true;
      });
      return;
    }
    const values = Array.isArray(listed.candidates)
      ? listed.candidates
          .filter(
            (entry): entry is Record<string, unknown> =>
              typeof entry === "object" && entry !== null && !Array.isArray(entry)
          )
      : [];
    const byId = new Map(
      values
        .map((entry) => [outputString(entry, "id"), entry] as const)
        .filter((entry): entry is [string, Record<string, unknown>] => Boolean(entry[0]))
    );
    await this.mutateArchive(brainId, (current) => {
      for (const candidate of current.candidates) {
        if (candidate.candidateKind === "source" || !candidate.workerCandidateId) continue;
        const snapshot = byId.get(candidate.workerCandidateId);
        if (!snapshot) {
          candidate.workerSyncError =
            "The worker candidate is absent from the authoritative candidate list.";
          continue;
        }
        candidate.workerSyncError = undefined;
        candidate.workerSnapshot = clone(snapshot);
        const status = outputString(snapshot, "status");
        candidate.workerStatus = status;
        const synchronizedState = workerState(status);
        if (
          synchronizedState &&
          !["stopped", "rolled-back"].includes(candidate.state)
        ) {
          candidate.state = synchronizedState;
        }
        const reason = outputString(snapshot, "reason");
        if (reason && ["rejected", "failed"].includes(candidate.state)) {
          candidate.error = reason;
        }
      }
      return true;
    });
  }

  async stop(brainId: string, runId: string): Promise<EvolutionRunRecord> {
    this.tools.cancel(brainId);
    const before = await this.loadArchive(brainId);
    const workerCandidates = before.candidates.filter(
      (candidate) =>
        candidate.runId === runId &&
        candidate.candidateKind !== "source" &&
        candidate.workerCandidateId &&
        !["promoted", "rolled-back", "rejected"].includes(candidate.state)
    );
    let workerError: string | undefined;
    for (const candidate of workerCandidates) {
      try {
        await this.assertWorkerPermission(brainId, "reject");
        await this.workerRequest(
          "evolution.reject",
          brainId,
          {
            candidateId: candidate.workerCandidateId,
            reason: "Stopped by operator."
          }
        );
      } catch (error) {
        workerError = errorMessage(error);
      }
    }
    return this.mutateArchive(brainId, (archive) => {
      const run = archive.runs.find((entry) => entry.id === runId);
      if (!run) throw new Error("The evolution run was not found.");
      if (["promoted", "rolled-back"].includes(run.state)) {
        throw new Error("A completed evolution run cannot be stopped.");
      }
      const now = new Date().toISOString();
      run.state = "stopped";
      run.error = workerError;
      run.updatedAt = now;
      for (const candidate of archive.candidates) {
        if (candidate.runId !== run.id || candidate.state === "promoted") continue;
        candidate.state = "stopped";
        if (workerError && candidate.candidateKind !== "source") {
          candidate.error = workerError;
        }
        candidate.updatedAt = now;
      }
      return run;
    });
  }

  async listCandidates(brainId: string, runId?: string): Promise<EvolutionCandidateRecord[]> {
    await this.syncWorkerArchive(brainId);
    const archive = await this.loadArchive(brainId);
    return clone(
      archive.candidates
        .filter((candidate) => !runId || candidate.runId === runId)
        .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
    );
  }

  async approve(request: EvolutionApprovalRequest): Promise<EvolutionCandidateRecord> {
    const candidate = (await this.listCandidates(request.brainId)).find(
      (entry) => entry.id === request.candidateId
    );
    if (!candidate) throw new Error("The evolution candidate was not found.");
    if (candidate.candidateKind !== "source") {
      return this.approveWorker(request, candidate);
    }
    if (!candidate.worktree) throw new Error("The evolution candidate has no isolated worktree.");
    if (!candidate.evaluatorSha256 || !candidate.proposalParentCommit) {
      throw new Error("The evolution candidate has no immutable evaluator identity.");
    }
    if (["stopped", "promoted", "rolled-back"].includes(candidate.state)) {
      throw new Error(`The ${candidate.state} evolution candidate cannot be promoted.`);
    }
    const evaluationTests = [
      ...new Set([...(request.tests ?? []), ...EVOLUTION_TEST_NAMES])
    ];

    let tested: ToolExecutionResult;
    try {
      tested = await this.tools.execute({
        brainId: request.brainId,
        toolId: "source.self-modify",
        action: "test",
        arguments: {
          worktree: candidate.worktree,
          tests: evaluationTests,
          timeoutMs: request.timeoutMs,
          expectedEvaluatorSha256: candidate.evaluatorSha256
        }
      });
    } catch (error) {
      const now = new Date().toISOString();
      tested = {
        id: randomUUID(),
        toolId: "source.self-modify",
        action: "test",
        state: "failed",
        startedAt: now,
        finishedAt: now,
        error: errorMessage(error)
      };
    }
    const testOutput = outputRecord(tested);
    const checks = outputChecks(testOutput, "checks");
    const baselineChecks = outputChecks(testOutput, "baselineChecks");
    const diffSha256 = outputString(testOutput, "diffSha256");
    const evaluatorSha256 = outputString(testOutput, "evaluatorSha256");
    const parentCommit = outputString(testOutput, "parentCommit");
    const benchmarkDomains = outputStringArray(testOutput, "benchmarkDomains");
    const regressions = outputStringArray(testOutput, "regressions");
    const requiredChecks = new Set<string>(evaluationTests);
    requiredChecks.add("diff-check");
    const checksComplete =
      checks.length > 0 &&
      [...requiredChecks].every((name) =>
        checks.some((check) => check.name === name && check.passed)
      ) &&
      checks.every((check) => check.passed);
    const rejectionReasons = [
      tested.state !== "complete" ? executionError(tested) : undefined,
      !outputBoolean(testOutput, "passed") ? "Candidate checks did not pass." : undefined,
      !outputBoolean(testOutput, "boundaryPassed")
        ? "Immutable evaluator boundary was not verified."
        : undefined,
      evaluatorSha256 !== candidate.evaluatorSha256
        ? "Evaluator hash changed between proposal and evaluation."
        : undefined,
      parentCommit !== candidate.proposalParentCommit
        ? "Candidate parent changed between proposal and evaluation."
        : undefined,
      !isSha256(diffSha256) ? "Evaluation did not return a valid diff hash." : undefined,
      !checksComplete ? "Required checks were missing or failed." : undefined,
      regressions.length ? `Detected regressions: ${regressions.join(", ")}.` : undefined
    ].filter((reason): reason is string => Boolean(reason));
    const evaluation: EvolutionEvaluationRecord = {
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      passed: rejectionReasons.length === 0,
      diffSha256,
      evaluatorVersion: candidate.evaluatorVersion,
      evaluatorSha256,
      parentCommit,
      boundaryPassed: outputBoolean(testOutput, "boundaryPassed"),
      benchmarkDomains,
      checks,
      baselineChecks,
      regressions,
      resources: outputResources(testOutput),
      rejectionReason: rejectionReasons.join(" ")
    };
    await this.mutateArchive(request.brainId, (archive) => {
      const current = archive.candidates.find((entry) => entry.id === candidate.id);
      const run = archive.runs.find((entry) => entry.id === candidate.runId);
      if (!current || !run) throw new Error("The evolution archive changed unexpectedly.");
      current.evaluations.push(evaluation);
      current.updatedAt = evaluation.createdAt;
      current.state = evaluation.passed ? "awaiting-review" : "rejected";
      current.error = evaluation.passed ? undefined : evaluation.rejectionReason;
      run.updatedAt = evaluation.createdAt;
      run.state = current.state;
      run.error = current.error;
      return current;
    });
    if (!evaluation.passed || !evaluation.diffSha256) {
      return (await this.listCandidates(request.brainId)).find(
        (entry) => entry.id === candidate.id
      )!;
    }

    const promotionExecution = await this.explicitExecution({
      brainId: request.brainId,
      toolId: "source.self-modify",
      action: "promote",
      arguments: {
        worktree: candidate.worktree,
        expectedDiffSha256: evaluation.diffSha256,
        expectedEvaluatorSha256: candidate.evaluatorSha256
      }
    });
    const promoted = promotionExecution.result;
    const promotedOutput = outputRecord(promoted);
    const updated = await this.mutateArchive(request.brainId, (archive) => {
      const current = archive.candidates.find((entry) => entry.id === candidate.id);
      const run = archive.runs.find((entry) => entry.id === candidate.runId);
      if (!current || !run) throw new Error("The evolution archive changed unexpectedly.");
      const now = new Date().toISOString();
      current.updatedAt = now;
      run.updatedAt = now;
      if (promoted.state !== "complete") {
        current.state = "failed";
        current.error = executionError(promoted);
        run.state = "failed";
        run.error = current.error;
        return current;
      }
      const commit = outputString(promotedOutput, "commit");
      const parentCommit = outputString(promotedOutput, "parentCommit");
      const promotedDiff = outputString(promotedOutput, "diffSha256");
      const promotedEvaluator = outputString(promotedOutput, "evaluatorSha256");
      if (
        !isCommit(commit) ||
        parentCommit !== current.proposalParentCommit ||
        promotedDiff !== evaluation.diffSha256 ||
        promotedEvaluator !== current.evaluatorSha256
      ) {
        const error =
          "The promoted candidate did not return verifiable parent, diff, and evaluator lineage.";
        current.state = "failed";
        current.error = error;
        run.state = "failed";
        run.error = error;
        return current;
      }
      const promotion: EvolutionPromotionRecord = {
        id: randomUUID(),
        candidateId: current.id,
        createdAt: now,
        commit,
        parentCommit: parentCommit!,
        diffSha256: evaluation.diffSha256!,
        evaluatorSha256: current.evaluatorSha256!,
        reason:
          `Passed immutable evaluator ${current.evaluatorSha256} across ` +
          `${evaluation.checks.length} checks with no recorded regression.`,
        approvalRequired: promotionExecution.approvalRequired,
        resources: evaluation.resources
      };
      current.promotion = promotion;
      current.promotionApprovalRequired = false;
      current.state = "promoted";
      current.error = undefined;
      run.state = "promoted";
      run.error = undefined;
      return current;
    });

    const archive = await this.loadArchive(request.brainId);
    const run = archive.runs.find((entry) => entry.id === candidate.runId);
    if (updated.state === "promoted" && run?.recursive) {
      try {
        const next = await this.startWithPolicy({
          brainId: request.brainId,
          objective:
            `Reassess the promoted change and improve the improvement process under the same immutable evaluator: ` +
            run.objective,
          recursive: true,
          parentCandidateId: updated.id
        }, false);
        await this.mutateArchive(request.brainId, (currentArchive) => {
          const current = currentArchive.candidates.find((entry) => entry.id === updated.id);
          if (!current) throw new Error("The recursive evolution parent disappeared.");
          current.recursiveNextRunId = next.id;
          current.recursiveError = next.state === "failed" ? next.error : undefined;
          current.updatedAt = new Date().toISOString();
          return current;
        });
      } catch (error) {
        await this.mutateArchive(request.brainId, (currentArchive) => {
          const current = currentArchive.candidates.find((entry) => entry.id === updated.id);
          if (!current) throw new Error("The recursive evolution parent disappeared.");
          current.recursiveError = errorMessage(error);
          current.updatedAt = new Date().toISOString();
          return current;
        });
      }
    }
    return (
      (await this.listCandidates(request.brainId)).find((entry) => entry.id === updated.id) ??
      updated
    );
  }

  private async approveWorker(
    request: EvolutionApprovalRequest,
    candidate: EvolutionCandidateRecord
  ): Promise<EvolutionCandidateRecord> {
    if (!candidate.workerCandidateId) {
      throw new Error("The worker evolution candidate has no isolated overlay identity.");
    }
    if (
      !candidate.workerBenchmarkSha256 ||
      !candidate.workerParentStateChecksum ||
      !candidate.workerCandidateStateChecksum ||
      !candidate.workerCandidateDiffSha256
    ) {
      throw new Error("The worker evolution candidate has incomplete immutable lineage.");
    }
    if (["stopped", "promoted", "rolled-back"].includes(candidate.state)) {
      throw new Error(`The ${candidate.state} evolution candidate cannot be promoted.`);
    }
    await this.assertWorkerPermission(request.brainId, "promote");
    const timeout =
      typeof request.timeoutMs === "number" &&
      Number.isFinite(request.timeoutMs) &&
      request.timeoutMs > 0
        ? request.timeoutMs
        : 30 * 60_000;
    let evaluated: Record<string, unknown> = {};
    let evaluationError: string | undefined;
    try {
      evaluated = await this.workerRequest(
        "evolution.evaluate",
        request.brainId,
        { candidateId: candidate.workerCandidateId },
        timeout
      );
    } catch (error) {
      evaluationError = errorMessage(error);
    }

    const checks = workerChecks(evaluated);
    const checksByName = new Map(checks.map((check) => [check.name, check.passed]));
    const benchmarkSha256 = outputString(evaluated, "benchmarkSha256");
    const evaluationSha256 = outputString(evaluated, "evaluationSha256");
    const evaluatedCandidateId = outputString(evaluated, "candidateId");
    const failures = outputStringArray(evaluated, "failures");
    const completeChecks = REQUIRED_WORKER_CHECKS.every(
      (name) => checksByName.has(name) && checksByName.get(name) === true
    );
    const boundaryPassed =
      checksByName.get("integrity") === true &&
      checksByName.get("architectureCompatible") === true &&
      checksByName.get("resources") === true &&
      checksByName.get("ternaryCoverage") === true;
    const rejectionReasons = [
      evaluationError,
      evaluatedCandidateId !== candidate.workerCandidateId
        ? "The worker evaluated a different candidate."
        : undefined,
      benchmarkSha256 !== candidate.workerBenchmarkSha256
        ? "The immutable neural benchmark hash changed after proposal."
        : undefined,
      !isSha256(evaluationSha256)
        ? "The worker evaluation has no verifiable checksum."
        : undefined,
      outputString(evaluated, "status") !== "evaluated"
        ? "The worker did not leave the candidate in evaluated state."
        : undefined,
      evaluated.passed !== true ? "The neural candidate did not pass evaluation." : undefined,
      !completeChecks ? "Required neural evaluation checks were missing or failed." : undefined,
      failures.length ? `Detected neural regressions: ${failures.join(", ")}.` : undefined
    ].filter((reason): reason is string => Boolean(reason));
    const evaluation: EvolutionEvaluationRecord = {
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      passed: rejectionReasons.length === 0,
      diffSha256: candidate.workerCandidateDiffSha256,
      evaluatorVersion: candidate.evaluatorVersion,
      evaluatorSha256: benchmarkSha256,
      boundaryPassed,
      benchmarkDomains: [...WORKER_BENCHMARK_DOMAINS],
      checks,
      baselineChecks: [],
      regressions: failures,
      resources: {},
      workerResources: clone(recordValue(evaluated.resources)),
      workerMetrics: clone(recordValue(evaluated.metrics)),
      workerEvaluationSha256: evaluationSha256,
      rejectionReason: rejectionReasons.join(" ")
    };
    await this.mutateArchive(request.brainId, (archive) => {
      const current = archive.candidates.find((entry) => entry.id === candidate.id);
      const run = archive.runs.find((entry) => entry.id === candidate.runId);
      if (!current || !run) throw new Error("The evolution archive changed unexpectedly.");
      current.evaluations.push(evaluation);
      current.workerStatus = outputString(evaluated, "status") ?? "evaluation-failed";
      current.updatedAt = evaluation.createdAt;
      current.state = evaluation.passed ? "awaiting-review" : "rejected";
      current.error = evaluation.passed ? undefined : evaluation.rejectionReason;
      run.updatedAt = evaluation.createdAt;
      run.state = current.state;
      run.error = current.error;
      return current;
    });
    if (!evaluation.passed) {
      try {
        await this.assertWorkerPermission(request.brainId, "reject");
        await this.workerRequest("evolution.reject", request.brainId, {
          candidateId: candidate.workerCandidateId,
          reason: evaluation.rejectionReason || "Neural evaluation failed."
        });
      } catch (error) {
        await this.mutateArchive(request.brainId, (archive) => {
          const current = archive.candidates.find((entry) => entry.id === candidate.id);
          if (!current) throw new Error("The evolution archive changed unexpectedly.");
          current.workerSyncError = `Worker rejection could not be recorded: ${errorMessage(error)}`;
          return current;
        });
      }
      return (
        (await this.listCandidates(request.brainId)).find(
          (entry) => entry.id === candidate.id
        ) ?? candidate
      );
    }

    let promoted: Record<string, unknown>;
    try {
      promoted = await this.workerRequest(
        "evolution.promote",
        request.brainId,
        { candidateId: candidate.workerCandidateId },
        timeout
      );
    } catch (error) {
      const message = errorMessage(error);
      return this.mutateArchive(request.brainId, (archive) => {
        const current = archive.candidates.find((entry) => entry.id === candidate.id);
        const run = archive.runs.find((entry) => entry.id === candidate.runId);
        if (!current || !run) throw new Error("The evolution archive changed unexpectedly.");
        current.state = "failed";
        current.error = message;
        current.updatedAt = new Date().toISOString();
        run.state = "failed";
        run.error = message;
        run.updatedAt = current.updatedAt;
        return current;
      });
    }
    const promotedCandidateId = outputString(promoted, "candidateId");
    const stateBefore = outputString(promoted, "stateChecksumBefore");
    const stateAfter = outputString(promoted, "stateChecksumAfter");
    const promotionValid =
      promoted.promoted === true &&
      promotedCandidateId === candidate.workerCandidateId &&
      stateBefore === candidate.workerParentStateChecksum &&
      stateAfter === candidate.workerCandidateStateChecksum &&
      promoted.rollbackAvailable === true;
    const updated = await this.mutateArchive(request.brainId, (archive) => {
      const current = archive.candidates.find((entry) => entry.id === candidate.id);
      const run = archive.runs.find((entry) => entry.id === candidate.runId);
      if (!current || !run) throw new Error("The evolution archive changed unexpectedly.");
      const now = new Date().toISOString();
      current.updatedAt = now;
      run.updatedAt = now;
      if (!promotionValid || !stateBefore || !stateAfter) {
        const error =
          "The worker promotion did not return the exact verified parent and candidate state lineage.";
        current.state = "failed";
        current.error = error;
        run.state = "failed";
        run.error = error;
        return current;
      }
      current.neuralPromotion = {
        id: randomUUID(),
        candidateId: current.id,
        createdAt: now,
        parameterChecksumBefore: outputString(promoted, "parameterChecksumBefore"),
        stateChecksumBefore: stateBefore,
        stateChecksumAfter: stateAfter,
        candidateDiffSha256: current.workerCandidateDiffSha256!,
        benchmarkSha256: current.workerBenchmarkSha256!,
        approvalRequired: candidate.promotionApprovalRequired === true,
        reason:
          `Passed immutable neural benchmark ${current.workerBenchmarkSha256} ` +
          `across ${evaluation.checks.length} transactional checks.`,
        rollbackAvailable: true
      };
      current.promotionApprovalRequired = false;
      current.workerStatus = "promoted";
      current.state = "promoted";
      current.error = undefined;
      run.state = "promoted";
      run.error = undefined;
      return current;
    });

    const archive = await this.loadArchive(request.brainId);
    const run = archive.runs.find((entry) => entry.id === candidate.runId);
    if (updated.state === "promoted" && run?.recursive) {
      try {
        const next = await this.startWithPolicy({
          brainId: request.brainId,
          objective:
            `Reassess the promoted ${candidate.candidateKind} overlay and improve the improvement process under the same immutable neural gates: ${run.objective}`,
          recursive: true,
          parentCandidateId: updated.id,
          candidateKind: candidate.candidateKind,
          sourceIds:
            candidate.candidateKind === "data" ? candidate.workerSourceIds : undefined,
          latentReplay: true,
          objectives: candidate.workerObjectives
        }, false);
        await this.mutateArchive(request.brainId, (currentArchive) => {
          const current = currentArchive.candidates.find((entry) => entry.id === updated.id);
          if (!current) throw new Error("The recursive evolution parent disappeared.");
          current.recursiveNextRunId = next.id;
          current.recursiveError = next.state === "failed" ? next.error : undefined;
          current.updatedAt = new Date().toISOString();
          return current;
        });
      } catch (error) {
        await this.mutateArchive(request.brainId, (currentArchive) => {
          const current = currentArchive.candidates.find((entry) => entry.id === updated.id);
          if (!current) throw new Error("The recursive evolution parent disappeared.");
          current.recursiveError = errorMessage(error);
          current.updatedAt = new Date().toISOString();
          return current;
        });
      }
    }
    return (
      (await this.listCandidates(request.brainId)).find(
        (entry) => entry.id === updated.id
      ) ?? updated
    );
  }

  async rollback(request: EvolutionRollbackRequest): Promise<EvolutionCandidateRecord> {
    const candidate = (await this.listCandidates(request.brainId)).find(
      (entry) => entry.id === request.candidateId
    );
    if (candidate && candidate.candidateKind !== "source") {
      return this.rollbackWorker(request, candidate);
    }
    if (!candidate?.promotion) throw new Error("The candidate has no promotion to roll back.");
    if (candidate.state === "rolled-back") return candidate;

    const rollbackExecution = await this.explicitExecution({
      brainId: request.brainId,
      toolId: "source.self-modify",
      action: "rollback",
      arguments: {
        expectedCommit: candidate.promotion.commit,
        parentCommit: candidate.promotion.parentCommit
      }
    });
    const rolledBack = rollbackExecution.result;
    const output = outputRecord(rolledBack);
    return this.mutateArchive(request.brainId, (archive) => {
      const current = archive.candidates.find((entry) => entry.id === candidate.id);
      const run = archive.runs.find((entry) => entry.id === candidate.runId);
      if (!current?.promotion || !run) throw new Error("The evolution archive changed unexpectedly.");
      const now = new Date().toISOString();
      current.updatedAt = now;
      run.updatedAt = now;
      if (rolledBack.state !== "complete") {
        current.error = executionError(rolledBack);
        return current;
      }
      const rollbackCommit = outputString(output, "rollbackCommit");
      if (!isCommit(rollbackCommit)) {
        throw new Error("Rollback did not return a verifiable commit.");
      }
      current.promotion.rollbackCommit = rollbackCommit;
      current.promotion.rolledBackAt = now;
      current.state = "rolled-back";
      current.error = undefined;
      run.state = "rolled-back";
      run.error = undefined;
      return current;
    });
  }

  private async rollbackWorker(
    request: EvolutionRollbackRequest,
    candidate: EvolutionCandidateRecord
  ): Promise<EvolutionCandidateRecord> {
    if (!candidate.workerCandidateId || !candidate.neuralPromotion) {
      throw new Error("The worker candidate has no neural promotion to roll back.");
    }
    if (candidate.state === "rolled-back") return candidate;
    await this.assertWorkerPermission(request.brainId, "rollback");
    const result = await this.workerRequest(
      "evolution.rollback",
      request.brainId,
      { candidateId: candidate.workerCandidateId }
    );
    const candidateId = outputString(result, "candidateId");
    const stateBefore = outputString(result, "stateChecksumBefore");
    const stateAfter = outputString(result, "stateChecksumAfter");
    if (
      result.rolledBack !== true ||
      candidateId !== candidate.workerCandidateId ||
      stateBefore !== candidate.neuralPromotion.stateChecksumAfter ||
      stateAfter !== candidate.neuralPromotion.stateChecksumBefore
    ) {
      throw new Error(
        "Worker rollback did not restore the exact verified parent neural state."
      );
    }
    return this.mutateArchive(request.brainId, (archive) => {
      const current = archive.candidates.find((entry) => entry.id === candidate.id);
      const run = archive.runs.find((entry) => entry.id === candidate.runId);
      if (!current?.neuralPromotion || !run) {
        throw new Error("The evolution archive changed unexpectedly.");
      }
      const now = new Date().toISOString();
      current.neuralPromotion.rollbackAvailable = false;
      current.neuralPromotion.rolledBackAt = now;
      current.neuralPromotion.restoredStateChecksum = stateAfter;
      current.workerStatus = "rolled-back";
      current.state = "rolled-back";
      current.error = undefined;
      current.updatedAt = now;
      run.state = "rolled-back";
      run.error = undefined;
      run.updatedAt = now;
      return current;
    });
  }
}
