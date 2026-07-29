import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import {
  EvolutionController,
  type EvolutionToolExecutor
} from "../src/main/evolutionController";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { EVOLUTION_BENCHMARK_DOMAINS } from "../src/main/evolutionPolicy";
import {
  DEFAULT_CONFIG,
  type BrainDocument,
  type ToolExecutionResult,
  type ToolInvocation
} from "../src/shared/types";

const PARENT_COMMIT = "c".repeat(40);
const PROMOTION_COMMIT = "b".repeat(40);
const EVALUATOR_SHA256 = "e".repeat(64);
const DIFF_SHA256 = "a".repeat(64);
const WORKER_CANDIDATE_ID = "1".repeat(32);
const WORKER_BENCHMARK_SHA256 = "2".repeat(64);
const WORKER_BASELINE_SHA256 = "3".repeat(64);
const WORKER_PARENT_PARAMETER_SHA256 = "4".repeat(64);
const WORKER_PARENT_STATE_SHA256 = "5".repeat(64);
const WORKER_CANDIDATE_PARAMETER_SHA256 = "6".repeat(64);
const WORKER_CANDIDATE_STATE_SHA256 = "7".repeat(64);
const WORKER_DIFF_SHA256 = "8".repeat(64);
const WORKER_EVALUATION_SHA256 = "9".repeat(64);

function workerCandidate(
  status = "ready",
  candidateType: "neural" | "architecture" = "neural"
): Record<string, unknown> {
  return {
    id: WORKER_CANDIDATE_ID,
    kind: "neural-evolution",
    candidateType,
    status,
    createdAt: new Date().toISOString(),
    benchmarkSha256: WORKER_BENCHMARK_SHA256,
    baselineManifestSha256: WORKER_BASELINE_SHA256,
    parentParameterChecksum: WORKER_PARENT_PARAMETER_SHA256,
    parentStateChecksum: WORKER_PARENT_STATE_SHA256,
    candidateParameterChecksum: WORKER_CANDIDATE_PARAMETER_SHA256,
    candidateStateChecksum: WORKER_CANDIDATE_STATE_SHA256,
    candidateDiffSha256: WORKER_DIFF_SHA256,
    sourceIds: ["retained-source"],
    objectives: ["language-prediction", "retention", "capability"],
    architectureMutation:
      candidateType === "architecture"
        ? {
            mutation: "grow-experts",
            addExperts: 1,
            expertCountBefore: 0,
            expertCountAfter: 1
          }
        : undefined
  };
}

function workerEvaluation(passed = true): Record<string, unknown> {
  return {
    brainId: "worker-fixture",
    candidateId: WORKER_CANDIDATE_ID,
    status: passed ? "evaluated" : "rejected",
    passed,
    benchmarkSha256: WORKER_BENCHMARK_SHA256,
    evaluationSha256: WORKER_EVALUATION_SHA256,
    checks: {
      integrity: true,
      architectureCompatible: true,
      resources: true,
      ternaryCoverage: true,
      objectiveNonRegression: passed,
      capabilityRetention: true,
      neuralRetention: true,
      changed: true
    },
    failures: passed ? [] : ["objectiveNonRegression"],
    metrics: { candidateObjectiveLoss: passed ? 0.5 : 50 },
    resources: { tensorCount: 20, tensorBytes: 4096 }
  };
}

function proposalOutput(worktree: string, branch: string, parentCommit = PARENT_COMMIT) {
  return {
    worktree,
    branch,
    parentCommit,
    evaluatorVersion: 1,
    evaluatorSha256: EVALUATOR_SHA256,
    benchmarkDomains: [...EVOLUTION_BENCHMARK_DOMAINS]
  };
}

function passingEvaluationOutput(
  checks: string[],
  parentCommit = PARENT_COMMIT
): Record<string, unknown> {
  return {
    passed: true,
    boundaryPassed: true,
    diffSha256: DIFF_SHA256,
    parentCommit,
    evaluatorVersion: 1,
    evaluatorSha256: EVALUATOR_SHA256,
    benchmarkDomains: [...EVOLUTION_BENCHMARK_DOMAINS],
    checks: [
      ...checks.map((name) => ({ name, passed: true, exitCode: 0, durationMs: 5 })),
      { name: "diff-check", passed: true, exitCode: 0, durationMs: 1 }
    ],
    baselineChecks: checks.map((name) => ({
      name,
      passed: true,
      exitCode: 0,
      durationMs: 6
    })),
    regressions: [],
    resources: {
      baselineDurationMs: checks.length * 6,
      candidateDurationMs: checks.length * 5 + 1,
      durationDeltaMs: 1 - checks.length,
      changedBytes: 128,
      changedPaths: 2,
      untrackedBytes: 32
    }
  };
}

function execution(
  invocation: ToolInvocation,
  state: ToolExecutionResult["state"],
  output?: unknown,
  approvalToken?: string,
  error?: string
): ToolExecutionResult {
  const now = new Date().toISOString();
  return {
    id: `${invocation.action}-${Math.random()}`,
    toolId: invocation.toolId,
    action: invocation.action,
    state,
    startedAt: now,
    finishedAt: state === "approval-required" ? undefined : now,
    output,
    approvalToken,
    error
  };
}

describe("EvolutionController", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;
  let brain: BrainDocument;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-evolution-controller-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
    brain = await repository.create({ ...DEFAULT_CONFIG, name: "Evolution fixture" });
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  async function setEvolutionPermission(
    level: "off" | "ask" | "auto" | "full"
  ): Promise<void> {
    const current = await repository.get(brain.id);
    current.toolPermissions = (current.toolPermissions ?? []).map((permission) =>
      permission.toolId === "source.self-modify"
        ? { ...permission, level, updatedAt: new Date().toISOString() }
        : permission
    );
    await repository.save(current);
    brain = current;
  }

  it("durably archives isolated candidates and stopping without deleting evidence", async () => {
    const tools: EvolutionToolExecutor = {
      execute: vi.fn(async (invocation) =>
        execution(
          invocation,
          "complete",
          proposalOutput(
            join(temporaryRoot, "worktree"),
            "omni-evolution/fixture"
          )
        )
      ),
      cancel: vi.fn(() => 1)
    };
    const controller = new EvolutionController(repository, tools);
    expect(controller.candidateRoutes()).toEqual([
      expect.objectContaining({ kind: "source", available: true }),
      expect.objectContaining({ kind: "neural", available: false }),
      expect.objectContaining({ kind: "data", available: false }),
      expect.objectContaining({ kind: "substrate", available: false }),
      expect.objectContaining({ kind: "architecture", available: false })
    ]);
    const run = await controller.start({
      brainId: brain.id,
      objective: "Reduce inference latency without retention loss"
    });
    expect(run).toMatchObject({
      state: "experimenting",
      recursive: false,
      generation: 0
    });

    const reloaded = new EvolutionController(repository, tools);
    expect(await reloaded.listCandidates(brain.id, run.id)).toEqual([
      expect.objectContaining({
        runId: run.id,
        objective: "Reduce inference latency without retention loss",
        worktree: join(temporaryRoot, "worktree"),
        branch: "omni-evolution/fixture"
      })
    ]);
    const stopped = await reloaded.stop(brain.id, run.id);
    expect(stopped.state).toBe("stopped");
    expect(tools.cancel).toHaveBeenCalledWith(brain.id);
    expect((await reloaded.listCandidates(brain.id, run.id))[0]?.state).toBe("stopped");
  });

  it("forwards typed source edits and archives their authored hash lineage", async () => {
    const sourceEdits = [
      {
        path: "src/cache-policy.ts",
        content: "export const cachePolicy = \"measured\";\n",
        expectedSha256: "d".repeat(64)
      },
      {
        path: "src/new-maintenance.ts",
        content: "export const maintenance = true;\n",
        expectedSha256: null
      }
    ];
    const sourceEditLineage = sourceEdits.map((edit) => ({
      path: edit.path,
      expectedSha256: edit.expectedSha256,
      resultSha256: edit.path.includes("new-") ? "1".repeat(64) : "2".repeat(64),
      bytes: Buffer.byteLength(edit.content)
    }));
    const execute = vi.fn(async (invocation: ToolInvocation) =>
      execution(invocation, "complete", {
        ...proposalOutput(
          join(temporaryRoot, "typed-worktree"),
          "omni-evolution/typed"
        ),
        sourceEditLineage,
        authoredChangedPaths: sourceEdits.map((edit) => edit.path).sort(),
        authoredDiffSha256: DIFF_SHA256,
        authoredBytes: sourceEditLineage.reduce(
          (total, edit) => total + edit.bytes,
          0
        )
      })
    );
    const controller = new EvolutionController(repository, {
      execute,
      cancel: vi.fn(() => 0)
    });

    const run = await controller.start({
      brainId: brain.id,
      objective: "Apply a bounded measured cache maintenance patch",
      candidateKind: "source",
      sourceEdits
    });

    expect(execute).toHaveBeenCalledWith(
      expect.objectContaining({
        toolId: "source.self-modify",
        action: "propose",
        arguments: expect.objectContaining({ sourceEdits })
      })
    );
    expect((await controller.listCandidates(brain.id, run.id))[0]).toMatchObject({
      state: "experimenting",
      sourceEditLineage,
      authoredChangedPaths: sourceEdits.map((edit) => edit.path).sort(),
      authoredDiffSha256: DIFF_SHA256,
      authoredBytes: sourceEditLineage.reduce(
        (total, edit) => total + edit.bytes,
        0
      )
    });
  });

  it("rejects an evaluator result that claims an empty source candidate passed", async () => {
    const execute = vi.fn(async (invocation: ToolInvocation) => {
      if (invocation.action === "propose") {
        return execution(
          invocation,
          "complete",
          proposalOutput(
            join(temporaryRoot, "empty-worktree"),
            "omni-evolution/empty"
          )
        );
      }
      if (invocation.action === "test") {
        return execution(invocation, "complete", {
          ...passingEvaluationOutput(["typecheck", "unit", "build"]),
          resources: {
            baselineDurationMs: 3,
            candidateDurationMs: 3,
            durationDeltaMs: 0,
            changedBytes: 0,
            changedPaths: 0,
            untrackedBytes: 0
          }
        });
      }
      return execution(
        invocation,
        "failed",
        undefined,
        undefined,
        "Promotion must never execute for an empty source candidate."
      );
    });
    const controller = new EvolutionController(repository, {
      execute,
      cancel: vi.fn(() => 0)
    });
    const run = await controller.start({
      brainId: brain.id,
      objective: "Do not promote an empty source fork"
    });
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;

    const rejected = await controller.approve({
      brainId: brain.id,
      candidateId: candidate.id
    });

    expect(rejected).toMatchObject({
      state: "rejected",
      error: expect.stringMatching(/empty source candidate cannot be promoted/i)
    });
    expect(execute.mock.calls.some(([invocation]) => invocation.action === "promote")).toBe(
      false
    );
  });

  it("bridges neural candidate proposal, list, evaluation, promotion, and rollback through dotted worker RPCs", async () => {
    await setEvolutionPermission("ask");
    let status = "ready";
    const request = vi.fn(
      async (
        method: string,
        params: Record<string, unknown>
      ): Promise<Record<string, unknown>> => {
        expect(params).toMatchObject({
          brainId: brain.id,
          storagePath: repository.brainDirectory(brain.id)
        });
        if (method === "evolution.propose") {
          expect(params).toMatchObject({
            texts: ["learn this causal example"],
            epochs: 2,
            latentReplay: false
          });
          return workerCandidate(status);
        }
        if (method === "evolution.list") {
          return { brainId: brain.id, candidates: [workerCandidate(status)] };
        }
        if (method === "evolution.evaluate") {
          expect(params.candidateId).toBe(WORKER_CANDIDATE_ID);
          status = "evaluated";
          return workerEvaluation(true);
        }
        if (method === "evolution.promote") {
          expect(params.candidateId).toBe(WORKER_CANDIDATE_ID);
          status = "promoted";
          return {
            brainId: brain.id,
            candidateId: WORKER_CANDIDATE_ID,
            promoted: true,
            parameterChecksumBefore: WORKER_PARENT_PARAMETER_SHA256,
            stateChecksumBefore: WORKER_PARENT_STATE_SHA256,
            stateChecksumAfter: WORKER_CANDIDATE_STATE_SHA256,
            rollbackAvailable: true,
            reloadRequired: true
          };
        }
        if (method === "evolution.rollback") {
          expect(params).not.toHaveProperty("force");
          status = "rolled-back";
          return {
            brainId: brain.id,
            candidateId: WORKER_CANDIDATE_ID,
            rolledBack: true,
            stateChecksumBefore: WORKER_CANDIDATE_STATE_SHA256,
            stateChecksumAfter: WORKER_PARENT_STATE_SHA256,
            reloadRequired: true
          };
        }
        throw new Error(`Unexpected worker method ${method}`);
      }
    );
    const tools: EvolutionToolExecutor = {
      execute: vi.fn(async () => {
        throw new Error("Worker evolution must not use the source worktree executor.");
      }),
      cancel: vi.fn(() => 0)
    };
    const controller = new EvolutionController(
      repository,
      tools,
      { request } as unknown as EngineSupervisor
    );
    expect(controller.candidateRoutes()).toEqual([
      expect.objectContaining({ kind: "source", available: true }),
      expect.objectContaining({
        kind: "neural",
        available: true,
        protocol: "evolution.*"
      }),
      expect.objectContaining({
        kind: "data",
        available: true,
        protocol: "evolution.*"
      }),
      expect.objectContaining({
        kind: "substrate",
        available: true,
        protocol: "evolution.*"
      }),
      expect.objectContaining({
        kind: "architecture",
        available: true,
        protocol: "evolution.*"
      })
    ]);

    const run = await controller.start({
      brainId: brain.id,
      objective: "Improve causal adaptation",
      candidateKind: "neural",
      texts: ["learn this causal example"],
      epochs: 2,
      latentReplay: false
    });
    expect(run).toMatchObject({
      candidateKind: "neural",
      state: "experimenting"
    });
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;
    expect(candidate).toMatchObject({
      candidateKind: "neural",
      workerProtocol: "evolution.*",
      workerCandidateId: WORKER_CANDIDATE_ID,
      workerBenchmarkSha256: WORKER_BENCHMARK_SHA256,
      workerParentStateChecksum: WORKER_PARENT_STATE_SHA256,
      workerCandidateStateChecksum: WORKER_CANDIDATE_STATE_SHA256,
      promotionPolicy: "ask",
      promotionApprovalRequired: true
    });
    expect(request.mock.calls.map(([method]) => method)).not.toContain(
      "evolution.evaluate"
    );

    const promoted = await controller.approve({
      brainId: brain.id,
      candidateId: candidate.id
    });
    expect(promoted).toMatchObject({
      state: "promoted",
      workerStatus: "promoted",
      neuralPromotion: {
        stateChecksumBefore: WORKER_PARENT_STATE_SHA256,
        stateChecksumAfter: WORKER_CANDIDATE_STATE_SHA256,
        candidateDiffSha256: WORKER_DIFF_SHA256,
        benchmarkSha256: WORKER_BENCHMARK_SHA256,
        approvalRequired: true,
        rollbackAvailable: true
      }
    });
    expect(promoted.evaluations[0]).toMatchObject({
      passed: true,
      boundaryPassed: true,
      evaluatorSha256: WORKER_BENCHMARK_SHA256,
      workerEvaluationSha256: WORKER_EVALUATION_SHA256
    });

    const rolledBack = await controller.rollback({
      brainId: brain.id,
      candidateId: candidate.id
    });
    expect(rolledBack).toMatchObject({
      state: "rolled-back",
      neuralPromotion: {
        rollbackAvailable: false,
        restoredStateChecksum: WORKER_PARENT_STATE_SHA256,
        rolledBackAt: expect.any(String)
      }
    });
    expect(
      request.mock.calls.map(([method]) => method)
    ).toEqual(
      expect.arrayContaining([
        "evolution.propose",
        "evolution.list",
        "evolution.evaluate",
        "evolution.promote",
        "evolution.rollback"
      ])
    );
    expect(tools.execute).not.toHaveBeenCalled();
  });

  it("uses the worker data route, rejects failed candidates, and keeps architecture/off routes closed", async () => {
    const offRequest = vi.fn();
    const offController = new EvolutionController(
      repository,
      {
        execute: vi.fn(),
        cancel: vi.fn(() => 0)
      },
      { request: offRequest } as unknown as EngineSupervisor
    );
    const disabled = await offController.start({
      brainId: brain.id,
      objective: "Attempt neural evolution while disabled",
      candidateKind: "neural",
      latentReplay: true
    });
    expect(disabled).toMatchObject({
      state: "failed",
      error: expect.stringMatching(/disabled/i)
    });
    expect(offRequest).not.toHaveBeenCalled();
    const disabledArchitecture = await offController.start({
      brainId: brain.id,
      objective: "Grow one compatible expert while disabled",
      candidateKind: "architecture",
      texts: ["disabled architecture fixture"]
    });
    expect(disabledArchitecture.state).toBe("failed");
    await expect(
      offController.start({
        brainId: brain.id,
        objective: "Invalid expert growth",
        candidateKind: "architecture",
        architectureChange: {
          mutation: "grow-experts",
          addExperts: 0
        }
      })
    ).rejects.toThrow(/positive grow-experts/i);

    await setEvolutionPermission("auto");
    let status = "ready";
    const request = vi.fn(
      async (
        method: string,
        params: Record<string, unknown>
      ): Promise<Record<string, unknown>> => {
        if (method === "evolution.propose") {
          expect(params).toMatchObject({
            sourceIds: ["retained-source"],
            latentReplay: false
          });
          return workerCandidate(status);
        }
        if (method === "evolution.list") {
          return { brainId: brain.id, candidates: [workerCandidate(status)] };
        }
        if (method === "evolution.evaluate") {
          status = "rejected";
          return workerEvaluation(false);
        }
        if (method === "evolution.reject") {
          expect(params).toMatchObject({ candidateId: WORKER_CANDIDATE_ID });
          status = "rejected";
          return {
            brainId: brain.id,
            candidateId: WORKER_CANDIDATE_ID,
            status,
            reason: params.reason
          };
        }
        throw new Error(`Unexpected worker method ${method}`);
      }
    );
    const controller = new EvolutionController(
      repository,
      {
        execute: vi.fn(),
        cancel: vi.fn(() => 0)
      },
      { request } as unknown as EngineSupervisor
    );
    const run = await controller.start({
      brainId: brain.id,
      objective: "Learn from retained source",
      candidateKind: "data",
      sourceIds: ["retained-source"],
      latentReplay: false
    });
    expect(run.state).toBe("rejected");
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;
    expect(candidate.candidateKind).toBe("data");
    expect(candidate).toMatchObject({
      state: "rejected",
      error: expect.stringMatching(/did not pass|missing or failed/i)
    });
    expect(request.mock.calls.map(([method]) => method)).toContain("evolution.reject");
    expect(request.mock.calls.map(([method]) => method)).not.toContain(
      "evolution.promote"
    );
  });

  it("auto-promotes a passing substrate overlay and preserves exact rollback lineage", async () => {
    await setEvolutionPermission("auto");
    let status = "ready";
    const request = vi.fn(
      async (
        method: string,
        params: Record<string, unknown>
      ): Promise<Record<string, unknown>> => {
        if (method === "evolution.propose") return workerCandidate(status);
        if (method === "evolution.list") {
          return { brainId: brain.id, candidates: [workerCandidate(status)] };
        }
        if (method === "evolution.evaluate") {
          status = "evaluated";
          return workerEvaluation(true);
        }
        if (method === "evolution.promote") {
          status = "promoted";
          return {
            brainId: brain.id,
            candidateId: WORKER_CANDIDATE_ID,
            promoted: true,
            parameterChecksumBefore: WORKER_PARENT_PARAMETER_SHA256,
            stateChecksumBefore: WORKER_PARENT_STATE_SHA256,
            stateChecksumAfter: WORKER_CANDIDATE_STATE_SHA256,
            rollbackAvailable: true
          };
        }
        if (method === "evolution.rollback") {
          status = "rolled-back";
          return {
            brainId: brain.id,
            candidateId: WORKER_CANDIDATE_ID,
            rolledBack: true,
            stateChecksumBefore: WORKER_CANDIDATE_STATE_SHA256,
            stateChecksumAfter: WORKER_PARENT_STATE_SHA256
          };
        }
        throw new Error(`Unexpected worker method ${method}`);
      }
    );
    const controller = new EvolutionController(
      repository,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { request } as unknown as EngineSupervisor
    );
    const run = await controller.start({
      brainId: brain.id,
      objective: "Automatically improve retained substrate capability",
      candidateKind: "substrate",
      texts: ["auto substrate promotion fixture"]
    });
    expect(run.state).toBe("promoted");
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;
    expect(candidate).toMatchObject({
      state: "promoted",
      candidateKind: "substrate",
      promotionPolicy: "auto",
      promotionApprovalRequired: false,
      neuralPromotion: {
        approvalRequired: false,
        stateChecksumBefore: WORKER_PARENT_STATE_SHA256,
        stateChecksumAfter: WORKER_CANDIDATE_STATE_SHA256
      }
    });
    const rolledBack = await controller.rollback({
      brainId: brain.id,
      candidateId: candidate.id
    });
    expect(rolledBack.state).toBe("rolled-back");
  });

  it("full authority auto-promotes only the concrete compatible expert-growth architecture", async () => {
    await setEvolutionPermission("full");
    let status = "ready";
    const request = vi.fn(
      async (
        method: string,
        params: Record<string, unknown>
      ): Promise<Record<string, unknown>> => {
        if (method === "evolution.propose") {
          expect(params.architectureChange).toEqual({
            mutation: "grow-experts",
            addExperts: 2
          });
          return workerCandidate(status, "architecture");
        }
        if (method === "evolution.list") {
          return {
            brainId: brain.id,
            candidates: [workerCandidate(status, "architecture")]
          };
        }
        if (method === "evolution.evaluate") {
          status = "evaluated";
          return {
            ...workerEvaluation(true),
            architectureMutation: {
              mutation: "grow-experts",
              addExperts: 2,
              expertCountBefore: 0,
              expertCountAfter: 2
            }
          };
        }
        if (method === "evolution.promote") {
          status = "promoted";
          return {
            brainId: brain.id,
            candidateId: WORKER_CANDIDATE_ID,
            candidateType: "architecture",
            promoted: true,
            stateChecksumBefore: WORKER_PARENT_STATE_SHA256,
            stateChecksumAfter: WORKER_CANDIDATE_STATE_SHA256,
            rollbackAvailable: true,
            architectureMutation: {
              mutation: "grow-experts",
              addExperts: 2
            }
          };
        }
        throw new Error(`Unexpected worker method ${method}`);
      }
    );
    const controller = new EvolutionController(
      repository,
      { execute: vi.fn(), cancel: vi.fn(() => 0) },
      { request } as unknown as EngineSupervisor
    );
    const run = await controller.start({
      brainId: brain.id,
      objective: "Grow two resource-checked residual experts",
      candidateKind: "architecture",
      architectureChange: {
        mutation: "grow-experts",
        addExperts: 2
      },
      texts: ["architecture promotion fixture"]
    });
    expect(run.state).toBe("promoted");
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;
    expect(candidate).toMatchObject({
      candidateKind: "architecture",
      state: "promoted",
      promotionPolicy: "full",
      promotionApprovalRequired: false,
      workerSnapshot: {
        candidateType: "architecture",
        architectureMutation: {
          mutation: "grow-experts"
        }
      },
      neuralPromotion: {
        approvalRequired: false
      }
    });
  });

  it("full authority auto-promotes source candidates while auto leaves source promotion for review", async () => {
    const execute = vi.fn(
      async (invocation: ToolInvocation): Promise<ToolExecutionResult> => {
        if (invocation.action === "propose") {
          return execution(
            invocation,
            "complete",
            proposalOutput(
              join(temporaryRoot, "policy-worktree"),
              "omni-evolution/policy"
            )
          );
        }
        if (invocation.action === "test") {
          return execution(
            invocation,
            "complete",
            passingEvaluationOutput(["typecheck", "unit", "build"])
          );
        }
        if (invocation.action === "promote") {
          return execution(invocation, "complete", {
            promoted: true,
            commit: PROMOTION_COMMIT,
            parentCommit: PARENT_COMMIT,
            diffSha256: DIFF_SHA256,
            evaluatorSha256: EVALUATOR_SHA256
          });
        }
        return execution(
          invocation,
          "failed",
          undefined,
          undefined,
          "Unexpected policy action."
        );
      }
    );
    const controller = new EvolutionController(repository, {
      execute,
      cancel: vi.fn(() => 0)
    });

    await setEvolutionPermission("auto");
    const held = await controller.start({
      brainId: brain.id,
      objective: "Keep source candidate awaiting explicit review",
      candidateKind: "source"
    });
    expect(held.state).toBe("experimenting");
    expect(execute.mock.calls.map(([invocation]) => invocation.action)).toEqual([
      "propose"
    ]);

    await setEvolutionPermission("full");
    const promoted = await controller.start({
      brainId: brain.id,
      objective: "Promote verified source candidate under full authority",
      candidateKind: "source"
    });
    expect(promoted.state).toBe("promoted");
    expect(execute.mock.calls.map(([invocation]) => invocation.action)).toEqual(
      expect.arrayContaining(["test", "promote"])
    );
  });

  it("records evaluations, consumes exact promotion approval, recurses, and rolls back by commit", async () => {
    let proposal = 0;
    const execute = vi.fn(async (invocation: ToolInvocation): Promise<ToolExecutionResult> => {
      if (invocation.action === "propose") {
        proposal += 1;
        return execution(
          invocation,
          "complete",
          proposalOutput(
            join(temporaryRoot, `worktree-${proposal}`),
            `omni-evolution/${proposal}`,
            proposal === 1 ? PARENT_COMMIT : PROMOTION_COMMIT
          )
        );
      }
      if (invocation.action === "test") {
        return execution(
          invocation,
          "complete",
          passingEvaluationOutput(["typecheck", "unit", "build"])
        );
      }
      if (invocation.action === "promote" && !invocation.approvalToken) {
        return execution(invocation, "approval-required", undefined, "promotion-token");
      }
      if (invocation.action === "promote") {
        expect(invocation.approvalToken).toBe("promotion-token");
        return execution(invocation, "complete", {
          promoted: true,
          commit: PROMOTION_COMMIT,
          parentCommit: PARENT_COMMIT,
          diffSha256: DIFF_SHA256,
          evaluatorSha256: EVALUATOR_SHA256
        });
      }
      if (invocation.action === "rollback" && !invocation.approvalToken) {
        return execution(invocation, "approval-required", undefined, "rollback-token");
      }
      if (invocation.action === "rollback") {
        expect(invocation.approvalToken).toBe("rollback-token");
        expect(invocation.arguments).toEqual({
          expectedCommit: PROMOTION_COMMIT,
          parentCommit: PARENT_COMMIT
        });
        return execution(invocation, "complete", {
          rolledBack: true,
          rollbackCommit: "d".repeat(40)
        });
      }
      return execution(invocation, "failed", undefined, undefined, "Unexpected action.");
    });
    const controller = new EvolutionController(repository, {
      execute,
      cancel: vi.fn(() => 0)
    });
    const run = await controller.start({
      brainId: brain.id,
      objective: "Improve the improvement evaluator",
      recursive: true
    });
    const initial = (await controller.listCandidates(brain.id, run.id))[0]!;
    const promoted = await controller.approve({
      brainId: brain.id,
      candidateId: initial.id,
      tests: ["typecheck", "unit"]
    });
    expect(promoted).toMatchObject({
      state: "promoted",
      promotion: {
        commit: PROMOTION_COMMIT,
        parentCommit: PARENT_COMMIT,
        diffSha256: DIFF_SHA256,
        evaluatorSha256: EVALUATOR_SHA256,
        approvalRequired: true
      }
    });
    expect(promoted.evaluations).toEqual([
      expect.objectContaining({
        passed: true,
        boundaryPassed: true,
        evaluatorSha256: EVALUATOR_SHA256,
        checks: [
          { name: "typecheck", passed: true, exitCode: 0, durationMs: 5 },
          { name: "unit", passed: true, exitCode: 0, durationMs: 5 },
          { name: "build", passed: true, exitCode: 0, durationMs: 5 },
          { name: "diff-check", passed: true, exitCode: 0, durationMs: 1 }
        ]
      })
    ]);
    const allCandidates = await controller.listCandidates(brain.id);
    expect(allCandidates).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          parentCandidateId: promoted.id,
          generation: 1,
          state: "experimenting"
        })
      ])
    );

    const rolledBack = await controller.rollback({
      brainId: brain.id,
      candidateId: promoted.id
    });
    expect(rolledBack).toMatchObject({
      state: "rolled-back",
      promotion: {
        rollbackCommit: "d".repeat(40),
        rolledBackAt: expect.any(String)
      }
    });
    expect(execute.mock.calls.filter(([invocation]) => invocation.action === "promote")).toHaveLength(2);
    expect(execute.mock.calls.filter(([invocation]) => invocation.action === "rollback")).toHaveLength(2);
  });

  it("derives limitation evidence from neural uncertainty, failed turns, traces, and resource events", async () => {
    const current = await repository.get(brain.id);
    const now = new Date().toISOString();
    current.concepts.uncertain = {
      id: "uncertain",
      label: "uncertain assembly",
      activation: 0.5,
      importance: 0.5,
      uncertainty: 0.92,
      exposures: 2,
      createdAt: now,
      lastActivatedAt: now,
      aliases: []
    };
    current.messages.push({
      id: "failed-turn",
      role: "brain",
      content: "Tool action failed while compiling.",
      createdAt: now,
      runtime: "adaptive-core",
      status: "error"
    });
    current.traces.push({
      id: "trace-with-loss",
      createdAt: now,
      input: "fixture",
      seed: 7,
      runtime: "adaptive-core",
      activatedConcepts: [],
      recalledIdeas: [],
      driveScores: { novelty: 0, coherence: 0, curiosity: 0 },
      branches: 1,
      selectedBranch: 0,
      steps: [
        {
          stage: "evaluation",
          detail: "Held-out benchmark failed with prediction error regression."
        }
      ],
      note: "measured fixture"
    });
    current.journal?.push({
      id: "resource-event",
      createdAt: now,
      kind: "system",
      summary: "Resource timeout under memory pressure."
    });
    await repository.save(current);

    const tools: EvolutionToolExecutor = {
      execute: vi.fn(async (invocation) =>
        execution(
          invocation,
          "complete",
          proposalOutput(join(temporaryRoot, "detected-worktree"), "omni-evolution/detected")
        )
      ),
      cancel: vi.fn(() => 0)
    };
    const controller = new EvolutionController(repository, tools);
    const limitations = await controller.detectLimitations(brain.id);
    expect(new Set(limitations.map((entry) => entry.kind))).toEqual(
      new Set([
        "failed-action",
        "prediction-error",
        "regression",
        "resource-pressure",
        "held-out-evaluation"
      ])
    );

    const run = await controller.start({
      brainId: brain.id,
      objective: "Reduce observed failures"
    });
    expect(run.limitations.length).toBeGreaterThan(0);
    expect(run.hypothesis).toMatch(/measured limitation/i);
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;
    expect(candidate).toMatchObject({
      candidateKind: "source",
      evaluatorSha256: EVALUATOR_SHA256,
      proposalParentCommit: PARENT_COMMIT
    });
    expect(candidate.limitations.map((entry) => entry.kind)).toEqual(
      expect.arrayContaining(["failed-action", "held-out-evaluation", "resource-pressure"])
    );
  });

  it("rejects evaluator identity drift and never attempts promotion", async () => {
    const execute = vi.fn(async (invocation: ToolInvocation): Promise<ToolExecutionResult> => {
      if (invocation.action === "propose") {
        return execution(
          invocation,
          "complete",
          proposalOutput(join(temporaryRoot, "drift-worktree"), "omni-evolution/drift")
        );
      }
      if (invocation.action === "test") {
        return execution(invocation, "complete", {
          ...passingEvaluationOutput(["typecheck", "unit", "build"]),
          evaluatorSha256: "f".repeat(64)
        });
      }
      return execution(invocation, "failed", undefined, undefined, "Promotion must not run.");
    });
    const controller = new EvolutionController(repository, {
      execute,
      cancel: vi.fn(() => 0)
    });
    const run = await controller.start({
      brainId: brain.id,
      objective: "Attempt evaluator drift"
    });
    const candidate = (await controller.listCandidates(brain.id, run.id))[0]!;
    const rejected = await controller.approve({
      brainId: brain.id,
      candidateId: candidate.id,
      tests: ["unit"]
    });
    expect(rejected).toMatchObject({
      state: "rejected",
      error: expect.stringMatching(/evaluator hash changed/i)
    });
    expect(rejected.evaluations[0]).toMatchObject({
      passed: false,
      boundaryPassed: true,
      rejectionReason: expect.stringMatching(/evaluator hash changed/i)
    });
    expect(execute.mock.calls.some(([invocation]) => invocation.action === "promote")).toBe(false);
  });

  it("persists permission and setup failures instead of claiming an experiment started", async () => {
    const controller = new EvolutionController(repository, {
      execute: vi.fn(async (invocation) =>
        execution(
          invocation,
          "failed",
          undefined,
          undefined,
          "This tool is disabled for the current brain."
        )
      ),
      cancel: vi.fn(() => 0)
    });
    const run = await controller.start({
      brainId: brain.id,
      objective: "Attempt without permission"
    });
    expect(run).toMatchObject({
      state: "failed",
      error: "This tool is disabled for the current brain."
    });
    expect((await controller.listCandidates(brain.id, run.id))[0]).toMatchObject({
      state: "failed",
      error: "This tool is disabled for the current brain."
    });
  });
});
