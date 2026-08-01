import { createHash, randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  BrainService,
  RuntimeJobManager
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import { ToolExecutor } from "../src/main/toolExecutor";
import {
  inspectRuntimeArtifacts,
  sha256File,
  type SourceRuntimeLifecycle,
  type SourceRuntimeManifest,
  type SourceRuntimeStageRequest
} from "../src/main/sourceRuntimeContract";
import {
  DEFAULT_CONFIG,
  type BrainDocument,
  type ChatResult,
  type ToolExecutionResult,
  type ToolPermissionLevel
} from "../src/shared/types";

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function run(
  executable: string,
  args: string[],
  cwd: string
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(
      executable,
      args,
      { cwd, encoding: "utf8", windowsHide: true },
      (error, stdout, stderr) => {
        if (error) {
          reject(
            new Error(
              `${executable} ${args.join(" ")} failed: ${stderr || error.message}`
            )
          );
          return;
        }
        resolve({ stdout, stderr });
      }
    );
  });
}

function completeOutput<T>(result: ToolExecutionResult): T {
  expect(result.state, result.error).toBe("complete");
  return result.output as T;
}

describe("ToolExecutor complete workflows", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;
  let brain: BrainDocument;
  let originalOmniEnvironment: Record<string, string>;

  beforeEach(async () => {
    originalOmniEnvironment = Object.fromEntries(
      Object.entries(process.env).filter(
        (entry): entry is [string, string] =>
          entry[0].startsWith("OMNI_") && typeof entry[1] === "string"
      )
    );
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-tool-workflow-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
    brain = await repository.create({ ...DEFAULT_CONFIG, name: "Workflow parent" });
  });

  afterEach(async () => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("OMNI_")) delete process.env[key];
    }
    Object.assign(process.env, originalOmniEnvironment);
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  async function setPermission(
    toolId: string,
    level: ToolPermissionLevel
  ): Promise<void> {
    const current = await repository.get(brain.id);
    const permission = current.toolPermissions?.find((entry) => entry.toolId === toolId);
    if (!permission) throw new Error(`Missing fixture permission for ${toolId}.`);
    permission.level = level;
    permission.updatedAt = new Date().toISOString();
    await repository.save(current);
  }

  function executorFor(
    chat?: BrainService["chat"],
    sourceRuntime?: SourceRuntimeLifecycle
  ): ToolExecutor {
    const service = {
      repository,
      listToolPermissions: async (brainId: string) => {
        const current = await repository.get(brainId);
        return [...(current.toolPermissions ?? [])];
      },
      chat
    } as unknown as BrainService;
    return new ToolExecutor(
      service,
      {
        generate: vi.fn(() => {
          throw new Error("Unexpected modality job.");
        })
      } as unknown as RuntimeJobManager,
      sourceRuntime
    );
  }

  it("validates, detects tampering, and promotes an isolated source evolution", async () => {
    const sourceRepository = join(temporaryRoot, "authorized-source");
    const evolutionRoot = join(temporaryRoot, "evolution-candidates");
    const runningBinary = join(sourceRepository, "release", "running-binary.exe");
    await Promise.all([
      mkdir(join(sourceRepository, "release"), { recursive: true }),
      mkdir(join(sourceRepository, "tests"), { recursive: true }),
      mkdir(join(sourceRepository, "node_modules"), { recursive: true })
    ]);
    await Promise.all([
      writeFile(
        join(sourceRepository, "package.json"),
        JSON.stringify({
          name: "omni-evolution-fixture",
          private: true,
          scripts: {
            test:
              "node -e \"const fs=require('node:fs');" +
              "const linked=fs.lstatSync('node_modules').isSymbolicLink();" +
              "const expected=process.env.OMNI_EVOLUTION_CANDIDATE_WORKTREE==='1';" +
              "if(linked!==expected)process.exit(3);" +
              "process.stdout.write('allowlisted unit test passed')\""
          }
        })
      ),
      writeFile(join(sourceRepository, "source.txt"), "original source\n"),
      writeFile(
        join(sourceRepository, "tests", "immutable-evaluator.test.js"),
        "export const immutableEvaluator = true;\n"
      ),
      writeFile(runningBinary, "running image remains unchanged\n")
    ]);
    await run("git", ["init"], sourceRepository);
    await run("git", ["config", "core.autocrlf", "false"], sourceRepository);
    await run("git", ["config", "core.eol", "lf"], sourceRepository);
    await run("git", ["config", "user.name", "Workflow Test"], sourceRepository);
    await run(
      "git",
      ["config", "user.email", "workflow-test@local.invalid"],
      sourceRepository
    );
    await run("git", ["add", "-A"], sourceRepository);
    await run("git", ["commit", "-m", "Initial fixture"], sourceRepository);
    const runningBinaryHash = sha256(await readFile(runningBinary));

    process.env.OMNI_SOURCE_REPOSITORY = sourceRepository;
    process.env.OMNI_EVOLUTION_ROOT = evolutionRoot;
    await setPermission("source.self-modify", "ask");
    const executor = executorFor();

    const proposal = completeOutput<{
      worktree: string;
      branch: string;
      taskFile: string;
      parentCommit: string;
      evaluatorSha256: string;
      benchmarkDomains: string[];
      sourceEditLineage: Array<{
        path: string;
        expectedSha256: string | null;
        resultSha256: string;
        bytes: number;
      }>;
      authoredChangedPaths: string[];
      authoredDiffSha256: string;
      authoredBytes: number;
      checks: Array<{ name: string; passed: boolean }>;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "propose",
        arguments: {
          objective: "Add a candidate capability without replacing the app.",
          sourceEdits: [
            {
              path: "source.txt",
              content: "candidate source\n",
              expectedSha256: sha256("original source\n")
            },
            {
              path: "new-capability.txt",
              content: "untracked version one\n",
              expectedSha256: null
            }
          ]
        }
      })
    );
    expect(proposal.worktree).toMatch(
      new RegExp(`^${evolutionRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`)
    );
    expect(proposal.branch).toMatch(/^omni-evolution\//);
    expect(proposal.parentCommit).toMatch(/^[a-f0-9]{40,64}$/);
    expect(proposal.evaluatorSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(proposal.benchmarkDomains).toEqual(
      expect.arrayContaining(["capability", "retention", "evolution-integrity"])
    );
    expect(proposal.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "isolated-worktree", passed: true }),
        expect.objectContaining({ name: "typed-source-authoring", passed: true })
      ])
    );
    expect(proposal.sourceEditLineage).toEqual([
      {
        path: "new-capability.txt",
        expectedSha256: null,
        resultSha256: sha256("untracked version one\n"),
        bytes: Buffer.byteLength("untracked version one\n")
      },
      {
        path: "source.txt",
        expectedSha256: sha256("original source\n"),
        resultSha256: sha256("candidate source\n"),
        bytes: Buffer.byteLength("candidate source\n")
      }
    ]);
    expect(proposal.authoredChangedPaths).toEqual([
      "new-capability.txt",
      "source.txt"
    ]);
    expect(proposal.authoredDiffSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(proposal.authoredBytes).toBe(
      Buffer.byteLength("untracked version one\n") +
        Buffer.byteLength("candidate source\n")
    );
    await expect(readFile(proposal.taskFile, "utf8")).resolves.toContain(
      "Add a candidate capability"
    );
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");

    const firstDiff = completeOutput<{
      diff: string;
      untracked: Array<{ path: string; sha256: string; bytes: number }>;
      sha256: string;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "diff",
        arguments: { worktree: proposal.worktree }
      })
    );
    expect(firstDiff.diff).toContain("candidate source");
    expect(firstDiff.untracked).toEqual([
      {
        path: "new-capability.txt",
        sha256: sha256("untracked version one\n"),
        bytes: Buffer.byteLength("untracked version one\n")
      }
    ]);

    await writeFile(
      join(proposal.worktree, "new-capability.txt"),
      "untracked version two\n"
    );
    const secondDiff = completeOutput<{
      diff: string;
      untracked: Array<{ path: string; sha256: string; bytes: number }>;
      sha256: string;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "diff",
        arguments: { worktree: proposal.worktree }
      })
    );
    expect(secondDiff.diff).toBe(firstDiff.diff);
    expect(secondDiff.sha256).not.toBe(firstDiff.sha256);
    expect(secondDiff.untracked[0]?.sha256).toBe(sha256("untracked version two\n"));

    await writeFile(
      join(proposal.worktree, "tests", "immutable-evaluator.test.js"),
      "export const immutableEvaluator = false;\n"
    );
    const evaluatorTamper = await executor.execute({
      brainId: brain.id,
      toolId: "source.self-modify",
      action: "test",
      arguments: {
        worktree: proposal.worktree,
        tests: ["unit"],
        expectedEvaluatorSha256: proposal.evaluatorSha256,
        timeoutMs: 30_000
      }
    });
    expect(evaluatorTamper).toMatchObject({
      state: "failed",
      error: expect.stringMatching(/immutable evaluator files/i)
    });
    await writeFile(
      join(proposal.worktree, "tests", "immutable-evaluator.test.js"),
      "export const immutableEvaluator = true;\n"
    );

    const validation = completeOutput<{
      passed: boolean;
      boundaryPassed: boolean;
      diffSha256: string;
      evaluatorSha256: string;
      parentCommit: string;
      baselineChecks: Array<{ name: string; passed: boolean }>;
      resources: {
        baselineDurationMs: number;
        candidateDurationMs: number;
        changedBytes: number;
      };
      checks: Array<{
        name: string;
        passed: boolean;
        stdout: string;
      }>;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "test",
        arguments: {
          worktree: proposal.worktree,
          tests: ["unit"],
          expectedEvaluatorSha256: proposal.evaluatorSha256,
          timeoutMs: 30_000
        }
      })
    );
    expect(validation.passed).toBe(true);
    expect(validation.boundaryPassed).toBe(true);
    expect(validation.diffSha256).toBe(secondDiff.sha256);
    expect(validation.evaluatorSha256).toBe(proposal.evaluatorSha256);
    expect(validation.parentCommit).toBe(proposal.parentCommit);
    expect(validation.baselineChecks).toEqual([
      expect.objectContaining({ name: "unit", passed: true })
    ]);
    expect(validation.resources).toMatchObject({
      baselineDurationMs: expect.any(Number),
      candidateDurationMs: expect.any(Number),
      changedBytes: expect.any(Number)
    });
    expect(validation.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "unit",
          passed: true,
          stdout: expect.stringContaining("allowlisted unit test passed")
        }),
        expect.objectContaining({ name: "diff-check", passed: true })
      ])
    );
    await expect(
      lstat(join(proposal.worktree, "node_modules"))
    ).rejects.toMatchObject({ code: "ENOENT" });

    await writeFile(
      join(proposal.worktree, "new-capability.txt"),
      "changed after validation\n"
    );
    const staleInvocation = {
      brainId: brain.id,
      toolId: "source.self-modify",
      action: "promote",
      arguments: {
        worktree: proposal.worktree,
        expectedDiffSha256: validation.diffSha256,
        expectedEvaluatorSha256: proposal.evaluatorSha256
      }
    };
    const staleChallenge = await executor.execute(staleInvocation);
    expect(staleChallenge.state).toBe("approval-required");
    const rejectedPromotion = await executor.execute({
      ...staleInvocation,
      approvalToken: staleChallenge.approvalToken
    });
    expect(rejectedPromotion.state).toBe("failed");
    expect(rejectedPromotion.error).toMatch(/changed after validation/i);
    await expect(readFile(join(sourceRepository, "source.txt"), "utf8")).resolves.toBe(
      "original source\n"
    );

    const revalidation = completeOutput<{
      passed: boolean;
      diffSha256: string;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "test",
        arguments: {
          worktree: proposal.worktree,
          tests: ["unit"],
          expectedEvaluatorSha256: proposal.evaluatorSha256,
          timeoutMs: 30_000
        }
      })
    );
    expect(revalidation.passed).toBe(true);
    expect(revalidation.diffSha256).not.toBe(validation.diffSha256);
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");

    const promotionInvocation = {
      brainId: brain.id,
      toolId: "source.self-modify",
      action: "promote",
      arguments: {
        worktree: proposal.worktree,
        expectedDiffSha256: revalidation.diffSha256,
        expectedEvaluatorSha256: proposal.evaluatorSha256
      }
    };
    const promotionChallenge = await executor.execute(promotionInvocation);
    expect(promotionChallenge).toMatchObject({
      state: "approval-required",
      approvalToken: expect.any(String)
    });
    const promotion = completeOutput<{
      promoted: boolean;
      commit: string;
      parentCommit: string;
      diffSha256: string;
      evaluatorSha256: string;
      note: string;
    }>(
      await executor.execute({
        ...promotionInvocation,
        approvalToken: promotionChallenge.approvalToken
      })
    );
    expect(promotion).toMatchObject({
      promoted: true,
      diffSha256: revalidation.diffSha256,
      evaluatorSha256: proposal.evaluatorSha256,
      note: expect.stringMatching(/did not activate or replace the running executable/i)
    });
    expect(promotion.commit).toMatch(/^[a-f0-9]{40,64}$/);
    expect(promotion.parentCommit).toMatch(/^[a-f0-9]{40,64}$/);
    await expect(readFile(join(sourceRepository, "source.txt"), "utf8")).resolves.toBe(
      "candidate source\n"
    );
    await expect(
      readFile(join(sourceRepository, "new-capability.txt"), "utf8")
    ).resolves.toBe("changed after validation\n");
    expect(sha256(await readFile(runningBinary))).toBe(runningBinaryHash);
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");

    const rollbackInvocation = {
      brainId: brain.id,
      toolId: "source.self-modify",
      action: "rollback",
      arguments: {
        expectedCommit: promotion.commit,
        parentCommit: promotion.parentCommit
      }
    };
    const rollbackChallenge = await executor.execute(rollbackInvocation);
    expect(rollbackChallenge).toMatchObject({
      state: "approval-required",
      approvalToken: expect.any(String)
    });
    const rollback = completeOutput<{
      rolledBack: boolean;
      revertedCommit: string;
      rollbackCommit: string;
    }>(
      await executor.execute({
        ...rollbackInvocation,
        approvalToken: rollbackChallenge.approvalToken
      })
    );
    expect(rollback).toMatchObject({
      rolledBack: true,
      revertedCommit: promotion.commit
    });
    expect(rollback.rollbackCommit).toMatch(/^[a-f0-9]{40,64}$/);
    await expect(readFile(join(sourceRepository, "source.txt"), "utf8")).resolves.toBe(
      "original source\n"
    );
    await expect(
      readFile(join(sourceRepository, "new-capability.txt"), "utf8")
    ).rejects.toMatchObject({ code: "ENOENT" });
    expect(sha256(await readFile(runningBinary))).toBe(runningBinaryHash);
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");
  }, 60_000);

  it("rejects unsafe or stale typed edits and never promotes an empty source candidate", async () => {
    const sourceRepository = join(temporaryRoot, "bounded-source");
    const evolutionRoot = join(temporaryRoot, "bounded-candidates");
    await Promise.all([
      mkdir(join(sourceRepository, "src"), { recursive: true }),
      mkdir(join(sourceRepository, "tests"), { recursive: true }),
      mkdir(join(sourceRepository, "node_modules"), { recursive: true })
    ]);
    await Promise.all([
      writeFile(
        join(sourceRepository, "package.json"),
        JSON.stringify({
          name: "omni-bounded-evolution-fixture",
          private: true,
          scripts: {
            test:
              "node -e \"const fs=require('node:fs');" +
              "const linked=fs.lstatSync('node_modules').isSymbolicLink();" +
              "const expected=process.env.OMNI_EVOLUTION_CANDIDATE_WORKTREE==='1';" +
              "if(linked!==expected)process.exit(3);" +
              "process.stdout.write('bounded evaluator passed')\""
          }
        })
      ),
      writeFile(
        join(sourceRepository, "src", "measured.ts"),
        "export const measured = 1;\n"
      ),
      writeFile(
        join(sourceRepository, "tests", "immutable-evaluator.test.js"),
        "export const immutableEvaluator = true;\n"
      )
    ]);
    await run("git", ["init"], sourceRepository);
    await run("git", ["config", "core.autocrlf", "false"], sourceRepository);
    await run("git", ["config", "core.eol", "lf"], sourceRepository);
    await run("git", ["config", "user.name", "Bounded Workflow Test"], sourceRepository);
    await run(
      "git",
      ["config", "user.email", "bounded-workflow@local.invalid"],
      sourceRepository
    );
    await run("git", ["add", "-A"], sourceRepository);
    await run("git", ["commit", "-m", "Initial bounded fixture"], sourceRepository);

    process.env.OMNI_SOURCE_REPOSITORY = sourceRepository;
    process.env.OMNI_EVOLUTION_ROOT = evolutionRoot;
    await setPermission("source.self-modify", "ask");
    const executor = executorFor();
    const propose = (sourceEdits: unknown) =>
      executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "propose",
        arguments: {
          objective: "Author one bounded measured maintenance candidate.",
          sourceEdits
        }
      });

    await expect(
      propose([
        {
          path: "../outside.ts",
          content: "export const escaped = true;\n",
          expectedSha256: null
        }
      ])
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/traverse|relative|portable/i)
    });
    await expect(
      propose([
        {
          path: "package.json",
          content: "{\"scripts\":{\"postinstall\":\"arbitrary setup\"}}\n",
          expectedSha256: sha256(
            JSON.stringify({
              name: "omni-bounded-evolution-fixture",
              private: true,
              scripts: {
                test:
                  "node -e \"const fs=require('node:fs');" +
                  "const linked=fs.lstatSync('node_modules').isSymbolicLink();" +
                  "const expected=process.env.OMNI_EVOLUTION_CANDIDATE_WORKTREE==='1';" +
                  "if(linked!==expected)process.exit(3);" +
                  "process.stdout.write('bounded evaluator passed')\""
              }
            })
          )
        }
      ])
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/package installation|execution manifests/i)
    });
    await expect(
      propose([
        {
          path: "release/running-binary.txt",
          content: "running image replacement\n",
          expectedSha256: null
        }
      ])
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/build, release, script, or control directories/i)
    });
    await expect(
      propose([
        {
          path: "tests/immutable-evaluator.test.js",
          content: "export const immutableEvaluator = false;\n",
          expectedSha256: sha256("export const immutableEvaluator = true;\n")
        }
      ])
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/immutable evaluator/i)
    });
    await expect(
      propose([
        {
          path: "src/measured.ts",
          content: "export const measured = 2;\n",
          expectedSha256: "0".repeat(64)
        }
      ])
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/checksum does not match/i)
    });
    await expect(
      propose([
        {
          path: "src/too-large.ts",
          content: "x".repeat(8 * 1024 * 1024 + 1),
          expectedSha256: null
        }
      ])
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/exceed.*byte proposal limit/i)
    });
    await expect(
      readFile(join(temporaryRoot, "outside.ts"), "utf8")
    ).rejects.toMatchObject({ code: "ENOENT" });
    await expect(
      readFile(join(sourceRepository, "src", "measured.ts"), "utf8")
    ).resolves.toBe("export const measured = 1;\n");
    await expect(
      readFile(
        join(sourceRepository, "tests", "immutable-evaluator.test.js"),
        "utf8"
      )
    ).resolves.toBe("export const immutableEvaluator = true;\n");

    const empty = completeOutput<{
      worktree: string;
      evaluatorSha256: string;
      authoredChangedPaths: string[];
      checks: Array<{ name: string; passed: boolean }>;
    }>(await propose([]));
    expect(empty.authoredChangedPaths).toEqual([]);
    expect(empty.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "typed-source-authoring", passed: false })
      ])
    );
    const evaluated = completeOutput<{
      passed: boolean;
      diffSha256: string;
      checks: Array<{ name: string; passed: boolean }>;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "test",
        arguments: {
          worktree: empty.worktree,
          tests: ["unit"],
          expectedEvaluatorSha256: empty.evaluatorSha256,
          timeoutMs: 30_000
        }
      })
    );
    expect(evaluated.passed).toBe(false);
    expect(evaluated.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "candidate-change", passed: false })
      ])
    );
    const promotionInvocation = {
      brainId: brain.id,
      toolId: "source.self-modify",
      action: "promote",
      arguments: {
        worktree: empty.worktree,
        expectedDiffSha256: evaluated.diffSha256,
        expectedEvaluatorSha256: empty.evaluatorSha256
      }
    };
    const challenge = await executor.execute(promotionInvocation);
    expect(challenge.state).toBe("approval-required");
    await expect(
      executor.execute({
        ...promotionInvocation,
        approvalToken: challenge.approvalToken
      })
    ).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/empty source candidate cannot be promoted/i)
    });
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");
  }, 60_000);

  it("stages and schedules only Full-Authority runtimes and fails closed before merge", async () => {
    const sourceRepository = join(temporaryRoot, "activation-source");
    const evolutionRoot = join(temporaryRoot, "activation-candidates");
    const runtimeRoot = join(temporaryRoot, "activation-runtimes");
    await Promise.all([
      mkdir(join(sourceRepository, "src"), { recursive: true }),
      mkdir(join(sourceRepository, "tests"), { recursive: true }),
      mkdir(join(sourceRepository, "node_modules"), { recursive: true }),
      mkdir(runtimeRoot, { recursive: true })
    ]);
    const unitScript =
      "node -e \"const fs=require('node:fs');" +
      "const linked=fs.lstatSync('node_modules').isSymbolicLink();" +
      "const expected=process.env.OMNI_EVOLUTION_CANDIDATE_WORKTREE==='1';" +
      "if(linked!==expected)process.exit(3);process.stdout.write('activation checks passed')\"";
    await Promise.all([
      writeFile(
        join(sourceRepository, "package.json"),
        JSON.stringify({
          name: "omni-activation-fixture",
          private: true,
          scripts: {
            pretest: "node -e \"process.exit(91)\"",
            test: unitScript
          }
        })
      ),
      writeFile(
        join(sourceRepository, "src", "runtime.txt"),
        "runtime version one\n"
      ),
      writeFile(
        join(sourceRepository, "tests", "immutable-evaluator.test.js"),
        "export const immutableEvaluator = true;\n"
      )
    ]);
    await run("git", ["init"], sourceRepository);
    await run("git", ["config", "core.autocrlf", "false"], sourceRepository);
    await run("git", ["config", "core.eol", "lf"], sourceRepository);
    await run("git", ["config", "user.name", "Activation Workflow Test"], sourceRepository);
    await run(
      "git",
      ["config", "user.email", "activation-workflow@local.invalid"],
      sourceRepository
    );
    await run("git", ["add", "-A"], sourceRepository);
    await run("git", ["commit", "-m", "Initial activation fixture"], sourceRepository);
    process.env.OMNI_SOURCE_REPOSITORY = sourceRepository;
    process.env.OMNI_EVOLUTION_ROOT = evolutionRoot;

    let failStaging = false;
    let tamperStaging = false;
    const stage = vi.fn(
      async (request: SourceRuntimeStageRequest) => {
        if (failStaging) throw new Error("native staging fixture failed");
        const linkedModules = join(request.worktree, "node_modules");
        expect((await lstat(linkedModules)).isSymbolicLink()).toBe(true);
        expect(await realpath(linkedModules)).toBe(
          await realpath(join(sourceRepository, "node_modules"))
        );
        const slotId = `source-fixture-${stage.mock.calls.length}`;
        const rootPath = join(runtimeRoot, slotId);
        const executablePath = join(rootPath, "desktop", "omni-test-runtime");
        await mkdir(dirname(executablePath), { recursive: true });
        await writeFile(executablePath, `runtime for ${request.diffSha256}\n`);
        const currentExecutableSha256 = (
          await sha256File(request.currentExecutablePath)
        ).sha256;
        const artifacts = await inspectRuntimeArtifacts(rootPath);
        const executableSha256 = (await sha256File(executablePath)).sha256;
        const manifest: SourceRuntimeManifest = {
          schemaVersion: 1,
          slotId,
          createdAt: new Date().toISOString(),
          state: "staged",
          platform: request.platform,
          architecture: request.architecture,
          lineage: {
            parentCommit: request.parentCommit,
            diffSha256: request.diffSha256,
            evaluatorSha256: request.evaluatorSha256,
            brainSnapshotId: request.brainSnapshotId
          },
          executableRelativePath: "desktop/omni-test-runtime",
          executableSha256,
          currentExecutableSha256,
          artifactSha256: artifacts.artifactSha256,
          artifacts: artifacts.artifacts,
          engineStrategy: "reused-current-worker"
        };
        const manifestPath = join(rootPath, "runtime-manifest.json");
        await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
        const result = {
          slotId,
          rootPath,
          manifestPath,
          manifestSha256: (await sha256File(manifestPath)).sha256,
          executablePath,
          manifest
        };
        if (tamperStaging) {
          await writeFile(executablePath, "tampered after manifest\n");
        }
        return result;
      }
    );
    const scheduleActivation = vi.fn(
      async (
        request: Parameters<SourceRuntimeLifecycle["scheduleActivation"]>[0]
      ) => {
        const manifest = await readFile(request.stage.manifestPath, "utf8").then(
          (value) => JSON.parse(value) as SourceRuntimeManifest
        );
        const updated: SourceRuntimeManifest = {
          ...manifest,
          state: "deferred",
          promotionCommit: request.promotionCommit,
          candidateCommit: request.candidateCommit,
          activation: {
            state: "deferred",
            scheduledAt: new Date().toISOString(),
            delayMs: request.delayMs,
            reason: "test host does not relaunch"
          }
        };
        await writeFile(
          request.stage.manifestPath,
          JSON.stringify(updated, null, 2)
        );
        return {
          state: "deferred" as const,
          slotId: request.stage.slotId,
          executablePath: request.stage.executablePath,
          manifestPath: request.stage.manifestPath,
          manifestSha256: (await sha256File(request.stage.manifestPath)).sha256,
          promotionCommit: request.promotionCommit,
          delayMs: request.delayMs,
          reason: "test host does not relaunch"
        };
      }
    );
    const abandonStage = vi.fn(async () => undefined);
    const lifecycle: SourceRuntimeLifecycle = {
      stage,
      scheduleActivation,
      abandonStage
    };
    const executor = executorFor(undefined, lifecycle);
    let currentContents = "runtime version one\n";
    const promote = async (
      level: "ask" | "auto" | "full",
      nextContents: string
    ): Promise<ToolExecutionResult> => {
      await setPermission("source.self-modify", level);
      const proposal = completeOutput<{
        worktree: string;
        evaluatorSha256: string;
      }>(
        await executor.execute({
          brainId: brain.id,
          toolId: "source.self-modify",
          action: "propose",
          arguments: {
            objective: `Promote ${level} activation fixture`,
            sourceEdits: [
              {
                path: "src/runtime.txt",
                content: nextContents,
                expectedSha256: sha256(currentContents)
              }
            ]
          }
        })
      );
      const validation = completeOutput<{ diffSha256: string }>(
        await executor.execute({
          brainId: brain.id,
          toolId: "source.self-modify",
          action: "test",
          arguments: {
            worktree: proposal.worktree,
            tests: ["unit"],
            expectedEvaluatorSha256: proposal.evaluatorSha256,
            timeoutMs: 30_000
          }
        })
      );
      const invocation = {
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "promote",
        arguments: {
          worktree: proposal.worktree,
          expectedDiffSha256: validation.diffSha256,
          expectedEvaluatorSha256: proposal.evaluatorSha256
        }
      };
      let result = await executor.execute(invocation);
      if (result.state === "approval-required") {
        result = await executor.execute({
          ...invocation,
          approvalToken: result.approvalToken
        });
      }
      if (result.state === "complete") currentContents = nextContents;
      return result;
    };

    const currentExecutableBefore = await sha256File(process.execPath);
    expect((await promote("ask", "runtime version ask\n")).state).toBe("complete");
    expect((await promote("auto", "runtime version auto\n")).state).toBe("complete");
    expect(stage).not.toHaveBeenCalled();
    expect(scheduleActivation).not.toHaveBeenCalled();

    const full = await promote("full", "runtime version full\n");
    expect(full.state, full.error).toBe("complete");
    expect(full.output).toMatchObject({
      promoted: true,
      brainSnapshotId: expect.any(String),
      runtimeActivation: {
        state: "deferred",
        slotId: expect.stringMatching(/^source-fixture-/),
        delayMs: 5_000,
        reason: "test host does not relaunch"
      }
    });
    expect(stage).toHaveBeenCalledTimes(1);
    expect(scheduleActivation).toHaveBeenCalledTimes(1);
    const fullWorktree = stage.mock.calls[0]![0].worktree;
    await expect(lstat(join(fullWorktree, "node_modules"))).rejects.toMatchObject({
      code: "ENOENT"
    });
    expect(await sha256File(process.execPath)).toEqual(currentExecutableBefore);
    expect((await repository.listSnapshots(brain.id)).length).toBeGreaterThan(0);

    const headBeforeFailure = (
      await run("git", ["rev-parse", "HEAD"], sourceRepository)
    ).stdout.trim();
    failStaging = true;
    const failedStage = await promote("full", "runtime version failed-stage\n");
    expect(failedStage).toMatchObject({
      state: "failed",
      error: expect.stringMatching(/native staging fixture failed/i)
    });
    expect(
      (await run("git", ["rev-parse", "HEAD"], sourceRepository)).stdout.trim()
    ).toBe(headBeforeFailure);
    await expect(
      readFile(join(sourceRepository, "src", "runtime.txt"), "utf8")
    ).resolves.toBe("runtime version full\n");

    failStaging = false;
    tamperStaging = true;
    const headBeforeTamper = (
      await run("git", ["rev-parse", "HEAD"], sourceRepository)
    ).stdout.trim();
    const tampered = await promote("full", "runtime version tampered-stage\n");
    expect(tampered).toMatchObject({
      state: "failed",
      error: expect.stringMatching(/artifact tree failed hash verification/i)
    });
    expect(
      (await run("git", ["rev-parse", "HEAD"], sourceRepository)).stdout.trim()
    ).toBe(headBeforeTamper);
    expect(abandonStage).toHaveBeenCalledTimes(1);
    expect(scheduleActivation).toHaveBeenCalledTimes(1);
    expect(await sha256File(process.execPath)).toEqual(currentExecutableBefore);
  }, 120_000);

  it("runs isolated subagent forks and leaves the parent's neural state unchanged", async () => {
    await setPermission("agent.fork", "full");
    const parentBefore = await repository.get(brain.id);
    const objective = "Explore three independent hypotheses";
    const chat = vi.fn(async (forkId: string, input: string): Promise<ChatResult> => {
      const fork = await repository.get(forkId);
      const marker = `branch-${forkId}`;
      const now = new Date().toISOString();
      fork.concepts[marker] = {
        id: marker,
        label: marker,
        activation: 1,
        importance: 0.5,
        uncertainty: 0.2,
        exposures: 1,
        createdAt: now,
        lastActivatedAt: now,
        aliases: []
      };
      const humanMessage = {
        id: randomUUID(),
        role: "human" as const,
        content: input,
        createdAt: now,
        runtime: "adaptive-core" as const
      };
      const brainMessage = {
        id: randomUUID(),
        role: "brain" as const,
        content: `candidate from ${forkId}`,
        createdAt: now,
        runtime: "adaptive-core" as const,
        status: "complete" as const
      };
      const trace = {
        id: randomUUID(),
        createdAt: now,
        input,
        seed: 1,
        runtime: "adaptive-core" as const,
        activatedConcepts: [],
        recalledIdeas: [],
        driveScores: { novelty: 1, coherence: 1, curiosity: 1 },
        branches: 1,
        selectedBranch: 0,
        steps: [],
        note: "isolated test trace"
      };
      fork.messages.push(humanMessage, brainMessage);
      fork.traces.push(trace);
      const saved = await repository.save(fork);
      return { brain: saved, humanMessage, brainMessage, trace };
    });
    const executor = executorFor(chat);

    const result = completeOutput<{
      state: string;
      objective: string;
      forkIds: string[];
      results: Array<{
        forkId: string;
        response: string;
        traceId: string;
        concepts: number;
      }>;
      mergePolicy: string;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "agent.fork",
        action: "start",
        arguments: { objective, workers: 3 }
      })
    );

    expect(result).toMatchObject({
      state: "complete",
      objective,
      mergePolicy: "ideas-evidence-replay-only"
    });
    expect(result.forkIds).toHaveLength(3);
    expect(new Set(result.forkIds).size).toBe(3);
    expect(result.results).toHaveLength(3);
    expect(chat).toHaveBeenCalledTimes(3);
    for (const forkId of result.forkIds) {
      expect(chat).toHaveBeenCalledWith(
        forkId,
        objective,
        expect.any(AbortSignal),
        undefined,
        undefined,
        96
      );
      const fork = await repository.get(forkId);
      expect(fork.lineage).toMatchObject({
        parentId: brain.id,
        rootId: brain.lineage.rootId,
        generation: brain.lineage.generation + 1
      });
      expect(fork.concepts[`branch-${forkId}`]).toBeDefined();
      for (const otherForkId of result.forkIds.filter((id) => id !== forkId)) {
        expect(fork.concepts[`branch-${otherForkId}`]).toBeUndefined();
      }
    }

    const parentAfter = await repository.get(brain.id);
    expect(parentAfter.concepts).toEqual(parentBefore.concepts);
    expect(parentAfter.synapses).toEqual(parentBefore.synapses);
    expect(parentAfter.ideas).toEqual(parentBefore.ideas);
    expect(parentAfter.messages).toEqual(parentBefore.messages);
    expect(parentAfter.traces.slice(0, -1)).toEqual(parentBefore.traces);
    expect(parentAfter.traces.at(-1)).toMatchObject({
      input: "agent.fork.start",
      steps: expect.arrayContaining([
        expect.objectContaining({ stage: "tool-invocation" }),
        expect.objectContaining({ stage: "tool-result" })
      ])
    });
    expect(parentAfter.counters).toEqual(parentBefore.counters);
    expect(parentAfter.originChecksum).toBe(parentBefore.originChecksum);
    expect(parentAfter.journal?.at(-1)).toMatchObject({
      kind: "tool",
      summary: "agent.fork.start: complete."
    });
  }, 60_000);
});
