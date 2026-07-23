import { createHash, randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  BrainService,
  RuntimeJobManager
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import { ToolExecutor } from "../src/main/toolExecutor";
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

  function executorFor(chat?: BrainService["chat"]): ToolExecutor {
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
      } as unknown as RuntimeJobManager
    );
  }

  it("validates, detects tampering, and promotes an isolated source evolution", async () => {
    const sourceRepository = join(temporaryRoot, "authorized-source");
    const evolutionRoot = join(temporaryRoot, "evolution-candidates");
    const runningBinary = join(sourceRepository, "release", "running-binary.exe");
    await mkdir(join(sourceRepository, "release"), { recursive: true });
    await Promise.all([
      writeFile(
        join(sourceRepository, "package.json"),
        JSON.stringify({
          name: "omni-evolution-fixture",
          private: true,
          scripts: {
            test: "node -e \"process.stdout.write('allowlisted unit test passed')\""
          }
        })
      ),
      writeFile(join(sourceRepository, "source.txt"), "original source\n"),
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
    await setPermission("source.self-modify", "full");
    const executor = executorFor();

    const proposal = completeOutput<{
      worktree: string;
      branch: string;
      taskFile: string;
      checks: Array<{ name: string; passed: boolean }>;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "propose",
        arguments: { objective: "Add a candidate capability without replacing the app." }
      })
    );
    expect(proposal.worktree).toMatch(
      new RegExp(`^${evolutionRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`)
    );
    expect(proposal.branch).toMatch(/^omni-evolution\//);
    expect(proposal.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "isolated-worktree", passed: true })
      ])
    );
    await expect(readFile(proposal.taskFile, "utf8")).resolves.toContain(
      "Add a candidate capability"
    );
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");

    await Promise.all([
      writeFile(join(proposal.worktree, "source.txt"), "candidate source\n"),
      writeFile(join(proposal.worktree, "new-capability.txt"), "untracked version one\n")
    ]);
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

    const validation = completeOutput<{
      passed: boolean;
      diffSha256: string;
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
          timeoutMs: 30_000
        }
      })
    );
    expect(validation.passed).toBe(true);
    expect(validation.diffSha256).toBe(secondDiff.sha256);
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

    await writeFile(
      join(proposal.worktree, "new-capability.txt"),
      "changed after validation\n"
    );
    const rejectedPromotion = await executor.execute({
      brainId: brain.id,
      toolId: "source.self-modify",
      action: "promote",
      arguments: {
        worktree: proposal.worktree,
        expectedDiffSha256: validation.diffSha256
      }
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
          timeoutMs: 30_000
        }
      })
    );
    expect(revalidation.passed).toBe(true);
    expect(revalidation.diffSha256).not.toBe(validation.diffSha256);
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");

    const promotion = completeOutput<{
      promoted: boolean;
      commit: string;
      diffSha256: string;
      note: string;
    }>(
      await executor.execute({
        brainId: brain.id,
        toolId: "source.self-modify",
        action: "promote",
        arguments: {
          worktree: proposal.worktree,
          expectedDiffSha256: revalidation.diffSha256
        }
      })
    );
    expect(promotion).toMatchObject({
      promoted: true,
      diffSha256: revalidation.diffSha256,
      note: expect.stringMatching(/running binary was not overwritten or restarted/i)
    });
    expect(promotion.commit).toMatch(/^[a-f0-9]{40,64}$/);
    await expect(readFile(join(sourceRepository, "source.txt"), "utf8")).resolves.toBe(
      "candidate source\n"
    );
    await expect(
      readFile(join(sourceRepository, "new-capability.txt"), "utf8")
    ).resolves.toBe("changed after validation\n");
    expect(sha256(await readFile(runningBinary))).toBe(runningBinaryHash);
    expect((await run("git", ["status", "--porcelain"], sourceRepository)).stdout).toBe("");
  });

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
        expect.any(AbortSignal)
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
  });
});
