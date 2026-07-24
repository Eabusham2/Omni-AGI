import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import {
  type BrainService,
  RuntimeJobManager
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import {
  buildInertBrowserDocument,
  ToolExecutor
} from "../src/main/toolExecutor";
import {
  DEFAULT_CONFIG,
  type BrainDocument,
  type ToolPermissionLevel
} from "../src/shared/types";

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

describe("ToolExecutor release gates", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;
  let brain: BrainDocument;
  let service: BrainService;
  let executor: ToolExecutor;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-tool-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
    brain = await repository.create({ ...DEFAULT_CONFIG, name: "Tool test brain" });
    service = {
      repository,
      listToolPermissions: async (brainId: string) => {
        const current = await repository.get(brainId);
        return [...(current.toolPermissions ?? [])];
      }
    } as unknown as BrainService;
    executor = new ToolExecutor(service, {
      generate: vi.fn(() => {
        throw new Error("Unexpected modality job.");
      })
    } as unknown as RuntimeJobManager);
  });

  afterEach(async () => {
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

  it("blocks Off tools before dispatch and records a failed audit event", async () => {
    const target = join(temporaryRoot, "off-secret.txt");
    await writeFile(target, "must not be returned");
    await setPermission("windows.files", "off");

    const result = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: target }
    });

    expect(result.state).toBe("failed");
    expect(result.error).toMatch(/disabled/i);
    expect(result.output).toBeUndefined();

    const saved = await repository.get(brain.id);
    const audit = saved.journal?.at(-1);
    expect(audit).toMatchObject({
      kind: "tool",
      summary: "windows.files.read: failed."
    });
    expect(JSON.parse(audit?.detail ?? "{}")).toMatchObject({
      argumentKeys: ["path"],
      paths: { path: target },
      error: "This tool is disabled for the current brain."
    });
  });

  it("requires an Ask approval exactly once and scopes it to the invocation", async () => {
    const target = join(temporaryRoot, "ask.txt");
    const substitutedTarget = join(temporaryRoot, "not-approved.txt");
    await writeFile(target, "approved contents");
    await writeFile(substitutedTarget, "different contents");
    await setPermission("windows.files", "ask");
    const invocation = {
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: target }
    };

    const challenge = await executor.execute(invocation);
    expect(challenge.state).toBe("approval-required");
    expect(challenge.approvalToken).toMatch(/^[a-f0-9-]{36}$/i);

    const substituted = await executor.execute({
      ...invocation,
      arguments: { path: substitutedTarget },
      approvalToken: challenge.approvalToken
    });
    expect(substituted.state).toBe("approval-required");

    const freshChallenge = await executor.execute(invocation);
    const approved = await executor.execute({
      ...invocation,
      approvalToken: freshChallenge.approvalToken
    });
    expect(approved.state).toBe("complete");
    expect(approved.output).toMatchObject({
      content: "approved contents",
      sha256: sha256("approved contents")
    });

    const replay = await executor.execute({
      ...invocation,
      approvalToken: challenge.approvalToken
    });
    expect(replay.state).toBe("approval-required");
    expect(replay.approvalToken).not.toBe(challenge.approvalToken);

    const saved = await repository.get(brain.id);
    expect(
      saved.journal?.filter((entry) => entry.summary === "windows.files.read: complete.")
    ).toHaveLength(1);
  });

  it("lets Auto perform reads but challenges risky writes", async () => {
    const target = join(repository.brainDirectory(brain.id), "auto.txt");
    const outside = join(temporaryRoot, "outside-auto.txt");
    const outsideDirectory = join(temporaryRoot, "outside-directory");
    const outsideViaLink = join(outsideDirectory, "linked-secret.txt");
    const escapeLink = join(repository.brainDirectory(brain.id), "escape-link");
    await writeFile(target, "before");
    await writeFile(outside, "outside");
    await mkdir(outsideDirectory);
    await writeFile(outsideViaLink, "outside through link");
    await symlink(
      outsideDirectory,
      escapeLink,
      process.platform === "win32" ? "junction" : "dir"
    );
    await setPermission("windows.files", "auto");

    const read = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: target }
    });
    expect(read.state).toBe("complete");

    const outsideRead = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: outside }
    });
    expect(outsideRead.state).toBe("approval-required");

    const symlinkEscape = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: join(escapeLink, "linked-secret.txt") }
    });
    expect(symlinkEscape.state).toBe("approval-required");
    expect(symlinkEscape.output).toBeUndefined();

    const write = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "write",
      arguments: { path: target, content: "after" }
    });
    expect(write.state).toBe("approval-required");
    await expect(readFile(target, "utf8")).resolves.toBe("before");

    const approvedWrite = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "write",
      arguments: { path: target, content: "after" },
      approvalToken: write.approvalToken
    });
    expect(approvedWrite.state).toBe("complete");
    await expect(readFile(target, "utf8")).resolves.toBe("after");
  });

  it("lets Full Authority write immediately while redacting content from the audit", async () => {
    const target = join(temporaryRoot, "full.txt");
    const secret = "PRIVATE-AUDIT-FIXTURE-7c3f";
    await setPermission("windows.files", "full");

    const result = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "write",
      arguments: { path: target, content: secret }
    });

    expect(result.state).toBe("complete");
    expect(result.approvalToken).toBeUndefined();
    await expect(readFile(target, "utf8")).resolves.toBe(secret);

    const saved = await repository.get(brain.id);
    const audit = saved.journal?.at(-1);
    const detailText = audit?.detail ?? "";
    const detail = JSON.parse(detailText) as Record<string, unknown>;
    expect(audit?.summary).toBe("windows.files.write: complete.");
    expect(detailText).not.toContain(secret);
    expect(detail).toMatchObject({
      argumentKeys: ["content", "path"],
      argumentSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      paths: { path: target },
      changedPath: target,
      outputSha256: sha256(secret)
    });
  });

  it("rejects relative paths and stale-write checksums without changing the target", async () => {
    const target = join(temporaryRoot, "guarded.txt");
    await writeFile(target, "original");
    await setPermission("windows.files", "full");

    const relative = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: "guarded.txt" }
    });
    expect(relative.state).toBe("failed");
    expect(relative.error).toMatch(/absolute/i);

    const stale = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "write",
      arguments: {
        path: target,
        content: "should not land",
        expectedSha256: sha256("different version")
      }
    });
    expect(stale.state).toBe("failed");
    expect(stale.error).toMatch(/checksum does not match/i);
    await expect(readFile(target, "utf8")).resolves.toBe("original");
  });

  it("cancels modality jobs and ignores a worker result that arrives afterward", async () => {
    let finishWorker: ((value: unknown) => void) | undefined;
    const workerResult = new Promise<unknown>((resolve) => {
      finishWorker = resolve;
    });
    const engine = Object.assign(new EventEmitter(), {
      request: vi.fn(() => workerResult),
      tryRequest: vi.fn(async () => ({ cancelled: true })),
      interruptAndRestart: vi.fn(async () => true)
    }) as unknown as EngineSupervisor;
    const jobs = new RuntimeJobManager(service, engine);
    executor = new ToolExecutor(service, jobs);
    await setPermission("modality.imagine", "auto");

    const pendingExecution = executor.execute({
      brainId: brain.id,
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image", conceptIds: ["concept-1"] }
    });
    await vi.waitFor(() => {
      expect(jobs.list(brain.id)).toEqual([
        expect.objectContaining({ kind: "image", state: "running" })
      ]);
    });

    expect(executor.cancel(brain.id)).toBe(1);
    const execution = await pendingExecution;
    expect(execution.state).toBe("failed");
    expect(execution.error).toMatch(/cancelled/i);
    expect(engine.interruptAndRestart).toHaveBeenCalledOnce();

    finishWorker?.({ artifactPath: join(temporaryRoot, "late.png") });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(jobs.list(brain.id)[0]?.state).toBe("cancelled");
  });

  it("waits for imagination and returns the completed artifact to chat tools", async () => {
    const artifactPath = join(temporaryRoot, "finished.png");
    const engine = Object.assign(new EventEmitter(), {
      request: vi.fn(async () => ({
        path: artifactPath,
        mimeType: "image/png",
        seed: 41,
        dataUrl: "data:image/png;base64,fixture"
      })),
      tryRequest: vi.fn(async () => undefined),
      interruptAndRestart: vi.fn(async () => true)
    }) as unknown as EngineSupervisor;
    const jobs = new RuntimeJobManager(service, engine);
    executor = new ToolExecutor(service, jobs);
    await setPermission("modality.imagine", "auto");

    const execution = await executor.execute({
      brainId: brain.id,
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image", conceptIds: ["concept-1"] }
    });

    expect(execution.state).toBe("complete");
    expect(execution.output).toMatchObject({
      state: "complete",
      artifactPath,
      path: artifactPath,
      mimeType: "image/png",
      seed: 41,
      dataUrl: "data:image/png;base64,fixture",
      jobId: expect.stringMatching(/^[a-f0-9-]{36}$/i)
    });
    expect(jobs.list(brain.id)[0]).toMatchObject({
      state: "complete",
      output: { path: artifactPath, mimeType: "image/png", seed: 41 }
    });
  });

  it("does not run fallback consolidation after cancelling neural training", async () => {
    let finishTraining: ((value: unknown) => void) | undefined;
    const trainingResult = new Promise<unknown>((resolve) => {
      finishTraining = resolve;
    });
    const engine = Object.assign(new EventEmitter(), {
      tryRequest: vi.fn(() => trainingResult),
      interruptAndRestart: vi.fn(async () => true)
    }) as unknown as EngineSupervisor;
    const consolidate = vi.fn(async () => repository.get(brain.id));
    const trainingService = {
      repository,
      consolidate
    } as unknown as BrainService;
    const jobs = new RuntimeJobManager(trainingService, engine);

    const training = jobs.startTraining({ brainId: brain.id, epochs: 2 });
    await vi.waitFor(() => expect(engine.tryRequest).toHaveBeenCalledOnce());
    const cancelled = await jobs.cancel(training.id);
    expect(cancelled.state).toBe("cancelled");

    finishTraining?.(undefined);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(consolidate).not.toHaveBeenCalled();
    expect(jobs.list(brain.id).find((job) => job.id === training.id)?.state).toBe(
      "cancelled"
    );
  });

  it("cancels a running code command and records the interrupted outcome", async () => {
    const script = join(temporaryRoot, "wait.js");
    await writeFile(script, "setTimeout(() => process.stdout.write('too late'), 30000);");
    await setPermission("code.execute", "full");

    const pending = executor.execute({
      brainId: brain.id,
      toolId: "code.execute",
      action: "run",
      arguments: {
        language: "javascript",
        entryPath: script,
        timeoutMs: 60_000
      }
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(executor.cancel(brain.id)).toBe(1);
    const result = await pending;
    expect(result.state).toBe("failed");
    expect(result.error).toMatch(/cancelled/i);

    const saved = await repository.get(brain.id);
    expect(saved.journal?.at(-1)?.summary).toBe("code.execute.run: failed.");
    expect(saved.traces.at(-1)).toMatchObject({
      input: "code.execute.run",
      steps: expect.arrayContaining([
        expect.objectContaining({
          stage: "tool-result",
          detail: "Tool execution was cancelled."
        })
      ])
    });
  });

  it("reports process deadlines as failures instead of successful killed commands", async () => {
    const script = join(temporaryRoot, "timeout.js");
    await writeFile(script, "setInterval(() => undefined, 30000);");
    await setPermission("code.execute", "full");

    const startedAt = Date.now();
    const result = await executor.execute({
      brainId: brain.id,
      toolId: "code.execute",
      action: "run",
      arguments: {
        language: "javascript",
        entryPath: script,
        timeoutMs: 1_000
      }
    });

    expect(result.state).toBe("failed");
    expect(result.error).toMatch(/timed out after 1000 ms/i);
    expect(Date.now() - startedAt).toBeLessThan(5_000);
  });

  it("converts hostile remote markup into an inert local browser document", () => {
    const inert = buildInertBrowserDocument(
      `<!doctype html>
       <html>
         <head>
           <title>Safe &amp; readable</title>
           <script>window.stolen = document.cookie</script>
           <link rel="stylesheet" href="https://tracker.invalid/a.css">
         </head>
         <body onload="steal()">
           <h1>Visible heading</h1>
           <img src="https://tracker.invalid/pixel" onerror="steal()">
           <a href="/guide?x=1#section">Guide &amp; docs</a>
           <a href="javascript:steal()">Unsafe link</a>
         </body>
       </html>`,
      "https://docs.example/base/"
    );

    expect(inert).toMatchObject({
      title: "Safe & readable",
      links: [
        {
          label: "Guide & docs",
          href: "https://docs.example/guide?x=1"
        }
      ]
    });
    expect(inert.text).toContain("Visible heading");
    expect(inert.text).not.toContain("document.cookie");
    expect(inert.document).not.toMatch(
      /<script|onload=|onerror=|javascript:|tracker\.invalid/i
    );
    expect(inert.document).toContain("Visible heading");
  });
});
