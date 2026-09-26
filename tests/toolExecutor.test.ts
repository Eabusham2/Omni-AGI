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
  BrainService,
  RuntimeJobManager
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import {
  buildInertBrowserDocument,
  parsePublicSearchRss,
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
      preflightStart: vi.fn(async () => undefined),
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
    await setPermission("system.files", "off");

    const result = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
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
      summary: "system.files.read: failed."
    });
    expect(JSON.parse(audit?.detail ?? "{}")).toMatchObject({
      argumentKeys: ["path"],
      paths: { path: target },
      error: "This tool is disabled for the current brain."
    });
  });

  it("uses a visible thirty-second Ask approval window by default", async () => {
    const target = join(temporaryRoot, "approval-window.txt");
    await writeFile(target, "visible after approval");
    await setPermission("system.files", "ask");
    const before = Date.now();
    const pending = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: target }
    });
    expect(pending.state).toBe("approval-required");
    expect(pending.approvalToken).toBeTruthy();
    const expires = Date.parse(pending.approvalExpiresAt ?? "");
    expect(expires - before).toBeGreaterThanOrEqual(29_000);
    expect(expires - before).toBeLessThanOrEqual(31_000);
  });

  it("canonicalizes a legacy file invocation before approval, result, and audit", async () => {
    const target = join(temporaryRoot, "legacy-alias.txt");
    await writeFile(target, "legacy permission, system presentation");
    await setPermission("system.files", "ask");

    const pending = await executor.execute({
      brainId: brain.id,
      toolId: "windows.files",
      action: "read",
      arguments: { path: target }
    });
    expect(pending).toMatchObject({
      state: "approval-required",
      toolId: "system.files"
    });

    const completed = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: target },
      approvalToken: pending.approvalToken
    });
    expect(completed).toMatchObject({
      state: "complete",
      toolId: "system.files"
    });
    const saved = await repository.get(brain.id);
    expect(saved.journal?.at(-1)?.summary).toBe(
      "system.files.read: complete."
    );
  });

  it("runs System shell through the native host adapter", async () => {
    await setPermission("system.shell", "full");
    const completed = await executor.execute({
      brainId: brain.id,
      toolId: "system.shell",
      action: "run",
      arguments: {
        command: process.platform === "win32"
          ? "Write-Output omni-native-shell"
          : "printf omni-native-shell",
        cwd: temporaryRoot
      }
    });

    expect(completed).toMatchObject({
      state: "complete",
      toolId: "system.shell",
      output: { exitCode: 0 }
    });
    expect((completed.output as { stdout: string }).stdout).toContain(
      "omni-native-shell"
    );
  });

  it("routes approved host input through the audited backend instead of renderer IPC", async () => {
    const deviceInput = {
      execute: vi.fn(async () => ({
        ok: true,
        state: "complete" as const,
        action: "move-pointer" as const,
        audit: {
          id: "device-audit-one",
          action: "move-pointer" as const,
          state: "complete" as const,
          platform: "darwin" as const,
          backend: "macos-accessibility" as const,
          startedAt: "2026-08-22T00:00:00.000Z",
          completedAt: "2026-08-22T00:00:00.010Z",
          durationMs: 10,
          request: { x: 40, y: 80 }
        }
      }))
    };
    executor = new ToolExecutor(
      service,
      {} as RuntimeJobManager,
      undefined,
      deviceInput
    );

    const request = {
      brainId: brain.id,
      toolId: "device.input",
      action: "move-pointer",
      arguments: { x: 40, y: 80 }
    };
    const pending = await executor.execute(request);
    expect(pending.state).toBe("approval-required");
    expect(deviceInput.execute).not.toHaveBeenCalled();

    const completed = await executor.execute({
      ...request,
      approvalToken: pending.approvalToken
    });
    expect(completed.state).toBe("complete");
    expect(deviceInput.execute).toHaveBeenCalledWith(
      { action: "move-pointer", x: 40, y: 80 },
      expect.objectContaining({
        authorization: expect.objectContaining({
          granted: true,
          policy: "ask",
          decisionId: expect.any(String)
        }),
        signal: expect.any(AbortSignal)
      })
    );
  });

  it("maps Full Authority host input to an explicit full-authority decision", async () => {
    await setPermission("device.input", "full");
    const deviceInput = {
      execute: vi.fn(async () => ({
        ok: true,
        state: "complete" as const,
        action: "scroll" as const,
        audit: {
          id: "device-audit-two",
          action: "scroll" as const,
          state: "complete" as const,
          platform: "win32" as const,
          backend: "windows-user32" as const,
          startedAt: "2026-08-22T00:00:00.000Z",
          completedAt: "2026-08-22T00:00:00.010Z",
          durationMs: 10,
          request: { deltaX: 0, deltaY: 120 }
        }
      }))
    };
    executor = new ToolExecutor(
      service,
      {} as RuntimeJobManager,
      undefined,
      deviceInput
    );

    const completed = await executor.execute({
      brainId: brain.id,
      toolId: "device.input",
      action: "scroll",
      arguments: { deltaY: 120 }
    });
    expect(completed.state).toBe("complete");
    expect(completed.approvalToken).toBeUndefined();
    expect(deviceInput.execute).toHaveBeenCalledWith(
      { action: "scroll", deltaY: 120 },
      expect.objectContaining({
        authorization: expect.objectContaining({ policy: "full-authority" })
      })
    );
  });

  it("keeps redacted native audit evidence when host input is denied by the OS", async () => {
    await setPermission("device.input", "full");
    const deviceInput = {
      execute: vi.fn(async () => ({
        ok: false,
        state: "denied" as const,
        action: "text" as const,
        errorCode: "permission-required" as const,
        error: "macOS Accessibility permission is required.",
        audit: {
          id: "device-audit-denied",
          action: "text" as const,
          state: "denied" as const,
          platform: "darwin" as const,
          backend: "macos-accessibility" as const,
          startedAt: "2026-08-22T00:00:00.000Z",
          completedAt: "2026-08-22T00:00:00.010Z",
          durationMs: 10,
          request: { utf8Bytes: 14, sha256: "a".repeat(64) }
        }
      }))
    };
    executor = new ToolExecutor(
      service,
      {} as RuntimeJobManager,
      undefined,
      deviceInput
    );

    const result = await executor.execute({
      brainId: brain.id,
      toolId: "device.input",
      action: "text",
      arguments: { text: "private phrase" }
    });
    expect(result).toMatchObject({
      state: "failed",
      error: "macOS Accessibility permission is required.",
      output: {
        state: "denied",
        audit: {
          id: "device-audit-denied",
          request: { utf8Bytes: 14, sha256: "a".repeat(64) }
        }
      }
    });
    expect(JSON.stringify(result)).not.toContain("private phrase");
    const saved = await repository.get(brain.id);
    const detail = JSON.parse(saved.journal?.at(-1)?.detail ?? "{}");
    expect(detail.deviceAudit).toMatchObject({
      id: "device-audit-denied",
      state: "denied"
    });
    expect(JSON.stringify(detail)).not.toContain("private phrase");
  });

  it("opens the local creativity workspace without external authority and audits it", async () => {
    const execution = await executor.execute({
      brainId: brain.id,
      toolId: "studio.ui",
      action: "open-creativity",
      arguments: { assemblyIds: ["story-scene"], organic: true }
    });

    expect(execution).toMatchObject({
      toolId: "studio.ui",
      action: "open-creativity",
      state: "complete",
      output: {
        workspace: "creativity",
        view: "imagine",
        local: true,
        reversible: true
      }
    });
    expect(execution.approvalToken).toBeUndefined();
    const saved = await repository.get(brain.id);
    expect(saved.journal?.at(-1)).toMatchObject({
      kind: "tool",
      summary: "studio.ui.open-creativity: complete."
    });
    expect(saved.traces.at(-1)?.steps).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ stage: "tool-permission", value: "complete" }),
        expect.objectContaining({
          stage: "tool-invocation",
          detail: expect.stringContaining("studio.ui.open-creativity")
        })
      ])
    );
  });

  it("lets the brain inspect access without granting or changing any permission", async () => {
    const before = await service.listToolPermissions(brain.id);
    const execution = await executor.execute({
      brainId: brain.id,
      toolId: "studio.settings",
      action: "inspect-access",
      arguments: {}
    });
    expect(execution).toMatchObject({
      state: "complete",
      output: {
        approvalTimeoutSeconds: 30,
        canRequestChange: true,
        grantsChanged: false,
        local: true
      }
    });
    expect(await service.listToolPermissions(brain.id)).toEqual(before);
  });

  it("lets the brain search only its own visible chat and audited actions", async () => {
    const current = await repository.get(brain.id);
    current.messages.push(
      {
        id: "human-history",
        role: "human",
        content: "Remember the copper lighthouse.",
        createdAt: "2026-08-13T01:00:00.000Z"
      },
      {
        id: "idle-history",
        role: "brain",
        content: "I chose to mention the copper lighthouse while idle.",
        createdAt: "2026-08-13T01:01:00.000Z",
        runtime: "adaptive-core"
      }
    );
    current.journal = [
      ...(current.journal ?? []),
      {
        id: "visible-action-history",
        kind: "tool",
        summary: "web.search.search: complete.",
        detail: JSON.stringify({ query: "copper lighthouse", resultCount: 2 }),
        createdAt: "2026-08-13T01:02:00.000Z"
      }
    ];
    await repository.save(current);
    await repository.appendChatDeliveryReceipt(brain.id, {
      schemaVersion: 1,
      turnId: "cancelled-history-turn",
      content: "Uncommitted copper lighthouse queue receipt.",
      createdAt: "2026-08-13T01:03:00.000Z",
      state: "cancelled"
    });

    const result = await executor.execute({
      brainId: brain.id,
      toolId: "brain.history",
      action: "search",
      arguments: { query: "copper lighthouse", includeActions: true, limit: 10 }
    });

    expect(result.state).toBe("complete");
    expect(result.approvalToken).toBeUndefined();
    expect(result.output).toMatchObject({
      matched: 3,
      scope: "this-brain-visible-history",
      privateReasoningIncluded: false,
      measuredTracesIncluded: false
    });
    const output = result.output as { entries: Array<Record<string, unknown>> };
    expect(output.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: "message", role: "human", id: "human-history" }),
      expect.objectContaining({ kind: "message", role: "brain", id: "idle-history" }),
      expect.objectContaining({ kind: "action", id: "visible-action-history" })
    ]));
    expect(JSON.stringify(result.output)).not.toContain("activatedConcepts");
    expect(JSON.stringify(result.output)).not.toContain("Uncommitted copper lighthouse");

    await setPermission("brain.history", "off");
    const disabled = await executor.execute({
      brainId: brain.id,
      toolId: "brain.history",
      action: "read",
      arguments: {}
    });
    expect(disabled.state).toBe("failed");
    expect(disabled.error).toMatch(/disabled/i);
  });

  it("pages complete directory listings instead of hiding entries after a fixed cutoff", async () => {
    const directory = join(temporaryRoot, "paged-files");
    await mkdir(directory);
    await Promise.all([
      writeFile(join(directory, "a.txt"), "a"),
      writeFile(join(directory, "b.txt"), "b"),
      writeFile(join(directory, "c.txt"), "c")
    ]);
    await setPermission("system.files", "full");

    const first = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "list",
      arguments: { path: directory, pageSize: 2 }
    });
    expect(first.state).toBe("complete");
    expect(first.output).toMatchObject({
      offset: 0,
      returned: 2,
      total: 3,
      hasMore: true,
      nextCursor: "2"
    });

    const second = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "list",
      arguments: { path: directory, pageSize: 2, cursor: "2" }
    });
    expect(second.state).toBe("complete");
    expect(second.output).toMatchObject({
      offset: 2,
      returned: 1,
      total: 3,
      hasMore: false
    });
    expect((second.output as { entries: Array<{ name: string }> }).entries[0]?.name).toBe("c.txt");
  });

  it("requires an Ask approval exactly once and scopes it to the invocation", async () => {
    const target = join(temporaryRoot, "ask.txt");
    const substitutedTarget = join(temporaryRoot, "not-approved.txt");
    await writeFile(target, "approved contents");
    await writeFile(substitutedTarget, "different contents");
    await setPermission("system.files", "ask");
    const invocation = {
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: target }
    };

    const challenge = await executor.execute(invocation);
    expect(challenge.state).toBe("approval-required");
    expect(challenge.approvalToken).toMatch(/^[a-f0-9-]{36}$/i);
    expect(executor.hasPendingOrActive(brain.id)).toBe(true);

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
      saved.journal?.filter((entry) => entry.summary === "system.files.read: complete.")
    ).toHaveLength(1);
  });

  it("invalidates Ask tokens across permission revisions and denied use", async () => {
    const target = join(temporaryRoot, "revoked-approval.txt");
    await writeFile(target, "must require current approval");
    await setPermission("system.files", "ask");
    const invocation = {
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: target }
    };

    const challenge = await executor.execute(invocation);
    expect(challenge.state).toBe("approval-required");

    await setPermission("system.files", "off");
    const denied = await executor.execute({
      ...invocation,
      approvalToken: challenge.approvalToken
    });
    expect(denied.state).toBe("failed");
    expect(denied.error).toMatch(/disabled/i);

    await setPermission("system.files", "ask");
    const resurrected = await executor.execute({
      ...invocation,
      approvalToken: challenge.approvalToken
    });
    expect(resurrected.state).toBe("approval-required");
    expect(resurrected.approvalToken).not.toBe(challenge.approvalToken);
    expect(resurrected.output).toBeUndefined();

    const current = await repository.get(brain.id);
    const permission = current.toolPermissions?.find(
      (entry) => entry.toolId === "system.files"
    );
    expect(permission).toBeTruthy();
    permission!.updatedAt = new Date(Date.now() + 1_000).toISOString();
    await repository.save(current);

    const staleRevision = await executor.execute({
      ...invocation,
      approvalToken: resurrected.approvalToken
    });
    expect(staleRevision.state).toBe("approval-required");
    expect(staleRevision.approvalToken).not.toBe(resurrected.approvalToken);
    expect(staleRevision.output).toBeUndefined();

    await setPermission("system.files", "full");
    const elevated = await executor.execute({
      ...invocation,
      approvalToken: staleRevision.approvalToken
    });
    expect(elevated.state).toBe("complete");
    await setPermission("system.files", "ask");
    const afterDowngrade = await executor.execute({
      ...invocation,
      approvalToken: staleRevision.approvalToken
    });
    expect(afterDowngrade.state).toBe("approval-required");
    expect(afterDowngrade.approvalToken).not.toBe(staleRevision.approvalToken);
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
    await setPermission("system.files", "auto");

    const read = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: target }
    });
    expect(read.state).toBe("complete");

    const outsideRead = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: outside }
    });
    expect(outsideRead.state).toBe("approval-required");

    const symlinkEscape = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: join(escapeLink, "linked-secret.txt") }
    });
    expect(symlinkEscape.state).toBe("approval-required");
    expect(symlinkEscape.output).toBeUndefined();

    const write = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "write",
      arguments: { path: target, content: "after" }
    });
    expect(write.state).toBe("approval-required");
    await expect(readFile(target, "utf8")).resolves.toBe("before");

    const approvedWrite = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
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
    await setPermission("system.files", "full");

    const result = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
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
    expect(audit?.summary).toBe("system.files.write: complete.");
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
    await setPermission("system.files", "full");

    const relative = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
      action: "read",
      arguments: { path: "guarded.txt" }
    });
    expect(relative.state).toBe("failed");
    expect(relative.error).toMatch(/absolute/i);

    const stale = await executor.execute({
      brainId: brain.id,
      toolId: "system.files",
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
    const engine = Object.assign(new EventEmitter(), {
      request: vi.fn((
        method: string,
        params: Record<string, unknown>,
        _timeout?: number,
        signal?: AbortSignal
      ) => {
        if (method === "cancel") {
          return Promise.resolve({
            jobId: params.jobId,
            cancelled: true,
            acknowledged: true
          });
        }
        return new Promise<unknown>((resolve, reject) => {
          finishWorker = resolve;
          signal?.addEventListener(
            "abort",
            () => reject(new Error("worker request cancelled")),
            { once: true }
          );
        });
      }),
      cancelRequest: vi.fn(async (requestId: string) => ({
        requestId,
        acknowledged: true,
        phase: "running" as const,
        workerTerminationAcknowledged: true
      })),
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
    expect(engine.cancelRequest).toHaveBeenCalledWith(
      expect.stringMatching(/^[a-f0-9-]{36}$/i)
    );
    expect(engine.interruptAndRestart).not.toHaveBeenCalled();

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
    expect(engine.request).toHaveBeenCalledWith(
      "generate_modality",
      expect.objectContaining({ settings: { outputMode: "auto" } }),
      3_600_000,
      expect.any(AbortSignal)
    );
  });

  it("does not expose direct training or consolidation runtime entry points", () => {
    expect(RuntimeJobManager.prototype).not.toHaveProperty("startTraining");
    expect(BrainService.prototype).not.toHaveProperty("consolidate");
  });

  it("keeps an unrelated runtime job active when another exact job is cancelled", async () => {
    const signals = new Map<string, AbortSignal>();
    const engine = Object.assign(new EventEmitter(), {
      request: vi.fn((
        method: string,
        params: Record<string, unknown>,
        _timeout?: number,
        signal?: AbortSignal
      ) => {
        if (method === "cancel") {
          return Promise.resolve({
            jobId: params.jobId,
            cancelled: true,
            acknowledged: true
          });
        }
        const jobId = String(params.jobId);
        if (signal) signals.set(jobId, signal);
        return new Promise<unknown>((_resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () => reject(new Error(`worker request ${jobId} cancelled`)),
            { once: true }
          );
        });
      }),
      cancelRequest: vi.fn(async (requestId: string) => ({
        requestId,
        acknowledged: true,
        phase: "running" as const,
        workerTerminationAcknowledged: true
      }))
    }) as unknown as EngineSupervisor;
    const modalityService = {
      repository,
      preflightStart: vi.fn(async () => undefined)
    } as unknown as BrainService;
    const jobs = new RuntimeJobManager(modalityService, engine);
    const first = jobs.generate({
      brainId: brain.id,
      modality: "image",
      prompt: "first cancellation fixture"
    });
    const second = jobs.generate({
      brainId: brain.id,
      modality: "audio",
      prompt: "unrelated cancellation fixture"
    });
    await vi.waitFor(() => expect(signals.size).toBe(2));

    engine.emit("activity", {
      requestId: second.id,
      owner: "modality",
      label: "Generating audio",
      method: "generate_modality",
      brainId: brain.id,
      jobId: second.id,
      state: "queued",
      queuePosition: 1,
      queuedBehind: {
        requestId: first.id,
        owner: "modality",
        label: "Generating image",
        method: "generate_modality",
        brainId: brain.id,
        jobId: first.id
      }
    });
    expect(jobs.list(brain.id).find((job) => job.id === second.id)).toMatchObject({
      state: "queued",
      queue: {
        position: 1,
        queuedBehind: { requestId: first.id, owner: "modality" }
      }
    });
    engine.emit("activity", {
      requestId: second.id,
      owner: "modality",
      label: "Generating audio",
      method: "generate_modality",
      brainId: brain.id,
      jobId: second.id,
      state: "running"
    });

    await expect(jobs.cancel(first.id)).resolves.toMatchObject({
      id: first.id,
      state: "cancelled"
    });
    expect(signals.get(first.id)?.aborted).toBe(true);
    expect(signals.get(second.id)?.aborted).toBe(false);
    const unrelated = jobs.list(brain.id).find((job) => job.id === second.id);
    expect(unrelated).toMatchObject({ state: "running" });
    expect(unrelated).not.toHaveProperty("error");
    expect(
      (engine.cancelRequest as ReturnType<typeof vi.fn>).mock.calls
    ).toEqual([[first.id]]);

    await expect(jobs.cancel(second.id)).resolves.toMatchObject({ state: "cancelled" });
  });

  it("cancels a running code command and records the interrupted outcome", async () => {
    const script = join(temporaryRoot, "wait.js");
    const started = join(temporaryRoot, "wait.started");
    await writeFile(
      script,
      `require("node:fs").writeFileSync(${JSON.stringify(started)}, "ready");` +
        "setTimeout(() => process.stdout.write('too late'), 30000);"
    );
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
    await vi.waitFor(async () => {
      expect(await readFile(started, "utf8")).toBe("ready");
    }, { timeout: 5_000 });
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

  it("cancels and acknowledges only the exact tool request id", async () => {
    const firstScript = join(temporaryRoot, "first-wait.js");
    const secondScript = join(temporaryRoot, "second-wait.js");
    const firstStarted = join(temporaryRoot, "first-wait.started");
    const secondStarted = join(temporaryRoot, "second-wait.started");
    await Promise.all([
      writeFile(
        firstScript,
        `require("node:fs").writeFileSync(${JSON.stringify(firstStarted)}, "ready");` +
          "setTimeout(() => process.stdout.write('first done'), 30000);"
      ),
      writeFile(
        secondScript,
        `require("node:fs").writeFileSync(${JSON.stringify(secondStarted)}, "ready");` +
          "setTimeout(() => process.stdout.write('second done'), 30000);"
      )
    ]);
    await setPermission("code.execute", "full");

    const first = executor.execute({
      brainId: brain.id,
      toolId: "code.execute",
      action: "run",
      arguments: { language: "javascript", entryPath: firstScript, timeoutMs: 60_000 }
    }, undefined, "chat-turn");
    const second = executor.execute({
      brainId: brain.id,
      toolId: "code.execute",
      action: "run",
      arguments: { language: "javascript", entryPath: secondScript, timeoutMs: 60_000 }
    }, undefined, "evolution-run");
    await vi.waitFor(async () => {
      expect(await readFile(firstStarted, "utf8")).toBe("ready");
      expect(await readFile(secondStarted, "utf8")).toBe("ready");
    }, { timeout: 5_000 });

    await expect(executor.cancelAndWait(brain.id, "chat-turn")).resolves.toBe(1);
    await expect(first).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/cancelled/i)
    });
    let unrelatedSettled = false;
    void second.finally(() => {
      unrelatedSettled = true;
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 25));
    expect(unrelatedSettled).toBe(false);

    await expect(executor.cancelAndWait(brain.id, "evolution-run")).resolves.toBe(1);
    await expect(second).resolves.toMatchObject({
      state: "failed",
      error: expect.stringMatching(/cancelled/i)
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

describe("public web-search fallback", () => {
  it("parses inert HTTPS RSS results and rejects executable or credential URLs", () => {
    const results = parsePublicSearchRss(`
      <rss><channel>
        <item><title>Primary paper</title><link>https://example.org/paper</link><description>&lt;b&gt;Evidence&lt;/b&gt;</description></item>
        <item><title>Unsafe</title><link>javascript:alert(1)</link><description>bad</description></item>
        <item><title>Credential URL</title><link>https://user:pass@example.org/</link><description>bad</description></item>
      </channel></rss>
    `);
    expect(results).toEqual([
      { title: "Primary paper", url: "https://example.org/paper", snippet: "Evidence" }
    ]);
  });
});
