/** One distinct revoke-during-setup regression. No process/model is spawned. */
import { expect, it, vi } from "vitest";
import { mkdtemp, rm, type FileHandle } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
import { startLocalTeacherChild } from "../src/main/localTeacherRunner";
const tracked = vi.hoisted(() => ({ files: [] as FileHandle[] }));
vi.mock("node:fs/promises", async original => {
  const actual = await original<typeof import("node:fs/promises")>();
  return { ...actual, open: async (...args: Parameters<typeof actual.open>) => { const file = await actual.open(...args);
    if (/process\.(?:stdout|stderr)\.bin$/.test(String(args[0]))) tracked.files.push(file); return file; } };
});
vi.mock("node:child_process", async original => ({ ...await original<typeof import("node:child_process")>(),
  spawn: vi.fn(() => { throw new Error("Unexpected fixture spawn: revoked authority must refuse first."); }) }));

it("rechecks revoked current policy after async setup, never spawns and closes both prepared owned FDs", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-teacher-prespawn-")); let current = true, checks = 0;
  try {
    await expect(startLocalTeacherChild({ root, relativeDirectory: "local-teachers/attempts/revoked-setup", executable: process.execPath,
      arguments: ["--version"], signal: new AbortController().signal, maxLogBytes: 1024, timeoutMs: 1000,
      requireCurrent: async () => { checks++; if (!current) throw new Error("Current local teacher policy was revoked during setup."); },
      checkResources: async () => { await Promise.resolve(); current = false; }
    })).rejects.toThrow(/revoked during setup/);
    expect(checks).toBe(2); expect(spawn).not.toHaveBeenCalled(); expect(tracked.files).toHaveLength(2);
    expect(tracked.files.every(file => file.fd === -1)).toBe(true);
  } finally { await Promise.allSettled(tracked.files.filter(file => file.fd !== -1).map(file => file.close())); await rm(root, { recursive: true, force: true }); }
});
