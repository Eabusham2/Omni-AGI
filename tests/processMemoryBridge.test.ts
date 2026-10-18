/** Real task-owned protocol children, synthetic OS receipts; no memory target. */
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { PROCESS_MEMORY_FORMAT, type ProcessMemoryNativeRequest, type ProcessMemoryReady } from "../src/shared/processMemory";
import { ProcessMemoryOperationError, processMemoryLaunch, runProcessMemoryHelper } from "../src/main/processMemoryBridge";
import { processMemoryHash } from "../src/main/processMemoryFormat";

const launch = (mode: string) => ({ command: process.execPath, arguments: [resolve("tests/fixtures/process-memory-protocol.cjs"), mode] });
const request = (operation: "read" | "write" = "read"): ProcessMemoryNativeRequest => ({ format: PROCESS_MEMORY_FORMAT,
  operationId: "bridge-fixture", operation, pid: 123456, access: "read-write", address: "0x1000", length: 4,
  expectedIdentity: { platform: "linux", pid: 123456, creation: "fixture-incarnation", executable: "/fixture/owned", executableIdentity: "1:2" },
  ...(operation === "write" ? { dataBase64: "V1hZWg==", expectedBeforeSha256: processMemoryHash(Buffer.from("abcd")) } : {}) });

describe("process-memory two-phase helper transport", () => {
  it("hands exact prepared identity to current authorization and validates the response", async () => {
    const beforeCommit = vi.fn(async (_ready: ProcessMemoryReady, _helperPid: number) => undefined), receipt = await runProcessMemoryHelper(launch("normal"), request(), {
      signal: new AbortController().signal, beforeCommit });
    expect(beforeCommit).toHaveBeenCalledTimes(1); expect(beforeCommit.mock.calls[0]?.[0]).toMatchObject({ operationId: "bridge-fixture", identity: request().expectedIdentity });
    expect(receipt).toMatchObject({ state: "complete", dataBase64: "YWJjZA==", transferredBytes: 4 });
  });
  it("wrong prepared input hash is rejected without calling the commit authorizer", async () => {
    const beforeCommit = vi.fn(async () => undefined);
    await expect(runProcessMemoryHelper(launch("mismatch"), request("write"), { signal: new AbortController().signal, beforeCommit })).rejects.toThrow(/exact request/);
    expect(beforeCommit).not.toHaveBeenCalled();
  });
  it("Stop while final review waits reports no committed write", async () => {
    const controller = new AbortController();
    const pending = runProcessMemoryHelper(launch("normal"), request("write"), { signal: controller.signal,
      beforeCommit: async () => { controller.abort(new Error("fixture Stop")); } });
    try { await pending; throw new Error("expected rejection"); }
    catch (error) { expect((error as ProcessMemoryOperationError).receipt).toMatchObject({ state: "cancelled", transferredBytes: 0, writeMayHaveOccurred: false }); }
  });
  it("Stop after a write commit retains the actual bounded result", async () => {
    const controller = new AbortController();
    const receipt = await runProcessMemoryHelper(launch("delayed"), request("write"), { signal: controller.signal,
      beforeCommit: async () => { setTimeout(() => controller.abort(), 10); } });
    expect(controller.signal.aborted).toBe(true);
    expect(receipt).toMatchObject({ state: "complete", transferredBytes: 4, writeMayHaveOccurred: true, retrySafe: false });
  });
  it("lost post-write result is uncertain, never a cancelled/no-effect receipt", async () => {
    try { await runProcessMemoryHelper(launch("lost"), request("write"), { signal: new AbortController().signal, beforeCommit: async () => undefined }, 100); throw new Error("expected rejection"); }
    catch (error) { expect((error as ProcessMemoryOperationError).receipt).toMatchObject({ state: "write-uncertain", transferredBytes: null, writeMayHaveOccurred: true, retrySafe: false }); }
  });
  it("truthful denial stays a denial and packaged mode has no interpreter fallback", async () => {
    const receipt = await runProcessMemoryHelper(launch("denied"), request(), { signal: new AbortController().signal, beforeCommit: async () => undefined });
    expect(receipt).toMatchObject({ state: "denied", osCode: 5, transferredBytes: 0 });
    expect(() => processMemoryLaunch({ appPath: process.cwd(), resourcesPath: "/nonexistent-fixture", packagedRequired: true })).toThrow(/No external interpreter fallback/);
  });
});
