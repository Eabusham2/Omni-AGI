/** Constructor-injected host fakes only. Never changes the test runner's real power or process priority. */
import { describe, expect, it, vi } from "vitest";
import { OsWorkLeaseService, type OsWorkPlatformAdapter } from "../src/main/osWorkLeaseService";

function fakePlatform(overrides: Partial<OsWorkPlatformAdapter> = {}) {
  const blockers = new Set<number>(), priorities = new Map<number, number>([[100, 0], [200, 5]]), calls: string[] = [];
  let nextBlocker = 40;
  const adapter: OsWorkPlatformAdapter = {
    platform: "fixture",
    startPowerSaveBlocker: () => { const id = ++nextBlocker; blockers.add(id); calls.push(`start:${id}`); return id; },
    stopPowerSaveBlocker: id => { blockers.delete(id); calls.push(`stop:${id}`); },
    isPowerSaveBlockerStarted: id => blockers.has(id),
    priorityProcessIds: () => [100, 200],
    getPriority: pid => priorities.get(pid)!,
    setPriority: (pid, priority) => { priorities.set(pid, priority); calls.push(`priority:${pid}:${priority}`); },
    requestedPriority: -7,
    ...overrides
  };
  return { adapter, blockers, priorities, calls };
}

describe("main-owned OS work lease", () => {
  it("reference-counts concurrent work, raises owned targets once, then restores exact original priority", () => {
    const host = fakePlatform(), service = new OsWorkLeaseService(host.adapter, () => new Date("2026-10-15T12:00:00Z"));
    const releaseA = service.acquire({ source: "host", id: "job-a", kind: "training", label: "Training" });
    const releaseA2 = service.acquire({ source: "host", id: "job-a", kind: "training", label: "Training · updated" });
    const releaseB = service.acquire({ source: "runtime-job", id: "job-b", kind: "video", label: "Video" });
    expect(service.status()).toMatchObject({ active: true, activeLeaseCount: 2, activeReferenceCount: 3,
      sleepPrevention: { active: true, outcome: "active" }, processPriority: { activeTargets: 2, outcome: "active" } });
    expect(host.calls.filter(value => value.startsWith("start:"))).toHaveLength(1);
    expect(host.calls.filter(value => value === "priority:100:-7")).toHaveLength(1);
    releaseA(); releaseB();
    expect(service.status()).toMatchObject({ active: true, activeLeaseCount: 1, activeReferenceCount: 1 });
    releaseA2();
    expect(service.status()).toMatchObject({ active: false, activeLeaseCount: 0,
      sleepPrevention: { active: false, outcome: "restored" }, processPriority: { activeTargets: 0, outcome: "restored" } });
    expect(host.priorities.get(100)).toBe(0); expect(host.priorities.get(200)).toBe(5); expect(host.blockers.size).toBe(0);
  });

  it("keeps denied and unsupported outcomes honest without blocking actual work", () => {
    const deniedError = Object.assign(new Error("operation not permitted"), { code: "EPERM" });
    const host = fakePlatform({ startPowerSaveBlocker: undefined, stopPowerSaveBlocker: undefined,
      isPowerSaveBlockerStarted: undefined, setPriority: () => { throw deniedError; } });
    const service = new OsWorkLeaseService(host.adapter), release = service.acquire({ source: "build", id: "build-a", kind: "build", label: "Build" });
    expect(service.status()).toMatchObject({ active: true, sleepPrevention: { active: false, outcome: "unsupported" },
      processPriority: { activeTargets: 0, deniedTargets: 2, outcome: "denied" } });
    release();
    expect(service.status()).toMatchObject({ active: false, processPriority: { activeTargets: 0, outcome: "denied" } });
  });

  it("leases only running/cancelling observations, refreshes a new owned worker, and isolates observer failures", () => {
    const pids = [100], diagnostics: string[] = [], host = fakePlatform({ priorityProcessIds: () => [...pids],
      diagnostic: message => { diagnostics.push(message); } });
    const service = new OsWorkLeaseService(host.adapter), listener = vi.fn();
    service.onStatus(() => { throw new Error("fixture observer failure"); }); service.onStatus(listener);
    const engine = (state: "queued" | "running" | "cancelling" | "complete") => service.observeEngineActivity({
      requestId: "engine-a", owner: "training", label: `Engine ${state}`, method: "ingest", brainId: "brain-a", state
    });
    engine("queued"); expect(service.status().active).toBe(false);
    engine("running"); expect(service.status()).toMatchObject({ active: true, activeLeaseCount: 1 });
    pids.push(200); service.refreshOwnedProcesses(); expect(service.status().processPriority.activeTargets).toBe(2);
    service.observeRuntimeJob({ id: "job-a", brainId: "brain-a", kind: "training", state: "running", progress: 0.5,
      label: "Training half", createdAt: new Date(0).toISOString(), updatedAt: new Date(0).toISOString() });
    engine("cancelling"); expect(service.status().activeLeaseCount).toBe(2);
    engine("complete"); expect(service.status().activeLeaseCount).toBe(1);
    service.observeRuntimeJob({ id: "job-a", brainId: "brain-a", kind: "training", state: "complete", progress: 1,
      label: "Training complete", createdAt: new Date(0).toISOString(), updatedAt: new Date(0).toISOString() });
    expect(service.status().active).toBe(false);
    const action = { id: "action-a", brainId: "brain-a", action: { kind: "tool" as const, source: "human" as const, arguments: {} },
      state: "running" as const, createdAt: new Date(0).toISOString(), updatedAt: new Date(0).toISOString() };
    service.observeActionEvent(action); expect(service.status().active).toBe(true);
    service.observeActionEvent({ ...action, state: "complete" }); expect(service.status().active).toBe(false);
    expect(diagnostics.length).toBeGreaterThan(0); expect(listener).toHaveBeenCalled();
  });

  it("reports a failed restoration and retries only its own retained target on Close", () => {
    let restoreFailures = 1;
    const host = fakePlatform({ priorityProcessIds: () => [100], setPriority: (pid, value) => {
      if (value === 0 && restoreFailures-- > 0) throw new Error("fixture restore failed");
      host.priorities.set(pid, value); host.calls.push(`priority:${pid}:${value}`);
    } });
    const service = new OsWorkLeaseService(host.adapter), release = service.acquire({ source: "host", id: "job-a", kind: "training", label: "Training" });
    release();
    expect(service.status()).toMatchObject({ active: false, processPriority: { activeTargets: 1, restoreFailedTargets: 1, outcome: "restore-failed" } });
    const closed = service.close();
    expect(closed).toMatchObject({ active: false, processPriority: { activeTargets: 0, restoreFailedTargets: 0, outcome: "restored" } });
    expect(host.priorities.get(100)).toBe(0);
    service.observeRuntimeJob({ id: "late", brainId: "brain-a", kind: "training", state: "running", progress: 0,
      label: "Late shutdown event", createdAt: new Date(0).toISOString(), updatedAt: new Date(0).toISOString() });
    expect(service.status().active).toBe(false);
  });

  it("normalizes long observed labels, never lowers a higher priority, and recaptures a reused owned PID", () => {
    const pids = [100], host = fakePlatform({ priorityProcessIds: () => [...pids] }); host.priorities.set(100, -14);
    const service = new OsWorkLeaseService(host.adapter), base = { id: "job-long", brainId: "brain-a", kind: "training" as const,
      progress: 0.2, createdAt: new Date(0).toISOString(), updatedAt: new Date(0).toISOString() };
    expect(() => service.observeRuntimeJob({ ...base, state: "running", label: `Reading ${"long-file-name-".repeat(100)}` })).not.toThrow();
    expect(host.priorities.get(100)).toBe(-14); expect(host.calls).not.toContain("priority:100:-7");
    pids.pop(); expect(service.retireOwnedProcess(100)).toBe(true); expect(host.calls).not.toContain("priority:100:-14");
    host.priorities.set(100, 4); pids.push(100); service.refreshOwnedProcesses();
    expect(host.priorities.get(100)).toBe(-7);
    service.observeRuntimeJob({ ...base, state: "complete", progress: 1, label: "Complete" });
    expect(host.priorities.get(100)).toBe(4);
  });
});
