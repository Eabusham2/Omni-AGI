/** Fake child events and fake OS adapter only: no process, worker, model, priority or blocker is started. */
import { EventEmitter } from "node:events";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";
import { OsWorkLeaseService } from "../src/main/osWorkLeaseService";
class FakeChild extends EventEmitter {
  readonly pid = 2345; killed = false; exitCode: number | null = null; signalCode = null;
  stdout = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  stderr = Object.assign(new EventEmitter(), { setEncoding: vi.fn() });
  stdin = { writable: true, write: vi.fn() };
}
it("refreshes an actually supervised fake child on spawn and retires it before PID reuse", () => {
  const engine = new EngineSupervisor({ appPath: "/inert-fixture" }), writes: Array<[number, number]> = [];
  const priorities = new Map([[101, 0], [2345, 4]]);
  const work = new OsWorkLeaseService({ platform: "fixture", priorityProcessIds: () => [101, ...engine.managedProcessIds],
    requestedPriority: -7, getPriority: pid => priorities.get(pid)!, setPriority: (pid, value) => { priorities.set(pid, value); writes.push([pid, value]); } });
  engine.on("worker-started", () => work.refreshOwnedProcesses());
  engine.on("worker-closed", ({ pid }) => work.retireOwnedProcess(pid));
  const release = work.acquire({ source: "host", id: "real-host-event", kind: "copy", label: "Copying" });
  const supervise = (child: FakeChild) => (engine as unknown as { superviseChild(value: ChildProcessWithoutNullStreams): void })
    .superviseChild(child as unknown as ChildProcessWithoutNullStreams);
  const child = new FakeChild(); supervise(child); child.emit("spawn");
  expect(writes).toContainEqual([2345, -7]);
  child.exitCode = 0; child.emit("close", 0, null);
  const count = writes.length; priorities.set(2345, 9); // Numeric PID is no longer owned.
  release(); expect(writes.slice(count)).toEqual([[101, 0]]); expect(priorities.get(2345)).toBe(9);
  work.close();
});
