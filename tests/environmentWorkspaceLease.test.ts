import { describe, expect, it } from "vitest";
import { withEnvironmentMutation, withEnvironmentSnapshot } from "../src/main/environmentWorkspaceLease";

describe("environment workspace snapshot lease", () => {
  it("waits for active writers and blocks new writers until capture releases", async () => {
    const events: string[] = []; let releaseFirst!: () => void, releaseSnapshot!: () => void;
    const firstGate = new Promise<void>(resolve => { releaseFirst = resolve; }), snapshotGate = new Promise<void>(resolve => { releaseSnapshot = resolve; });
    const first = withEnvironmentMutation("/fixture/brain", async () => { events.push("first-start"); await firstGate; events.push("first-end"); });
    await Promise.resolve();
    const snapshot = withEnvironmentSnapshot("/fixture/brain", async () => { events.push("snapshot-start"); await snapshotGate; events.push("snapshot-end"); });
    const second = withEnvironmentMutation("/fixture/brain", async () => { events.push("second"); });
    await Promise.resolve(); expect(events).toEqual(["first-start"]);
    releaseFirst(); await first; await Promise.resolve(); expect(events).toEqual(["first-start", "first-end", "snapshot-start"]);
    releaseSnapshot(); await snapshot; await second; expect(events).toEqual(["first-start", "first-end", "snapshot-start", "snapshot-end", "second"]);
  });
});
