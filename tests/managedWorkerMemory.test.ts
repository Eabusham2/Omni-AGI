/** Pure launcher environment fixture; never launch a worker or probe RAM. */
import { describe, expect, it } from "vitest";
import { managedWorkerMemoryEnvironment } from "../src/main/engineSupervisor";

describe("trusted managed-family memory ownership", () => {
  it("always binds the app's actual PID, not a supplied environment owner", () => {
    const caller = { OMNI_MEMORY_OWNER_PID: "1", CUSTOM_KEY: "fixture" };
    const actual = managedWorkerMemoryEnvironment(caller);
    expect(actual.OMNI_MEMORY_OWNER_PID).toBe(String(process.pid));
    expect(actual.CUSTOM_KEY).toBe("fixture");
    expect(caller.OMNI_MEMORY_OWNER_PID).toBe("1");
  });
});
