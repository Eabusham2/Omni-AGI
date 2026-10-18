import { describe, expect, it } from "vitest";
import {
  engineBuildProgressEvent,
  hostCommittedBuildProgressEvent
} from "../src/main/buildProgressEvents";

describe("host-owned build completion fence", () => {
  it("maps native-core completion to readiness 99 before a distinct durable host commit reaches 100", () => {
    const timing = {
      startedAt: "2026-10-13T12:00:00.000Z",
      updatedAt: "2026-10-13T12:01:00.000Z",
      elapsedMs: 60_000
    };
    const core = engineBuildProgressEvent({
      event: {
        type: "build-progress",
        brainId: "brain-a",
        streamId: "stream-a",
        sequence: 7,
        progress: 1,
        message: "Core complete",
        data: { phase: "complete", metrics: { recordsVisited: 12 } }
      },
      sequence: 11,
      fallbackLabel: "Building native core",
      timing
    });

    expect(core).toMatchObject({
      brainId: "brain-a",
      streamId: "stream-a",
      sequence: 11,
      phase: "readiness",
      progress: 0.99,
      elapsedMs: 60_000,
      data: { recordsVisited: 12 }
    });
    expect(core.label).toContain("finalizing documentation");

    expect(hostCommittedBuildProgressEvent({
      brainId: "brain-a",
      sequence: 12,
      timing: { ...timing, updatedAt: "2026-10-13T12:01:05.000Z", elapsedMs: 65_000 }
    })).toMatchObject({
      brainId: "brain-a",
      sequence: 12,
      phase: "complete",
      progress: 1,
      elapsedMs: 65_000,
      label: "OmniCortex ready · initialization durably committed"
    });
  });
});
