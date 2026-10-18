import { describe, expect, it, vi } from "vitest";
import { subscribeAppObservations } from "../src/renderer/src/mcpAppObservationChannel";
import type { McpAppsApi, McpAppObservationEvent } from "../src/shared/mcpApps";

describe("read-only App observation listener sharing", () => {
  it("uses one preload listener for many App cards and tears it down after the last", () => {
    const close = vi.fn(); let emit!: (value: McpAppObservationEvent) => void;
    const onObservation = vi.fn((listener: typeof emit) => { emit = listener; return close; });
    const api: McpAppsApi = { describe: vi.fn(), open: vi.fn(), request: vi.fn(), close: vi.fn(), onObservation };
    const first = vi.fn(), second = vi.fn(), stopFirst = subscribeAppObservations(api, first), stopSecond = subscribeAppObservations(api, second);
    const event = { brainId: "brain", artifactId: "artifact", error: "Explicit Load required" }; emit(event);
    expect(first).toHaveBeenCalledWith(event); expect(second).toHaveBeenCalledWith(event); expect(onObservation).toHaveBeenCalledOnce();
    stopFirst(); expect(close).not.toHaveBeenCalled(); stopSecond(); expect(close).toHaveBeenCalledOnce();
  });
});
