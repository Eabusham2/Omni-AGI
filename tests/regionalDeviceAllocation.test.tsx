import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { DeviceAllocationCards } from "../src/renderer/src/DeviceAllocationSelector";
import type { DeviceResourceSettings, DeviceResourceTopology } from "../src/shared/deviceResources";

it("allows verified CUDA storage separately while preserving explicit packed compute route acknowledgement", () => {
  const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-15T00:00:00Z", platform: "fixture", architecture: "fixture",
    devices: [{ id: "cuda:fixture", label: "Fixture CUDA", kind: "gpu", backend: "cuda", executionDevice: "cuda:0", memoryGroupId: "gpu:fixture",
      totalBytes: 8 * 1024 ** 3, availableBytes: 7 * 1024 ** 3, allocatedBytes: 0, reservedBytes: 0, unifiedMemory: false,
      nativeComputeSupported: false, unavailableReason: "Native CUDA packed kernel unavailable" }], links: [], measurementRequested: false, neuralModelConstructed: false, warnings: [] };
  const settings: DeviceResourceSettings = { version: 1, allocations: [{ deviceId: "cuda:fixture", enabled: true, mode: "manual", percent: 70, autoIncrease: false }],
    storageDirectories: [], allowStorageSpill: true, autoIncreaseStorage: false };
  const render = () => renderToStaticMarkup(<DeviceAllocationCards topology={topology} settings={settings} ramMode="auto" ramPercent={0} onChange={() => undefined} />);
  expect(render()).not.toContain('aria-label="Fixture CUDA allocation policy" disabled=""');
  expect(render()).toContain("Enabling storage does not authorize a compute fallback route");
  settings.regionalExecution = { version: 1, enabled: true, mode: "manual", routes: [{ regionId: "blocks.0", deviceId: "cuda:fixture",
    executionMode: "packed-cpu-tile-fallback", acknowledgePackedCpuFallback: true }] };
  const acknowledged = render();
  expect(acknowledged).not.toContain('aria-label="Fixture CUDA allocation policy" disabled=""');
  expect(acknowledged).toContain("Native CUDA packed kernel unavailable");
  expect(acknowledged).toContain("does not provide a native CUDA kernel or a measured speedup");
  settings.regionalExecution.enabled = false;
  expect(render()).not.toContain('aria-label="Fixture CUDA allocation policy" disabled=""');
  expect(topology.devices[0]!.nativeComputeSupported).toBe(false);
});
