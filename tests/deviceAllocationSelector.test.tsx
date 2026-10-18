import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { DeviceAllocationCards, deviceAllocationChoice, deviceAvailability, synchronizeDeviceRam } from "../src/renderer/src/DeviceAllocationSelector";
import { DEFAULT_DEVICE_RESOURCE_SETTINGS, type DeviceResourceSettings } from "../src/shared/deviceResources";
import { devicePlanFixture, deviceTopologyFixture } from "./fixtures/deviceResourceTopology";

describe("per-device allocation review", () => {
  it("retains one RAM policy without mutating preferences or creating authority", () => {
    const settings: DeviceResourceSettings = { ...DEFAULT_DEVICE_RESOURCE_SETTINGS, allocations: [{ deviceId: "ram:system", enabled: true, mode: "manual", percent: 90, autoIncrease: false }], storageDirectories: ["/imported/untrusted/path"] };
    const updated = synchronizeDeviceRam(settings, "manual", 45)!;
    expect(updated.allocations[0]?.percent).toBe(45); expect(settings.allocations[0]?.percent).toBe(90);
    expect(updated.storageDirectories).toEqual(settings.storageDirectories);
    expect(Object.keys(updated)).not.toContain("selectionId");
    expect(synchronizeDeviceRam(undefined, "auto", 65)).toBeUndefined();
    const auto = { ...settings, allocations: [{ ...settings.allocations[0]!, mode: "auto" as const, percent: 60 }] };
    expect(synchronizeDeviceRam(auto, "auto", 65)).toEqual(auto); // merely opening a page cannot create a dirty override
    expect(deviceAllocationChoice(deviceTopologyFixture().devices[0]!, undefined, "manual", 55).percent).toBe(55);
  });
  it("disables unavailable execution and unverified/read-only storage without inventing capacities", () => {
    const topology = deviceTopologyFixture();
    expect(deviceAvailability(topology.devices[3]!)).toBe("Native packed kernels are not available.");
    expect(deviceAllocationChoice(topology.devices[3]!, undefined, "auto", 65).enabled).toBe(false);
    expect(deviceAvailability({ ...topology.devices[4]!, readOnly: true })).toContain("read-only");
    expect(deviceAvailability({ ...topology.devices[4]!, filesystemDeviceId: undefined })).toContain("not verified");
  });
  it("labels actual shared pools, missing measurements, per-device controls and transfer links", () => {
    const markup = renderToStaticMarkup(<DeviceAllocationCards topology={deviceTopologyFixture()} ramMode="auto" ramPercent={65} onChange={() => undefined} />);
    expect(markup).toContain("Same physical pool as Unified GPU"); expect(markup).toContain("These bytes are not added together");
    expect(markup).toContain("Not measured"); expect(markup).toContain("Not reported");
    expect(markup).toContain('aria-label="Discrete GPU allocation percent"');
    expect(markup).toContain('aria-label="Allow automatic increases for Discrete GPU"');
    expect(markup).toMatch(/aria-label="Enable Unavailable NPU"[^>]*disabled/);
    expect(markup).toContain("Ceiling follows the shared Auto RAM policy");
    expect(markup).not.toContain('aria-label="System RAM allocation percent"');
    expect(markup).toContain("Measured transfer links"); expect(markup).toContain("8.0 GiB/s");
    expect(markup).toContain("Default checkpoint/spill volume; select this folder to customize its spill budget.");
    expect(markup).not.toContain('aria-label="Enable Primary drive"');
    const selected = renderToStaticMarkup(<DeviceAllocationCards topology={deviceTopologyFixture(true)}
      settings={{ ...DEFAULT_DEVICE_RESOURCE_SETTINGS, allocations: [], storageDirectories: ["/fixture/chosen"] }}
      ramMode="auto" ramPercent={65} onChange={() => undefined} />);
    expect(selected).toContain('aria-label="Enable Chosen drive"');
    expect(selected).toContain('aria-label="Chosen drive allocation percent"');
  });
  it("mounts in Build and runtime while passing picker proof only through the transient request", async () => {
    const app = await readFile(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    expect(app.match(/<DeviceAllocationSelector/g)).toHaveLength(2);
    expect(app).toContain("spillPoolSelectionId: extras.spillPoolSelectionId");
    expect(app).toContain("deviceResources: effectiveDeviceResources");
    expect(app).toContain("brainId={brain.id} settings={effectiveDeviceResources}");
    expect(app).toContain("synchronizeDeviceRam(deviceResources, systemRamMode, systemRamSharePercent)");
    expect(app.indexOf("<BestWorkSettings")).toBeLessThan(app.indexOf('id="active-mode-heading"'));
  });
  it("uses the applied shared-volume quota instead of diagnostic free-space projections", () => {
    const settings = { ...DEFAULT_DEVICE_RESOURCE_SETTINGS, allocations: [], storageDirectories: ["/fixture/chosen"] };
    const render = (plan = devicePlanFixture()) => renderToStaticMarkup(<DeviceAllocationCards topology={deviceTopologyFixture(true)}
      settings={settings} ramMode="auto" ramPercent={65} plan={plan} onChange={() => undefined} />);
    const markup = render();
    const chosen = markup.match(/<article[^>]*aria-label="Chosen drive allocation"[^>]*>(.*?)<\/article>/)?.[1] ?? "";
    const primary = markup.match(/<article[^>]*aria-label="Primary drive allocation"[^>]*>(.*?)<\/article>/)?.[1] ?? "";
    expect(chosen).toContain("Shared volume ceiling</dt><dd>48.0 GiB");
    expect(chosen).not.toContain("96.0 GiB");
    expect(chosen.split("<details>")[0]).not.toContain("Charged storage");
    expect(chosen).toContain("Charged storage</dt><dd>2.0 GiB");
    expect(chosen).toContain("Pending storage</dt><dd>1.0 GiB");
    expect(chosen).toContain("Unmaterialized sparse promises</dt><dd>4.0 GiB");
    expect(primary).not.toContain("Shared volume ceiling");
    expect(primary).not.toContain("Charged storage");
    const unavailable = render({ ...devicePlanFixture(), appliedStorage: undefined });
    expect(unavailable).toContain("Shared volume ceiling</dt><dd>Not reported");
    expect(unavailable).not.toContain("96.0 GiB");
  });
});
