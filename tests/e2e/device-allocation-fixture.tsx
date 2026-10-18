import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { DeviceAllocationSelector, synchronizeDeviceRam } from "../../src/renderer/src/DeviceAllocationSelector";
import type { DeviceResourceSettings } from "../../src/shared/deviceResources";
import { devicePlanFixture, deviceTopologyFixture } from "../fixtures/deviceResourceTopology";
import "../../src/renderer/src/styles.css";
const calls: Array<{ kind: string; input: unknown }> = [];
const plan = devicePlanFixture();
Object.assign(window, { __deviceCalls: calls });
const catalog = {
  deviceTopology: async (request?: { measure?: boolean; storageDirectories?: string[] }) => {
    calls.push({ kind: "topology", input: request });
    const topology = deviceTopologyFixture(Boolean(request?.storageDirectories?.includes("/fixture/chosen")));
    return { ...topology, measurementRequested: request?.measure ?? false };
  },
  selectSpillPools: async (brainId?: string) => { calls.push({ kind: "picker", input: brainId ?? null }); return { selectionId: "native-picker-only", directories: ["/fixture/chosen"] }; }
};
function Fixture() {
  const [runtime, setRuntime] = useState(false);
  const [settings, setSettings] = useState<DeviceResourceSettings>();
  const [ramMode, setRamMode] = useState<"auto" | "manual">("auto");
  const [ramPercent, setRamPercent] = useState(65), [token, setToken] = useState<string>();
  const effective = useMemo(() => synchronizeDeviceRam(settings, ramMode, ramPercent), [settings, ramMode, ramPercent]);
  Object.assign(window, { __deviceReviewed: { settings: effective, selectionId: token, ramMode, ramPercent } });
  return <main style={{ maxWidth: 1150, margin: "0 auto", padding: 24 }}>
    <p>Per-device synthetic UI fixture. No brain is constructed or trained.</p>
    <button type="button" className="button" onClick={() => { setRuntime((value) => !value); setToken(undefined); }}>Switch to {runtime ? "Build" : "runtime"}</button>
    <label style={{ display: "block", margin: "16px 0" }}>Existing RAM percentage<input aria-label="Existing RAM percentage" type="range" min={30} max={100} value={ramPercent}
      onChange={(event) => { setRamMode("manual"); setRamPercent(Number(event.target.value)); }} /></label>
    <DeviceAllocationSelector catalog={catalog} brainId={runtime ? "existing-brain" : undefined} settings={effective} ramMode={ramMode} ramPercent={ramPercent} plan={plan}
      onChange={(next, selectionId) => { setSettings(next); const ram = next.allocations.find((entry) => entry.deviceId === "ram:system");
        if (ram) { setRamMode(ram.mode); setRamPercent(ram.percent); } if (selectionId) setToken(selectionId); }} />
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
