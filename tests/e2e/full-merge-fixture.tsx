import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { FullMergePanel } from "../../src/renderer/src/FullMergePanel";
import { DEFAULT_CONFIG, type BrainSummary, type OmniApi } from "../../src/shared/types";
import type { FullMergeProgress, FullMergeStartRequest } from "../../src/shared/fullMerge";
import "../../src/renderer/src/styles.css";

const parents: BrainSummary[] = ["Aster", "Lumen"].map((name, index) => ({ id: `parent-${index}`, name,
  preset: "whole-brain", runtime: "adaptive-core", updatedAt: "2026-10-07T00:00:00Z", concepts: 4, synapses: 8,
  activeMode: false, generation: 0, rootId: `parent-${index}` }));
let operations: FullMergeProgress[] = JSON.parse(localStorage.getItem("merge-fixture-operations") ?? "[]");
const listeners = new Set<(value: FullMergeProgress) => void>();
const waiting = new Map<string, (reason: Error) => void>();
const requests: FullMergeStartRequest[] = [];
Object.assign(window, { __fullMergeRequests: requests });
const summaries = () => [...parents, ...operations.filter((value) => value.state === "complete").map((value) => ({ ...parents[0]!, id: value.targetBrainId, name: value.name! }))];
const publish = (value: FullMergeProgress) => {
  operations = [value, ...operations.filter((item) => item.operationId !== value.operationId)];
  localStorage.setItem("merge-fixture-operations", JSON.stringify(operations));
  for (const listener of listeners) listener(value);
  return value;
};
const find = (id: string) => operations.find((value) => value.operationId === id)!;
const api = { brain: {
  list: async () => summaries(), listFullMerges: async () => operations,
  fullMergeStatus: async (id: string) => find(id),
  onFullMerge: (listener: (value: FullMergeProgress) => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; },
  get: async (id: string) => ({ id, config: { ...DEFAULT_CONFIG, name: "Parent settings" }, toolPermissions: [] }),
  startFullMerge: (request: FullMergeStartRequest) => {
    requests.push(request);
    publish({ operationId: request.operationId, targetBrainId: `merged-${request.operationId}`, name: request.name, state: "copying",
      filesDiscovered: 8, filesCopied: 3, logicalBytes: 8192, copiedBytes: 3072, uniqueBlobBytes: 2048,
      parents: parents.map((parent) => ({ brainId: parent.id, name: parent.name, generationSha256: "a".repeat(64), ancestry: [] })),
      reviewedSettings: request.config, reviewedAccess: request.accessPolicy, hardwareTier: request.hardwareTier });
    return new Promise((_resolve, reject) => { waiting.set(request.operationId, reject); });
  },
  pauseFullMerge: async (id: string) => { const value = publish({ ...find(id), state: "paused" }); waiting.get(id)?.(new Error("Paused at the fixture safe boundary.")); waiting.delete(id); return value; },
  resumeFullMerge: async (id: string) => { publish({ ...find(id), state: "verifying", filesCopied: 8, copiedBytes: 8192 }); await new Promise((resolve) => setTimeout(resolve, 200)); return publish({ ...find(id), state: "complete" }); },
  cancelFullMerge: async (id: string) => { const value = publish({ ...find(id), state: "cancelled" }); waiting.get(id)?.(new Error("Stopped and discarded incomplete fixture output.")); waiting.delete(id); return value; }
}, catalog: { selectSpillPools: async () => ({ selectionId: "fixture-native-picker", directories: ["/fixture/selected-drive"] }) } } as unknown as OmniApi;
function Fixture() {
  const [brains, setBrains] = useState(summaries);
  return <main style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
    <p>Synthetic UI transport fixture — no model is loaded.</p>
    <div role="status" data-testid="parent-count">{parents.length} original parents preserved · {brains.length} library instances</div>
    <FullMergePanel summaries={brains} api={api} onPublished={() => setBrains(summaries())} onOpen={() => undefined} />
  </main>;
}
document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
