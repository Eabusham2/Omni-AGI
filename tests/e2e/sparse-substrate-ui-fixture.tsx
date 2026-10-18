import React, { useState } from "react";
import type { SparseSubstrateJobView, SparseSubstratePlanView, SparseSubstrateUiApi, SparseSubstrateUiEvent } from "../../src/shared/sparseSubstrateTransfer";
import type { OmniApi, ToolExecutionResult } from "../../src/shared/types";
import { SparseSubstrateJobsPanel, SparseSubstrateJobsProvider } from "../../src/renderer/src/SparseSubstrateJobs";

// Inert transport for production components. No brain/core/model/host effect.
const owner = "sparse-ui-fixture", checksum = "e".repeat(64), listeners = new Set<(event: SparseSubstrateUiEvent) => void>();
const timestamp = "2026-10-15T12:00:00Z";
const makeJob = (id: string, state: SparseSubstrateJobView["state"], revision = 1): SparseSubstrateJobView => ({
  format: "omni-sparse-substrate-job-view", version: 1, scope: "native-sparse-assemblies", id, brainId: owner, originalBrainId: owner,
  requestId: `request-${id}`, requestSha256: checksum, planId: "plan-fixture", planSha256: checksum, state, revision,
  createdAt: timestamp, updatedAt: timestamp, phase: state === "queued" ? "queued" : state === "applying" ? "native-apply" : "terminal", liveAuthorityAvailable: false
});
const receipt = (id: string): NonNullable<SparseSubstrateJobView["receipt"]> => ({ format: "omni-sparse-substrate-applied", version: 1, scope: "native-sparse-assemblies",
  brainId: owner, jobId: id, requestSha256: checksum, planSha256: checksum, bundleSha256: checksum, selectedPreimagesSha256: checksum, changesSha256: checksum,
  appliedRows: 48, committedAt: timestamp, receiptSha256: checksum, generationSha256: checksum, duplicate: false, weightsAveraged: false, cortexParametersChanged: false, originChanged: false });
let jobs: SparseSubstrateJobView[] = [makeJob("pending-fixture", "queued"), { ...makeJob("finished-fixture", "complete", 5), receipt: receipt("finished-fixture") }];
const plan = (id: string): SparseSubstratePlanView => ({ format: "omni-sparse-substrate-plan-view", version: 1, scope: "native-sparse-assemblies", id, brainId: owner,
  bundleId: "bundle-fixture", bundleSha256: checksum, mode: "replace-exact", operation: id === "inverse-fixture" ? "rollback" : "import", sha256: checksum, bytes: 4096,
  manifestSha256: checksum, rows: 128, additions: 0, replacements: 48, identical: 80, conflicts: 0, targetGenerationSha256: checksum, selectedPreimagesSha256: checksum, changesSha256: checksum });
function publish(value: SparseSubstrateJobView): void { jobs = [...jobs.filter(job => job.id !== value.id), value];
  listeners.forEach(listener => listener({ format: "omni-sparse-substrate-ui-event", version: 1, brainId: owner, kind: "job", job: value, observedAt: timestamp })); }
const api: SparseSubstrateUiApi = {
  jobs: async () => ({ entries: structuredClone(jobs), total: jobs.length }),
  job: async (_owner, id) => { const value = jobs.find(job => job.id === id); if (!value) throw new Error("Synthetic update not found"); return structuredClone(value); },
  plan: async (_owner, id) => plan(id),
  stop: async (_owner, id) => { const value = jobs.find(job => job.id === id)!;
    const stopped = { ...value, state: "cancelled" as const, phase: "terminal" as const, revision: value.revision + 1 }; publish(stopped); return stopped; },
  onEvent: (_owner, listener) => { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
const tools: Pick<OmniApi["tool"], "execute" | "declineApproval"> = {
  execute: async request => {
    const base = { id: "synthetic-only", toolId: request.toolId, action: request.action, startedAt: timestamp };
    if (!request.approvalToken) return { ...base, state: "approval-required", approvalToken: "synthetic-exact-review" };
    if (request.action === "prepare-rollback") return { ...base, state: "complete", output: { format: "omni-sparse-substrate-plan", scope: "native-sparse-assemblies", id: "inverse-fixture" } };
    const queued = makeJob("inverse-update-fixture", "queued"); publish(queued);
    return { ...base, state: "complete", output: { ...queued, format: "omni-sparse-substrate-job" } } as ToolExecutionResult;
  },
  declineApproval: async () => true
};
export function SparseSubstrateUiFixture() {
  const [notice, setNotice] = useState("");
  return <SparseSubstrateJobsProvider brainId={owner} api={api} tools={tools}>
    <p>No native state changes here. These are synthetic job events and reviews.</p>
    <div className="automation-buttons"><button className="button" onClick={() => { const old = jobs.find(job => job.id === "pending-fixture")!;
      publish({ ...old, state: "applying", phase: "native-apply", revision: old.revision + 1 }); setNotice("Synthetic native phase event delivered."); }}>Begin synthetic phase</button>
      <button className="button" onClick={() => { const old = jobs.find(job => job.id === "pending-fixture")!;
        publish({ ...old, state: "complete", phase: "terminal", revision: old.revision + 1, receipt: receipt(old.id) }); setNotice("Synthetic receipt delivered."); }}>Finish synthetic phase</button></div>
    {notice ? <p role="status">{notice}</p> : null}
    <SparseSubstrateJobsPanel />
  </SparseSubstrateJobsProvider>;
}
