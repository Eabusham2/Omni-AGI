import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { SparseJobsController, type SparseJobsSnapshot } from "../src/renderer/src/sparseSubstrateJobsController";
import { SparseSubstrateJobCard } from "../src/renderer/src/SparseSubstrateJobs";
import { sparseJobForAction } from "../src/renderer/src/sparseSubstratePresentation";
import type { SparseSubstrateJobView, SparseSubstrateUiApi, SparseSubstrateUiEvent } from "../src/shared/sparseSubstrateTransfer";
import type { ActionEvent } from "../src/shared/types";

const job = (revision = 1, state: SparseSubstrateJobView["state"] = "queued"): SparseSubstrateJobView => ({
  format: "omni-sparse-substrate-job-view", version: 1, scope: "native-sparse-assemblies", id: "job", brainId: "owner", originalBrainId: "owner",
  actionEventId: "event", requestId: "request", requestSha256: "a".repeat(64), planId: "plan", planSha256: "b".repeat(64), state,
  phase: state === "applying" ? "native-apply" : state === "queued" ? "queued" : "terminal", revision,
  createdAt: "2026-10-15T12:00:00Z", updatedAt: "2026-10-15T12:00:01Z", liveAuthorityAvailable: false
});
function apiFixture() {
  let listener: (event: SparseSubstrateUiEvent) => void = () => {};
  const unsubscribe = vi.fn();
  const api: SparseSubstrateUiApi = {
    jobs: vi.fn(async () => ({ entries: [job()], total: 1 })), job: vi.fn(async () => job()), plan: vi.fn(async () => { throw new Error("unused"); }),
    stop: vi.fn(async () => job(3, "cancelled")), onEvent: vi.fn((_brainId, next) => { listener = next; return unsubscribe; })
  };
  return { api, unsubscribe, emit: (value: SparseSubstrateJobView) => listener({ format: "omni-sparse-substrate-ui-event", version: 1, brainId: value.brainId, kind: "job", job: value, observedAt: value.updatedAt }) };
}

describe("sparse job display/controller (no native worker)", () => {
  it("keeps a newer live revision when the initial page resolves late and ignores foreign owners", async () => {
    const f = apiFixture(); let resolve!: (page: { entries: SparseSubstrateJobView[]; total: number }) => void;
    vi.mocked(f.api.jobs).mockImplementation(() => new Promise(next => { resolve = next; }));
    let current!: SparseJobsSnapshot; const controller = new SparseJobsController("owner", f.api, value => { current = value; });
    controller.start(); f.emit(job(2, "applying")); f.emit({ ...job(99), brainId: "foreign" });
    resolve({ entries: [job(1)], total: 1 }); await controller.refresh();
    expect(current.jobs).toHaveLength(1); expect(current.jobs[0]).toMatchObject({ revision: 2, state: "applying" });
    controller.stop(); expect(f.unsubscribe).toHaveBeenCalledOnce();
  });

  it("coalesces exact Stop, preserves a commit that wins the race, and drops late stopped-owner reads", async () => {
    const f = apiFixture(); let finish!: (value: SparseSubstrateJobView) => void;
    vi.mocked(f.api.stop).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    let current!: SparseJobsSnapshot; const changed = vi.fn((value: SparseJobsSnapshot) => { current = value; });
    const controller = new SparseJobsController("owner", f.api, changed); controller.start(); await controller.refresh();
    const first = controller.stopJob("job"), second = controller.stopJob("job"); expect(first).toBe(second);
    f.emit(job(4, "complete")); finish(job(3, "cancelled")); await first;
    expect(current.jobs[0]).toMatchObject({ revision: 4, state: "complete" }); expect(f.api.stop).toHaveBeenCalledOnce();
    let resolve!: (value: SparseSubstrateJobView) => void;
    vi.mocked(f.api.job).mockImplementation(() => new Promise(next => { resolve = next; }));
    const late = controller.refreshJob("other"); controller.stop(); const count = changed.mock.calls.length;
    resolve({ ...job(5), id: "other" }); await late; expect(changed).toHaveBeenCalledTimes(count);
  });

  it("shows uncertainty/stopped state truthfully, folds exact IDs and does not invent a percent", () => {
    const unknown = renderToStaticMarkup(<SparseSubstrateJobCard job={job(3, "uncertain")} />);
    expect(unknown).toContain("Outcome unconfirmed"); expect(unknown).toContain("Connections and receipt details"); expect(unknown).toContain("Plan checksum");
    expect(unknown).not.toContain("percent"); expect(unknown).not.toContain("width:"); expect(unknown).not.toContain("Approve exact action");
    expect(renderToStaticMarkup(<SparseSubstrateJobCard job={job(4, "cancelled")} />)).toContain("Previously committed work is preserved");
  });

  it("uses only actual result/main correlation for inline placement, not model-supplied job operands", () => {
    const event = { id: "event", brainId: "owner", action: { kind: "tool", toolId: "brain.substrate", action: "apply", arguments: { jobId: "invented" } } } as ActionEvent;
    expect(sparseJobForAction(event, [job()])).toBe("job"); expect(sparseJobForAction(event, [])).toBeUndefined();
    expect(sparseJobForAction(event, [{ ...job(), originalBrainId: "parent" }])).toBeUndefined();
    const result = { ...event, execution: { output: { format: "omni-sparse-substrate-job", scope: "native-sparse-assemblies", brainId: "owner", id: "real-output" } } } as ActionEvent;
    expect(sparseJobForAction(result, [])).toBe("real-output");
  });
});
