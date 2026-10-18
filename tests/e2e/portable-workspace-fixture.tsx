import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { PortableWorkspaceControls } from "../../src/renderer/src/PortableWorkspaceControls";
import type { BrainSummary, WorkingMemoryResourcePlan } from "../../src/shared/types";
import type { PortableDestinationReview, PortablePresentationSettings, PortableWorkspaceApi,
  PortableWorkspaceCreateResult, PortableWorkspaceStatus } from "../../src/shared/portableWorkspace";
import "../../src/renderer/src/styles.css";

type CreateOutcome = "success" | "cancelled" | "error";

const BRAIN_ID = "portable-fixture-brain";
const GIB = 1024 ** 3;
const PRESENTATION: PortablePresentationSettings = {
  appearance: { schemaVersion: 1, mode: "dark", palette: "aqua", layout: "glass" },
  workspace: { schemaVersion: 1, arrangement: "split", notificationsEnabled: true, profiles: [] },
  theme: null
};
const SUMMARIES = [{ id: BRAIN_ID, name: "Synthetic portable instance", preset: "micro", runtime: "cpu",
  updatedAt: "2026-10-15T12:00:00.000Z", concepts: 0, synapses: 0, activeMode: false }] as unknown as BrainSummary[];

const status = (state: PortableWorkspaceStatus["state"], patch: Partial<PortableWorkspaceStatus> = {}): PortableWorkspaceStatus => ({
  portable: true, state, workspaceId: "portable-workspace-fixture", brainId: BRAIN_ID, name: "Synthetic portable instance",
  presentation: PRESENTATION, settingsSha256: "a".repeat(64),
  reservation: { requestedPoolBytes: 96 * GIB, admittedPoolBytes: state === "ready" ? 96 * GIB : null,
    mandatoryFreeDiskBytes: 20 * GIB, kind: "logical-shared-pool", allocatedPaddingBytes: 0 }, ...patch
});
const plan = (allowed: boolean): WorkingMemoryResourcePlan => ({
  schemaVersion: 1, allowed,
  blockers: allowed ? [] : ["The selected shared pool needs 12 GiB more free space."],
  warnings: ["The x64 neural worker is emulated in this synthetic Windows ARM64 review."],
  context: { selectedTokens: 131072 },
  resources: { systemRamBudgetBytes: 24 * GIB, sharedStoragePoolBytes: 96 * GIB }
} as WorkingMemoryResourcePlan);
const review = (allowed: boolean): PortableDestinationReview => ({
  reviewId: allowed ? "fit-allowed" : "fit-blocked", reviewSha256: (allowed ? "b" : "c").repeat(64),
  workspaceId: "portable-workspace-fixture", brainId: BRAIN_ID, plan: plan(allowed),
  sourceDriveSelectionsDiscarded: true, copiedExternalAuthorityDormant: true, credentialsTransferred: false
});

function Fixture() {
  const [hostStatus, setHostStatus] = useState(() => status("preparation-required"));
  const [fitAllowed, setFitAllowed] = useState(false);
  const [createOutcome, setCreateOutcome] = useState<CreateOutcome>("success");
  const [apiEpoch, setApiEpoch] = useState(0);
  const [logs, setLogs] = useState<string[]>(["Synthetic API ready · no picker, files, preload, engine, model, or neural work"]);
  const statusRef = useRef(hostStatus), fitRef = useRef(fitAllowed), outcomeRef = useRef(createOutcome);
  statusRef.current = hostStatus; fitRef.current = fitAllowed; outcomeRef.current = createOutcome;
  const log = (value: string): void => setLogs(current => [...current.slice(-7), value]);
  const api = useMemo<PortableWorkspaceApi>(() => ({
    status: async () => structuredClone(statusRef.current),
    prepare: async () => {
      const next = status("preparing", { phase: "synthetic-copy", label: "Preparing synthetic saved files…" });
      setHostStatus(next); log("Synthetic prepare callback · no file or neural operation"); return next;
    },
    reviewDestination: async () => { log(`Synthetic fit review · ${fitRef.current ? "allowed" : "blocked"}`); return review(fitRef.current); },
    approveDestination: async input => {
      if (!fitRef.current || input.reviewId !== "fit-allowed") throw new Error("Synthetic destination remains blocked until allowed fit is reviewed.");
      const next = status("ready"); setHostStatus(next); log("Synthetic destination approval · no runtime Load"); return next;
    },
    create: async request => {
      log(`Synthetic create callback · ${outcomeRef.current} · ${request.brainId} · no picker or filesystem`);
      if (outcomeRef.current === "cancelled") return null;
      if (outcomeRef.current === "error") throw new Error("Synthetic portable creation error");
      return { destination: "C:\\Synthetic\\Omni Portable", workspaceId: "created-fixture", brainId: request.brainId,
        files: 7, bytes: 4096, destinationReviewRequired: true };
    }
  }), [apiEpoch]);
  const refreshOwner = (next: PortableWorkspaceStatus): void => { setHostStatus(next); setApiEpoch(value => value + 1); };

  return <main style={{ height: "100dvh", overflowY: "auto", maxWidth: 1000, margin: "0 auto", padding: 24, boxSizing: "border-box" }}>
    <header>
      <p className="eyebrow">No-engine portable UI fixture</p>
      <h1>Production portable workspace controls</h1>
      <p><strong>Synthetic typed API only.</strong> No picker, filesystem, preload, brain, Load, model, training, provider, or real portable folder.</p>
    </header>
    <section aria-label="Synthetic portable host controls" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBlock: 16 }}>
      <button className="button" type="button" onClick={() => refreshOwner(status("preparation-required"))}>Preparation required</button>
      <button className="button" type="button" onClick={() => refreshOwner(status("review-required"))}>Advance to destination review</button>
      <button className="button" type="button" aria-pressed={fitAllowed} onClick={() => setFitAllowed(value => !value)}>{fitAllowed ? "Fit allowed" : "Fit blocked"}</button>
      {(["success", "cancelled", "error"] as const).map(value => <button className="button" type="button" key={value}
        aria-pressed={createOutcome === value} onClick={() => setCreateOutcome(value)}>Create · {value}</button>)}
      <button className="button" type="button" onClick={() => {
        document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark";
      }}>Switch appearance</button>
    </section>
    <PortableWorkspaceControls key={apiEpoch} api={api} summaries={SUMMARIES} presentation={() => PRESENTATION}
      onCreated={(result: PortableWorkspaceCreateResult) => log(`Synthetic created callback · ${result.workspaceId} · destination review still required`)}
      onEditSettings={brainId => log(`Synthetic edit-settings callback · ${brainId} · no settings changed`)} />
    <section aria-labelledby="portable-fixture-log">
      <h2 id="portable-fixture-log">Synthetic callback log</h2>
      <ol aria-live="polite">{logs.map((entry, index) => <li key={`${index}-${entry}`}>{entry}</li>)}</ol>
      <p><strong>Filesystem writes:</strong> 0 · <strong>Neural work:</strong> 0 · <strong>Provider calls:</strong> 0</p>
    </section>
  </main>;
}

document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
