import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

describe("initialization recovery surface", () => {
  it("keeps non-ready minds out of chat and exposes the typed retry bridge", async () => {
    const [app, presentation, styles, main, preload, ipc] = await Promise.all([
      readFile("src/renderer/src/App.tsx", "utf8"),
      readFile("src/renderer/src/initializationRecoveryPresentation.ts", "utf8"),
      readFile("src/renderer/src/styles.css", "utf8"),
      readFile("src/main/index.ts", "utf8"),
      readFile("src/preload/index.ts", "utf8"),
      readFile("src/shared/ipc.ts", "utf8")
    ]);

    expect(presentation).toContain('readiness.state === "initializing"');
    expect(app).toContain('activeBrain?.readiness.state === "failed"');
    expect(app).toContain("initializationIsRunning(");
    expect(app).toContain("initializationRunningPresentation(activeBrain.id, buildProgress)");
    expect(app).toContain(") : initializationRunning ? (");
    expect(app).toContain("initializationRunning!.detail");
    expect(app).toContain("buildProgress && !initializationRunningActive");
    expect(app).toContain("initialMaterialsPresentation.detail");
    expect(app).not.toContain('buildProgress.data?.recordsVisited ?? "…"');
    expect(app).not.toContain('buildProgress.data?.neuronDelta?.toLocaleString() ?? "…"');
    expect(app).toContain('role="progressbar"');
    expect(app).toContain("initializationRecovery!.actionLabel");
    expect(presentation).toContain("export function initializationIsRunning(");
    expect(presentation).toContain("export function activeInitializationTrainingJob(");
    expect(presentation).toContain("jobProgress?.committedRecords");
    expect(presentation).toContain('title: "Learning paused"');
    expect(presentation).toContain('actionLabel: "Resume training"');
    expect(presentation).toContain('title: foundationFailure ? "Setup failed" : "Initial learning failed"');
    expect(presentation).toContain("The OmniCortex native core could not be completed.");
    expect(presentation).not.toContain("ground-up neural core");
    expect(app).toContain("window.omni.brain.retryInitialization(activeBrain.id)");
    expect(app).toContain('stopActiveInitialization("pause")');
    expect(app).toContain('stopActiveInitialization("cancel")');
    expect(app).toContain("await window.omni.data.pause(job.id)");
    expect(app).toContain("await window.omni.data.cancel(job.id)");
    expect(app).toContain("Confirm cancel training");
    expect(app).toContain('aria-modal="true"');
    expect(app).toContain("Nothing is deleted.");
    const runningStart = app.indexOf(") : initializationRunning ? (");
    const recoveryStart = app.indexOf(") : activeBrain?.readiness.state === \"failed\" ? (", runningStart);
    expect(runningStart).toBeGreaterThan(-1);
    expect(recoveryStart).toBeGreaterThan(runningStart);
    const runningSurface = app.slice(runningStart, recoveryStart);
    expect(runningSurface).toContain("Pause training");
    expect(runningSurface).toContain("Cancel training");
    expect(runningSurface).not.toContain("Back to library");
    expect(styles).toContain(".initialization-cancel-dialog");
    expect(styles).toContain("@media (max-width: 520px)");
    expect(main).toContain("initialization.recoverAll(");
    expect(main).toContain("initialLearningBuildProgressEvent(");
    expect(preload).toContain("invoke(IPC.brain.retryInitialization, id)");
    expect(preload).toContain("invoke(IPC.data.pause, jobId)");
    expect(preload).toContain("invoke(IPC.data.cancel, jobId)");
    expect(ipc).toContain('retryInitialization: "omni:brain:retry-initialization"');
    expect(ipc).toContain('pause: "omni:data:pause"');
    expect(ipc).toContain('cancel: "omni:data:cancel"');
  });
});
