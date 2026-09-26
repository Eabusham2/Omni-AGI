import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const source = (path: string): string =>
  readFileSync(resolve(import.meta.dirname, "..", path), "utf8");

describe("restart-safe paused dataset discovery", () => {
  it("wires authoritative disk discovery through IPC and preload", () => {
    const ipc = source("src/shared/ipc.ts");
    const preload = source("src/preload/index.ts");
    const mainIpc = source("src/main/ipc.ts");
    const ingestion = source("src/main/dataIngestion.ts");

    expect(ipc).toContain('resumable: "omni:data:resumable"');
    expect(preload).toContain("resumable: (brainId) => invoke(IPC.data.resumable, brainId)");
    expect(mainIpc).toContain("service.datasets.latestResumable(requireId(brainId))");
    expect(ingestion).toContain("async latestResumable(");
    expect(ingestion).toContain('progress.cursor.state === "running"');
  });

  it("shows and resumes the current brain's persisted manifest", () => {
    const app = source("src/renderer/src/App.tsx");

    expect(app).toContain("window.omni.data.resumable(brain.id)");
    expect(app).toContain('className="dataset-resume-note"');
    expect(app).toContain("Resume dataset");
    expect(app).toContain("manifestId: resumableManifest.manifestId");
    expect(app).toContain("epochs: resumableManifest.requestedEpochs");
    expect(app).toContain("setResumableManifest(null)");
  });
});
