import { EventEmitter } from "node:events";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BuildResourceSelectionStore } from "../src/main/buildResourceSelections";
import { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import type { DatasetManifest } from "../src/shared/types";

describe("initial Build resource persistence", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  it("restores renderer-safe selections and persists the brain/job/manifest lifecycle", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-build-selection-"));
    roots.push(root);
    const source = join(root, "dataset");
    await mkdir(source);
    const index = join(root, "state", "pending-build-resources.json");
    const createdAt = new Date().toISOString();
    const first = new BuildResourceSelectionStore(index);
    await first.put({
      id: "selection-a",
      kind: "folder",
      label: "dataset",
      itemCount: 1,
      paths: [source],
      createdAt,
      updatedAt: createdAt,
      state: "selected"
    });

    const afterRendererReload = new BuildResourceSelectionStore(index);
    await expect(afterRendererReload.list()).resolves.toEqual([
      {
        id: "selection-a",
        kind: "folder",
        label: "dataset",
        itemCount: 1
      }
    ]);
    expect(JSON.stringify(await afterRendererReload.list())).not.toContain(source);

    await afterRendererReload.claim("selection-a", "brain-a");
    await afterRendererReload.attachRuntimeJob("selection-a", "job-a");
    await afterRendererReload.commitManifest("selection-a", "manifest-a");
    await afterRendererReload.complete("selection-a");

    const afterMainRestart = new BuildResourceSelectionStore(index);
    await expect(afterMainRestart.list()).resolves.toEqual([]);
    await expect(afterMainRestart.listTasks("brain-a")).resolves.toMatchObject([
      {
        id: "selection-a",
        brainId: "brain-a",
        runtimeJobId: "job-a",
        manifestId: "manifest-a",
        state: "complete",
        paths: [source]
      }
    ]);
  });

  it("preserves retry material while clearing failed process-local job ownership", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-build-selection-retry-"));
    roots.push(root);
    const source = join(root, "validation.parquet");
    const index = join(root, "state", "pending-build-resources.json");
    const createdAt = new Date().toISOString();
    const store = new BuildResourceSelectionStore(index);
    await store.put({
      id: "selection-retry",
      kind: "files",
      label: "validation.parquet",
      itemCount: 1,
      paths: [source],
      createdAt,
      updatedAt: createdAt,
      state: "selected"
    });
    await store.claim("selection-retry", "brain-a");
    await store.commitManifest("selection-retry", "manifest-a");
    await store.attachRuntimeJob("selection-retry", "stale-job");

    await store.releaseForRetry("selection-retry", "brain-a");

    const restarted = new BuildResourceSelectionStore(index);
    await expect(restarted.get("selection-retry")).resolves.toMatchObject({
      state: "retryable",
      brainId: "brain-a",
      manifestId: "manifest-a",
      paths: [source]
    });
    expect((await restarted.get("selection-retry"))?.runtimeJobId).toBeUndefined();
    await restarted.attachRuntimeJob("selection-retry", "replacement-job");
    await expect(restarted.get("selection-retry")).resolves.toMatchObject({
      state: "training",
      runtimeJobId: "replacement-job",
      manifestId: "manifest-a"
    });
  });

  it("returns an observable job before hashing and transitions into training only after commit", async () => {
    let releaseManifest = (): void => undefined;
    const manifestReady = new Promise<void>((resolve) => {
      releaseManifest = resolve;
    });
    const manifest: DatasetManifest = {
      schemaVersion: 1,
      id: "manifest-a",
      brainId: "brain-a",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      roots: ["/dataset"],
      entryFile: "entries.ndjson",
      discoveredFiles: 3,
      discoveredBytes: 12_345,
      manifestHash: "a".repeat(64)
    };
    const order: string[] = [];
    const service = {
      previewDataset: vi.fn(async (
        _brainId: string,
        _paths: string[],
        options: { onProgress?: (value: Record<string, unknown>) => void }
      ) => {
        options.onProgress?.({
          phase: "hashing",
          discoveredFiles: 1,
          discoveredBytes: 12_345,
          hashedFiles: 0,
          hashedBytes: 8_192,
          currentFile: "dataset/first.parquet",
          currentFileBytes: 12_345,
          currentFileHashedBytes: 8_192
        });
        order.push("hashing");
        await manifestReady;
        order.push("manifest");
        return manifest;
      }),
      ingestManifest: vi.fn(async () => {
        order.push("training");
        return {
          manifest,
          coverage: {
            schemaVersion: 1,
            manifestId: manifest.id,
            discoveredFiles: 3,
            processedFiles: 3,
            rejectedFiles: 0,
            discoveredRecords: 7,
            processedRecords: 7,
            rejectedRecords: 0,
            discoveredBytes: 12_345,
            processedBytes: 12_345,
            shards: 1,
            modalityCounts: { parquet: 7 },
            errors: [],
            complete: true,
            updatedAt: new Date().toISOString()
          },
          results: [],
          paused: false
        };
      })
    } as unknown as BrainService;
    const engine = Object.assign(new EventEmitter(), {
      interruptAndRestart: vi.fn()
    }) as unknown as EngineSupervisor;
    const jobs = new RuntimeJobManager(service, engine);
    const events: string[] = [];
    jobs.on("event", ({ job }) => events.push(`${job.state}:${job.label}`));

    const job = jobs.startBuildResource(
      {
        brainId: "brain-a",
        selectionId: "selection-a",
        policy: "pretrain"
      },
      ["/dataset"],
      async (manifestId) => {
        expect(manifestId).toBe(manifest.id);
        order.push("committed");
      },
      async () => {
        order.push("complete");
      }
    );

    expect(job.kind).toBe("ingestion");
    expect(["queued", "running"]).toContain(job.state);
    expect(service.ingestManifest).not.toHaveBeenCalled();
    expect(jobs.list("brain-a")[0]?.output).toMatchObject({
      phase: "snapshot",
      preview: { hashedBytes: 8_192 }
    });

    releaseManifest();
    await expect(jobs.wait(job.id)).resolves.toMatchObject({ state: "complete" });
    expect(order).toEqual(["hashing", "manifest", "committed", "training", "complete"]);
    expect(events.some((value) => value.includes("Hashing initial source"))).toBe(true);
  });

  it("cancels snapshot hashing cooperatively without restarting unrelated neural work", async () => {
    const service = {
      previewDataset: vi.fn((
        _brainId: string,
        _paths: string[],
        options: { signal?: AbortSignal }
      ) => new Promise<never>((_resolve, reject) => {
        options.signal?.addEventListener("abort", () => {
          const error = new Error("cancelled");
          error.name = "AbortError";
          reject(error);
        }, { once: true });
      })),
      ingestManifest: vi.fn()
    } as unknown as BrainService;
    const engine = Object.assign(new EventEmitter(), {
      interruptAndRestart: vi.fn()
    }) as unknown as EngineSupervisor;
    const jobs = new RuntimeJobManager(service, engine);
    const job = jobs.startBuildResource(
      { brainId: "brain-a", selectionId: "selection-a" },
      ["/dataset"]
    );

    await expect(jobs.cancel(job.id)).resolves.toMatchObject({ state: "cancelled" });
    expect(engine.interruptAndRestart).not.toHaveBeenCalled();
    expect(service.ingestManifest).not.toHaveBeenCalled();
  });

  it("fails closed when resumable whole-dataset learning pauses before full coverage", async () => {
    const manifest: DatasetManifest = {
      schemaVersion: 1,
      id: "manifest-paused",
      brainId: "brain-paused",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      roots: ["/dataset"],
      entryFile: "entries.ndjson",
      discoveredFiles: 1,
      discoveredBytes: 12_345,
      manifestHash: "b".repeat(64)
    };
    const pauseReason = "Paused before /dataset/part.parquet: worker lost";
    const onCompleted = vi.fn(async () => undefined);
    const service = {
      previewDataset: vi.fn(async () => manifest),
      ingestManifest: vi.fn(async () => ({
        manifest,
        coverage: {
          complete: false,
          discoveredFiles: 1,
          processedFiles: 0
        },
        results: [],
        paused: true,
        pauseReason
      }))
    } as unknown as BrainService;
    const engine = Object.assign(new EventEmitter(), {
      interruptAndRestart: vi.fn()
    }) as unknown as EngineSupervisor;
    const jobs = new RuntimeJobManager(service, engine);

    const job = jobs.startBuildResource(
      { brainId: "brain-paused", selectionId: "selection-paused" },
      ["/dataset"],
      async () => undefined,
      onCompleted
    );

    await expect(jobs.wait(job.id)).resolves.toMatchObject({
      state: "failed",
      progress: expect.any(Number),
      error: pauseReason,
      output: { paused: true, pauseReason }
    });
    expect(jobs.list("brain-paused")[0]!.progress).toBeLessThan(1);
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it.each([
    ["missing", undefined, false],
    ["ambiguous", {}, false],
    ["missing with the entire worker output", undefined, true]
  ])("fails closed when the completion receipt is %s", async (_label, coverage, missingOutput) => {
    const manifest: DatasetManifest = {
      schemaVersion: 1,
      id: "manifest-no-receipt",
      brainId: "brain-no-receipt",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      roots: ["/dataset"],
      entryFile: "entries.ndjson",
      discoveredFiles: 1,
      discoveredBytes: 10,
      manifestHash: "c".repeat(64)
    };
    const service = {
      previewDataset: vi.fn(async () => manifest),
      ingestManifest: vi.fn(async () => missingOutput
        ? undefined
        : {
            manifest,
            ...(coverage === undefined ? {} : { coverage }),
            results: [],
            paused: false
          })
    } as unknown as BrainService;
    const jobs = new RuntimeJobManager(
      service,
      Object.assign(new EventEmitter(), {
        interruptAndRestart: vi.fn()
      }) as unknown as EngineSupervisor
    );

    const job = jobs.startBuildResource(
      { brainId: "brain-no-receipt", selectionId: "selection-no-receipt" },
      ["/dataset"]
    );

    await expect(jobs.wait(job.id)).resolves.toMatchObject({
      state: "failed",
      error: missingOutput
        ? "Dataset learning ended without a complete manifest coverage receipt."
        : "Dataset learning ended before complete manifest coverage was committed."
    });
    expect(jobs.list("brain-no-receipt")[0]!.progress).toBeLessThan(1);
  });
});
