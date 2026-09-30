import { createServer, type RequestListener, type Server } from "node:http";
import { EventEmitter } from "node:events";
import {
  appendFile,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  realpath,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { performance } from "node:perf_hooks";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  CrawlFrontierStore,
  DatasetManifestStore,
  detectDatasetFormat,
  streamUtf8Text,
} from "../src/main/dataIngestion";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import {
  EngineRequestError,
  type EngineSupervisor,
} from "../src/main/engineSupervisor";
import {
  DEFAULT_CONFIG,
  type DatasetManifestEntry,
} from "../src/shared/types";

describe("whole-dataset persistence", { timeout: 30_000 }, () => {
  let root: string;
  let brainDirectory: string;
  let servers: Server[];
  let priorAllowLocal: string | undefined;
  let priorPacing: string | undefined;
  let priorBackoff: string | undefined;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-dataset-test-"));
    brainDirectory = join(root, "brain");
    await mkdir(brainDirectory, { recursive: true });
    servers = [];
    priorAllowLocal = process.env.OMNI_ALLOW_LOCAL_URLS;
    priorPacing = process.env.OMNI_CRAWL_MIN_DELAY_MS;
    priorBackoff = process.env.OMNI_CRAWL_BACKOFF_BASE_MS;
  });

  afterEach(async () => {
    await Promise.all(
      servers.map(
        (server) =>
          new Promise<void>((resolveClose) => {
            server.close(() => resolveClose());
            server.closeAllConnections();
          }),
      ),
    );
    if (priorAllowLocal === undefined) delete process.env.OMNI_ALLOW_LOCAL_URLS;
    else process.env.OMNI_ALLOW_LOCAL_URLS = priorAllowLocal;
    if (priorPacing === undefined) delete process.env.OMNI_CRAWL_MIN_DELAY_MS;
    else process.env.OMNI_CRAWL_MIN_DELAY_MS = priorPacing;
    if (priorBackoff === undefined)
      delete process.env.OMNI_CRAWL_BACKOFF_BASE_MS;
    else process.env.OMNI_CRAWL_BACKOFF_BASE_MS = priorBackoff;
    await rm(root, { recursive: true, force: true });
  });

  async function listen(handler: RequestListener): Promise<string> {
    const server = createServer(handler);
    servers.push(server);
    await new Promise<void>((resolveListen, rejectListen) => {
      server.once("error", rejectListen);
      server.listen(0, "127.0.0.1", () => {
        server.off("error", rejectListen);
        resolveListen();
      });
    });
    const address = server.address() as AddressInfo;
    return `http://127.0.0.1:${address.port}`;
  }

  async function crawlerService(name: string): Promise<{
    brainId: string;
    service: BrainService;
    engineRequests: Array<Record<string, unknown>>;
  }> {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name });
    const engineRequests: Array<Record<string, unknown>> = [];
    const neuralIngest = async (
      _method: string,
      params: Record<string, unknown>,
    ) => {
      engineRequests.push(params);
      return {
        source: {
          learned_ideas: 1,
          learned_concepts: 1,
          plasticity_events: 1,
        },
      };
    };
    const service = new BrainService(repository, {
      tryRequest: neuralIngest,
      request: neuralIngest,
    } as unknown as EngineSupervisor);
    return { brainId: brain.id, service, engineRequests };
  }

  const externalAuditRoot = process.env.OMNI_LARGE_DATASET_ROOT;
  if (externalAuditRoot) {
    it("commits an exhaustive external-folder manifest without beginning training", async () => {
      const store = new DatasetManifestStore(() => brainDirectory);
      const manifest = await store.create("brain-a", [externalAuditRoot]);
      let entries = 0;
      let explicitRejections = 0;
      const formats = new Map<string, number>();
      for await (const entry of store.entries("brain-a", manifest.id)) {
        entries += 1;
        if (entry.rejection) explicitRejections += 1;
        formats.set(entry.format, (formats.get(entry.format) ?? 0) + 1);
      }
      expect(entries).toBe(manifest.discoveredFiles);
      expect(entries).toBeGreaterThan(2_000);
      expect(formats.get("jsonl")).toBeGreaterThanOrEqual(2);
      expect(formats.get("parquet")).toBeGreaterThanOrEqual(1);
      expect(explicitRejections).toBeGreaterThan(0);
    }, 180_000);
  }

  it("classifies every stable Office, OpenDocument, and compressed upload format", () => {
    for (const extension of ["docx", "pptx", "xlsx", "odt", "ods", "odp"]) {
      expect(detectDatasetFormat(`fixture.${extension}`)).toBe("office");
    }
    for (const extension of ["gz", "bz2", "xz"]) {
      expect(detectDatasetFormat(`corpus.txt.${extension}`)).toBe("archive");
    }
  });

  it("walks beyond the former 2,000-file ceiling and persists an exhaustive cursor", async () => {
    const dataset = join(root, "many-files");
    await mkdir(dataset);
    for (let index = 0; index < 2_005; index += 1) {
      await writeFile(
        join(dataset, `${String(index).padStart(4, "0")}.txt`),
        `row ${index}`,
      );
    }
    const store = new DatasetManifestStore(() => brainDirectory);
    const manifest = await store.create("brain-a", [dataset]);
    expect(manifest.discoveredFiles).toBe(2_005);
    expect(manifest.discoveredBytes).toBeGreaterThan(2_005);

    let visited = 0;
    for await (const entry of store.entries("brain-a", manifest.id)) {
      expect(entry.index).toBe(visited);
      if (visited === 0) expect(entry.relativePath).toBe("0000.txt");
      visited += 1;
    }
    expect(visited).toBe(2_005);

    const cursor = await store.cursor("brain-a", manifest.id);
    const coverage = await store.coverage("brain-a", manifest.id);
    cursor.nextEntry = 1_337;
    cursor.processedFiles = 1_337;
    cursor.state = "paused";
    coverage.processedFiles = 1_337;
    await store.saveProgress("brain-a", cursor, coverage);
    await expect(store.cursor("brain-a", manifest.id)).resolves.toMatchObject({
      nextEntry: 1_337,
      processedFiles: 1_337,
      state: "paused",
    });
  }, 90_000);

  it("reads only the checksummed progress generation when legacy mirrors disagree", async () => {
    const path = join(root, "authoritative.txt");
    await writeFile(path, "authoritative progress fixture");
    const store = new DatasetManifestStore(() => brainDirectory);
    const manifest = await store.create("brain-a", [path]);
    const cursor = await store.cursor("brain-a", manifest.id);
    const coverage = await store.coverage("brain-a", manifest.id);
    cursor.state = "paused";
    const committed = await store.saveProgress("brain-a", cursor, coverage);
    const directory = join(brainDirectory, "datasets", manifest.id);

    await writeFile(join(directory, "cursor.json"), "{not-json");
    await writeFile(
      join(directory, "coverage.json"),
      JSON.stringify({ complete: true }),
    );

    await expect(store.progress("brain-a", manifest.id)).resolves.toMatchObject(
      {
        generation: committed.generation,
        contentSha256: committed.contentSha256,
        cursor: { state: "paused", nextEntry: 0 },
        coverage: { complete: false, processedFiles: 0 },
      },
    );
  });

  it("returns the latest checksum-bound completed manifest coverage without inventing an active job", async () => {
    const path = join(root, "completed-records.jsonl");
    await writeFile(path, '{"text":"fixture"}\n');
    const store = new DatasetManifestStore(() => brainDirectory);
    const manifest = await store.create("brain-a", [path]);
    const cursor = await store.cursor("brain-a", manifest.id);
    const coverage = await store.coverage("brain-a", manifest.id);
    Object.assign(cursor, {
      currentEpoch: 1,
      nextEntry: 0,
      nextRecord: 0,
      processedFiles: 1,
      processedRecords: 21_990,
      processedBytes: manifest.discoveredBytes,
      state: "complete"
    });
    Object.assign(coverage, {
      completedEpochs: 1,
      discoveredRecords: 21_990,
      processedFiles: 1,
      processedRecords: 21_990,
      processedBytes: manifest.discoveredBytes,
      complete: true,
      modalityCounts: { text: 21_990 }
    });
    const committed = await store.saveProgress("brain-a", cursor, coverage);

    await expect(store.latestCompletedCoverage("brain-a")).resolves.toEqual({
      brainId: "brain-a",
      manifestId: manifest.id,
      manifestHash: manifest.manifestHash,
      source: "validated-dataset-progress",
      discoveredFiles: 1,
      processedFiles: 1,
      rejectedFiles: 0,
      discoveredRecords: 21_990,
      processedRecords: 21_990,
      rejectedRecords: 0,
      complete: true,
      updatedAt: committed.coverage.updatedAt
    });

    await writeFile(
      join(brainDirectory, "datasets", manifest.id, "entries.ndjson"),
      '{"tampered":true}\n',
      { flag: "a" }
    );
    await expect(
      store.latestCompletedCoverage("brain-a")
    ).rejects.toThrow(/entry checksum failed/i);
  });

  it("ignores reserved crawl/cache directories but rejects a broken manifest directory", async () => {
    const store = new DatasetManifestStore(() => brainDirectory);
    await mkdir(join(brainDirectory, "datasets", "crawls"), { recursive: true });
    await mkdir(join(brainDirectory, "datasets", "web-cache"), { recursive: true });
    await mkdir(join(brainDirectory, "datasets", "web-spool"), { recursive: true });
    await expect(store.latestCompletedCoverage("brain-a")).resolves.toBeUndefined();

    await mkdir(join(brainDirectory, "datasets", "broken-manifest"), {
      recursive: true
    });
    await expect(store.latestCompletedCoverage("brain-a")).rejects.toMatchObject({
      code: "ENOENT"
    });
  });

  it("rediscovers a paused manifest from its authoritative progress after restart", async () => {
    const path = join(root, "restart-paused.jsonl");
    await writeFile(path, '{"text":"resume me"}\n');
    const store = new DatasetManifestStore(() => brainDirectory);
    const manifest = await store.create("brain-a", [path]);
    const progress = await store.progress("brain-a", manifest.id);
    await store.saveCursor("brain-a", {
      ...progress.cursor,
      state: "paused",
      updatedAt: new Date(Date.parse(progress.updatedAt) + 1_000).toISOString(),
    });

    const restarted = new DatasetManifestStore(() => brainDirectory);
    await expect(restarted.latestResumable("brain-a")).resolves.toMatchObject({
      brainId: "brain-a",
      manifestId: manifest.id,
      manifestHash: manifest.manifestHash,
      state: "paused",
      discoveredFiles: 1,
      processedFiles: 0,
      rejectedFiles: 0,
      currentEpoch: 0,
      requestedEpochs: 1,
      updatedAt: expect.any(String),
    });

    const latest = await restarted.progress("brain-a", manifest.id);
    await restarted.saveCursor("brain-a", {
      ...latest.cursor,
      state: "running",
      updatedAt: new Date(Date.parse(latest.updatedAt) + 1_000).toISOString(),
    });
    await expect(restarted.latestResumable("brain-a")).resolves.toMatchObject({
      manifestId: manifest.id,
      state: "interrupted",
    });
  });

  it("keeps authoritative progress readable when compatibility materialization fails midway", async () => {
    const path = join(root, "compatibility-fault.txt");
    await writeFile(path, "compatibility mirrors are disposable");
    const phases: string[] = [];
    const store = new DatasetManifestStore(() => brainDirectory, {
      compatibilityMaterializationHook: (phase) => {
        phases.push(phase);
        if (phase === "after-cursor") throw new Error("simulated mirror crash");
      },
    });

    const manifest = await store.create("brain-a", [path]);
    const progress = await store.progress("brain-a", manifest.id);

    expect(phases).toEqual(["after-authoritative", "after-cursor"]);
    expect(progress).toMatchObject({
      generation: 1,
      cursor: { nextEntry: 0, state: "ready" },
      coverage: { processedFiles: 0, complete: false },
    });
  });

  it("snapshots committed SQLite WAL state without mutating the selected database", async () => {
    const source = join(root, "wal-source.sqlite3");
    const canonicalSource = await realpath(root).then((directory) =>
      join(directory, "wal-source.sqlite3"),
    );
    const database = new DatabaseSync(source);
    try {
      database.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;");
      database.exec("CREATE TABLE examples(value TEXT NOT NULL);");
      database.exec(
        "INSERT INTO examples(value) VALUES ('main'), ('committed-wal');",
      );
      const sourceBefore = await stat(source);
      const bytesBefore = await readFile(source);
      const store = new DatasetManifestStore(() => brainDirectory);

      const manifest = await store.create("brain-a", [source]);
      const entries = [];
      for await (const entry of store.entries("brain-a", manifest.id))
        entries.push(entry);
      expect(entries).toHaveLength(1);
      expect(entries[0]).toMatchObject({
        sourcePath: canonicalSource,
        snapshotKind: "sqlite",
        format: "sqlite",
        contentSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      });
      expect(entries[0]!.path).not.toBe(source);
      expect(entries[0]!.path).toContain(
        join("datasets", manifest.id, "snapshots"),
      );

      const snapshot = new DatabaseSync(entries[0]!.path, { readOnly: true });
      try {
        const count = snapshot
          .prepare("SELECT COUNT(*) AS count FROM examples")
          .get() as {
          count: number;
        };
        expect(count.count).toBe(2);
      } finally {
        snapshot.close();
      }
      expect(await readFile(source)).toEqual(bytesBefore);
      await expect(stat(source)).resolves.toMatchObject({
        size: sourceBefore.size,
        mtimeMs: sourceBefore.mtimeMs,
      });
    } finally {
      database.close();
    }
  });

  it("commits the same deterministic entry hash regardless of selected-root order", async () => {
    const first = join(root, "a.jsonl");
    const second = join(root, "b.parquet");
    await writeFile(first, '{"text":"first"}\n');
    await writeFile(second, "bounded fixture bytes");
    const store = new DatasetManifestStore(() => brainDirectory);

    const left = await store.create("brain-a", [second, first]);
    const right = await store.create("brain-a", [first, second]);

    expect(left.roots).toEqual([first, second]);
    expect(right.roots).toEqual(left.roots);
    expect(right.manifestHash).toBe(left.manifestHash);
    const entries = [];
    for await (const entry of store.entries("brain-a", left.id))
      entries.push(entry);
    expect(entries.map((entry) => entry.relativePath)).toEqual([
      "a.jsonl",
      "b.parquet",
    ]);
    expect(
      entries.every((entry) => typeof entry.lastModifiedMs === "number"),
    ).toBe(true);
    expect(
      entries.every((entry) =>
        /^[a-f0-9]{64}$/.test(entry.contentSha256 ?? ""),
      ),
    ).toBe(true);
  });

  it("streams manifest discovery/hash progress and removes a cooperatively cancelled snapshot", async () => {
    const dataset = join(root, "cancel-preview");
    await mkdir(dataset);
    const large = join(dataset, "large.bin");
    const handle = await open(large, "w");
    await handle.truncate(24 * 1024 * 1024);
    await handle.close();
    const controller = new AbortController();
    const events: Array<{
      phase: string;
      discoveredFiles: number;
      hashedBytes: number;
    }> = [];
    const store = new DatasetManifestStore(() => brainDirectory);

    await expect(
      store.create("brain-a", [dataset], {
        signal: controller.signal,
        onProgress: (progress) => {
          events.push({
            phase: progress.phase,
            discoveredFiles: progress.discoveredFiles,
            hashedBytes: progress.hashedBytes,
          });
          if (progress.hashedBytes >= 8 * 1024 * 1024) controller.abort();
        },
      }),
    ).rejects.toMatchObject({ name: "AbortError" });

    expect(events.some((event) => event.phase === "hashing")).toBe(true);
    expect(events.some((event) => event.discoveredFiles === 1)).toBe(true);
    expect(
      Math.max(...events.map((event) => event.hashedBytes)),
    ).toBeGreaterThanOrEqual(8 * 1024 * 1024);
    expect(await readdir(join(brainDirectory, "datasets"))).toEqual([]);
  });

  it("keeps an unreadable or missing selection as an explicit manifest rejection", async () => {
    const missing = join(root, "missing-dataset.jsonl");
    const store = new DatasetManifestStore(() => brainDirectory);

    const manifest = await store.create("brain-a", [missing]);
    const entries = [];
    for await (const entry of store.entries("brain-a", manifest.id))
      entries.push(entry);

    expect(manifest.discoveredFiles).toBe(1);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      index: 0,
      path: missing,
      relativePath: "missing-dataset.jsonl",
      format: "jsonl",
      bytes: 0,
      rejection: expect.stringMatching(/could not inspect dataset path/i),
    });
  });

  it("rejects hidden/download sidecars and skips hidden caches without training them", async () => {
    const folder = join(root, "dataset-with-sidecars");
    await mkdir(join(folder, ".git", "objects"), { recursive: true });
    await mkdir(join(folder, ".cache", "downloads"), { recursive: true });
    await writeFile(join(folder, "000_00000.parquet"), "trainable fixture");
    await writeFile(join(folder, "000_00000.parquet.lock"), "");
    await writeFile(
      join(folder, "000_00000.parquet.metadata"),
      "download state",
    );
    await writeFile(
      join(folder, "001_00000.parquet.incomplete"),
      "unfinished shard",
    );
    await writeFile(
      join(folder, "002_00000.parquet.inprogress"),
      "unfinished shard",
    );
    await writeFile(
      join(folder, "003_00000.parquet.pending"),
      "unfinished shard",
    );
    await writeFile(
      join(folder, ".hidden.jsonl"),
      '{"text":"must not train"}\n',
    );
    await writeFile(join(folder, ".gitignore"), "*.lock\n");
    await writeFile(join(folder, ".DS_Store"), "finder metadata");
    await writeFile(
      join(folder, ".git", "objects", "pack"),
      "repository internals",
    );
    await writeFile(
      join(folder, ".cache", "downloads", "cached.jsonl"),
      '{"text":"cache"}\n',
    );
    const store = new DatasetManifestStore(() => brainDirectory);

    const manifest = await store.create("brain-a", [folder]);
    const entries = [];
    for await (const entry of store.entries("brain-a", manifest.id))
      entries.push(entry);

    expect(entries.map((entry) => entry.relativePath)).toEqual([
      ".DS_Store",
      ".gitignore",
      ".hidden.jsonl",
      "000_00000.parquet",
      "000_00000.parquet.lock",
      "000_00000.parquet.metadata",
      "001_00000.parquet.incomplete",
      "002_00000.parquet.inprogress",
      "003_00000.parquet.pending",
    ]);
    expect(
      entries.find((entry) => entry.relativePath === "000_00000.parquet")
        ?.rejection,
    ).toBeUndefined();
    for (const entry of entries.filter(
      (entry) => entry.relativePath !== "000_00000.parquet",
    )) {
      expect(entry.rejection).toMatch(/not training data/i);
      expect(entry.contentSha256).toBeUndefined();
    }
    expect(
      entries.some((entry) => entry.relativePath.includes(".git/objects")),
    ).toBe(false);
    expect(
      entries.some((entry) => entry.relativePath.includes(".cache/downloads")),
    ).toBe(false);
  });

  it("never sends hidden or incomplete folder artifacts to the neural worker", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Artifact filter",
    });
    const folder = join(root, "artifact-filter");
    await mkdir(join(folder, ".cache"), { recursive: true });
    await writeFile(join(folder, "records.txt"), "the only trainable record");
    await writeFile(join(folder, ".hidden.jsonl"), '{"text":"hidden"}\n');
    await writeFile(join(folder, "shard.parquet.lock"), "lock state");
    await writeFile(join(folder, "shard.parquet.incomplete"), "partial bytes");
    await writeFile(join(folder, ".cache", "cached.txt"), "cache bytes");
    const workerPaths: string[] = [];
    const service = new BrainService(repository, {
      request: async (_method: string, params: Record<string, unknown>) => {
        workerPaths.push(String(params.path));
        return {
          transactionKey: params.transactionKey,
          source: {
            learned_ideas: 1,
            learned_concepts: 1,
            synaptic_update_events: 1,
            parameter_update_steps: 1,
            parameter_checksum_changed: true,
          },
          coverage: {
            discoveredFiles: 1,
            completedFiles: 1,
            rejectedFiles: 0,
            discoveredRecords: 1,
            processedRecords: 1,
            rejectedRecords: 0,
            processedBytes: 25,
            shards: 0,
            modalityCounts: { text: 1 },
            errors: [],
            complete: true,
          },
        };
      },
    } as unknown as EngineSupervisor);

    const manifest = await service.previewDataset(brain.id, [folder]);
    const result = await service.ingestManifest(brain.id, manifest.id);

    expect(
      workerPaths.map((path) => basename(path)),
    ).toEqual(["records.txt"]);
    expect(result.coverage).toMatchObject({
      discoveredFiles: 4,
      processedFiles: 1,
      rejectedFiles: 3,
      discoveredRecords: 4,
      processedRecords: 1,
      rejectedRecords: 3,
      complete: true,
    });
  });

  it("streams all text beyond the former 16-million-character truncation", async () => {
    const path = join(root, "large.txt");
    const payload = `${"omnibrain ".repeat(1_800_000)}tail-sentinel`;
    await writeFile(path, payload);
    let rebuilt = "";
    for await (const chunk of streamUtf8Text(path)) rebuilt += chunk;
    expect(rebuilt.length).toBe(payload.length);
    expect(rebuilt.endsWith("tail-sentinel")).toBe(true);
  });

  it("accepts files beyond the former 128 MB limit and records worker coverage", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Large dataset",
    });
    const path = join(root, "large.parquet");
    const handle = await open(path, "w");
    await handle.truncate(129 * 1024 * 1024);
    await handle.close();
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 17,
          learned_concepts: 23,
          plasticity_events: 41,
          synaptic_update_events: 37,
          parameter_update_steps: 9,
          parameter_checksum_changed: true,
        },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 1,
          discoveredRecords: 500,
          processedRecords: 500,
          rejectedRecords: 0,
          processedBytes: 129 * 1024 * 1024,
          shards: 1,
          modalityCounts: { parquet: 500 },
          errors: [],
        },
      }),
    } as unknown as EngineSupervisor);
    const results = await service.ingestPaths(brain.id, [path]);
    expect(results).toHaveLength(1);
    expect(results[0]!.source).toMatchObject({
      kind: "parquet",
      bytes: 129 * 1024 * 1024,
      learnedIdeas: 17,
      learnedConcepts: 23,
      learnedSynapses: 37,
      learnedParameterSteps: 9,
      parametersChanged: true,
    });
    expect(results[0]!.coverage).toMatchObject({
      processedFiles: 1,
      discoveredRecords: 500,
      processedRecords: 500,
      rejectedRecords: 0,
    });
  });

  it("preserves JSONL and TSV parser kinds while keeping coarse source ledger kinds", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Exact parser contracts",
    });
    const jsonl = join(root, "records.jsonl");
    const tsv = join(root, "records.tsv");
    await writeFile(jsonl, '{"text":"first"}\n{"text":"second"}\n');
    await writeFile(tsv, "prompt\tresponse\nhello\tworld\n");
    const requests: Array<Record<string, unknown>> = [];
    const service = new BrainService(repository, {
      request: async (_method: string, params: Record<string, unknown>) => {
        requests.push(params);
        const parserKind = String(params.kind);
        return {
          source: {
            kind: parserKind,
            learned_ideas: 2,
            learned_concepts: 2,
            plasticity_events: 2,
          },
          coverage: {
            discoveredFiles: 1,
            completedFiles: 1,
            discoveredRecords: 2,
            processedRecords: 2,
            rejectedRecords: 0,
            processedBytes: Number((await stat(String(params.path))).size),
            shards: 0,
            modalityCounts: { [parserKind]: 2 },
            errors: [],
          },
        };
      },
    } as unknown as EngineSupervisor);

    const results = await service.ingestPaths(brain.id, [jsonl, tsv]);

    expect(requests.map((request) => request.kind)).toEqual(["jsonl", "tsv"]);
    expect(results.map((result) => result.source.kind)).toEqual([
      "json",
      "csv",
    ]);
  });

  it("maps entry-local worker progress across the whole folder by committed bytes", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Folder progress",
    });
    const folder = join(root, "folder-progress");
    await mkdir(folder);
    await writeFile(join(folder, "a.jsonl"), '{"text":"aaaaaaaa"}\n');
    await writeFile(join(folder, "b.jsonl"), '{"text":"bbbbbbbb"}\n');
    const engine = new EventEmitter() as EngineSupervisor;
    engine.request = (async <T>(
      _method: string,
      params: Record<string, unknown>,
    ): Promise<T> => {
      engine.emit("event", {
        type: "job-progress",
        jobId: params.jobId,
        progress: 0.98,
        message: `entry ${String(params.path).endsWith("a.jsonl") ? "a" : "b"} at 98%`,
        data: {
          datasetProgress: {
            coverage: {
              discoveredFiles: 1,
              completedFiles: 0,
              discoveredRecords: 1,
              processedRecords: 1,
              rejectedRecords: 0,
              processedBytes: 16,
              shards: 0,
              modalityCounts: { jsonl: 1 },
              errors: [],
            },
            currentRecord: 1,
            committedRecords: 0,
            checkpointCommitted: false,
            recordTotalKnown: false,
          },
        },
      });
      engine.emit("event", {
        type: "job-progress",
        jobId: params.jobId,
        progress: 0.25,
        message: `entry ${String(params.path).endsWith("a.jsonl") ? "a" : "b"} entered a later phase`,
        data: {
          datasetProgress: {
            coverage: {
              discoveredFiles: 1,
              completedFiles: 0,
              discoveredRecords: 1,
              processedRecords: 1,
              rejectedRecords: 0,
              processedBytes: 16,
              shards: 0,
              modalityCounts: { jsonl: 1 },
              errors: [],
            },
            currentRecord: 1,
            committedRecords: 0,
            checkpointCommitted: false,
            recordTotalKnown: false,
          },
        },
      });
      return {
        transactionKey: params.transactionKey,
        source: { learned_ideas: 1, learned_concepts: 1, plasticity_events: 3 },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 1,
          discoveredRecords: 1,
          processedRecords: 1,
          rejectedRecords: 0,
          processedBytes: 20,
          shards: 0,
          modalityCounts: { jsonl: 1 },
          errors: [],
        },
      } as T;
    }) as EngineSupervisor["request"];
    const manifest = await new BrainService(repository, engine).previewDataset(
      brain.id,
      [folder],
    );
    const samples: Array<{ value: number; message: string; records?: number }> =
      [];
    const service = new BrainService(repository, engine);

    const result = await service.ingestManifest(
      brain.id,
      manifest.id,
      "encode",
      () => false,
      (value, message, detail) =>
        samples.push({
          value,
          message,
          records: detail?.coverage?.processedRecords,
        }),
      false,
      1,
      true,
      "folder-job",
    );

    const firstEntry98 = samples.find((sample) =>
      sample.message.includes("entry a at 98%"),
    );
    expect(firstEntry98?.value).toBeGreaterThan(0.45);
    expect(firstEntry98?.value).toBeLessThan(0.55);
    expect(firstEntry98?.records).toBe(1);
    expect(result.coverage).toMatchObject({
      processedFiles: 2,
      discoveredRecords: 2,
      processedRecords: 2,
      complete: true,
    });
    expect(samples.at(-1)).toMatchObject({ value: 1 });
    expect(samples.slice(0, -1).every((sample) => sample.value < 1)).toBe(true);
    expect(samples.map((sample) => sample.value)).toEqual(
      [...samples.map((sample) => sample.value)].sort(
        (left, right) => left - right,
      ),
    );
  });

  it("persists the worker's committed record cursor when a large entry pauses", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Record cursor bridge",
    });
    const path = join(root, "checkpointed.jsonl");
    await writeFile(
      path,
      Array.from({ length: 4 }, (_, index) =>
        JSON.stringify({ text: `record-${index}` }),
      ).join("\n"),
    );
    const engine = new EventEmitter() as EngineSupervisor;
    engine.request = (async <T>(
      _method: string,
      params: Record<string, unknown>,
    ): Promise<T> => {
      engine.emit("event", {
        type: "job-progress",
        jobId: params.jobId,
        progress: 0.5,
        message: "Committed 2 learned records",
        data: {
          datasetProgress: {
            coverage: {
              discoveredFiles: 1,
              completedFiles: 0,
              discoveredRecords: 2,
              processedRecords: 2,
              rejectedRecords: 0,
              processedBytes: 40,
              shards: 0,
              modalityCounts: { jsonl: 2 },
              errors: [],
            },
            currentRecord: 2,
            committedRecords: 2,
            checkpointCommitted: true,
            recordTotalKnown: false,
          },
        },
      });
      throw new Error("simulated resource pause after worker checkpoint");
    }) as EngineSupervisor["request"];
    const service = new BrainService(repository, engine);
    const manifest = await service.previewDataset(brain.id, [path]);

    const result = await service.ingestManifest(
      brain.id,
      manifest.id,
      "encode",
      () => false,
      () => undefined,
      false,
      1,
      true,
      "checkpoint-job",
    );

    expect(result.paused).toBe(true);
    const store = new DatasetManifestStore((id) =>
      repository.brainDirectory(id),
    );
    await expect(store.cursor(brain.id, manifest.id)).resolves.toMatchObject({
      nextEntry: 0,
      nextRecord: 2,
      state: "paused",
    });
  });

  it("reports the exact durable checkpoint baseline before resumed worker traversal", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Telemetry resume baseline",
    });
    const path = join(root, "resume.parquet");
    await writeFile(path, Buffer.alloc(100, 7));
    let workerStarted = false;
    let workerDeadline: number | undefined;
    const engine = new EventEmitter() as EngineSupervisor;
    engine.request = (async <T>(
      _method: string,
      params: Record<string, unknown>,
      timeoutMs?: number,
    ): Promise<T> => {
      workerStarted = true;
      workerDeadline = timeoutMs;
      return {
        transactionKey: params.transactionKey,
        source: {
          kind: "parquet",
          learned_ideas: 1,
          learned_concepts: 1,
          plasticity_events: 1,
        },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 1,
          discoveredRecords: 21_990,
          processedRecords: 21_990,
          rejectedRecords: 0,
          processedBytes: 100,
          shards: 0,
          modalityCounts: { parquet: 21_990 },
          errors: [],
        },
      } as T;
    }) as EngineSupervisor["request"];
    const service = new BrainService(repository, engine);
    const manifest = await service.previewDataset(brain.id, [path]);
    let entry: DatasetManifestEntry | undefined;
    for await (const candidate of service.datasets.entries(brain.id, manifest.id)) {
      entry = candidate;
      break;
    }
    expect(entry).toBeDefined();
    const initial = await service.datasets.progress(brain.id, manifest.id);
    await service.datasets.saveProgress(
      brain.id,
      {
        ...initial.cursor,
        nextRecord: 5_632,
        state: "paused",
      },
      initial.coverage,
      initial.lastEntryReceipt,
      initial.generation,
    );
    const checkpointIdentity = "d".repeat(64);
    await mkdir(join(repository.brainDirectory(brain.id), "engine"), {
      recursive: true,
    });
    await writeFile(
      join(repository.brainDirectory(brain.id), "engine", "brain.json"),
      JSON.stringify({
        brain_id: brain.id,
        ingestion_checkpoints: {
          [checkpointIdentity]: {
            format: "omni-record-ingestion-checkpoint",
            formatVersion: 2,
            status: "active",
            sourceIdentity: checkpointIdentity,
            contentHash: entry!.contentSha256,
            policy: "pretrain",
            epoch: 0,
            sourceBytes: entry!.bytes,
            resolvedKind: "parquet",
            committedRecords: 5_632,
            visitedRecords: 5_632,
            processedRecords: 5_632,
            rejectedRecords: 0,
            processedBytes: 42,
            coverageAtCommit: {
              complete: false,
              discoveredFiles: 1,
              processedFiles: 0,
              rejectedFiles: 0,
              discoveredRecords: 5_632,
              processedRecords: 5_632,
              rejectedRecords: 0,
              processedBytes: 42,
              shards: 0,
              modalityCounts: { parquet: 5_632 },
              errors: [],
              errorCount: 0,
              errorsTruncated: false,
            },
          },
        },
      }),
      "utf8",
    );
    const samples: Array<{
      beforeWorker: boolean;
      detail?: Record<string, unknown>;
    }> = [];

    await service.ingestManifest(
      brain.id,
      manifest.id,
      "pretrain",
      () => false,
      (_value, _message, detail) => {
        samples.push({
          beforeWorker: !workerStarted,
          detail: detail as unknown as Record<string, unknown>,
        });
      },
      false,
      1,
      true,
      "resume-baseline-job",
    );

    const baseline = samples.find((sample) => sample.detail?.durableBaseline === true);
    expect(workerDeadline).toBe(0);
    expect(baseline).toMatchObject({
      beforeWorker: true,
      detail: {
        recordsCompleted: 5_632,
        currentRecord: 5_632,
        committedRecords: 5_632,
        checkpointCommitted: true,
        physicalSourceBytesComparable: false,
        rateBaseline: true,
        durableBaseline: true,
        coverage: {
          processedRecords: 5_632,
          processedBytes: 42,
        },
      },
    });
  });

  it("persists the neural worker's animated-GIF video reroute", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "GIF routing",
    });
    const path = join(root, "motion.gif");
    await writeFile(path, Buffer.from("GIF89a-temporal-fixture"));
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          kind: "video",
          learned_ideas: 1,
          learned_concepts: 2,
          plasticity_events: 3,
        },
      }),
    } as unknown as EngineSupervisor);

    const [result] = await service.ingestPaths(brain.id, [path]);
    expect(result?.source).toMatchObject({
      kind: "video",
      name: "motion.gif",
    });
    expect((await repository.get(brain.id)).trainingSources[0]?.kind).toBe(
      "video",
    );
  });

  it("keeps the cursor on an uncommitted file when neural training pauses", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Resume exactly",
    });
    const path = join(root, "resource-pause.txt");
    await writeFile(path, "This record must be retried, never skipped.");
    const service = new BrainService(repository, {
      request: async () => {
        throw new Error(
          "SubstrateResourcePause: neural substrate growth paused at the host resource reserve",
        );
      },
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);
    expect(run.paused).toBe(true);
    expect(run.coverage).toMatchObject({
      processedFiles: 0,
      rejectedFiles: 0,
      complete: false,
    });

    const store = new DatasetManifestStore((id) =>
      repository.brainDirectory(id),
    );
    await expect(store.cursor(brain.id, manifest.id)).resolves.toMatchObject({
      nextEntry: 0,
      processedFiles: 0,
      state: "paused",
    });
  });

  it("never reports a recoverable ingestPaths pause as an empty success", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Visible upload pause",
    });
    const path = join(root, "visible-pause.txt");
    await writeFile(path, "This source remains available for resume.");
    const service = new BrainService(repository, {
      request: async () => {
        throw new Error("NeuralStateResourcePause: retry after memory pressure");
      },
    } as unknown as EngineSupervisor);

    await expect(service.ingestPaths(brain.id, [path])).rejects.toThrow(
      /Paused before .*Resume dataset manifest [a-f0-9-]{36}; no source was reported as learned\./,
    );
    const manifests = await readdir(
      join(repository.brainDirectory(brain.id), "datasets"),
    );
    expect(manifests).toHaveLength(1);
    const progress = await service.datasets.progress(brain.id, manifests[0]!);
    expect(progress.cursor.state).toBe("paused");
    expect(progress.coverage).toMatchObject({
      processedFiles: 0,
      rejectedFiles: 0,
      complete: false,
    });
  });

  it("explicitly rejects a file changed after its deterministic manifest snapshot", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Snapshot identity",
    });
    const path = join(root, "changing.jsonl");
    await writeFile(path, '{"text":"original"}\n');
    let workerCalls = 0;
    const service = new BrainService(repository, {
      request: async () => {
        workerCalls += 1;
        throw new Error("the worker must not see a changed manifest entry");
      },
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);
    await appendFile(path, '{"text":"changed"}\n');

    const run = await service.ingestManifest(brain.id, manifest.id);

    expect(workerCalls).toBe(0);
    expect(run.paused).toBe(false);
    expect(run.coverage).toMatchObject({
      discoveredFiles: 1,
      processedFiles: 0,
      rejectedFiles: 1,
      discoveredRecords: 1,
      processedRecords: 0,
      rejectedRecords: 1,
      errorCount: 1,
      complete: true,
    });
    expect(run.coverage.errors[0]?.message).toMatch(
      /source changed since the manifest/i,
    );
    const store = new DatasetManifestStore((id) =>
      repository.brainDirectory(id),
    );
    await expect(store.cursor(brain.id, manifest.id)).resolves.toMatchObject({
      nextEntry: 0,
      processedFiles: 1,
      state: "complete",
    });
  });

  it("reports a wholly unsupported worker source as rejected instead of processed", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Explicit rejection",
    });
    const path = join(root, "weights.safetensors");
    await writeFile(path, Buffer.from([0, 255, 0, 255]));
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 0,
          learned_concepts: 0,
          plasticity_events: 0,
        },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 0,
          rejectedFiles: 1,
          discoveredRecords: 1,
          processedRecords: 0,
          rejectedRecords: 1,
          processedBytes: 0,
          shards: 0,
          modalityCounts: {},
          errors: [
            {
              source: "weights.safetensors",
              message: "unsupported binary dataset format",
            },
          ],
        },
      }),
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);

    expect(run.coverage).toMatchObject({
      discoveredFiles: 1,
      processedFiles: 0,
      rejectedFiles: 1,
      discoveredRecords: 1,
      processedRecords: 0,
      rejectedRecords: 1,
      complete: true,
    });
  });

  it("persists exhaustive aggregate rejection counts without millions of error objects", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Aggregate rejection",
    });
    const path = join(root, "python-edu.parquet");
    await writeFile(path, "small schema fixture");
    const rejectedRows = 3_839_224;
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 0,
          learned_concepts: 0,
          plasticity_events: 0,
        },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 1,
          rejectedFiles: 0,
          discoveredRecords: rejectedRows,
          processedRecords: 0,
          rejectedRecords: rejectedRows,
          processedBytes: 20,
          shards: 0,
          modalityCounts: {},
          errorCount: rejectedRows,
          errors: [
            {
              source: "python-edu.parquet#all-rows",
              message: "metadata-only Parquet schema has no trainable content",
              count: rejectedRows,
            },
          ],
        },
      }),
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);

    expect(run.coverage).toMatchObject({
      discoveredRecords: rejectedRows,
      processedRecords: 0,
      rejectedRecords: rejectedRows,
      errorCount: rejectedRows,
      complete: true,
    });
    expect(run.coverage.errors).toHaveLength(1);
    expect(run.coverage.errors[0]).toMatchObject({ count: rejectedRows });
    const errorLog = join(
      repository.brainDirectory(brain.id),
      "datasets",
      manifest.id,
      "errors.ndjson",
    );
    const log = await readFile(errorLog, "utf8");
    expect(log.trim().split("\n")).toHaveLength(1);
    expect(log).toContain(`"count":${rejectedRows}`);
  });

  it("pauses before persistence when worker traversal totals are incomplete", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Honest coverage",
    });
    const path = join(root, "partial.jsonl");
    await writeFile(path, '{"first":true}\n{"second":true}\n');
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 1,
          learned_concepts: 1,
          plasticity_events: 1,
        },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 1,
          discoveredRecords: 2,
          processedRecords: 1,
          rejectedRecords: 0,
          processedBytes: 15,
          shards: 0,
          modalityCounts: { jsonl: 1 },
          errors: [],
        },
      }),
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);

    expect(run.paused).toBe(true);
    expect(run.coverage).toMatchObject({
      processedFiles: 0,
      rejectedFiles: 0,
      discoveredRecords: 0,
      processedRecords: 0,
      rejectedRecords: 0,
      complete: false,
    });
    expect((await repository.get(brain.id)).trainingSources).toEqual([]);
    const store = new DatasetManifestStore((id) =>
      repository.brainDirectory(id),
    );
    await expect(store.cursor(brain.id, manifest.id)).resolves.toMatchObject({
      nextEntry: 0,
      processedFiles: 0,
      state: "paused",
    });
  });

  it("resumes from a durable worker receipt without training the entry twice", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Receipt recovery",
    });
    const path = join(root, "receipt-recovery.txt");
    await writeFile(
      path,
      "a worker acknowledgement must survive an Electron crash",
    );
    let workerCalls = 0;
    const engine = {
      request: async (_method: string, params: Record<string, unknown>) => {
        workerCalls += 1;
        return {
          transactionKey: params.transactionKey,
          source: {
            learned_ideas: 5,
            learned_concepts: 7,
            plasticity_events: 11,
            synaptic_update_events: 8,
            parameter_update_steps: 3,
            parameter_checksum_changed: true,
          },
        };
      },
    } as unknown as EngineSupervisor;
    const service = new BrainService(repository, engine);
    const manifest = await service.previewDataset(brain.id, [path]);
    const saveProgress = service.datasets.saveProgress.bind(service.datasets);
    let injectFault = true;
    service.datasets.saveProgress = async (...args) => {
      const committed = await saveProgress(...args);
      if (injectFault && args[3]?.outcome === "learned") {
        injectFault = false;
        throw new Error("simulated crash after durable worker receipt");
      }
      return committed;
    };

    await expect(
      service.ingestManifest(brain.id, manifest.id),
    ).rejects.toThrow();
    expect(workerCalls).toBe(1);
    expect((await repository.get(brain.id)).trainingSources).toEqual([]);

    const restarted = new BrainService(repository, engine);
    const resumed = await restarted.ingestManifest(brain.id, manifest.id);
    expect(resumed.paused).toBe(false);
    expect(resumed.coverage).toMatchObject({
      processedFiles: 1,
      complete: true,
    });
    expect(workerCalls).toBe(1);
    const recoveredBrain = await repository.get(brain.id);
    expect(recoveredBrain.trainingSources).toHaveLength(1);
    expect(recoveredBrain.trainingSources[0]).toMatchObject({
      learnedIdeas: 5,
      learnedConcepts: 7,
      learnedSynapses: 8,
      learnedParameterSteps: 3,
      parametersChanged: true,
    });
    expect(
      (recoveredBrain.journal ?? []).filter(
        (event) => event.kind === "learning",
      ),
    ).toHaveLength(1);
  });

  it("makes repository persistence idempotent when the cursor commit is interrupted", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Cursor recovery",
    });
    const path = join(root, "cursor-recovery.txt");
    await writeFile(
      path,
      "repository persistence must not duplicate replay evidence",
    );
    let workerCalls = 0;
    const engine = {
      request: async (_method: string, params: Record<string, unknown>) => {
        workerCalls += 1;
        return {
          transactionKey: params.transactionKey,
          source: {
            learned_ideas: 2,
            learned_concepts: 3,
            plasticity_events: 5,
          },
        };
      },
    } as unknown as EngineSupervisor;
    const service = new BrainService(repository, engine);
    const manifest = await service.previewDataset(brain.id, [path]);
    const saveProgress = service.datasets.saveProgress.bind(service.datasets);
    let injectFault = true;
    service.datasets.saveProgress = async (...args) => {
      if (injectFault && args[1].nextEntry === 1) {
        injectFault = false;
        throw new Error("simulated crash before cursor advancement");
      }
      return saveProgress(...args);
    };

    await expect(service.ingestManifest(brain.id, manifest.id)).rejects.toThrow(
      /before cursor advancement/,
    );
    expect(workerCalls).toBe(1);
    expect((await repository.get(brain.id)).trainingSources).toHaveLength(1);

    const restarted = new BrainService(repository, engine);
    const resumed = await restarted.ingestManifest(brain.id, manifest.id);
    expect(resumed.coverage).toMatchObject({
      processedFiles: 1,
      complete: true,
    });
    expect(workerCalls).toBe(1);
    const recoveredBrain = await repository.get(brain.id);
    expect(recoveredBrain.trainingSources).toHaveLength(1);
    expect(
      (recoveredBrain.journal ?? []).filter(
        (event) => event.kind === "learning",
      ),
    ).toHaveLength(1);
  });

  it("propagates worker record errors into the durable coverage report", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Coverage errors",
    });
    const path = join(root, "mixed.jsonl");
    await writeFile(path, '{"ok":true}\nnot-json\n');
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 1,
          learned_concepts: 2,
          plasticity_events: 3,
        },
        coverage: {
          discoveredFiles: 1,
          completedFiles: 1,
          discoveredRecords: 2,
          processedRecords: 1,
          rejectedRecords: 1,
          processedBytes: 21,
          shards: 0,
          modalityCounts: { jsonl: 1 },
          errors: [{ source: "mixed.jsonl#line-2", message: "invalid JSONL" }],
        },
      }),
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);
    expect(run.coverage).toMatchObject({
      processedFiles: 1,
      rejectedFiles: 0,
      discoveredRecords: 2,
      processedRecords: 1,
      rejectedRecords: 1,
      complete: true,
    });
    expect(run.coverage.errors).toContainEqual({
      source: "mixed.jsonl#line-2",
      message: "invalid JSONL",
    });
    const errorLog = join(
      repository.brainDirectory(brain.id),
      "datasets",
      manifest.id,
      "errors.ndjson",
    );
    await expect(readFile(errorLog, "utf8")).resolves.toContain(
      '"source":"mixed.jsonl#line-2"',
    );
  });

  it("visits every file in every requested epoch and explicitly replays later epochs", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Two epochs",
    });
    const path = join(root, "epochs.txt");
    await writeFile(path, "epoch traversal fixture");
    const calls: Array<Record<string, unknown>> = [];
    const service = new BrainService(repository, {
      request: async (_method: string, params: Record<string, unknown>) => {
        calls.push(params);
        return {
          source: {
            learned_ideas: 1,
            learned_concepts: 1,
            plasticity_events: 1,
          },
          coverage: {
            discoveredFiles: 1,
            completedFiles: 1,
            discoveredRecords: 1,
            processedRecords: 1,
            rejectedRecords: 0,
            processedBytes: 23,
            shards: 0,
            modalityCounts: { text: 1 },
            errors: [],
          },
        };
      },
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(
      brain.id,
      manifest.id,
      "encode",
      () => false,
      () => undefined,
      true,
      2,
      true,
      "desktop-ingestion-job",
    );
    expect(calls).toHaveLength(2);
    expect(calls.map((params) => params.allowReplay)).toEqual([true, true]);
    expect(calls.map((params) => params.epoch)).toEqual([0, 1]);
    expect(calls.map((params) => params.jobId)).toEqual([
      "desktop-ingestion-job",
      "desktop-ingestion-job",
    ]);
    expect(calls.map((params) => params.contentHash)).toEqual([
      expect.stringMatching(/^[a-f0-9]{64}$/),
      expect.stringMatching(/^[a-f0-9]{64}$/),
    ]);
    expect(calls[1]?.contentHash).toBe(calls[0]?.contentHash);
    expect(run.coverage).toMatchObject({
      requestedEpochs: 2,
      completedEpochs: 2,
      discoveredFiles: 2,
      processedFiles: 2,
      complete: true,
    });
    expect(run.results).toHaveLength(2);
    expect(calls[0]?.transactionKey).not.toBe(calls[1]?.transactionKey);
    await service.ingestManifest(brain.id, manifest.id, "encode");
    expect(calls).toHaveLength(2);

    // Explicit restart keeps this exact source snapshot, but gives each epoch
    // a new durable identity. Resume of that restarted run keeps its identity.
    await service.datasets.reset(brain.id, manifest.id, 2);
    const restartId = (await service.datasets.cursor(brain.id, manifest.id)).runId;
    expect(restartId).toEqual(expect.any(String));
    const restarted = await service.ingestManifest(
      brain.id, manifest.id, "encode", () => false, () => undefined, false, 2,
    );
    expect(restarted.coverage.complete).toBe(true);
    expect(calls).toHaveLength(4);
    expect(new Set(calls.map((params) => params.transactionKey)).size).toBe(4);
    const committed = await service.datasets.progress(brain.id, manifest.id);
    expect(committed.cursor.runId).toBe(restartId);
    expect(committed.lastEntryReceipt?.runId).toBe(restartId);
    await service.ingestManifest(brain.id, manifest.id, "encode");
    expect(calls).toHaveLength(4);
  });

  it("replays a prelearned source in an explicit manifest but deduplicates one-off uploads", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Replay semantics",
    });
    const path = join(root, "prelearned.txt");
    await writeFile(
      path,
      "explicit whole-dataset traversal must revisit this source",
    );
    const calls: Array<Record<string, unknown>> = [];
    const service = new BrainService(repository, {
      request: async (_method: string, params: Record<string, unknown>) => {
        calls.push(params);
        return {
          source: {
            learned_ideas: 1,
            learned_concepts: 2,
            plasticity_events: 3,
          },
        };
      },
    } as unknown as EngineSupervisor);

    const firstUpload = await service.ingestPaths(brain.id, [path]);
    const duplicateUpload = await service.ingestPaths(brain.id, [path]);
    expect(firstUpload).toHaveLength(1);
    expect(duplicateUpload[0]!.warnings.join(" ")).toMatch(/already encoded/i);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ allowReplay: false, epoch: 0 });

    const manifest = await service.previewDataset(brain.id, [path]);
    const explicitRun = await service.ingestManifest(brain.id, manifest.id);
    expect(calls).toHaveLength(2);
    expect(calls[1]).toMatchObject({ allowReplay: true, epoch: 0 });
    expect(explicitRun.coverage).toMatchObject({
      processedFiles: 1,
      rejectedFiles: 0,
      complete: true,
    });
    expect((await repository.get(brain.id)).trainingSources).toHaveLength(1);
  });

  it("persists an unbounded crawl frontier in SQLite and resumes in-flight pages", async () => {
    const start = "https://example.com/";
    const frontier = await CrawlFrontierStore.create(
      brainDirectory,
      "brain-a",
      start,
      "crawl-a",
      false,
    );
    frontier.enqueue(
      Array.from({ length: 5_100 }, (_, index) => ({
        url: `https://example.com/page/${index}`,
        depth: 1,
      })),
    );
    const claimed = frontier.next(8);
    expect(claimed).toHaveLength(8);
    frontier.visited(claimed[0]!.url, 123);
    frontier.skipped(claimed[1]!.url, "fixture failure");
    frontier.close();

    const resumed = await CrawlFrontierStore.create(
      brainDirectory,
      "brain-a",
      start,
      "crawl-a",
      true,
    );
    const counts = resumed.counts();
    expect(counts).toMatchObject({
      queued: 5_099,
      visited: 1,
      skipped: 1,
      processedBytes: 123,
    });
    expect(resumed.warnings()).toEqual([
      "https://example.com/page/0: fixture failure",
    ]);
    resumed.close();
  });

  it("crawls an explicit loopback HTTP source without a test-only environment bypass", async () => {
    delete process.env.OMNI_ALLOW_LOCAL_URLS;
    const origin = await listen((_request, response) => {
      response.setHeader("content-type", "text/html");
      response.end("<p>local private learning source</p>");
    });
    const { brainId, service, engineRequests } = await crawlerService(
      "Loopback HTTP crawler"
    );

    const result = await service.crawlWeb({
      brainId,
      url: `${origin}/index.html`,
      maxPages: 1,
      respectRobots: false,
      quarantine: true,
    });

    expect(result).toMatchObject({
      startUrl: `${origin}/index.html`,
      visited: 1,
      skipped: 0,
      stopped: false,
      coverage: { complete: true, processedRecords: 1 },
    });
    expect(engineRequests).toHaveLength(1);
    expect(engineRequests[0]).toMatchObject({
      url: `${origin}/index.html`,
      kind: "text",
      policy: "archive",
    });
    const accepted = await service.ingestWeb({
      brainId, url: `${origin}/index.html`, quarantine: false, policy: "pretrain",
    });
    expect(engineRequests).toHaveLength(2);
    expect(engineRequests[1]).toMatchObject({ policy: "pretrain", allowReplay: true });
    expect(accepted.source.id).toBe(result.results[0]?.source.id);
    expect(accepted.source.policy).toBe("pretrain");
  });

  it("does not fetch a redirect target outside the default crawl origin", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    let targetPageHits = 0;
    const targetOrigin = await listen((request, response) => {
      if (request.url === "/robots.txt") {
        response.end("User-agent: *\nDisallow:\n");
        return;
      }
      targetPageHits += 1;
      response.setHeader("content-type", "text/html");
      response.end("<p>must not be fetched</p>");
    });
    const sourceOrigin = await listen((request, response) => {
      if (request.url === "/robots.txt") {
        response.end("User-agent: *\nDisallow:\n");
        return;
      }
      response.statusCode = 302;
      response.setHeader("location", `${targetOrigin}/outside`);
      response.end();
    });
    const { brainId, service } = await crawlerService("Redirect boundary");

    const result = await service.crawlWeb({
      brainId,
      url: `${sourceOrigin}/`,
      maxPages: 1,
      quarantine: true,
    });

    expect(result).toMatchObject({
      visited: 0,
      skipped: 1,
      frontierRemaining: 0,
      stopped: false,
    });
    expect(targetPageHits).toBe(0);
    expect(result.warnings.join(" ")).toMatch(
      /outside the same-site crawl origin/i,
    );
  });

  it("checks destination robots before following an allowed cross-origin redirect", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    let targetRobotsHits = 0;
    let targetPageHits = 0;
    const targetOrigin = await listen((request, response) => {
      if (request.url === "/robots.txt") {
        targetRobotsHits += 1;
        response.end("User-agent: *\nDisallow: /private\n");
        return;
      }
      targetPageHits += 1;
      response.end("must not be fetched");
    });
    const sourceOrigin = await listen((request, response) => {
      if (request.url === "/robots.txt") {
        response.end("User-agent: *\nDisallow:\n");
        return;
      }
      response.statusCode = 302;
      response.setHeader("location", `${targetOrigin}/private`);
      response.end();
    });
    const { brainId, service } = await crawlerService("Redirect robots");

    const result = await service.crawlWeb({
      brainId,
      url: `${sourceOrigin}/`,
      maxPages: 1,
      followExternalLinks: true,
      quarantine: true,
    });

    expect(result).toMatchObject({ visited: 0, skipped: 1, stopped: false });
    expect(targetRobotsHits).toBe(1);
    expect(targetPageHits).toBe(0);
    expect(result.warnings.join(" ")).toMatch(
      /robots\.txt disallows this path/i,
    );
  });

  it("paces same-domain requests and retries throttled pages with backoff", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    // Keep a comfortable wall-clock margin for heavily loaded CI runners
    // while still proving requests are serialized and paced per domain.
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "50";
    process.env.OMNI_CRAWL_BACKOFF_BASE_MS = "30";
    const requestTimes: Array<{ path: string; at: number }> = [];
    let throttledHits = 0;
    const origin = await listen((request, response) => {
      requestTimes.push({ path: request.url ?? "", at: performance.now() });
      response.setHeader("content-type", "text/html");
      if (request.url === "/") {
        response.end('<a href="/throttled">one</a><a href="/other">two</a>');
        return;
      }
      if (request.url === "/throttled" && throttledHits++ === 0) {
        response.statusCode = 503;
        response.end("retry later");
        return;
      }
      response.end(`<p>${request.url}</p>`);
    });
    const { brainId, service } = await crawlerService("Paced crawler");

    const result = await service.crawlWeb({
      brainId,
      url: `${origin}/`,
      maxPages: 3,
      maxDepth: 1,
      concurrency: 2,
      respectRobots: false,
      quarantine: true,
    });

    expect(result.warnings).toEqual([]);
    expect(result).toMatchObject({ visited: 3, skipped: 0, stopped: false });
    expect(throttledHits).toBe(2);
    expect(requestTimes).toHaveLength(4);
    const requestGaps = requestTimes
      .slice(1)
      .map((value, index) => value.at - requestTimes[index]!.at);
    expect(
      Math.min(...requestGaps),
      `requests=${JSON.stringify(requestTimes)} gaps=${JSON.stringify(requestGaps)} configured=${process.env.OMNI_CRAWL_MIN_DELAY_MS}`,
    ).toBeGreaterThanOrEqual(25);
  });

  it("discovers and neurally trains linked image, audio, and video responses", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    const origin = await listen((request, response) => {
      if (request.url === "/") {
        response.setHeader("content-type", "text/html");
        response.end(
          '<img src="/scene.png"><audio src="/sound.wav"></audio>' +
            '<video poster="/poster.webp"><source src="/clip.mp4"></video>',
        );
        return;
      }
      if (request.url === "/scene.png" || request.url === "/poster.webp") {
        response.setHeader(
          "content-type",
          request.url.endsWith(".webp") ? "image/webp" : "image/png",
        );
        response.end(
          request.url.endsWith(".webp")
            ? Buffer.from("RIFF-WEBP-media-fixture")
            : Buffer.from([0x89, 0x50, 0x4e, 0x47, 1, 2, 3]),
        );
        return;
      }
      if (request.url === "/sound.wav") {
        response.setHeader("content-type", "audio/wav");
        response.end(Buffer.from("RIFF-media-fixture"));
        return;
      }
      response.setHeader("content-type", "video/mp4");
      response.end(Buffer.from("ftyp-video-fixture"));
    });
    const { brainId, service, engineRequests } =
      await crawlerService("Multimodal crawler");

    const result = await service.crawlWeb({
      brainId,
      url: `${origin}/`,
      maxPages: 5,
      maxDepth: 1,
      concurrency: 4,
      respectRobots: false,
      quarantine: false,
      policy: "pretrain",
    });

    expect(result).toMatchObject({
      visited: 5,
      skipped: 0,
      resultCount: 5,
      resultsTruncated: false,
      warningCount: 0,
      warningsTruncated: false,
      coverage: {
        complete: true,
        modalityCounts: { text: 1, image: 2, audio: 1, video: 1 },
      },
    });
    expect(
      engineRequests
        .map((params) => params.kind)
        .filter((kind) => ["image", "audio", "video"].includes(String(kind))),
    ).toEqual(expect.arrayContaining(["image", "image", "audio", "video"]));
    expect(
      engineRequests
        .filter((params) => params.kind === "image")
        .map((params) => String(params.path)),
    ).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/\.png$/),
        expect.stringMatching(/\.webp$/),
      ]),
    );
    const saved = await service.repository.get(brainId);
    expect(
      saved.trainingSources
        .filter((source) => ["image", "audio", "video"].includes(source.kind))
        .map((source) => ({
          kind: source.kind,
          provenanceUrl: source.provenanceUrl,
          blobHash: source.blobHash,
        })),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "image",
          provenanceUrl: `${origin}/scene.png`,
          blobHash: undefined,
        }),
        expect.objectContaining({
          kind: "audio",
          provenanceUrl: `${origin}/sound.wav`,
          blobHash: undefined,
        }),
        expect.objectContaining({
          kind: "video",
          provenanceUrl: `${origin}/clip.mp4`,
          blobHash: undefined,
        }),
      ]),
    );
    // The locked human-like policy keeps neural changes, content hashes, and
    // provenance, then releases accepted crawl bytes instead of maintaining a
    // second exact image/audio/video archive.
    expect(
      saved.trainingSources
        .filter((source) => ["image", "audio", "video"].includes(source.kind))
        .every((source) => source.contentHash && !source.blobHash),
    ).toBe(true);
  });

  it("keeps crawl diagnostics bounded while receipts preserve exact resumed coverage", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    const origin = await listen((request, response) => {
      response.setHeader(
        "content-type",
        request.url === "/" ? "text/html" : "text/plain",
      );
      if (request.url === "/") {
        response.end(
          Array.from(
            { length: 47 },
            (_, index) => `<a href="/page-${index + 1}">page ${index + 1}</a>`,
          ).join(""),
        );
        return;
      }
      response.end(`unique learned page ${request.url}`);
    });
    const { brainId, service } = await crawlerService("Bounded crawl receipts");

    const first = await service.crawlWeb({
      brainId,
      url: `${origin}/`,
      maxPages: 8,
      maxDepth: 1,
      concurrency: 8,
      respectRobots: false,
      quarantine: false,
    });
    expect(first).toMatchObject({
      visited: 8,
      resultCount: 8,
      resultsTruncated: false,
      frontierRemaining: 40,
      coverage: { modalityCounts: { text: 8 } },
    });

    const resumed = await service.crawlWeb({
      brainId,
      url: `${origin}/`,
      crawlId: first.crawlId,
      maxPages: 48,
      maxDepth: 1,
      concurrency: 8,
      respectRobots: false,
      quarantine: false,
      resume: true,
    });
    expect(resumed).toMatchObject({
      visited: 48,
      resultCount: 48,
      resultsTruncated: true,
      frontierRemaining: 0,
      coverage: {
        processedFiles: 48,
        processedRecords: 48,
        modalityCounts: { text: 48 },
        complete: true,
      },
    });
    expect(resumed.results).toHaveLength(32);
    expect(resumed.resultLog).toBe(
      join("datasets", "crawls", `${resumed.crawlId}.sqlite3`),
    );
  });

  it("pauses and preserves the frontier when a response would consume disk reserve", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    const origin = await listen((_request, response) => {
      response.setHeader("content-type", "text/plain");
      response.setHeader("content-length", String(Number.MAX_SAFE_INTEGER));
      response.end();
    });
    const { brainId, service } = await crawlerService("Resource pause");

    const result = await service.crawlWeb({
      brainId,
      url: `${origin}/`,
      maxPages: 1,
      respectRobots: false,
      quarantine: true,
    });

    expect(result).toMatchObject({
      visited: 0,
      skipped: 0,
      frontierRemaining: 1,
      stopped: true,
      coverage: { complete: false },
    });
    expect(result.warnings.join(" ")).toMatch(/disk reserve/i);
  });

  it("preserves a crawled page on both resource pauses and learner failures", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    const origin = await listen((_request, response) => {
      response.setHeader("content-type", "text/plain");
      response.end("small crawled learning fixture");
    });

    const pausedFixture = await crawlerService("Worker resource pause");
    const pausedEngine = (
      pausedFixture.service as unknown as {
        engine: {
          request(...arguments_: unknown[]): Promise<unknown>;
        };
      }
    ).engine;
    pausedEngine.request = async () => {
      throw new EngineRequestError(
        "Neural learning paused at the allocator watermark.",
        -32020,
        {
          recoverable: true,
          resourcePause: {
            recoverable: true,
            reason: "Active neural state reached its safe RAM reserve.",
            userAction: "Close memory-heavy applications, then resume this crawl.",
          },
        },
      );
    };
    const paused = await pausedFixture.service.crawlWeb({
      brainId: pausedFixture.brainId,
      url: `${origin}/pause`,
      maxPages: 1,
      respectRobots: false,
      quarantine: false,
    });
    expect(paused).toMatchObject({
      visited: 0,
      skipped: 0,
      frontierRemaining: 1,
      stopped: true,
      coverage: { complete: false },
    });
    expect(paused.warnings.join(" ")).toMatch(/allocator watermark/i);
    expect(paused.warnings.join(" ")).toMatch(/close memory-heavy applications/i);

    const failedFixture = await crawlerService("Worker hard failure");
    const failedEngine = (
      failedFixture.service as unknown as {
        engine: {
          request(...arguments_: unknown[]): Promise<unknown>;
        };
      }
    ).engine;
    failedEngine.request = async () => {
      throw new EngineRequestError(
        "RuntimeError: unrelated worker failure",
        -32000,
        { recoverable: false },
      );
    };
    const failed = await failedFixture.service.crawlWeb({
      brainId: failedFixture.brainId,
      url: `${origin}/failure`,
      maxPages: 1,
      respectRobots: false,
      quarantine: false,
    });
    expect(failed).toMatchObject({
      visited: 0,
      skipped: 0,
      frontierRemaining: 1,
      stopped: true,
      coverage: { complete: false },
    });
    expect(failed.warnings.join(" ")).toMatch(/unrelated worker failure/i);
  });
});
