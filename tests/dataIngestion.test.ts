import { createServer, type RequestListener, type Server } from "node:http";
import { mkdir, mkdtemp, open, readFile, rm, writeFile } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  CrawlFrontierStore,
  DatasetManifestStore,
  streamUtf8Text
} from "../src/main/dataIngestion";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG } from "../src/shared/types";

describe("whole-dataset persistence", () => {
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
          })
      )
    );
    if (priorAllowLocal === undefined) delete process.env.OMNI_ALLOW_LOCAL_URLS;
    else process.env.OMNI_ALLOW_LOCAL_URLS = priorAllowLocal;
    if (priorPacing === undefined) delete process.env.OMNI_CRAWL_MIN_DELAY_MS;
    else process.env.OMNI_CRAWL_MIN_DELAY_MS = priorPacing;
    if (priorBackoff === undefined) delete process.env.OMNI_CRAWL_BACKOFF_BASE_MS;
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
      params: Record<string, unknown>
    ) => {
      engineRequests.push(params);
      return {
        source: {
          learned_ideas: 1,
          learned_concepts: 1,
          plasticity_events: 1
        }
      };
    };
    const service = new BrainService(repository, {
      tryRequest: neuralIngest,
      request: neuralIngest
    } as unknown as EngineSupervisor);
    return { brainId: brain.id, service, engineRequests };
  }

  it("walks beyond the former 2,000-file ceiling and persists an exhaustive cursor", async () => {
    const dataset = join(root, "many-files");
    await mkdir(dataset);
    for (let index = 0; index < 2_005; index += 1) {
      await writeFile(join(dataset, `${String(index).padStart(4, "0")}.txt`), `row ${index}`);
    }
    const store = new DatasetManifestStore(() => brainDirectory);
    const manifest = await store.create("brain-a", [dataset]);
    expect(manifest.discoveredFiles).toBe(2_005);
    expect(manifest.discoveredBytes).toBeGreaterThan(2_005);

    let visited = 0;
    for await (const entry of store.entries("brain-a", manifest.id)) {
      expect(entry.index).toBe(visited);
      visited += 1;
    }
    expect(visited).toBe(2_005);

    const cursor = await store.cursor("brain-a", manifest.id);
    cursor.nextEntry = 1_337;
    cursor.processedFiles = 1_337;
    cursor.state = "paused";
    await store.saveCursor("brain-a", cursor);
    await expect(store.cursor("brain-a", manifest.id)).resolves.toMatchObject({
      nextEntry: 1_337,
      processedFiles: 1_337,
      state: "paused"
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
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Large dataset" });
    const path = join(root, "large.parquet");
    const handle = await open(path, "w");
    await handle.truncate(129 * 1024 * 1024);
    await handle.close();
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 17,
          learned_concepts: 23,
          plasticity_events: 41
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
          errors: []
        }
      })
    } as unknown as EngineSupervisor);
    const results = await service.ingestPaths(brain.id, [path]);
    expect(results).toHaveLength(1);
    expect(results[0]!.source).toMatchObject({
      kind: "parquet",
      bytes: 129 * 1024 * 1024,
      learnedIdeas: 17,
      learnedConcepts: 23,
      learnedSynapses: 41
    });
    expect(results[0]!.coverage).toMatchObject({
      processedFiles: 1,
      discoveredRecords: 500,
      processedRecords: 500,
      rejectedRecords: 0
    });
  });

  it("keeps the cursor on an uncommitted file when neural training pauses", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Resume exactly" });
    const path = join(root, "resource-pause.txt");
    await writeFile(path, "This record must be retried, never skipped.");
    const service = new BrainService(repository, {
      request: async () => {
        throw new Error(
          "SubstrateResourcePause: neural substrate growth paused at the host resource reserve"
        );
      }
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);
    expect(run.paused).toBe(true);
    expect(run.coverage).toMatchObject({
      processedFiles: 0,
      rejectedFiles: 0,
      complete: false
    });

    const store = new DatasetManifestStore((id) => repository.brainDirectory(id));
    await expect(store.cursor(brain.id, manifest.id)).resolves.toMatchObject({
      nextEntry: 0,
      processedFiles: 0,
      state: "paused"
    });
  });

  it("propagates worker record errors into the durable coverage report", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Coverage errors" });
    const path = join(root, "mixed.jsonl");
    await writeFile(path, '{"ok":true}\nnot-json\n');
    const service = new BrainService(repository, {
      request: async () => ({
        source: {
          learned_ideas: 1,
          learned_concepts: 2,
          plasticity_events: 3
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
          errors: [{ source: "mixed.jsonl#line-2", message: "invalid JSONL" }]
        }
      })
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(brain.id, manifest.id);
    expect(run.coverage).toMatchObject({
      processedFiles: 1,
      rejectedFiles: 0,
      discoveredRecords: 2,
      processedRecords: 1,
      rejectedRecords: 1,
      complete: true
    });
    expect(run.coverage.errors).toContainEqual({
      source: "mixed.jsonl#line-2",
      message: "invalid JSONL"
    });
    const errorLog = join(
      repository.brainDirectory(brain.id),
      "datasets",
      manifest.id,
      "errors.ndjson"
    );
    await expect(readFile(errorLog, "utf8")).resolves.toContain(
      '"source":"mixed.jsonl#line-2"'
    );
  });

  it("visits every file in every requested epoch and explicitly replays later epochs", async () => {
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Two epochs" });
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
            plasticity_events: 1
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
            errors: []
          }
        };
      }
    } as unknown as EngineSupervisor);
    const manifest = await service.previewDataset(brain.id, [path]);

    const run = await service.ingestManifest(
      brain.id,
      manifest.id,
      "encode",
      () => false,
      () => undefined,
      true,
      2
    );
    expect(calls).toHaveLength(2);
    expect(calls.map((params) => params.allowReplay)).toEqual([false, true]);
    expect(calls.map((params) => params.epoch)).toEqual([0, 1]);
    expect(run.coverage).toMatchObject({
      requestedEpochs: 2,
      completedEpochs: 2,
      discoveredFiles: 2,
      processedFiles: 2,
      complete: true
    });
    expect(run.results).toHaveLength(2);
  });

  it("persists an unbounded crawl frontier in SQLite and resumes in-flight pages", async () => {
    const start = "https://example.com/";
    const frontier = await CrawlFrontierStore.create(
      brainDirectory,
      "brain-a",
      start,
      "crawl-a",
      false
    );
    frontier.enqueue(
      Array.from({ length: 5_100 }, (_, index) => ({
        url: `https://example.com/page/${index}`,
        depth: 1
      }))
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
      true
    );
    const counts = resumed.counts();
    expect(counts).toMatchObject({
      queued: 5_099,
      visited: 1,
      skipped: 1,
      processedBytes: 123
    });
    expect(resumed.warnings()).toEqual([
      "https://example.com/page/0: fixture failure"
    ]);
    resumed.close();
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
      quarantine: true
    });

    expect(result).toMatchObject({
      visited: 0,
      skipped: 1,
      frontierRemaining: 0,
      stopped: false
    });
    expect(targetPageHits).toBe(0);
    expect(result.warnings.join(" ")).toMatch(/outside the same-site crawl origin/i);
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
      quarantine: true
    });

    expect(result).toMatchObject({ visited: 0, skipped: 1, stopped: false });
    expect(targetRobotsHits).toBe(1);
    expect(targetPageHits).toBe(0);
    expect(result.warnings.join(" ")).toMatch(/robots\.txt disallows this path/i);
  });

  it("paces same-domain requests and retries throttled pages with backoff", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "20";
    process.env.OMNI_CRAWL_BACKOFF_BASE_MS = "30";
    const requestTimes: number[] = [];
    let throttledHits = 0;
    const origin = await listen((request, response) => {
      requestTimes.push(Date.now());
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
      quarantine: true
    });

    expect(result.warnings).toEqual([]);
    expect(result).toMatchObject({ visited: 3, skipped: 0, stopped: false });
    expect(throttledHits).toBe(2);
    expect(requestTimes).toHaveLength(4);
    for (let index = 1; index < requestTimes.length; index += 1) {
      expect(requestTimes[index]! - requestTimes[index - 1]!).toBeGreaterThanOrEqual(12);
    }
  });

  it("discovers and neurally trains linked image, audio, and video responses", async () => {
    process.env.OMNI_ALLOW_LOCAL_URLS = "1";
    process.env.OMNI_CRAWL_MIN_DELAY_MS = "1";
    const origin = await listen((request, response) => {
      if (request.url === "/") {
        response.setHeader("content-type", "text/html");
        response.end(
          '<img src="/scene.png"><audio src="/sound.wav"></audio>' +
            '<video poster="/poster.webp"><source src="/clip.mp4"></video>'
        );
        return;
      }
      if (request.url === "/scene.png" || request.url === "/poster.webp") {
        response.setHeader(
          "content-type",
          request.url.endsWith(".webp") ? "image/webp" : "image/png"
        );
        response.end(
          request.url.endsWith(".webp")
            ? Buffer.from("RIFF-WEBP-media-fixture")
            : Buffer.from([0x89, 0x50, 0x4e, 0x47, 1, 2, 3])
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
    const { brainId, service, engineRequests } = await crawlerService(
      "Multimodal crawler"
    );

    const result = await service.crawlWeb({
      brainId,
      url: `${origin}/`,
      maxPages: 5,
      maxDepth: 1,
      concurrency: 4,
      respectRobots: false,
      quarantine: false,
      policy: "pretrain"
    });

    expect(result).toMatchObject({
      visited: 5,
      skipped: 0,
      coverage: {
        complete: true,
        modalityCounts: { text: 1, image: 2, audio: 1, video: 1 }
      }
    });
    expect(
      engineRequests
        .map((params) => params.kind)
        .filter((kind) => ["image", "audio", "video"].includes(String(kind)))
    ).toEqual(expect.arrayContaining(["image", "image", "audio", "video"]));
    expect(
      engineRequests
        .filter((params) => params.kind === "image")
        .map((params) => String(params.path))
    ).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/\.png$/),
        expect.stringMatching(/\.webp$/)
      ])
    );
    const saved = await service.repository.get(brainId);
    expect(
      saved.trainingSources
        .filter((source) => ["image", "audio", "video"].includes(source.kind))
        .map((source) => ({
          kind: source.kind,
          provenanceUrl: source.provenanceUrl,
          blobHash: source.blobHash
        }))
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "image",
          provenanceUrl: `${origin}/scene.png`,
          blobHash: expect.stringMatching(/^[a-f0-9]{64}$/)
        }),
        expect.objectContaining({
          kind: "audio",
          provenanceUrl: `${origin}/sound.wav`,
          blobHash: expect.stringMatching(/^[a-f0-9]{64}$/)
        }),
        expect.objectContaining({
          kind: "video",
          provenanceUrl: `${origin}/clip.mp4`,
          blobHash: expect.stringMatching(/^[a-f0-9]{64}$/)
        })
      ])
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
      quarantine: true
    });

    expect(result).toMatchObject({
      visited: 0,
      skipped: 0,
      frontierRemaining: 1,
      stopped: true,
      coverage: { complete: false }
    });
    expect(result.warnings.join(" ")).toMatch(/disk reserve/i);
  });
});
