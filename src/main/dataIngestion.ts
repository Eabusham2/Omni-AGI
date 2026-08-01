import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import {
  access,
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  stat,
  writeFile
} from "node:fs/promises";
import { createInterface } from "node:readline";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";
import { once } from "node:events";
import { DatabaseSync } from "node:sqlite";
import type {
  DatasetCursor,
  DatasetFormat,
  DatasetManifest,
  DatasetManifestEntry,
  TrainingCoverage,
  TrainingCoverageError
} from "../shared/types";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const STREAM_TEXT_CHARS = 256 * 1024;

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.next`;
  await writeFile(temporary, JSON.stringify(value, null, 2), {
    encoding: "utf8",
    flag: "wx",
    mode: 0o600
  });
  await rename(temporary, path);
}

async function writeLine(
  stream: ReturnType<typeof createWriteStream>,
  value: string
): Promise<void> {
  if (!stream.write(value)) await once(stream, "drain");
}

export function detectDatasetFormat(path: string): DatasetFormat {
  const lower = basename(path).toLocaleLowerCase();
  const extension = extname(lower);
  if (lower.endsWith(".tar.gz") || lower.endsWith(".tar.bz2") || lower.endsWith(".tar.xz")) {
    return "archive";
  }
  if (
    lower.endsWith(".hf.json") ||
    lower.endsWith(".dataset.json") ||
    lower.endsWith(".manifest.json") ||
    lower === "dataset_info.json" ||
    lower === "dataset_infos.json"
  ) {
    return "huggingface";
  }
  if (extension === ".pdf") return "pdf";
  if (extension === ".epub") return "epub";
  if ([".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp"].includes(extension)) {
    return "office";
  }
  if (extension === ".csv") return "csv";
  if (extension === ".tsv") return "tsv";
  if (extension === ".json") return "json";
  if ([".jsonl", ".ndjson"].includes(extension)) return "jsonl";
  if (extension === ".parquet") return "parquet";
  if ([".arrow", ".feather", ".ipc"].includes(extension)) return "arrow";
  if ([".sqlite", ".sqlite3", ".db"].includes(extension)) return "sqlite";
  if ([".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"].includes(extension)) {
    return extension === ".tar" || extension === ".tgz" ? "webdataset" : "archive";
  }
  if (
    [
      ".txt",
      ".md",
      ".mdx",
      ".rst",
      ".py",
      ".js",
      ".jsx",
      ".ts",
      ".tsx",
      ".rs",
      ".go",
      ".java",
      ".c",
      ".cc",
      ".cpp",
      ".h",
      ".hpp",
      ".cs",
      ".swift",
      ".kt",
      ".sql",
      ".html",
      ".htm",
      ".css",
      ".scss",
      ".yaml",
      ".yml",
      ".toml"
    ].includes(extension)
  ) {
    return "text";
  }
  if (
    [
      ".png",
      ".jpg",
      ".jpeg",
      ".webp",
      ".gif",
      ".bmp",
      ".tif",
      ".tiff",
      ".avif",
      ".heic",
      ".heif"
    ].includes(extension)
  ) {
    return "image";
  }
  if (
    [
      ".wav",
      ".mp3",
      ".flac",
      ".m4a",
      ".aac",
      ".ogg",
      ".oga",
      ".opus",
      ".aiff",
      ".aif",
      ".wma"
    ].includes(extension)
  ) {
    return "audio";
  }
  if (
    [
      ".mp4",
      ".webm",
      ".mov",
      ".mkv",
      ".avi",
      ".m4v",
      ".mpeg",
      ".mpg",
      ".wmv",
      ".flv"
    ].includes(extension)
  ) {
    return "video";
  }
  return "unknown";
}

async function* walkRegularFiles(
  roots: string[]
): AsyncGenerator<{ path: string; root: string }> {
  const seen = new Set<string>();
  const queue = [...new Set(roots.map((value) => resolve(value)))].map((path) => ({
    path,
    root: path
  }));
  while (queue.length > 0) {
    const next = queue.shift();
    if (!next) break;
    const info = await lstat(next.path);
    if (info.isSymbolicLink()) continue;
    const canonical = await realpath(next.path);
    if (seen.has(canonical)) continue;
    seen.add(canonical);
    if (info.isFile()) {
      yield { path: canonical, root: next.root };
      continue;
    }
    if (!info.isDirectory()) continue;
    const children = await readdir(canonical);
    for (const name of children.sort((left, right) => left.localeCompare(right))) {
      queue.push({ path: join(canonical, name), root: next.root });
    }
  }
}

export async function hashFile(path: string): Promise<string> {
  const digest = createHash("sha256");
  const stream = createReadStream(path);
  for await (const chunk of stream) digest.update(chunk as Buffer);
  return digest.digest("hex");
}

export async function* streamUtf8Text(path: string): AsyncGenerator<string> {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let pending = "";
  for await (const chunk of createReadStream(path)) {
    pending += decoder.decode(chunk as Buffer, { stream: true }).replace(/\0/g, "");
    while (pending.length >= STREAM_TEXT_CHARS) {
      const splitAt = Math.max(
        pending.lastIndexOf("\n", STREAM_TEXT_CHARS),
        pending.lastIndexOf(" ", STREAM_TEXT_CHARS)
      );
      const boundary = splitAt > 0 ? splitAt : STREAM_TEXT_CHARS;
      yield pending.slice(0, boundary);
      pending = pending.slice(boundary);
    }
  }
  pending += decoder.decode();
  if (pending) yield pending;
}

export class DatasetManifestStore {
  constructor(private readonly brainDirectory: (brainId: string) => string) {}

  private datasetRoot(brainId: string): string {
    return join(this.brainDirectory(brainId), "datasets");
  }

  private manifestDirectory(brainId: string, manifestId: string): string {
    if (!SAFE_ID.test(manifestId)) throw new Error("Invalid dataset manifest id.");
    return join(this.datasetRoot(brainId), manifestId);
  }

  private manifestPath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "manifest.json");
  }

  private cursorPath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "cursor.json");
  }

  private coveragePath(brainId: string, manifestId: string): string {
    return join(this.manifestDirectory(brainId, manifestId), "coverage.json");
  }

  async create(brainId: string, inputPaths: string[]): Promise<DatasetManifest> {
    if (inputPaths.length === 0) throw new Error("Choose at least one dataset path.");
    const roots = [...new Set(inputPaths.map((value) => resolve(value)))];
    const id = randomUUID();
    const directory = this.manifestDirectory(brainId, id);
    const entryPath = join(directory, "entries.ndjson");
    await mkdir(directory, { recursive: true });
    const output = createWriteStream(entryPath, { encoding: "utf8", flags: "wx", mode: 0o600 });
    const digest = createHash("sha256");
    let discoveredFiles = 0;
    let discoveredBytes = 0;
    try {
      for await (const entry of walkRegularFiles(roots)) {
        const info = await stat(entry.path);
        if (!info.isFile()) continue;
        const relativePath =
          roots.length === 1 && roots[0] === entry.root && (await lstat(entry.root)).isFile()
            ? basename(entry.path)
            : relative(entry.root, entry.path).split(sep).join("/");
        const record: DatasetManifestEntry = {
          index: discoveredFiles,
          path: entry.path,
          relativePath,
          format: detectDatasetFormat(entry.path),
          bytes: info.size
        };
        const line = `${JSON.stringify(record)}\n`;
        digest.update(line);
        await writeLine(output, line);
        discoveredFiles += 1;
        discoveredBytes += info.size;
      }
    } finally {
      output.end();
      await once(output, "close");
    }
    const now = new Date().toISOString();
    const manifest: DatasetManifest = {
      schemaVersion: 1,
      id,
      brainId,
      createdAt: now,
      updatedAt: now,
      roots,
      entryFile: "entries.ndjson",
      discoveredFiles,
      discoveredBytes,
      manifestHash: digest.digest("hex")
    };
    const cursor: DatasetCursor = {
      schemaVersion: 1,
      manifestId: id,
      currentEpoch: 0,
      requestedEpochs: 1,
      nextEntry: 0,
      nextRecord: 0,
      processedFiles: 0,
      processedRecords: 0,
      processedBytes: 0,
      state: "ready",
      updatedAt: now
    };
    const coverage: TrainingCoverage = {
      schemaVersion: 1,
      manifestId: id,
      requestedEpochs: 1,
      completedEpochs: 0,
      discoveredFiles,
      processedFiles: 0,
      rejectedFiles: 0,
      discoveredRecords: 0,
      processedRecords: 0,
      rejectedRecords: 0,
      discoveredBytes,
      processedBytes: 0,
      shards: 0,
      modalityCounts: {},
      errors: [],
      errorLog: "errors.ndjson",
      complete: discoveredFiles === 0,
      updatedAt: now
    };
    await Promise.all([
      atomicJson(this.manifestPath(brainId, id), manifest),
      atomicJson(this.cursorPath(brainId, id), cursor),
      atomicJson(this.coveragePath(brainId, id), coverage)
    ]);
    return manifest;
  }

  async manifest(brainId: string, manifestId: string): Promise<DatasetManifest> {
    const parsed = JSON.parse(
      await readFile(this.manifestPath(brainId, manifestId), "utf8")
    ) as DatasetManifest;
    if (
      parsed.schemaVersion !== 1 ||
      parsed.id !== manifestId ||
      parsed.brainId !== brainId ||
      !/^[a-f0-9]{64}$/.test(parsed.manifestHash)
    ) {
      throw new Error("Dataset manifest is invalid or belongs to another brain.");
    }
    return parsed;
  }

  async cursor(brainId: string, manifestId: string): Promise<DatasetCursor> {
    const parsed = JSON.parse(
      await readFile(this.cursorPath(brainId, manifestId), "utf8")
    ) as DatasetCursor;
    if (parsed.schemaVersion !== 1 || parsed.manifestId !== manifestId) {
      throw new Error("Dataset cursor is invalid.");
    }
    parsed.nextRecord = Math.max(0, Math.round(parsed.nextRecord ?? 0));
    parsed.currentEpoch = Math.max(0, Math.round(parsed.currentEpoch ?? 0));
    parsed.requestedEpochs = Math.max(
      1,
      Math.round(parsed.requestedEpochs ?? 1)
    );
    return parsed;
  }

  async coverage(brainId: string, manifestId: string): Promise<TrainingCoverage> {
    const parsed = JSON.parse(
      await readFile(this.coveragePath(brainId, manifestId), "utf8")
    ) as TrainingCoverage;
    if (parsed.schemaVersion !== 1 || parsed.manifestId !== manifestId) {
      throw new Error("Dataset coverage report is invalid.");
    }
    return parsed;
  }

  async *entries(
    brainId: string,
    manifestId: string,
    startIndex = 0
  ): AsyncGenerator<DatasetManifestEntry> {
    const manifest = await this.manifest(brainId, manifestId);
    const entryPath = join(this.manifestDirectory(brainId, manifestId), manifest.entryFile);
    const actual = await hashFile(entryPath);
    if (actual !== manifest.manifestHash) {
      throw new Error("Dataset manifest entry checksum failed.");
    }
    const lines = createInterface({
      input: createReadStream(entryPath, { encoding: "utf8" }),
      crlfDelay: Infinity
    });
    for await (const line of lines) {
      if (!line.trim()) continue;
      const entry = JSON.parse(line) as DatasetManifestEntry;
      if (entry.index < startIndex) continue;
      yield entry;
    }
  }

  async saveCursor(brainId: string, cursor: DatasetCursor): Promise<void> {
    cursor.updatedAt = new Date().toISOString();
    await atomicJson(this.cursorPath(brainId, cursor.manifestId), cursor);
  }

  async saveCoverage(brainId: string, coverage: TrainingCoverage): Promise<void> {
    coverage.updatedAt = new Date().toISOString();
    await atomicJson(this.coveragePath(brainId, coverage.manifestId), coverage);
  }

  async recordError(
    brainId: string,
    coverage: TrainingCoverage,
    error: TrainingCoverageError
  ): Promise<void> {
    const path = join(
      this.manifestDirectory(brainId, coverage.manifestId),
      coverage.errorLog ?? "errors.ndjson"
    );
    await mkdir(dirname(path), { recursive: true });
    const stream = createWriteStream(path, { encoding: "utf8", flags: "a", mode: 0o600 });
    stream.end(`${JSON.stringify(error)}\n`);
    await once(stream, "close");
    coverage.errors.push(error);
  }

  async reset(
    brainId: string,
    manifestId: string,
    requestedEpochs = 1
  ): Promise<void> {
    const manifest = await this.manifest(brainId, manifestId);
    const now = new Date().toISOString();
    const epochs = Math.max(1, Math.round(requestedEpochs));
    await writeFile(
      join(this.manifestDirectory(brainId, manifestId), "errors.ndjson"),
      "",
      { encoding: "utf8", mode: 0o600 }
    );
    await Promise.all([
      this.saveCursor(brainId, {
        schemaVersion: 1,
        manifestId,
        currentEpoch: 0,
        requestedEpochs: epochs,
        nextEntry: 0,
        nextRecord: 0,
        processedFiles: 0,
        processedRecords: 0,
        processedBytes: 0,
        state: "ready",
        updatedAt: now
      }),
      this.saveCoverage(brainId, {
        schemaVersion: 1,
        manifestId,
        requestedEpochs: epochs,
        completedEpochs: 0,
        discoveredFiles: manifest.discoveredFiles * epochs,
        processedFiles: 0,
        rejectedFiles: 0,
        discoveredRecords: 0,
        processedRecords: 0,
        rejectedRecords: 0,
        discoveredBytes: manifest.discoveredBytes * epochs,
        processedBytes: 0,
        shards: 0,
        modalityCounts: {},
        errors: [],
        errorLog: "errors.ndjson",
        complete: manifest.discoveredFiles === 0,
        updatedAt: now
      })
    ]);
  }

  async hasManifest(brainId: string, manifestId: string): Promise<boolean> {
    return pathExists(this.manifestPath(brainId, manifestId));
  }
}

export interface CrawlFrontierEntry {
  url: string;
  depth: number;
}

export interface CrawlFrontierCounts {
  queued: number;
  visited: number;
  skipped: number;
  processedBytes: number;
  resultCount: number;
  warningCount: number;
  modalityCounts: Partial<Record<DatasetFormat, number>>;
}

export interface CrawlResultReceipt {
  sourceId: string;
  sourceName: string;
  kind: DatasetFormat;
  bytes: number;
  contentHash?: string;
  learnedIdeas: number;
  learnedConcepts: number;
  learnedSynapses: number;
  warnings: string[];
}

export class CrawlFrontierStore {
  readonly id: string;
  private readonly database: DatabaseSync;

  constructor(
    brainDirectory: string,
    brainId: string,
    startUrl: string,
    requestedId?: string,
    resume = true
  ) {
    this.id =
      requestedId && SAFE_ID.test(requestedId)
        ? requestedId
        : createHash("sha256")
            .update(`${brainId}\0${startUrl}`)
            .digest("hex")
            .slice(0, 32);
    const directory = join(brainDirectory, "datasets", "crawls");
    this.database = new DatabaseSync(join(directory, `${this.id}.sqlite3`), {
      open: false
    });
    // DatabaseSync cannot create a missing parent directory itself.
    // mkdirSync is intentionally avoided elsewhere, so open happens after the
    // caller has prepared this known brain-local directory.
    this.prepare(directory, brainId, startUrl, resume);
  }

  private prepare(
    directory: string,
    brainId: string,
    startUrl: string,
    resume: boolean
  ): void {
    // DatabaseSync.open is synchronous; creating only this exact app-owned
    // directory before construction is handled by CrawlFrontierStore.create.
    this.database.open();
    this.database.exec(`
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS frontier (
        url TEXT PRIMARY KEY,
        depth INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'queued',
        added_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS visited (
        url TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        visited_at TEXT NOT NULL,
        error TEXT,
        bytes INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS warnings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        message TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS result_receipts (
        url TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        source_name TEXT NOT NULL,
        kind TEXT NOT NULL,
        bytes INTEGER NOT NULL DEFAULT 0,
        content_hash TEXT,
        learned_ideas INTEGER NOT NULL DEFAULT 0,
        learned_concepts INTEGER NOT NULL DEFAULT 0,
        learned_synapses INTEGER NOT NULL DEFAULT 0,
        warnings_json TEXT NOT NULL DEFAULT '[]',
        recorded_at TEXT NOT NULL
      );
    `);
    const visitedColumns = this.database
      .prepare("PRAGMA table_info(visited)")
      .all() as unknown as Array<{ name: string }>;
    if (!visitedColumns.some((column) => column.name === "bytes")) {
      this.database.exec("ALTER TABLE visited ADD COLUMN bytes INTEGER NOT NULL DEFAULT 0");
    }
    const existing = this.database
      .prepare("SELECT value FROM meta WHERE key = 'startUrl'")
      .get() as { value?: string } | undefined;
    if (existing?.value && existing.value !== startUrl) {
      throw new Error("The crawl id belongs to a different start URL.");
    }
    if (!resume) {
      this.database.exec(
        "DELETE FROM frontier; DELETE FROM visited; DELETE FROM warnings; DELETE FROM result_receipts;"
      );
    }
    const insertMeta = this.database.prepare(
      "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)"
    );
    insertMeta.run("brainId", brainId);
    insertMeta.run("startUrl", startUrl);
    insertMeta.run("updatedAt", new Date().toISOString());
    this.database
      .prepare("UPDATE frontier SET state = 'queued' WHERE state = 'inflight'")
      .run();
    this.enqueue([{ url: startUrl, depth: 0 }]);
  }

  static async create(
    brainDirectory: string,
    brainId: string,
    startUrl: string,
    requestedId?: string,
    resume = true
  ): Promise<CrawlFrontierStore> {
    await mkdir(join(brainDirectory, "datasets", "crawls"), { recursive: true });
    return new CrawlFrontierStore(
      brainDirectory,
      brainId,
      startUrl,
      requestedId,
      resume
    );
  }

  private transact(operation: () => void): void {
    this.database.exec("BEGIN IMMEDIATE");
    try {
      operation();
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }

  enqueue(entries: CrawlFrontierEntry[]): void {
    const insert = this.database.prepare(`
      INSERT OR IGNORE INTO frontier(url, depth, state, added_at)
      SELECT ?, ?, 'queued', ?
      WHERE NOT EXISTS (SELECT 1 FROM visited WHERE url = ?)
    `);
    this.transact(() => {
      const now = new Date().toISOString();
      for (const entry of entries) {
        insert.run(entry.url, Math.max(0, Math.round(entry.depth)), now, entry.url);
      }
    });
  }

  next(limit: number): CrawlFrontierEntry[] {
    const entries = this.database
      .prepare(
        "SELECT url, depth FROM frontier WHERE state = 'queued' ORDER BY rowid LIMIT ?"
      )
      .all(Math.max(1, Math.round(limit))) as unknown as CrawlFrontierEntry[];
    const mark = this.database.prepare(
      "UPDATE frontier SET state = 'inflight' WHERE url = ?"
    );
    this.transact(() => {
      for (const entry of entries) mark.run(entry.url);
    });
    return entries;
  }

  visited(url: string, bytes = 0, receipt?: CrawlResultReceipt): void {
    this.finish(url, "visited", undefined, bytes, receipt);
  }

  skipped(url: string, message?: string): void {
    this.finish(url, "skipped", message);
  }

  retry(url: string): void {
    this.database
      .prepare("UPDATE frontier SET state = 'queued' WHERE url = ?")
      .run(url);
  }

  private finish(
    url: string,
    status: "visited" | "skipped",
    error?: string,
    bytes = 0,
    receipt?: CrawlResultReceipt
  ): void {
    this.transact(() => {
      this.database.prepare("DELETE FROM frontier WHERE url = ?").run(url);
      this.database
        .prepare(
          "INSERT OR REPLACE INTO visited(url, status, visited_at, error, bytes) VALUES (?, ?, ?, ?, ?)"
        )
        .run(url, status, new Date().toISOString(), error ?? null, Math.max(0, bytes));
      if (error) {
        this.database
          .prepare("INSERT INTO warnings(url, message) VALUES (?, ?)")
          .run(url, error);
      }
      if (receipt) {
        this.database
          .prepare(
            `INSERT OR REPLACE INTO result_receipts(
              url, source_id, source_name, kind, bytes, content_hash,
              learned_ideas, learned_concepts, learned_synapses,
              warnings_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
          )
          .run(
            url,
            receipt.sourceId,
            receipt.sourceName,
            receipt.kind,
            Math.max(0, Math.round(receipt.bytes)),
            receipt.contentHash ?? null,
            Math.max(0, Math.round(receipt.learnedIdeas)),
            Math.max(0, Math.round(receipt.learnedConcepts)),
            Math.max(0, Math.round(receipt.learnedSynapses)),
            JSON.stringify(receipt.warnings),
            new Date().toISOString()
          );
      }
      this.database
        .prepare("INSERT OR REPLACE INTO meta(key, value) VALUES ('updatedAt', ?)")
        .run(new Date().toISOString());
    });
  }

  recordWarning(url: string, message: string): void {
    this.database
      .prepare("INSERT INTO warnings(url, message) VALUES (?, ?)")
      .run(url, message);
  }

  counts(): CrawlFrontierCounts {
    const queued = this.database
      .prepare("SELECT COUNT(*) AS count FROM frontier")
      .get() as { count: number };
    const counts = this.database
      .prepare(
        "SELECT status, COUNT(*) AS count, COALESCE(SUM(bytes), 0) AS bytes FROM visited GROUP BY status"
      )
      .all() as unknown as Array<{ status: string; count: number; bytes: number }>;
    const receiptCounts = this.database
      .prepare("SELECT kind, COUNT(*) AS count FROM result_receipts GROUP BY kind")
      .all() as unknown as Array<{ kind: DatasetFormat; count: number }>;
    const warnings = this.database
      .prepare("SELECT COUNT(*) AS count FROM warnings")
      .get() as { count: number };
    const modalityCounts: Partial<Record<DatasetFormat, number>> = {};
    for (const entry of receiptCounts) {
      modalityCounts[entry.kind] = Number(entry.count);
    }
    return {
      queued: Number(queued.count),
      visited: Number(counts.find((entry) => entry.status === "visited")?.count ?? 0),
      skipped: Number(counts.find((entry) => entry.status === "skipped")?.count ?? 0),
      processedBytes: Number(
        counts.find((entry) => entry.status === "visited")?.bytes ?? 0
      ),
      resultCount: receiptCounts.reduce((sum, entry) => sum + Number(entry.count), 0),
      warningCount: Number(warnings.count),
      modalityCounts
    };
  }

  warnings(limit = 64): string[] {
    const records = this.database
      .prepare("SELECT url, message FROM warnings ORDER BY id DESC LIMIT ?")
      .all(Math.max(1, Math.round(limit))) as unknown as Array<{
      url: string;
      message: string;
    }>;
    return records.reverse().map((entry) => `${entry.url}: ${entry.message}`);
  }

  close(): void {
    this.database.close();
  }
}
