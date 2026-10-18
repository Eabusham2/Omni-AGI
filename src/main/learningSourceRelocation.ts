import { createHash } from "node:crypto";
import { lstat, mkdir, realpath } from "node:fs/promises";
import { isAbsolute, join, relative } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { DatasetManifestEntry } from "../shared/types";

export interface LearningSourceRelocation {
  originalPath: string;
  originalSourceIdentity: string;
  identityPaths?: string[];
  sourceName: string;
  contentSha256: string;
  bytes: number;
  relativePath: string;
  originalManifestId?: string;
  parentId?: string;
}
const SHA = /^[a-f0-9]{64}$/;
export const learningRelocationPath = (root: string): string => join(root, "learning", "source-relocations.sqlite3");
function checked(value: unknown): LearningSourceRelocation {
  const row = value as LearningSourceRelocation;
  if (!row || typeof row !== "object" || typeof row.originalPath !== "string" || row.originalPath.includes("\0") ||
      !SHA.test(row.originalSourceIdentity) || !SHA.test(row.contentSha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 0 ||
      typeof row.sourceName !== "string" || !row.sourceName || row.sourceName.includes("\0") ||
      typeof row.relativePath !== "string" || !/^(?:learning\/source-blobs\/[a-f0-9]{64}\.data|datasets\/[A-Za-z0-9._-]+\/(?:snapshots\/)?[a-f0-9]{64}\.sqlite|datasets\/web-cache(?:\/[A-Za-z0-9._-]+)?\/[a-f0-9]{64}\.[a-z0-9]+)$/.test(row.relativePath) ||
      row.identityPaths !== undefined && (!Array.isArray(row.identityPaths) || row.identityPaths.some(path => typeof path !== "string" || !path || path.includes("\0"))) ||
      ![row.originalPath, ...(row.identityPaths ?? [])].some(path => typeof path === "string" && row.originalSourceIdentity === createHash("sha256").update(`file\0${path}`).digest("hex"))) throw new Error("Saved source relocation lost its exact original source identity.");
  return row;
}
export async function openLearningRelocations(root: string, brainId: string, write = false): Promise<LearningSourceRelocations | undefined> {
  const path = learningRelocationPath(root), info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
  if (!info && !write) return undefined;
  if (info && (!info.isFile() || info.isSymbolicLink())) throw new Error("Unsafe saved source-relocation index.");
  if (write) await mkdir(join(root, "learning"), { recursive: true, mode: 0o700 });
  const database = new DatabaseSync(path, { readOnly: !write });
  try {
    if (write) database.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS bindings(manifest_id TEXT NOT NULL,entry_index INTEGER NOT NULL,body TEXT NOT NULL,PRIMARY KEY(manifest_id,entry_index)); CREATE TABLE IF NOT EXISTS paths(relative_path TEXT PRIMARY KEY,body TEXT NOT NULL);");
    const owner = database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value;
    if (owner !== undefined && owner !== brainId) throw new Error("Saved source relocation belongs to another instance.");
    if (write) database.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
    return new LearningSourceRelocations(root, database);
  } catch (error) { database.close(); throw error; }
}

export class LearningSourceRelocations {
  constructor(private readonly root: string, private readonly database: DatabaseSync) {}
  close(): void { this.database.close(); }
  add(manifestId: string, entryIndex: number, value: LearningSourceRelocation): void {
    const row = checked(value);
    const existing = this.database.prepare("SELECT body FROM bindings WHERE manifest_id=? AND entry_index=?").get(manifestId, entryIndex);
    if (existing && existing.body !== JSON.stringify(row)) throw new Error("A retained source binding changed during adoption.");
    this.database.prepare("INSERT OR IGNORE INTO bindings VALUES(?,?,?)").run(manifestId, entryIndex, JSON.stringify(row));
    this.addGeneric(row);
  }
  /** Adoption may extend the path lineage, never replace a source's bytes. */
  adopt(manifestId: string, entryIndex: number, value: LearningSourceRelocation): void {
    const row = checked(value), old = this.source(manifestId, entryIndex);
    if (old) {
      if (old.originalPath !== row.originalPath || old.contentSha256 !== row.contentSha256 || old.bytes !== row.bytes || old.sourceName !== row.sourceName ||
          old.originalManifestId !== row.originalManifestId || old.parentId !== row.parentId) throw new Error("Retained source adoption changed its canonical identity.");
      const incoming = new Set([row.originalPath, ...(row.identityPaths ?? [])]);
      if (![old.originalPath, ...(old.identityPaths ?? []), join(this.root, ...old.relativePath.split("/"))].every(path => incoming.has(path))) throw new Error("Retained source adoption lost a prior path identity.");
      this.database.prepare("UPDATE bindings SET body=? WHERE manifest_id=? AND entry_index=?").run(JSON.stringify(row), manifestId, entryIndex);
      this.addGeneric(row);
    } else this.add(manifestId, entryIndex, row);
  }
  addGeneric(value: LearningSourceRelocation): void {
    const row = checked(value);
    const path = this.database.prepare("SELECT body FROM paths WHERE relative_path=?").get(row.relativePath);
    // Shared identical bytes may have different original path identities.
    // Dataset lookup remains exact; ambiguous generic lookup is never guessed.
    if (!path) this.database.prepare("INSERT INTO paths VALUES(?,?)").run(row.relativePath, JSON.stringify(row));
    else if (path.body !== JSON.stringify(row)) {
      const old = path.body === "null" ? undefined : checked(JSON.parse(String(path.body)));
      if (old && old.originalPath === row.originalPath && old.contentSha256 === row.contentSha256 && old.sourceName === row.sourceName && old.parentId === row.parentId)
        this.database.prepare("UPDATE paths SET body=? WHERE relative_path=?").run(JSON.stringify(row), row.relativePath);
      else this.database.prepare("UPDATE paths SET body='null' WHERE relative_path=?").run(row.relativePath);
    }
  }
  source(manifestId: string, entryIndex: number): LearningSourceRelocation | undefined {
    const row = this.database.prepare("SELECT body FROM bindings WHERE manifest_id=? AND entry_index=?").get(manifestId, entryIndex);
    return row ? checked(JSON.parse(String(row.body))) : undefined;
  }
  generic(path: string): LearningSourceRelocation | undefined {
    const fromRoot = relative(this.root, path).replaceAll("\\", "/");
    if (isAbsolute(fromRoot) || fromRoot.startsWith("../")) return undefined;
    const row = this.database.prepare("SELECT body FROM paths WHERE relative_path=?").get(fromRoot);
    return row && row.body !== "null" ? checked(JSON.parse(String(row.body))) : undefined;
  }
  async resolve(manifestId: string, entry: DatasetManifestEntry): Promise<DatasetManifestEntry> {
    const source = this.source(manifestId, entry.index);
    if (!source) return entry;
    if (source.originalPath !== entry.path || source.contentSha256 !== entry.contentSha256 || source.bytes !== entry.bytes) throw new Error("Retained source binding differs from its canonical manifest entry.");
    const path = join(this.root, ...source.relativePath.split("/")), info = await lstat(path);
    const actual = await realpath(path), owner = await realpath(this.root), fromRoot = relative(owner, actual);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== source.bytes || isAbsolute(fromRoot) || fromRoot.startsWith("..")) throw new Error("Retained dataset source is missing or escaped its owned instance.");
    return { ...entry, path, lastModifiedMs: info.mtimeMs, sourceRelocation: source };
  }
}
