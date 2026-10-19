import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { DEFAULT_INSTANCE_VOICE_SETTINGS, instanceVoiceSettingsFromLegacy, validateInstanceVoiceSettings,
  type InstanceVoicePreferences, type InstanceVoiceSettings } from "../shared/voiceRouting";
import type { LiveVoicePreferences } from "../shared/liveVoice";
import { sqliteFilename } from "./sqliteFilename";

export const VOICE_PREFERENCES_RELATIVE_PATH = "voice/workspace.sqlite3";
const FORMAT = "omni-instance-voice-preferences-1";
type VoicePreferenceSource = "fresh-auto" | "import-auto" | "copy-auto" | "merged-auto" | "legacy-global-v1" | "manual";
const sources = new Set<VoicePreferenceSource>(["fresh-auto", "import-auto", "copy-auto", "merged-auto", "legacy-global-v1", "manual"]);
const id = (value: unknown): string => {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid voice preference owner.");
  return value;
};
const canonical = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  const text = JSON.stringify(value); if (text === undefined) throw new Error("Voice preference history requires explicit JSON."); return text;
};
async function exists(path: string): Promise<boolean> {
  try { const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Voice preferences require a regular owned file."); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}

interface StoredPreference {
  revision: number; preferences: InstanceVoiceSettings; updatedAt: string; source: VoicePreferenceSource;
}
function validateStored(row: Record<string, unknown>): StoredPreference {
  const value = JSON.parse(String(row.payload)) as unknown;
  if (!Number.isSafeInteger(row.revision) || Number(row.revision) < 1 || typeof row.updated_at !== "string" ||
      !Number.isFinite(Date.parse(row.updated_at)) || !sources.has(row.source as VoicePreferenceSource)) {
    throw new Error("Saved instance voice preferences are invalid.");
  }
  return { revision: Number(row.revision), preferences: validateInstanceVoiceSettings(value),
    updatedAt: row.updated_at, source: row.source as VoicePreferenceSource };
}

/** Independent host settings: neural writes cannot overwrite a later choice. */
export class VoicePreferencesStore {
  private constructor(readonly db: DatabaseSync, readonly brainId: string) {}

  static async open(root: string, brainId: string, create = false): Promise<VoicePreferencesStore | undefined> {
    id(brainId);
    const path = join(root, VOICE_PREFERENCES_RELATIVE_PATH);
    const present = await exists(path);
    if (!present && !create) return undefined;
    const parent = join(root, "voice");
    if (create) await mkdir(parent, { recursive: true, mode: 0o700 });
    const info = await lstat(parent);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Voice preference directory is not owned.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: !create });
    try {
      db.exec("PRAGMA cache_size=-64; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      if (create) {
        db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; BEGIN IMMEDIATE;");
        try {
          db.exec("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); " +
            "CREATE TABLE IF NOT EXISTS preferences(singleton INTEGER PRIMARY KEY CHECK(singleton=1),revision INTEGER NOT NULL,payload TEXT NOT NULL,source TEXT NOT NULL,updated_at TEXT NOT NULL); " +
            "CREATE TABLE IF NOT EXISTS inherited(id TEXT PRIMARY KEY,parent_id TEXT NOT NULL,payload TEXT NOT NULL);");
          db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(FORMAT);
          db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
          if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT ||
              db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Voice preferences belong to another instance or format.");
          db.exec("COMMIT");
        } catch (error) { db.exec("ROLLBACK"); throw error; }
      } else if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT ||
          db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) {
        throw new Error("Voice preferences belong to another instance or format.");
      }
      return new VoicePreferencesStore(db, brainId);
    } catch (error) { db.close(); throw error; }
  }

  close(): void { this.db.close(); }
  stored(): StoredPreference | undefined {
    const size = this.db.prepare("SELECT LENGTH(payload) AS bytes FROM preferences WHERE singleton=1").get();
    if (size && (!Number.isSafeInteger(size.bytes) || Number(size.bytes) > 4096)) throw new Error("Saved voice settings exceed their typed record envelope.");
    const row = this.db.prepare("SELECT revision,payload,source,updated_at FROM preferences WHERE singleton=1").get();
    return row ? validateStored(row) : undefined;
  }
  snapshot(): InstanceVoicePreferences {
    const stored = this.stored();
    const inherited = Number(this.db.prepare("SELECT COUNT(*) AS n FROM inherited").get()?.n ?? 0);
    if (!Number.isSafeInteger(inherited) || inherited < 0) throw new Error("Voice preference history count is invalid.");
    return { brainId: this.brainId, revision: stored?.revision ?? 0, updatedAt: stored?.updatedAt ?? null,
      preferences: stored?.preferences ?? { ...DEFAULT_INSTANCE_VOICE_SETTINGS }, initialized: !!stored,
      legacyMigrationAllowed: !stored, inheritedVariants: inherited };
  }
  transaction<T>(work: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try { const result = work(); this.db.exec("COMMIT"); return result; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  private put(preferences: InstanceVoiceSettings, revision: number, source: VoicePreferenceSource): void {
    this.db.prepare("INSERT INTO preferences VALUES(1,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET revision=excluded.revision,payload=excluded.payload,source=excluded.source,updated_at=excluded.updated_at")
      .run(revision, canonical(preferences), source, new Date().toISOString());
  }
  initialize(source: Exclude<VoicePreferenceSource, "manual" | "legacy-global-v1">): InstanceVoicePreferences {
    return this.transaction(() => { if (!this.stored()) this.put({ ...DEFAULT_INSTANCE_VOICE_SETTINGS }, 1, source); return this.snapshot(); });
  }
  update(expectedRevision: number, preferences: InstanceVoiceSettings): InstanceVoicePreferences {
    const checked = validateInstanceVoiceSettings(preferences);
    if (!Number.isSafeInteger(expectedRevision) || expectedRevision < 0 || expectedRevision >= Number.MAX_SAFE_INTEGER) throw new Error("Invalid voice preference revision.");
    return this.transaction(() => {
      const current = this.stored();
      if ((current?.revision ?? 0) !== expectedRevision) throw new Error("Voice preferences changed; reload the current choices before saving.");
      this.put(checked, expectedRevision + 1, "manual"); return this.snapshot();
    });
  }
  migrateLegacy(legacy: LiveVoicePreferences): InstanceVoicePreferences {
    const checked = instanceVoiceSettingsFromLegacy(legacy);
    return this.transaction(() => { if (!this.stored()) this.put(checked, 1, "legacy-global-v1"); return this.snapshot(); });
  }
}

export async function instanceVoicePreferences(root: string, brainId: string): Promise<InstanceVoicePreferences> {
  const store = await VoicePreferencesStore.open(root, brainId);
  try { return store?.snapshot() ?? { brainId: id(brainId), revision: 0, updatedAt: null,
    preferences: { ...DEFAULT_INSTANCE_VOICE_SETTINGS }, initialized: false, legacyMigrationAllowed: true, inheritedVariants: 0 }; }
  finally { store?.close(); }
}
export async function initializeInstanceVoice(root: string, brainId: string,
  source: "fresh-auto" | "import-auto" | "copy-auto" | "merged-auto"): Promise<void> {
  const store = (await VoicePreferencesStore.open(root, brainId, true))!;
  try { store.initialize(source); } finally { store.close(); }
}
export async function validateVoiceWorkspace(root: string, brainId: string): Promise<void> {
  const store = await VoicePreferencesStore.open(root, brainId);
  if (!store) return;
  try {
    if (store.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Voice preference database failed integrity verification.");
    store.snapshot();
    if (store.db.prepare("SELECT 1 AS invalid FROM inherited WHERE LENGTH(payload)>4096 LIMIT 1").get()) throw new Error("Inherited voice settings exceed their typed record envelope.");
    for (const row of store.db.prepare("SELECT id,parent_id,payload FROM inherited").iterate()) {
      id(row.parent_id);
      const body = JSON.parse(String(row.payload));
      if (createHash("sha256").update(canonical(body)).digest("hex") !== row.id || body.parentId !== row.parent_id ||
          typeof body.updatedAt !== "string" || !Number.isFinite(Date.parse(body.updatedAt)) ||
          !Number.isSafeInteger(body.revision) || body.revision < 1 || !sources.has(body.source)) throw new Error("Inherited voice preferences are invalid.");
      validateInstanceVoiceSettings(body.preferences);
    }
  } finally { store.close(); }
}
/** Staged isolated copy only. Original parent choices/history stay unchanged. */
export async function rekeyVoiceWorkspace(root: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  id(targetBrainId); await validateVoiceWorkspace(root, sourceBrainId);
  const store = await VoicePreferencesStore.open(root, sourceBrainId, true);
  try { store!.transaction(() => { store!.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId); }); }
  finally { store?.close(); }
  await validateVoiceWorkspace(root, targetBrainId);
}
/** New merged identity keeps Auto; parent settings are retained, not last-wins. */
export async function mergeVoiceWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentId: string): Promise<void> {
  id(parentId); await validateVoiceWorkspace(sourceRoot, owner);
  const source = await VoicePreferencesStore.open(sourceRoot, owner);
  try { await initializeInstanceVoice(targetRoot, owner, "merged-auto"); }
  catch (error) { source?.close(); throw error; }
  if (!source) return;
  const target = (await VoicePreferencesStore.open(targetRoot, owner, true))!;
  try { target.transaction(() => {
    const stored = source.stored();
    if (stored) {
      const body = { parentId, ...stored }, payload = canonical(body), digest = createHash("sha256").update(payload).digest("hex");
      target.db.prepare("INSERT OR IGNORE INTO inherited VALUES(?,?,?)").run(digest, parentId, payload);
    }
    for (const row of source.db.prepare("SELECT id,parent_id,payload FROM inherited").iterate())
      target.db.prepare("INSERT OR IGNORE INTO inherited VALUES(?,?,?)").run(String(row.id), String(row.parent_id), String(row.payload));
  }); } finally { source.close(); target.close(); }
  await validateVoiceWorkspace(targetRoot, owner);
}
