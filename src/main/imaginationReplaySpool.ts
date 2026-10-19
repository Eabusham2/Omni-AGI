import { createHash } from "node:crypto";
import { lstat, open } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { ImaginationCapturedFragment, ImaginationRecording } from "../shared/imaginationCapture";
import type { ImaginationAssemblyManifest, ImaginationAssemblyManifestSummary, ImaginationAssemblyOptions, ImaginationReplaySpan, ImaginationReplayTrack } from "../shared/imaginationAssembly";
import { nativeMediaTimeline, imaginationDeliveryAnchor } from "../shared/nativeMediaTimeline";
import { checkedAssemblyOptions, deliveredReplayState } from "./imaginationAssemblyPlan";
import { replayFileHash, type InspectedReplaySource } from "./imaginationReplayCodec";
import { sqliteFilename } from "./sqliteFilename";

type Fragment = Omit<ImaginationCapturedFragment, "mediaUrl" | "available">;
export interface ReplaySpoolSeal { jsonSha256: string; sqliteSha256: string; sourceSha256: string }
export function replayReadBudget(bytes: number): void { if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes * 12 > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Replay metadata awaits bounded parse/transfer headroom; no recording rows were truncated."); }
function parsed<T>(text: string): T { replayReadBudget(Buffer.byteLength(text)); return JSON.parse(text) as T; }
class ManifestWriter {
  private readonly digest = createHash("sha256"); private pending = "";
  constructor(private readonly file: Awaited<ReturnType<typeof open>>, private readonly reserve: (bytes: number) => Promise<void>) {}
  async append(text: string): Promise<void> { this.digest.update(text); this.pending += text; if (Buffer.byteLength(this.pending) >= 65536) await this.flush(); }
  private async flush(): Promise<void> {
    if (!this.pending) return;
    const bytes = Buffer.from(this.pending); await this.reserve(bytes.byteLength + 65536);
    for (let offset = 0; offset < bytes.byteLength;) {
      const { bytesWritten } = await this.file.write(bytes, offset, bytes.byteLength - offset);
      if (!bytesWritten) throw new Error("Replay manifest write made no progress; no complete manifest was published.");
      offset += bytesWritten;
    }
    this.pending = "";
  }
  async finish(): Promise<string> { await this.flush(); await this.file.sync(); return this.digest.digest("hex"); }
}
/** Disk-spooled plan: bounded row parsing, no recording-sized arrays or writer lease. */
export async function buildReplaySpool(input: {
  directory: string; recording: ImaginationRecording; options?: Partial<ImaginationAssemblyOptions>; refreshHz?: number;
  inspect(fragment: Fragment): Promise<InspectedReplaySource>;
  admit(bytes: number): Promise<void>; reserve(bytes: number): Promise<void>; signal: AbortSignal;
  progress?(state:"sources"|"spooling",completed:number,total:number):void;
}): Promise<{ summary: ImaginationAssemblyManifestSummary; seal: ReplaySpoolSeal; estimatedPeakBytes: number; timingIssueCount: number }> {
  const sourcePath = join(input.directory, "source.sqlite3"), source = new DatabaseSync(sqliteFilename(sourcePath), { readOnly: true }), path = join(input.directory, "manifest.sqlite3"), db = new DatabaseSync(sqliteFilename(path));
  db.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA cache_size=-256; PRAGMA temp_store=FILE; CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE fragments(idx INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,stream_id TEXT NOT NULL,offset REAL NOT NULL,frame INTEGER,body TEXT NOT NULL); CREATE TABLE sources(sha TEXT PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE tracks(slot INTEGER PRIMARY KEY,stream_id TEXT UNIQUE NOT NULL,body TEXT NOT NULL); CREATE TABLE events(frame INTEGER NOT NULL,stream_id TEXT NOT NULL,idx INTEGER NOT NULL,PRIMARY KEY(frame,stream_id)); CREATE INDEX event_stream ON events(stream_id,frame); CREATE TABLE spans(idx INTEGER PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE intervals(value REAL NOT NULL);");
  db.exec("BEGIN IMMEDIATE");
  let count = 0, maxWidth = 2, maxHeight = 2, sourceFps = 1, audioOnly = true, last = 0, timingIssueCount = 0;
  try {
    input.progress?.("sources",0,input.recording.fragmentCount);
    const raw = source.prepare("SELECT fragment_index,body,sha FROM fragments WHERE recording_id=? ORDER BY fragment_index");
    for (const item of raw.iterate(input.recording.id)) {
      input.signal.throwIfAborted();
      const body = item.body;
      if (typeof body !== "string" || typeof item.sha !== "string" || typeof item.fragment_index !== "number" || !Number.isSafeInteger(item.fragment_index)) throw new Error("Recorded source fragment has a missing or corrupt SQLite body, checksum or index.");
      const row = parsed<Fragment>(body);
      if (row.recordingId !== input.recording.id || row.index !== count || row.index !== item.fragment_index || createHash("sha256").update(JSON.stringify(row)).digest("hex") !== item.sha || !/^[a-f0-9]{64}$/.test(row.sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 1 || !Number.isFinite(row.offsetMs) || row.offsetMs < 0 || row.mediaTimeline !== undefined && !nativeMediaTimeline(row.mediaTimeline) || row.deliveryAnchor !== undefined && !imaginationDeliveryAnchor(row.deliveryAnchor)) throw new Error("Revision-bound source fragment validation failed.");
      if (!db.prepare("SELECT sha FROM sources WHERE sha=?").get(row.sha256)) { const measured = await input.inspect(row), { path: _privatePath, ...metadata } = measured; await input.reserve(Buffer.byteLength(JSON.stringify(metadata)) + 65536); db.prepare("INSERT INTO sources VALUES(?,?)").run(row.sha256, JSON.stringify(metadata)); }
      const measured = parsed<Omit<InspectedReplaySource,"path">>(String(db.prepare("SELECT body FROM sources WHERE sha=?").get(row.sha256)!.body));
      if (row.mediaTimeline?.unit === "frames" && measured.frames !== row.mediaTimeline.count || row.mediaTimeline?.unit === "samples" && !measured.hasAudio || row.mediaTimeline?.containerAudio && !measured.hasAudio) throw new Error("A shared-content revision disagrees with actual decoded media.");
      maxWidth = Math.max(maxWidth, measured.width ?? 2); maxHeight = Math.max(maxHeight, measured.height ?? 2); if (row.mediaTimeline?.unit === "frames") sourceFps = Math.max(sourceFps, row.mediaTimeline.timebaseDenominator / row.mediaTimeline.timebaseNumerator);
      audioOnly &&= row.modality === "audio"; last = Math.max(last, row.offsetMs);
      await input.reserve(Buffer.byteLength(body) * 3 + 4096); db.prepare("INSERT INTO fragments VALUES(?,?,?,?,NULL,?)").run(count, row.id, row.streamId, row.offsetMs, body); count++;
      input.progress?.("sources",count,input.recording.fragmentCount);
    }
    if (!count || count !== input.recording.fragmentCount) throw new Error("The complete selected recording has no exact source traversal.");
    let previous: number | undefined; for (const row of db.prepare("SELECT offset FROM fragments ORDER BY offset,idx").iterate()) { const offset = Number(row.offset); if (previous !== undefined && offset > previous) db.prepare("INSERT INTO intervals VALUES(?)").run(offset - previous); previous = offset; }
    const intervalCount = Number(db.prepare("SELECT COUNT(*) AS n FROM intervals").get()!.n), middle = intervalCount ? Number(db.prepare("SELECT value FROM intervals ORDER BY value LIMIT 1 OFFSET ?").get(Math.floor(intervalCount / 2))!.value) : undefined;
    const distinct = Number(db.prepare("SELECT COUNT(DISTINCT stream_id) AS n FROM fragments").get()!.n), columns = Math.ceil(Math.sqrt(distinct)), rows = Math.ceil(distinct / columns), cadence = middle ? 1000 / middle : sourceFps;
    const refresh = input.refreshHz && Number.isFinite(input.refreshHz) && input.refreshHz > 0 ? input.refreshHz : sourceFps;
    const options = checkedAssemblyOptions({ width: Math.ceil(maxWidth * columns / 2) * 2, height: Math.ceil(maxHeight * rows / 2) * 2, fps: Math.ceil(Math.max(sourceFps, Math.min(cadence, refresh))), ...input.options });
    if (!audioOnly && (options.width < columns * 2 || options.height < rows * 2)) throw new Error("The exact output cannot represent every source stream's minimum tile; choose larger dimensions.");
    const estimatedPeakBytes = audioOnly ? distinct * 262144 + 65536 : options.width * options.height * 32 * (distinct + 2) + distinct * 16384;
    await input.admit(estimatedPeakBytes);
    const measuredEnd = input.recording.timelineEndMs !== undefined; if (!measuredEnd && options.historicalTailMs === undefined) throw new Error("Historical recording has no measured end; declare its presentation tail explicitly.");
    const end = input.recording.timelineEndMs ?? last + options.historicalTailMs!, rate = audioOnly ? 48000 : options.fps;
    if (!Number.isFinite(end) || end <= 0 || end < last) throw new Error("The composition end does not cover its source timeline.");
    const endFrame = Math.max(1, Math.round(end * rate / 1000)); if (!Number.isSafeInteger(endFrame)) throw new Error("Timeline exceeds the exact output tick representation.");
    const explicit = new Map((options.presentationOffsets ?? []).map(row => [row.streamId, row.offsetMs]));
    for (const key of explicit.keys()) if (!db.prepare("SELECT idx FROM fragments WHERE stream_id=? LIMIT 1").get(key)) throw new Error("An explicit offset names an absent stream.");
    let slot = 0, indexed=0;input.progress?.("spooling",0,count); for (const item of db.prepare("SELECT body FROM fragments ORDER BY offset,idx").iterate()) {
      const row = parsed<Fragment>(String(item.body)); if (!db.prepare("SELECT slot FROM tracks WHERE stream_id=?").get(row.streamId)) { const track: ImaginationReplayTrack = { streamId: row.streamId, slot: slot++, firstDeliveryMs: row.offsetMs, truncatedStart: row.deliveryAnchor?.truncatedStart ?? true,
        anchor: explicit.has(row.streamId) ? "explicit-retained-source-offset" : row.deliveryAnchor && !row.deliveryAnchor.truncatedStart ? "first-verified-main-delivery" : "unknown" }; db.prepare("INSERT INTO tracks VALUES(?,?,?)").run(track.slot, row.streamId, JSON.stringify(track)); }
      const frame = Math.min(endFrame, Math.max(0, Math.round(row.offsetMs * rate / 1000))); db.prepare("UPDATE fragments SET frame=? WHERE idx=?").run(frame, row.index); db.prepare("INSERT INTO events VALUES(?,?,?) ON CONFLICT(frame,stream_id) DO UPDATE SET idx=excluded.idx").run(frame, row.streamId, row.index);
      input.progress?.("spooling",++indexed,count);
    }
    const boundary = db.prepare("SELECT frame,LEAD(frame) OVER (ORDER BY frame) AS next FROM (SELECT 0 AS frame UNION SELECT ? UNION SELECT frame FROM events) ORDER BY frame"); let spanCount = 0;
    for (const row of boundary.iterate(endFrame)) {
      if (row.next === null) continue;
      // Bounded codec work checkpoints, not a total duration/recording limit.
      for (let startFrame = Number(row.frame); startFrame < Number(row.next); startFrame += Math.max(1, Math.floor(rate * 10))) {
      input.signal.throwIfAborted(); const endFrame = Math.min(Number(row.next), startFrame + Math.max(1, Math.floor(rate * 10))), startMs = startFrame * 1000 / rate, endMs = endFrame * 1000 / rate, states: ImaginationReplaySpan["states"] = [];
      for (const current of db.prepare("SELECT t.body AS track,f.body AS fragment FROM tracks t JOIN events e ON e.stream_id=t.stream_id AND e.frame=(SELECT MAX(q.frame) FROM events q WHERE q.stream_id=t.stream_id AND q.frame<=?) JOIN fragments f ON f.idx=e.idx ORDER BY t.slot").iterate(startFrame)) {
        const track = parsed<ImaginationReplayTrack>(String(current.track)), fragment = parsed<Fragment>(String(current.fragment)); replayReadBudget((states.length + 1) * 1024);
        const state = deliveredReplayState(fragment, track, track.firstDeliveryMs, explicit.get(track.streamId), startMs); if (state.issue) timingIssueCount++; states.push(state);
      }
      const span: ImaginationReplaySpan = { startFrame, endFrame, startMs, endMs, states }, body = JSON.stringify(span); await input.reserve(Buffer.byteLength(body) + 4096); db.prepare("INSERT INTO spans VALUES(?,?)").run(spanCount++, body);
      }
    }
    db.exec("COMMIT");
    const collisionCount = count - Number(db.prepare("SELECT COUNT(*) AS n FROM events").get()!.n), presentation: ImaginationAssemblyManifestSummary["presentation"] = { layout: "fixed-stream-grid", missingVisual: "neutral-black", exhaustedVisual: "last-verified-frame-hold", missingAudio: "declared-silence", independentAudio: "delivery-timed-equal-weight-mix", quantization: audioOnly ? "nearest-output-sample" : "nearest-output-frame", maximumQuantizationMs: 1000 / rate, nativeSynchronizationClaimed: false, supersededInSameFrameCount: collisionCount };
    const base = { format: "omni-delivered-timeline-composition" as const, version: 1 as const, policy: "delivered-timeline-v1" as const, outputKind: audioOnly ? "audio-only" as const : "audiovisual" as const, recording: input.recording, options, durationMs: endFrame * 1000 / rate, measuredEnd, captureGapCount: input.recording.gapCount, fragmentCount: count, trackCount: distinct, spanCount, presentation };
    const manifest = await open(join(input.directory, "manifest.json"), "wx", 0o600), writer = new ManifestWriter(manifest, input.reserve); let jsonSha256: string;
    try {
      await writer.append(`${JSON.stringify(base).slice(0, -1)},"fragments":[`); let comma = false;
      for (const row of db.prepare("SELECT body FROM fragments ORDER BY idx").iterate()) { await writer.append(`${comma ? "," : ""}${row.body}`); comma = true; }
      await writer.append('],"tracks":['); comma = false; for (const row of db.prepare("SELECT body FROM tracks ORDER BY slot").iterate()) { await writer.append(`${comma ? "," : ""}${row.body}`); comma = true; }
      await writer.append('],"spans":['); comma = false; for (const row of db.prepare("SELECT body FROM spans ORDER BY idx").iterate()) { await writer.append(`${comma ? "," : ""}${row.body}`); comma = true; }
      await writer.append('],"supersededInSameFrame":['); comma = false; for (const row of db.prepare("SELECT id FROM fragments WHERE idx NOT IN (SELECT idx FROM events) ORDER BY idx").iterate()) { await writer.append(`${comma ? "," : ""}${JSON.stringify(row.id)}`); comma = true; }
      await writer.append("]}"); jsonSha256 = await writer.finish();
    } finally { await manifest.close(); }
    const summary: ImaginationAssemblyManifestSummary = { ...base, manifestSha256: jsonSha256! };
    db.prepare("INSERT INTO meta VALUES('summary',?)").run(JSON.stringify(summary)); db.prepare("INSERT INTO meta VALUES('format','omni-replay-spool-1')").run();
    db.close(); source.close(); return { summary, seal: { jsonSha256: jsonSha256!, sqliteSha256: await replayFileHash(path), sourceSha256: await replayFileHash(sourcePath) }, estimatedPeakBytes, timingIssueCount };
  } catch (error) { try { db.close(); } catch { /* Preserve the failed private spool for the owner. */ } try { source.close(); } catch {} throw error; }
}

export class ReplaySpool {
  readonly db: DatabaseSync;
  constructor(readonly directory: string, readonly summary: ImaginationAssemblyManifestSummary) {
    this.db = new DatabaseSync(sqliteFilename(join(directory, "manifest.sqlite3")), { readOnly: true }); this.db.exec("PRAGMA cache_size=-256; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;");
    const objects = this.db.prepare("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").all();
    if (objects.length !== 8 || objects.some(row => row.type !== "table" && row.name !== "event_stream") || this.db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-replay-spool-1" || this.db.prepare("SELECT value FROM meta WHERE key='summary'").get()?.value !== JSON.stringify(summary)) { this.db.close(); throw new Error("Replay spool schema or bound summary changed."); }
  }
  close(): void { this.db.close(); }
  *fragments(): Generator<Fragment> { for (const row of this.db.prepare("SELECT body FROM fragments ORDER BY idx").iterate()) yield parsed<Fragment>(String(row.body)); }
  span(index: number): ImaginationReplaySpan { const row = this.db.prepare("SELECT body FROM spans WHERE idx=?").get(index); if (!row) throw new Error("Missing committed replay span."); return parsed<ImaginationReplaySpan>(String(row.body)); }
  source(root: string, fragmentId: string): InspectedReplaySource { const row = this.db.prepare("SELECT f.body AS fragment,s.body AS measured FROM fragments f JOIN sources s ON s.sha=json_extract(f.body,'$.sha256') WHERE f.id=?").get(fragmentId); if (!row) throw new Error("Replay state names an absent source.");
    const fragment = parsed<Fragment>(String(row.fragment)), measured = parsed<Omit<InspectedReplaySource,"path">>(String(row.measured)); return { ...measured, path: join(root, "imagination", "blobs", `${fragment.sha256}.data`), timeline: fragment.mediaTimeline }; }
  page(section: "fragments" | "tracks" | "spans", after = -1, limit = 20) { if (!Number.isSafeInteger(after) || after < -1 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100 || !["fragments", "tracks", "spans"].includes(section)) throw new Error("Invalid bounded replay manifest page.");
    const index = section === "tracks" ? "slot" : "idx", total = section === "fragments" ? this.summary.fragmentCount : section === "tracks" ? this.summary.trackCount : this.summary.spanCount;
    replayReadBudget(Number(this.db.prepare(`SELECT SUM(bytes) AS n FROM (SELECT length(CAST(body AS BLOB)) AS bytes FROM ${section} WHERE ${index}>? ORDER BY ${index} LIMIT ?)`).get(after, limit)?.n ?? 0));
    const rows = this.db.prepare(`SELECT ${index} AS idx,body FROM ${section} WHERE ${index}>? ORDER BY ${index} LIMIT ?`).all(after, limit), entries = rows.map(row => parsed<any>(String(row.body)));
    return { entries, total, ...(rows.length === limit ? { nextAfterIndex: Number(rows.at(-1)!.idx) } : {}) }; }
}
export async function verifyReplaySpool(directory: string, seal: ReplaySpoolSeal): Promise<void> {
  for (const [name, sha] of [["manifest.json", seal.jsonSha256], ["manifest.sqlite3", seal.sqliteSha256], ["source.sqlite3", seal.sourceSha256]]) { const path = join(directory, name!), info = await lstat(path); if (!info.isFile() || info.isSymbolicLink() || await replayFileHash(path) !== sha) throw new Error("An immutable replay manifest/source spool changed."); }
}
