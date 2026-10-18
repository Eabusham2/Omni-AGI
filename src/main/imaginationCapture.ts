import { randomUUID } from "node:crypto";
import { lstat, mkdir, mkdtemp, readdir, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import type { AuthorizedMediaArtifact } from "./mediaArtifactRegistry";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";
import { captureHash, captureSeconds, ImaginationCaptureStore } from "./imaginationCaptureStore";
import { registerImaginationWorkspaceBoundary, imaginationFileHash } from "./imaginationCaptureWorkspace";
import type { ImaginationCapturedFragment, ImaginationCaptureEvent, ImaginationCaptureOrigin, ImaginationCaptureStatus, ImaginationRecording } from "../shared/imaginationCapture";

export interface DeliveredImaginationFragment {
  brainId: string; streamId: string; origin: ImaginationCaptureOrigin; kind: "preview-fragment" | "final-artifact";
  modality: "image" | "audio" | "video"; revision?: number; width?: number; height?: number; durationMs?: number; fps?: number; frameCount?: number;
  mediaTimeline?: import("../shared/nativeMediaTimeline").NativeMediaTimeline;
  deliveryAnchor?: import("../shared/nativeMediaTimeline").ImaginationDeliveryAnchor;
  /** Main registry rechecks immutable file identity, MIME and checksum. */
  resolve(): Promise<AuthorizedMediaArtifact>;
}
type Buffered = { id: string; inheritedId?: string; at: number; expiresAt: number; observedAt: string; source: DeliveredImaginationFragment; path: string; sha256: string; mimeType: string; bytes: number };
interface Owner { root: string; rolling: string; buffer: Buffered[]; anchors: Map<string, { at: number; wall: string; truncatedStart: boolean }>; activeId?: string; activeStartAt?: number; frozenAt?: number; unregister?: () => void; pendingGaps: number; autoPaused: boolean; stopped: boolean; pending: Promise<void>; pendingCount: number }
export interface ImaginationCaptureOptions {
  brainDirectory(id: string): string;
  reserve(root: string, bytes: number): Promise<void>;
  admit(bytes: number): void;
  mediaUrl(brainId: string, sha256: string, mimeType: string): Promise<string>;
  now?(): number;
  wall?(): string;
}
/** Actual native events only. No generator/model/prompt API is retained. */
export class ImaginationCaptureService {
  private readonly owners = new Map<string, Promise<Owner>>();
  private readonly listeners = new Set<(event: ImaginationCaptureEvent) => void>();
  private readonly blocked = new Set<string>();
  private readonly activated = new Set<string>();
  activate(id: string): void { this.blocked.delete(id); this.activated.add(id); void this.owners.get(id)?.then(owner => this.serial(owner, async () => {
    owner.stopped = false;
    if (owner.frozenAt !== undefined) { const delta = this.now() - owner.frozenAt; for (const fragment of owner.buffer) { fragment.at += delta; fragment.expiresAt += delta; } for (const anchor of owner.anchors.values()) anchor.at += delta; owner.frozenAt = undefined; }
  })); }
  private readonly tick: NodeJS.Timeout;
  constructor(private readonly options: ImaginationCaptureOptions) { this.tick = setInterval(() => { for (const [id, promise] of this.owners) void promise.then(owner => this.serial(owner, () => this.prune(id, owner))).catch(() => undefined); }, 1000); this.tick.unref?.(); }
  subscribe(listener: (event: ImaginationCaptureEvent) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  private now(): number { return this.options.now?.() ?? performance.now(); }
  private wall(): string { return this.options.wall?.() ?? new Date().toISOString(); }
  private emit(brainId: string, type: ImaginationCaptureEvent["type"], recordingId?: string, detail?: string): void {
    for (const listener of this.listeners) try { listener({ brainId, type, at: this.wall(), ...(recordingId ? { recordingId } : {}), ...(detail ? { detail } : {}) }); } catch { /* Display listeners cannot change native work. */ }
  }
  private owner(id: string): Promise<Owner> {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(id)) return Promise.reject(new Error("Invalid imagination owner."));
    let promise = this.owners.get(id); if (!promise) { const root = this.options.brainDirectory(id); promise = (async () => {
      for (const path of [root, join(root, "brain.json")]) { const info = await lstat(path); if (info.isSymbolicLink() || (path === root ? !info.isDirectory() : !info.isFile())) throw new Error("Imagination needs its existing managed owner."); }
      for (const parent of [join(root, "engine"), join(root, "engine", ".imagination-rolling")]) {
        await mkdir(parent, { recursive: true, mode: 0o700 }); const info = await lstat(parent); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Unsafe rolling capture owner.");
      }
      const rolling = await mkdtemp(join(root, "engine", ".imagination-rolling", "capture-"));
      const store = (await ImaginationCaptureStore.open(root, id, true))!;
      const buffer: Buffered[] = [], frozenAt = this.activated.has(id) ? undefined : this.now();
      try { for (const row of store.db.prepare("SELECT id FROM recordings WHERE json_extract(body,'$.state')='recording'").iterate()) {
        const old = store.get(String(row.id))!; store.update(old.id, old.revision, value => { value.state = "dormant"; });
      }
        const window = store.window(), clock = store.windowClock(), wallAge = clock.paused ? 0 : Math.max(0, Date.parse(this.wall()) - Date.parse(clock.at)); this.options.admit(window.length * 4096);
        for (const item of window) { const remaining = Math.max(0, item.remainingMs - wallAge); if (!remaining) continue;
          const sourcePath = join(root, "imagination", "window-blobs", `${item.sha256}.data`), info = await lstat(sourcePath);
          if (!info.isFile() || info.isSymbolicLink() || info.size !== item.bytes || await imaginationFileHash(sourcePath) !== item.sha256) throw new Error("A retained live-window fragment changed.");
          await this.options.reserve(root, item.bytes); const path = join(rolling, `${item.id}.data`); await copyMutableFileIsolated(sourcePath, path, { expectedSha256: item.sha256 });
          buffer.push({ ...item, at: this.now() - item.ageMs - wallAge, expiresAt: this.now() + remaining, path,
            source: { ...item.source, resolve: async () => ({ path, sha256: item.sha256, mimeType: item.mimeType, size: item.bytes }) } });
        }
      } finally { store.close(); }
      buffer.sort((a, b) => a.at - b.at);
      const anchors = new Map<string, { at: number; wall: string; truncatedStart: boolean }>();
      for (const fragment of buffer) if (fragment.source.deliveryAnchor) anchors.set(fragment.source.streamId, { at: fragment.at - fragment.source.deliveryAnchor.elapsedMs, wall: fragment.source.deliveryAnchor.firstObservedAt, truncatedStart: fragment.source.deliveryAnchor.truncatedStart });
      const owner: Owner = { root, rolling, buffer, anchors, frozenAt, pendingGaps: 0, autoPaused: false, stopped: false, pending: Promise.resolve(), pendingCount: 0 };
      return owner;
    })(); this.owners.set(id, promise);
      const prepared = promise, unregister = registerImaginationWorkspaceBoundary(root, async work => {
        const owner = await prepared; return this.serial(owner, async () => { if (!owner.stopped) await this.checkpointWindow(id, owner, false); return work(); });
      });
      void prepared.then(owner => { owner.unregister = unregister; }, () => { unregister(); if (this.owners.get(id) === prepared) this.owners.delete(id); });
    } return promise;
  }
  private serial<T>(owner: Owner, operation: () => Promise<T>): Promise<T> { const result = owner.pending.then(operation); owner.pending = result.then(() => undefined, () => undefined); return result; }
  async delivered(source: DeliveredImaginationFragment): Promise<void> {
    if (source.origin.kind !== "spontaneous" || this.blocked.has(source.brainId)) return;
    const deliveredAt = this.now(), deliveredWall = this.wall();
    const owner = await this.owner(source.brainId); if (owner.stopped) return;
    try { this.options.admit((owner.pendingCount + owner.buffer.length + 1) * 4096); } catch { if (owner.activeId) owner.pendingGaps++;
      this.emit(source.brainId, "gap", owner.activeId, "Capture queue declined a fragment at the current memory budget; native generation was unchanged."); return; }
    owner.pendingCount++;
    await this.serial(owner, async () => {
      if (owner.stopped) return;
      if (owner.frozenAt !== undefined) { const delta = this.now() - owner.frozenAt; for (const fragment of owner.buffer) { fragment.at += delta; fragment.expiresAt += delta; } for (const anchor of owner.anchors.values()) anchor.at += delta; owner.frozenAt = undefined; }
      const store = (await ImaginationCaptureStore.open(owner.root, source.brainId, true))!;
      try {
        if (owner.activeId && owner.pendingGaps) { store.addGaps(owner.activeId, owner.pendingGaps); owner.pendingGaps = 0; }
        await this.prune(source.brainId, owner, store); const resolved = await source.resolve();
        if (!Number.isSafeInteger(resolved.size) || resolved.size <= 0 || !/^[a-f0-9]{64}$/.test(resolved.sha256) || !/^(?:image|audio|video)\/[A-Za-z0-9.+-]+$/.test(resolved.mimeType)) throw new Error("The delivered native fragment has invalid immutable metadata.");
        const id = captureHash({ streamId: source.streamId, kind: source.kind, revision: source.revision ?? null, sha256: resolved.sha256 });
        if (owner.buffer.some(fragment => fragment.id === id)) return;
        await this.options.reserve(owner.root, resolved.size + 65536); const path = join(owner.rolling, `${id}.data`);
        await copyMutableFileIsolated(resolved.path, path, { expectedSha256: resolved.sha256 });
        let anchor = owner.anchors.get(source.streamId); if (!anchor) { anchor = { at: deliveredAt, wall: deliveredWall, truncatedStart: source.kind !== "preview-fragment" || source.revision !== 0 }; owner.anchors.set(source.streamId, anchor); }
        const deliveredSource: DeliveredImaginationFragment = { ...source, deliveryAnchor: { version: 1, kind: "first-verified-main-delivery", firstObservedAt: anchor.wall, elapsedMs: Math.max(0, deliveredAt - anchor.at), truncatedStart: anchor.truncatedStart } };
        const fragment: Buffered = { id, at: deliveredAt, expiresAt: deliveredAt + store.settings().bufferSeconds * 1000, observedAt: deliveredWall, source: deliveredSource, path, sha256: resolved.sha256, mimeType: resolved.mimeType, bytes: resolved.size };
        owner.buffer.push(fragment);
        const settings = store.settings();
        if (!owner.activeId && settings.alwaysSave && !settings.dormant && !owner.autoPaused) { owner.activeId = store.create(`always-${randomUUID()}`, "always-save", 0, 0, fragment.observedAt).id; owner.activeStartAt = fragment.at; }
        if (owner.activeId) await this.archive(source.brainId, owner, store, owner.activeId, fragment);
        this.emit(source.brainId, "changed", owner.activeId);
      } catch (error) {
        if (owner.activeId) { const current = store.get(owner.activeId); if (current?.state === "recording") store.update(current.id, current.revision, value => { value.gapCount++; value.state = "resource-paused"; value.endedAt = this.wall(); value.timelineEndMs = Math.max(0, this.now() - (owner.activeStartAt ?? this.now())); }); owner.activeId = undefined; }
        owner.autoPaused = true; this.emit(source.brainId, "gap", undefined, "A fragment could not be retained at the current resource or integrity boundary. Existing fragments and native work are unchanged.");
      } finally { store.close(); }
    }).finally(() => { owner.pendingCount--; });
  }
  private async archive(id: string, owner: Owner, store: ImaginationCaptureStore, recordingId: string, fragment: Buffered): Promise<void> {
    await this.options.reserve(owner.root, fragment.bytes + 65536);
    const destination = join(owner.root, "imagination", "blobs", `${fragment.sha256}.data`); await mkdir(join(owner.root, "imagination", "blobs"), { recursive: true, mode: 0o700 });
    const directory = await lstat(join(owner.root, "imagination", "blobs")); if (!directory.isDirectory() || directory.isSymbolicLink()) throw new Error("Unsafe imagination archive directory.");
    const existing = await lstat(destination).catch(error => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!existing) await copyMutableFileIsolated(fragment.path, destination, { expectedSha256: fragment.sha256 });
    else { if (!existing.isFile() || existing.isSymbolicLink() || existing.size !== fragment.bytes) throw new Error("Retained imagination blob changed.");
      const hash = createHash("sha256"); for await (const chunk of createReadStream(destination)) hash.update(chunk);
      if (hash.digest("hex") !== fragment.sha256) throw new Error("Retained imagination blob changed."); }
    const { source } = fragment, observedOffset = Math.max(0, fragment.at - (owner.activeStartAt ?? fragment.at));
    const metadataFields = ["revision", "width", "height", "durationMs", "fps", "frameCount", "mediaTimeline", "deliveryAnchor"] as const;
    store.append(recordingId, { id: fragment.id, streamId: source.streamId, observedAt: fragment.observedAt, offsetMs: observedOffset,
      origin: source.origin, kind: source.kind, modality: source.modality, mimeType: fragment.mimeType, sha256: fragment.sha256, bytes: fragment.bytes,
      ...Object.fromEntries(metadataFields.filter(key => source[key] !== undefined).map(key => [key, source[key]])) });
  }
  private async prune(id: string, owner: Owner, opened?: ImaginationCaptureStore): Promise<void> {
    const store = opened ?? await ImaginationCaptureStore.open(owner.root, id); if (!store) return;
    try { const current = owner.frozenAt ?? this.now(), oldest = current - store.settings().bufferSeconds * 1000;
      const retired = owner.buffer.filter(fragment => fragment.at < oldest || fragment.expiresAt < current); owner.buffer = owner.buffer.filter(fragment => fragment.at >= oldest && fragment.expiresAt >= current);
      for (const fragment of retired) await rm(fragment.path, { force: true });
      if (retired.length) { const retiredIds = new Set(retired.map(value => value.id)), kept = store.window().filter(value => !retiredIds.has(value.id));
        const clock = store.windowClock(); store.setWindow(kept, clock.paused, clock.at);
        const keptHashes = new Set(kept.map(value => value.sha256)); for (const value of retired) if (!keptHashes.has(value.sha256)) await rm(join(owner.root, "imagination", "window-blobs", `${value.sha256}.data`), { force: true });
      }
    } finally { if (!opened) store.close(); }
  }
  async status(id: string): Promise<ImaginationCaptureStatus> { const owner = await this.owner(id); return this.serial(owner, async () => {
    const store = (await ImaginationCaptureStore.open(owner.root, id, true))!; try { await this.prune(id, owner, store); const first = owner.buffer[0];
      return { brainId: id, settings: store.settings(), ...(owner.activeId ? { recording: store.get(owner.activeId) } : {}), bufferedFragments: owner.buffer.length,
        bufferedBytes: owner.buffer.reduce((sum, fragment) => sum + fragment.bytes, 0), availableLookbackSeconds: first ? Math.max(0, ((owner.frozenAt ?? this.now()) - first.at) / 1000) : 0,
        autoPaused: owner.autoPaused, unclassifiedChatIntent: true };
    } finally { store.close(); } }); }
  async settings(request: { brainId: string; expectedRevision: number; bufferSeconds: number; alwaysSave: boolean }): Promise<ImaginationCaptureStatus> {
    const owner = await this.owner(request.brainId); await this.serial(owner, async () => { if (owner.stopped) throw new Error("Capture owner changed; refresh before applying settings."); await this.options.reserve(owner.root, 65536); const store = (await ImaginationCaptureStore.open(owner.root, request.brainId, true))!;
      try { store.setSettings(request.expectedRevision, request.bufferSeconds, request.alwaysSave); owner.autoPaused = false;
        if (!request.alwaysSave && owner.activeId) { const active = store.get(owner.activeId); if (active?.mode === "always-save") { store.update(active.id, active.revision, value => { value.state = "complete"; value.endedAt = this.wall(); value.timelineEndMs = Math.max(0, this.now() - (owner.activeStartAt ?? this.now())); }); owner.activeId = undefined; } }
        await this.prune(request.brainId, owner, store);
      } finally { store.close(); } }); this.emit(request.brainId, "changed"); return this.status(request.brainId);
  }
  async record(request: { brainId: string; requestId: string; lookbackSeconds: number }): Promise<ImaginationRecording> {
    if (request.lookbackSeconds !== 0) captureSeconds(request.lookbackSeconds); const owner = await this.owner(request.brainId);
    return this.serial(owner, async () => { if (owner.stopped) throw new Error("Capture owner changed; refresh before recording."); await this.options.reserve(owner.root, 65536); const store = (await ImaginationCaptureStore.open(owner.root, request.brainId, true))!;
      try { const prior = store.request(request.requestId); if (prior) return prior; if (owner.activeId) throw new Error("An imagination recording is already active.");
        const settings = store.settings(); if (request.lookbackSeconds > settings.bufferSeconds) throw new Error("That lookback exceeds the selected rolling window; expired fragments cannot be recreated.");
        await this.prune(request.brainId, owner, store); const currentTime = owner.frozenAt ?? this.now(), selected = owner.buffer.filter(fragment => fragment.at >= currentTime - request.lookbackSeconds * 1000);
        const available = selected[0] ? Math.min(request.lookbackSeconds, (currentTime - selected[0].at) / 1000) : 0;
        const current = store.create(request.requestId, "manual", request.lookbackSeconds, available, selected[0]?.observedAt ?? this.wall()); owner.activeId = current.id; owner.activeStartAt = selected[0]?.at ?? this.now(); owner.autoPaused = false;
        try { for (const fragment of selected) await this.archive(request.brainId, owner, store, current.id, fragment); }
        catch (error) { const last = store.get(current.id)!; store.update(last.id, last.revision, value => { value.state = "resource-paused"; value.gapCount++; });
          owner.activeId = undefined; owner.autoPaused = true; this.emit(request.brainId, "gap", current.id, "Back-record stopped at a storage boundary; committed fragments remain."); throw error; }
        this.emit(request.brainId, "recording", current.id); return store.get(current.id)!;
      } finally { store.close(); } });
  }
  private async finish(request: { brainId: string; recordingId: string; expectedRevision: number }, state: "complete" | "cancelled"): Promise<ImaginationRecording> {
    const owner = await this.owner(request.brainId); return this.serial(owner, async () => { if (owner.stopped) throw new Error("Capture owner changed; refresh before its control boundary."); const store = (await ImaginationCaptureStore.open(owner.root, request.brainId, true))!;
      try { await this.options.reserve(owner.root, 65536); if (owner.activeId && owner.pendingGaps) { store.addGaps(owner.activeId, owner.pendingGaps); owner.pendingGaps = 0; } const current = store.get(request.recordingId);
        if (!current || !Number.isSafeInteger(request.expectedRevision) || request.expectedRevision < current.controlRevision || request.expectedRevision > current.revision || owner.activeId !== current.id) throw new Error("Recording control changed; refresh before applying.");
        const result = store.update(request.recordingId, current.revision, value => {
        if (value.state !== "recording") throw new Error("This recording is not active."); value.state = state; value.endedAt = this.wall(); value.timelineEndMs = Math.max(0, this.now() - (owner.activeStartAt ?? this.now())); });
        if (owner.activeId === result.id) owner.activeId = undefined; owner.autoPaused = true; this.emit(request.brainId, "recording", result.id); return result;
      } finally { store.close(); } });
  }
  stop(request: { brainId: string; recordingId: string; expectedRevision: number }) { return this.finish(request, "complete"); }
  cancel(request: { brainId: string; recordingId: string; expectedRevision: number }) { return this.finish(request, "cancelled"); }
  async resume(request: { brainId: string; recordingId: string; expectedRevision: number }): Promise<ImaginationRecording> {
    const owner = await this.owner(request.brainId); return this.serial(owner, async () => {
      if (owner.stopped) throw new Error("Capture owner changed; refresh before resuming.");
      if (owner.activeId) throw new Error("Another recording is already active."); await this.options.reserve(owner.root, 65536);
      const store = (await ImaginationCaptureStore.open(owner.root, request.brainId, true))!; try {
        const current = store.get(request.recordingId); if (!current || !["dormant", "resource-paused"].includes(current.state)) throw new Error("This saved recording cannot resume.");
        const result = store.update(current.id, request.expectedRevision, value => { value.state = "recording"; delete value.endedAt; });
        const last = store.db.prepare("SELECT body FROM fragments WHERE recording_id=? ORDER BY fragment_index DESC LIMIT 1").get(current.id);
        owner.activeId = current.id; owner.activeStartAt = this.now() - (current.timelineEndMs ?? (last ? Number(JSON.parse(String(last.body)).offsetMs) : 0)); owner.autoPaused = false;
        this.emit(request.brainId, "recording", result.id); return result;
      } finally { store.close(); }
    });
  }
  async recordings(id: string, query: { beforeSequence?: number; limit?: number } = {}) { const owner = await this.owner(id); return this.serial(owner, async () => { const store = (await ImaginationCaptureStore.open(owner.root, id, true))!; try { return store.list(query.beforeSequence, query.limit); } finally { store.close(); } }); }
  /** Short SQLite snapshot boundary; no unbounded fragment array or codec work holds this queue. */
  async snapshotTimelineDatabase(id: string, recordingId: string, expectedRevision: number, destination: string): Promise<ImaginationRecording> {
    const owner = await this.owner(id); return this.serial(owner, async () => {
      const store = (await ImaginationCaptureStore.open(owner.root, id, true))!; try {
        const current = store.get(recordingId); if (!current || current.revision !== expectedRevision) throw new Error("Recording changed before assembly; refresh its exact revision.");
        const recording = { ...current }; if (owner.activeId === current.id) recording.timelineEndMs = Math.max(0, this.now() - (owner.activeStartAt ?? this.now()));
        await this.options.reserve(owner.root, (await lstat(join(owner.root, "imagination", "workspace.sqlite3"))).size + (await lstat(join(owner.root, "imagination", "workspace.sqlite3-wal")).catch(() => ({ size: 0 }))).size + 65536);
        await snapshotMutableSqliteIsolated(join(owner.root, "imagination", "workspace.sqlite3"), destination);
        return recording;
      } finally { store.close(); }
    });
  }
  async fragments(id: string, recordingId: string, query: { afterIndex?: number; limit?: number } = {}) { const owner = await this.owner(id); return this.serial(owner, async () => {
    const store = (await ImaginationCaptureStore.open(owner.root, id, true))!;
    try { const page = store.fragments(recordingId, query.afterIndex, query.limit), entries: ImaginationCapturedFragment[] = [];
      for (const fragment of page.entries) { try { entries.push({ ...fragment, available: true, mediaUrl: await this.options.mediaUrl(id, fragment.sha256, fragment.mimeType) }); }
        catch { entries.push({ ...fragment, available: false }); } } return { ...page, entries };
    } finally { store.close(); } }); }
  private async checkpointWindow(id: string, owner: Owner, paused: boolean): Promise<void> {
    const store = (await ImaginationCaptureStore.open(owner.root, id, true))!; try {
      if (owner.activeId && owner.pendingGaps) { store.addGaps(owner.activeId, owner.pendingGaps); owner.pendingGaps = 0; }
      await this.prune(id, owner, store); const at = owner.frozenAt ?? this.now(), entries = [];
      await mkdir(join(owner.root, "imagination", "window-blobs"), { recursive: true, mode: 0o700 });
      const directory = await lstat(join(owner.root, "imagination", "window-blobs")); if (!directory.isDirectory() || directory.isSymbolicLink()) throw new Error("Unsafe live-window custody directory.");
      for (const fragment of owner.buffer) { const destination = join(owner.root, "imagination", "window-blobs", `${fragment.sha256}.data`); await this.options.reserve(owner.root, fragment.bytes + 65536);
        await copyMutableFileIsolated(fragment.path, destination, { expectedSha256: fragment.sha256 }); const { resolve: _resolve, ...source } = fragment.source;
        entries.push({ id: fragment.id, ...(fragment.inheritedId ? { inheritedId: fragment.inheritedId } : {}), observedAt: fragment.observedAt, ageMs: Math.max(0, at - fragment.at), remainingMs: Math.max(0, fragment.expiresAt - at), sha256: fragment.sha256, mimeType: fragment.mimeType, bytes: fragment.bytes, source });
      } store.setWindow(entries, paused || owner.frozenAt !== undefined, this.wall());
      const wanted = new Set(entries.map(value => `${value.sha256}.data`)); for (const name of await readdir(join(owner.root, "imagination", "window-blobs"))) {
        if (!/^[a-f0-9]{64}\.data$/.test(name)) throw new Error("Unowned live-window blob found; cleanup refused."); if (!wanted.has(name)) await rm(join(owner.root, "imagination", "window-blobs", name), { force: true });
      }
    } finally { store.close(); }
  }
  /** Unload pauses live-window lifetime; Fresh deliberately retires its current context only. */
  async release(id: string, discardWindow = false): Promise<void> { this.blocked.add(id); this.activated.delete(id); const promise = this.owners.get(id); if (!promise) return; const owner = await promise; owner.stopped = true; owner.frozenAt ??= this.now();
    await this.serial(owner, async () => { if (!discardWindow) await this.checkpointWindow(id, owner, true);
      else { const store = (await ImaginationCaptureStore.open(owner.root, id, true))!; try { const old = store.window(); store.setWindow([], true, this.wall()); for (const item of old) await rm(join(owner.root, "imagination", "window-blobs", `${item.sha256}.data`), { force: true }); } finally { store.close(); } }
      if (owner.activeId) { const store = (await ImaginationCaptureStore.open(owner.root, id, true))!;
      try { const record = store.get(owner.activeId); if (record?.state === "recording") store.update(record.id, record.revision, value => { value.state = "dormant"; value.timelineEndMs = Math.max(0, (owner.frozenAt ?? this.now()) - (owner.activeStartAt ?? this.now())); }); } finally { store.close(); } }
      for (const name of await readdir(owner.rolling)) { if (!/^[a-f0-9]{64}\.data$/.test(name)) throw new Error("Rolling capture cleanup found an unowned payload."); }
      await rm(owner.rolling, { recursive: true, force: true }); owner.buffer = []; }); owner.unregister?.(); this.owners.delete(id); this.emit(id, "changed"); }
  async dispose(): Promise<void> { clearInterval(this.tick); for (const id of this.owners.keys()) await this.release(id); }
}
