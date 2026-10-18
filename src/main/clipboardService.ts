import type { ToolPermissionLevel } from "../shared/types";
import { CLIPBOARD_FORMAT, CLIPBOARD_READ_PAGE_BYTES, type ClipboardCapabilities, type ClipboardMutationAction,
  type ClipboardMutationReceipt, type ClipboardReadReceipt, type ClipboardScope } from "../shared/clipboard";
import { clipboardCanonical, clipboardHash, clipboardId, clipboardKeys, clipboardObject, clipboardScope, clipboardText, clipboardWhole } from "./clipboardFormat";
import { ClipboardStore, type ClipboardEffectRecord } from "./clipboardStore";
import { withClipboardMutation } from "./clipboardWorkspaceLease";
import type { HostClipboardAdapter, HostClipboardCursor, HostClipboardHolderLease } from "./clipboardHostBridge";

export interface ClipboardServiceDependencies {
  brainDirectory(brainId: string): Promise<string>;
  assertBrainOwner(brainId: string): Promise<void>;
  requireActiveOwner?(brainId: string): Promise<void>;
  currentPermission(brainId: string, toolId: "studio.clipboard" | "system.clipboard"): Promise<{ level: ToolPermissionLevel; revision: string }>;
  checkResources(brainId: string, bytes: number, purpose: string, childPids?: readonly number[]): Promise<void>;
  reserveWrite(root: string, bytes: number): Promise<void>;
  host: HostClipboardAdapter;
  onError?(error: unknown, brainId: string): void;
}
interface ActiveClipboardOperation { brainId: string; toolId: "studio.clipboard" | "system.clipboard"; controller: AbortController; promise: Promise<unknown> }
interface RetainedClipboardHolder { brainId: string; holder: HostClipboardHolderLease; monitor: NodeJS.Timeout }
interface CursorBody { format: typeof CLIPBOARD_FORMAT; scope: ClipboardScope; versionId?: string; revision?: number; offset: number; contentSha256?: string; adapter?: string; prefixSha256?: string }

export class ClipboardOperationError extends Error {
  constructor(message: string, readonly receipt: ClipboardMutationReceipt) { super(message); this.name = "ClipboardOperationError"; }
}
const now = (): string => new Date().toISOString();
function cursor(body: CursorBody): string { const text = clipboardCanonical(body), envelope = { body, sha256: clipboardHash(text) }; return Buffer.from(clipboardCanonical(envelope)).toString("base64url"); }
function parseCursor(value: unknown, scope: ClipboardScope): CursorBody | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length > 2048) throw new Error("Clipboard continuation cursor is invalid.");
  let row: Record<string, unknown>; try { row = clipboardObject(JSON.parse(Buffer.from(value, "base64url").toString("utf8")), "clipboard cursor"); } catch { throw new Error("Clipboard continuation cursor is invalid."); }
  const body = clipboardObject(row.body, "clipboard cursor body") as unknown as CursorBody;
  if (row.sha256 !== clipboardHash(clipboardCanonical(body)) || body.format !== CLIPBOARD_FORMAT || body.scope !== scope || !Number.isSafeInteger(body.offset) || body.offset < 0) throw new Error("Clipboard continuation cursor failed integrity/scope validation.");
  if (body.versionId !== undefined) clipboardId(body.versionId, "clipboard cursor version"); if (body.contentSha256 !== undefined && !/^[a-f0-9]{64}$/.test(body.contentSha256)) throw new Error("Clipboard cursor content checksum is invalid.");
  if (body.prefixSha256 !== undefined && !/^[a-f0-9]{64}$/.test(body.prefixSha256)) throw new Error("Clipboard cursor prefix checksum is invalid.");
  if (scope === "instance" && (!body.versionId || body.revision === undefined || !Number.isSafeInteger(body.revision) || body.revision < 0 || !body.contentSha256 || body.adapter !== undefined || body.prefixSha256 !== undefined)) throw new Error("Instance clipboard cursor is incomplete or contains host authority.");
  if (scope === "host" && (!body.adapter || !body.prefixSha256 || body.versionId !== undefined || body.revision !== undefined || body.contentSha256 !== undefined)) throw new Error("Host clipboard cursor is incomplete or contains instance authority.");
  return body;
}
function requestPayload(scope: ClipboardScope, action: ClipboardMutationAction, text: string): { scope: ClipboardScope; action: ClipboardMutationAction; requestSha256: string; contentSha256: string; contentBytes: number } {
  const bytes = Buffer.from(action === "clear" ? "" : text, "utf8"), contentSha256 = clipboardHash(bytes);
  const base = { scope, action, contentSha256, contentBytes: bytes.byteLength };
  return { ...base, requestSha256: clipboardHash(clipboardCanonical(base)) };
}
function uncertain(effect: ClipboardEffectRecord, message: string, mayHaveOccurred: boolean): ClipboardMutationReceipt {
  return { format: CLIPBOARD_FORMAT, version: 1, action: effect.action, requestId: effect.requestId, requestSha256: effect.requestSha256,
    scope: "host", state: mayHaveOccurred ? "write-uncertain" : "cancelled", contentType: "text/plain", contentBytes: effect.contentBytes,
    contentSha256: effect.contentSha256, writeMayHaveOccurred: mayHaveOccurred, retrySafe: false,
    provenance: { kind: "host-os-clipboard", capturedAt: now() }, error: message };
}

export class ClipboardService {
  private readonly active = new Map<string, ActiveClipboardOperation>();
  private readonly generations = new Map<string, number>();
  private readonly recovered = new Map<string, Promise<void>>();
  private readonly retainedHolders = new Map<string, RetainedClipboardHolder>();
  private closing = false;
  constructor(private readonly dependencies: ClipboardServiceDependencies) {}
  private key(brainId: string, requestId: string): string { return `${brainId}\0${requestId}`; }
  private generation(brainId: string, toolId: string): number { return this.generations.get(`${brainId}\0${toolId}`) ?? 0; }
  private async root(brainId: string): Promise<string> { return this.dependencies.brainDirectory(brainId); }
  private async store(brainId: string): Promise<ClipboardStore> { return ClipboardStore.open(await this.root(brainId), brainId); }
  private async ensureRecovered(brainId: string): Promise<void> {
    let pending = this.recovered.get(brainId); if (!pending) { pending = (async () => { const root = await this.root(brainId); await withClipboardMutation(root, async () => {
      const store = await ClipboardStore.open(root, brainId); try { await store.removeUnownedBlobs(); store.recoverInterruptedHostEffects(); } finally { store.close(); }
    }); })(); this.recovered.set(brainId, pending); }
    try { await pending; } catch (error) { this.recovered.delete(brainId); throw error; }
  }
  permissionChanged(brainId: string, toolId: string): void {
    if (toolId !== "studio.clipboard" && toolId !== "system.clipboard") return;
    const key = `${brainId}\0${toolId}`; this.generations.set(key, this.generation(brainId, toolId) + 1);
    for (const item of this.active.values()) if (item.brainId === brainId && item.toolId === toolId && !item.controller.signal.aborted) item.controller.abort(new Error("Clipboard permission changed before the operation settled."));
  }
  async forget(brainId: string): Promise<void> {
    this.recovered.delete(brainId); for (const item of this.active.values()) if (item.brainId === brainId) item.controller.abort(new Error("Clipboard instance ownership changed."));
    await Promise.allSettled([...this.active.values()].filter(item => item.brainId === brainId).map(item => item.promise));
  }
  private retainHolder(brainId: string, bytes: number, holder: HostClipboardHolderLease): void {
    if (this.retainedHolders.has(holder.id)) return;
    const check = async (): Promise<void> => { try { await this.dependencies.checkResources(brainId, bytes * 6 + 32 * 1024 * 1024,
      "retained Linux clipboard owner", [holder.pid]); } catch (error) { this.dependencies.onError?.(error, brainId); await holder.stop(); } };
    const monitor = setInterval(() => { void check(); }, 250); monitor.unref();
    this.retainedHolders.set(holder.id, { brainId, holder, monitor });
    void holder.exit.finally(() => { const current = this.retainedHolders.get(holder.id); if (current?.holder === holder) { clearInterval(current.monitor); this.retainedHolders.delete(holder.id); } });
    void check();
  }
  async shutdown(): Promise<void> {
    this.closing = true; for (const item of this.active.values()) item.controller.abort(new Error("Studio is closing."));
    await Promise.allSettled([...this.active.values()].map(item => item.promise)); await this.dependencies.host.shutdown?.();
    await Promise.allSettled([...this.retainedHolders.values()].map(item => Promise.race([item.holder.exit,
      new Promise<void>((_, reject) => setTimeout(() => reject(new Error(`Clipboard holder ${item.holder.pid} did not close before shutdown.`)), 2_000))])));
    for (const item of this.retainedHolders.values()) clearInterval(item.monitor); this.retainedHolders.clear();
  }
  async execute(brainId: string, toolId: "studio.clipboard" | "system.clipboard", action: string, argsValue: Record<string, unknown>, requestIdValue: string, signal: AbortSignal): Promise<unknown> {
    clipboardId(brainId, "clipboard brain owner"); const requestId = clipboardId(requestIdValue, "clipboard request ID"), args = structuredClone(clipboardObject(argsValue, "clipboard arguments"));
    const expectedScope: ClipboardScope = toolId === "studio.clipboard" ? "instance" : "host", scope = clipboardScope(args.scope);
    if (scope !== expectedScope) throw new Error(`${toolId} accepts only scope=${expectedScope}; scope cannot bypass the other tool's policy.`);
    if (!['capabilities', 'read', 'write', 'clear'].includes(action)) throw new Error("Unsupported clipboard action.");
    await this.ensureRecovered(brainId); const key = this.key(brainId, requestId);
    if (this.active.has(key)) throw new Error("This exact clipboard request is already active.");
    const controller = new AbortController(), abort = (): void => controller.abort(signal.reason); signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort();
    const promise = this.perform(brainId, toolId, action, scope, args, requestId, controller).finally(() => { signal.removeEventListener("abort", abort); this.active.delete(key); });
    this.active.set(key, { brainId, toolId, controller, promise }); return promise;
  }
  private async perform(brainId: string, toolId: "studio.clipboard" | "system.clipboard", action: string, scope: ClipboardScope,
    args: Record<string, unknown>, requestId: string, controller: AbortController): Promise<unknown> {
    const signal = controller.signal;
    const initial = await this.dependencies.currentPermission(brainId, toolId), generation = this.generation(brainId, toolId), pids: number[] = [];
    const requireCurrent = async (): Promise<void> => {
      signal.throwIfAborted(); await this.dependencies.assertBrainOwner(brainId); await this.dependencies.requireActiveOwner?.(brainId);
      const current = await this.dependencies.currentPermission(brainId, toolId); signal.throwIfAborted();
      if (this.closing || current.level === "off" || current.level !== initial.level || current.revision !== initial.revision || this.generation(brainId, toolId) !== generation) throw new Error("Clipboard authority changed before/after the awaited operation.");
    };
    await requireCurrent(); if (initial.level === "off") throw new Error(`${toolId} is Off.`);
    if (action === "capabilities") {
      clipboardKeys(args, scope === "instance" ? ["scope", "afterVersion", "limit"] : ["scope"]);
      if (scope === "host") return this.dependencies.host.capabilities();
      const store = await this.store(brainId); try { const current = store.version(), page = store.versions(args.afterVersion === undefined ? "" : clipboardId(args.afterVersion), clipboardWhole(args.limit, "clipboard version page", 200, 50));
        const result: ClipboardCapabilities = { format: CLIPBOARD_FORMAT, scope, implemented: true, adapter: "portable-instance-store", contentTypes: ["text/plain"], maxReadPageBytes: CLIPBOARD_READ_PAGE_BYTES,
          readIsBounded: true, write: true, clear: true, authority: "brain-portable-no-os-grant", detail: "Complete unsanitized text content and version provenance are brain-owned and portable; no host clipboard grant is implied.",
          currentVersionId: current.id, retainedVersions: page.entries, ...(page.nextAfter ? { nextAfterVersion: page.nextAfter } : {}) }; return result;
      } finally { store.close(); }
    }
    if (action === "read") {
      clipboardKeys(args, ["scope", "cursor", "maxBytes", "versionId"]); const maximum = clipboardWhole(args.maxBytes, "clipboard read page bytes", CLIPBOARD_READ_PAGE_BYTES, 64 * 1024); if (maximum < 4) throw new Error("Clipboard pages must allow at least four UTF-8 bytes.");
      const continued = parseCursor(args.cursor, scope);
      if (scope === "instance") {
        const selected = args.versionId === undefined ? continued?.versionId : clipboardId(args.versionId, "clipboard version ID"); if (continued && args.versionId !== undefined && selected !== continued.versionId) throw new Error("Clipboard version differs from its continuation cursor.");
        const store = await this.store(brainId); try { const page = await store.read(selected, continued?.offset ?? 0, maximum);
          if (continued && (continued.versionId !== page.version.id || continued.revision !== page.version.revision || continued.contentSha256 !== page.version.contentSha256)) throw new Error("Instance clipboard changed before this continuation.");
          await this.dependencies.checkResources(brainId, maximum * 8 + 65536, "bounded instance clipboard read"); await requireCurrent();
          const receipt: ClipboardReadReceipt = { format: CLIPBOARD_FORMAT, version: 1, action: "read", requestId, scope, state: page.complete ? "complete" : "partial", contentType: "text/plain",
            text: page.text, chunkBytes: page.bytes, chunkSha256: clipboardHash(page.text), complete: page.complete, totalBytes: page.version.bytes, contentSha256: page.version.contentSha256,
            versionId: page.version.id, revision: page.version.revision, provenance: page.version.provenance,
            ...(!page.complete ? { cursor: cursor({ format: CLIPBOARD_FORMAT, scope, versionId: page.version.id, revision: page.version.revision, offset: page.nextOffset!, contentSha256: page.version.contentSha256 }) } : {}) };
          return receipt;
        } finally { store.close(); }
      }
      if (args.versionId !== undefined) throw new Error("Host clipboard reads do not accept an instance version ID.");
      const capabilities = await this.dependencies.host.capabilities(); if (!capabilities.implemented) throw new Error(capabilities.detail);
      if (continued?.adapter && continued.adapter !== capabilities.adapter) throw new Error("Host clipboard adapter changed before continuation.");
      const monitor = setInterval(() => { void this.dependencies.checkResources(brainId, maximum * 12 + 32 * 1024 * 1024, "isolated host clipboard read", pids).catch(error => controller.abort(error)); }, 100); monitor.unref();
      try {
        const page = await this.dependencies.host.read(continued ? { afterBytes: continued.offset, prefixSha256: continued.prefixSha256! } : undefined, maximum,
          { signal, onSpawn: pid => { pids.push(pid); void this.dependencies.checkResources(brainId, maximum * 12 + 32 * 1024 * 1024, "host clipboard helper", pids).catch(error => controller.abort(error)); } });
        await requireCurrent(); const provenance = { kind: "host-os-clipboard" as const, capturedAt: page.capturedAt, adapter: page.adapter };
        const receipt: ClipboardReadReceipt = { format: CLIPBOARD_FORMAT, version: 1, action: "read", requestId, scope, state: page.complete ? "complete" : "partial", contentType: "text/plain",
          text: page.text, chunkBytes: page.bytes, chunkSha256: page.chunkSha256, complete: page.complete, provenance,
          ...(page.next ? { cursor: cursor({ format: CLIPBOARD_FORMAT, scope, offset: page.next.afterBytes, prefixSha256: page.next.prefixSha256, adapter: page.adapter }) } : {}) };
        return receipt;
      } finally { clearInterval(monitor); }
    }
    clipboardKeys(args, action === "write" ? ["scope", "contentType", "text"] : ["scope"]); if (action === "write" && args.contentType !== "text/plain") throw new Error("Clipboard v1 supports only text/plain.");
    const text = action === "clear" ? "" : clipboardText(args.text), payload = requestPayload(scope, action as ClipboardMutationAction, text), root = await this.root(brainId);
    await this.dependencies.checkResources(brainId, payload.contentBytes * 6 + 65536, `${scope} clipboard mutation`); await this.dependencies.reserveWrite(root, payload.contentBytes * 2 + 65536); await requireCurrent();
    if (scope === "instance") return withClipboardMutation(root, async () => { const store = await ClipboardStore.open(root, brainId); try {
      return await store.mutateInstance(requestId, payload, text, { kind: "instance-owned", capturedAt: now(), brainId });
    } finally { store.close(); } });
    return withClipboardMutation(root, async () => {
      const capabilities = await this.dependencies.host.capabilities();
      if (!capabilities.implemented || action === "write" && !capabilities.write || action === "clear" && !capabilities.clear) throw new Error(capabilities.detail);
      const store = await ClipboardStore.open(root, brainId); let effect: ClipboardEffectRecord;
      try { effect = store.prepareHostEffect(requestId, payload); if (effect.receipt) {
        if (effect.receipt.state === "complete") return { ...effect.receipt, repeated: true };
        throw new ClipboardOperationError(effect.receipt.error ?? "This host clipboard mutation is retained and was not replayed.", { ...effect.receipt, repeated: true });
      } } finally { store.close(); }
      let dispatched = effect.phase === "dispatched";
      if (dispatched) { const receipt = uncertain(effect, "An earlier host clipboard mutation lost its terminal acknowledgement and was not replayed.", true);
        try { const journal = await ClipboardStore.open(root, brainId); try { journal.terminal(requestId, receipt); } finally { journal.close(); } }
        catch (error) { throw new ClipboardOperationError(`${receipt.error} The local journal still needs recovery: ${error instanceof Error ? error.message : String(error)}`, receipt); }
        throw new ClipboardOperationError(receipt.error!, receipt); }
      const monitor = setInterval(() => { void this.dependencies.checkResources(brainId, payload.contentBytes * 6 + 32 * 1024 * 1024, "isolated host clipboard mutation", pids).catch(error => controller.abort(error)); }, 100); monitor.unref();
      let result;
      try {
        const hostContext = { signal, onSpawn: (pid: number): void => { pids.push(pid); void this.dependencies.checkResources(brainId, payload.contentBytes * 6 + 32 * 1024 * 1024, "host clipboard helper", pids).catch(error => controller.abort(error)); },
          onRetainedHolder: (holder: HostClipboardHolderLease): void => this.retainHolder(brainId, payload.contentBytes, holder), beforeDispatch: async (): Promise<void> => {
          await requireCurrent(); const journal = await ClipboardStore.open(root, brainId); try { journal.dispatched(requestId); dispatched = true; } finally { journal.close(); } await requireCurrent(); } };
        result = action === "write" ? await this.dependencies.host.write(text, hostContext) : await this.dependencies.host.clear(hostContext);
      } catch (error) {
        if (error instanceof ClipboardOperationError) throw error; const receipt = uncertain(effect, error instanceof Error ? error.message : String(error), dispatched);
        try { const journal = await ClipboardStore.open(root, brainId); try { journal.terminal(requestId, receipt); } finally { journal.close(); } }
        catch (journalError) { throw new ClipboardOperationError(`${receipt.error} The local journal still needs recovery: ${journalError instanceof Error ? journalError.message : String(journalError)}`, receipt); }
        throw new ClipboardOperationError(receipt.error!, receipt);
      } finally { clearInterval(monitor); }
      const receipt: ClipboardMutationReceipt = { format: CLIPBOARD_FORMAT, version: 1, action: action as ClipboardMutationAction, requestId, requestSha256: payload.requestSha256,
        scope, state: "complete", contentType: "text/plain", contentBytes: payload.contentBytes, contentSha256: payload.contentSha256,
        writeMayHaveOccurred: true, retrySafe: false, provenance: { kind: "host-os-clipboard", capturedAt: result.completedAt, adapter: result.adapter } };
      try { const journal = await ClipboardStore.open(root, brainId); try { journal.terminal(requestId, receipt); } finally { journal.close(); } }
      catch (error) { throw new ClipboardOperationError(`The host clipboard mutation completed, but its effect journal could not finish; do not repeat it: ${error instanceof Error ? error.message : String(error)}`, receipt); }
      try { await requireCurrent(); } catch (error) { throw new ClipboardOperationError(error instanceof Error ? error.message : String(error), receipt); }
      return receipt;
    });
  }
}
