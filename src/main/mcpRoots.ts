import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, lstat, mkdir, open, realpath, rename, unlink } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { McpRequestOrigin } from "../shared/mcpInteractions";
import type { McpFileDisclosurePolicy, McpHostConnection, McpRoot, McpRootDisclosureReview, McpRootsStatus } from "../shared/mcpHostServices";

const MAX_BYTES = 1024 * 1024;
const SHA = /^[a-f0-9]{64}$/;
const canonical = (value: unknown) => JSON.stringify(value);
const sha = (value: string) => createHash("sha256").update(value).digest("hex");
const key = (brainId: string, serverId: string) => {
  for (const value of [brainId, serverId]) if (typeof value !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(value) || /[\r\n\0]/.test(value)) throw new Error("Invalid MCP roots owner.");
  return sha(canonical([brainId, serverId]));
};
interface SelectedRoot extends McpRoot { path: string; identity: string; }
interface Selection {
  format: "omni-device-mcp-roots"; version: 1; brainId: string; serverId: string;
  endpoint: string; configurationSha256: string; revision: number; roots: SelectedRoot[];
}
interface LiveDisclosure { connection: McpHostConnection; revision: number; policyRevision: string; selectionSha256: string; }
const selectionSha = (selection: Selection) => sha(canonical({ revision: selection.revision, configurationSha256: selection.configurationSha256, roots: selection.roots }));
export interface McpRootsOptions {
  /** Device-owned private settings; never inside a brain or portable closure. */
  directory: string;
  connection(brainId: string, serverId: string): McpHostConnection;
  requireConnection(connection: McpHostConnection): void;
  filePolicy(brainId: string, signal: AbortSignal): Promise<McpFileDisclosurePolicy>;
  approve(connection: McpHostConnection, review: McpRootDisclosureReview, signal: AbortSignal): Promise<boolean>;
  admit(bytes: number, brainId: string): Promise<void>;
  checkDisk(bytes: number): Promise<void>;
  /** Returns false when the exact existing session is no longer ready. No connect. */
  notify(connection: McpHostConnection): Promise<boolean>;
}

/** Roots disclose selected URI boundaries. They do not sandbox a local process,
 * grant remote file reads, change tool policy, or become portable authority. */
export class McpRootsService {
  private readonly live = new Map<string, LiveDisclosure>();
  private readonly pending = new Map<string, { brainId: string; controller: AbortController }>();
  private readonly tails = new Map<string, Promise<void>>();
  private readonly notifications = new Map<string, McpRootsStatus["notification"]>();
  private closed = false;
  constructor(private readonly options: McpRootsOptions) {}
  private file(brainId: string, serverId: string) { return join(this.options.directory, key(brainId, serverId) + ".json"); }
  private same(a: McpHostConnection, b: McpHostConnection) {
    return a.brainId === b.brainId && a.serverId === b.serverId && a.endpoint === b.endpoint &&
      a.configurationSha256 === b.configurationSha256 && a.connectionEpoch === b.connectionEpoch && a.connectionId === b.connectionId;
  }
  private check(connection: McpHostConnection, signal: AbortSignal, ready = true): void {
    signal.throwIfAborted(); if (this.closed) throw new Error("MCP root disclosure service is closed.");
    const current = this.options.connection(connection.brainId, connection.serverId);
    if (!this.same(connection, current) || !current.enabled) throw new Error("MCP roots connection changed or lost access; review the current endpoint.");
    if (ready) this.options.requireConnection(connection);
    signal.throwIfAborted();
  }
  private async policy(connection: McpHostConnection, signal: AbortSignal, ready = true): Promise<McpFileDisclosurePolicy> {
    this.check(connection, signal, ready); const policy = await this.options.filePolicy(connection.brainId, signal); this.check(connection, signal, ready);
    if (!policy || !["off", "ask", "auto", "full"].includes(policy.level) || typeof policy.revision !== "string" || !policy.revision) throw new Error("Current file disclosure policy is unavailable.");
    return policy;
  }
  private async inspect(path: string, expected?: string): Promise<{ identity: string; root: McpRoot }> {
    if (typeof path !== "string" || !path || /[\0\r\n]/.test(path) || path.length > 32_000 || resolve(path) !== path) throw new Error("MCP roots require canonical main-selected absolute paths.");
    const before = await lstat(path, { bigint: true });
    if (before.isSymbolicLink() || !before.isFile() && !before.isDirectory() || resolve(await realpath(path)) !== path) throw new Error("The selected MCP root is a link, symbolic ancestor, or unsupported file type.");
    const identity = `${before.dev}:${before.ino}:${before.mode & 0o170000n}`;
    if (before.ino <= 0n || expected !== undefined && expected !== identity) throw new Error("The selected MCP root was replaced; a fresh disclosure review is required.");
    await access(path, constants.R_OK | (before.isDirectory() ? constants.X_OK : 0));
    const after = await lstat(path, { bigint: true });
    if (`${after.dev}:${after.ino}:${after.mode & 0o170000n}` !== identity || resolve(await realpath(path)) !== path) throw new Error("The selected MCP root changed during accessibility checks.");
    return { identity, root: { uri: pathToFileURL(path).href, name: basename(path) || path } };
  }
  private async read(brainId: string, serverId: string): Promise<Selection | undefined> {
    const path = this.file(brainId, serverId); let input;
    try {
      const info = await lstat(path, { bigint: true });
      if (!info.isFile() || info.isSymbolicLink() || info.size > BigInt(MAX_BYTES) || resolve(await realpath(path)) !== resolve(path)) throw new Error("Unsafe device-owned MCP roots settings.");
      await this.options.admit(Number(info.size) * 8 + 65536, brainId);
      input = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
      const raw = await input.readFile(); const after = await input.stat({ bigint: true }), current = await lstat(path, { bigint: true });
      const identity = (value: typeof info) => [value.dev, value.ino, value.size, value.mtimeNs, value.ctimeNs].join(":");
      if (BigInt(raw.length) !== info.size || identity(after) !== identity(info) || identity(current) !== identity(info)) throw new Error("Device roots settings changed while reading.");
      const value = JSON.parse(raw.toString("utf8")) as Selection;
      if (!value || value.format !== "omni-device-mcp-roots" || value.version !== 1 || value.brainId !== brainId || value.serverId !== serverId ||
          typeof value.endpoint !== "string" || !SHA.test(value.configurationSha256) || !Number.isSafeInteger(value.revision) || value.revision < 1 || !Array.isArray(value.roots)) throw new Error("Malformed device-owned MCP roots selection.");
      const seen = new Set<string>();
      for (const root of value.roots) {
        if (!root || typeof root.path !== "string" || resolve(root.path) !== root.path || /[\r\n\0]/.test(root.path) ||
            root.uri !== pathToFileURL(root.path).href || typeof root.name !== "string" || typeof root.identity !== "string" || !root.identity || seen.has(root.uri)) throw new Error("Device roots lost their exact selected URI/source identity.");
        seen.add(root.uri);
      }
      return value;
    } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
    finally { await input?.close(); }
  }
  private async mutate(brainId: string, serverId: string, expectedRevision: number, selection: Selection, signal: AbortSignal): Promise<void> {
    const id = key(brainId, serverId), previous = this.tails.get(id) ?? Promise.resolve();
    const operation = previous.then(async () => {
      signal.throwIfAborted(); const saved = await this.read(brainId, serverId);
      if ((saved?.revision ?? 0) !== expectedRevision) throw new Error("MCP roots selection changed; review the current selection instead of rebasing approval.");
      const contents = Buffer.from(canonical(selection));
      if (contents.length > MAX_BYTES) throw new Error("The roots list exceeds its protocol response envelope; no tail was silently omitted.");
      await this.options.admit(contents.length * 8 + 65536, brainId); await this.options.checkDisk(contents.length * 2 + 65536); signal.throwIfAborted();
      await mkdir(this.options.directory, { recursive: true, mode: 0o700 });
      if (resolve(await realpath(this.options.directory)) !== resolve(this.options.directory)) throw new Error("Device roots settings directory has a symbolic ancestor.");
      const temporary = this.file(brainId, serverId) + "." + randomUUID() + ".next", output = await open(temporary, "wx", 0o600);
      try { await output.writeFile(contents); await output.sync(); await output.close(); signal.throwIfAborted(); await rename(temporary, this.file(brainId, serverId)); }
      finally { await output.close().catch(() => undefined); await unlink(temporary).catch(() => undefined); }
    });
    const tail = operation.then(() => undefined, () => undefined); this.tails.set(id, tail);
    try { await operation; } finally { if (this.tails.get(id) === tail) this.tails.delete(id); }
  }
  private async notify(connection: McpHostConnection): Promise<void> {
    const id = key(connection.brainId, connection.serverId);
    try { this.notifications.set(id, await this.options.notify(connection) ? "sent" : "not-ready"); }
    catch { this.notifications.set(id, "failed"); }
  }
  private begin(connection: McpHostConnection, signal: AbortSignal) {
    const id = key(connection.brainId, connection.serverId);
    this.pending.get(id)?.controller.abort(new Error("The roots review was replaced or revoked."));
    const controller = new AbortController(); this.pending.set(id, { brainId: connection.brainId, controller });
    return { id, controller, signal: AbortSignal.any([signal, controller.signal]), finish: () => { if (this.pending.get(id)?.controller === controller) this.pending.delete(id); } };
  }
  private revoke(id: string): McpHostConnection | undefined { const previous = this.live.get(id); this.live.delete(id); return previous?.connection; }
  async status(brainId: string, serverId: string): Promise<McpRootsStatus> {
    const connection = this.options.connection(brainId, serverId), saved = await this.read(brainId, serverId), id = key(brainId, serverId), live = this.live.get(id);
    const roots: McpRootsStatus["roots"] = [];
    let allAccessible = true;
    for (const root of saved?.roots ?? []) {
      try { await this.inspect(root.path, root.identity); roots.push({ uri: root.uri, name: root.name, accessible: true }); }
      catch (error) { allAccessible = false; roots.push({ uri: root.uri, name: root.name, accessible: false, reason: error instanceof Error ? error.message : String(error) }); }
    }
    const policy = await this.options.filePolicy(brainId, new AbortController().signal);
    const latest = this.options.connection(brainId, serverId);
    const current = Boolean(!this.closed && saved && live && policy.level !== "off" && policy.revision === live.policyRevision && allAccessible && latest.ready && latest.enabled && this.same(connection, latest) && this.same(latest, live.connection) && this.live.get(id) === live && saved.revision === live.revision && saved.configurationSha256 === latest.configurationSha256 && selectionSha(saved) === live.selectionSha256);
    return { format: "omni-mcp-roots-status", version: 1, brainId, serverId, endpoint: latest.endpoint, revision: saved?.revision ?? 0, roots,
      state: !roots.length ? "empty" : current ? "shared" : "review-required", connectionReady: latest.ready,
      ...(current ? { sharedConnectionEpoch: latest.connectionEpoch } : {}), ...(this.notifications.has(id) ? { notification: this.notifications.get(id)! } : {}),
      disclosureOnly: true, filesystemSandbox: false, permissionsPortable: false };
  }
  /** `paths` are exclusively a main picker result, never tool/renderer arguments. */
  async select(brainId: string, serverId: string, paths: string[], signal: AbortSignal): Promise<McpRootsStatus> {
    const connection = this.options.connection(brainId, serverId), work = this.begin(connection, signal);
    try {
      const policy = await this.policy(connection, work.signal, false); if (policy.level === "off") throw new Error("File access is Off; filesystem root disclosure is disabled.");
      const old = this.revoke(work.id); if (old) await this.notify(old);
      if (!Array.isArray(paths)) throw new Error("Select MCP roots through the trusted main picker.");
      let estimated = 0; for (const path of paths) { if (typeof path !== "string") throw new Error("Invalid selected root."); estimated += Buffer.byteLength(path) * 8 + 4096; }
      if (estimated > MAX_BYTES * 8) throw new Error("Selected roots exceed the protocol response envelope."); await this.options.admit(estimated + 65536, brainId); this.check(connection, work.signal, false);
      const roots: SelectedRoot[] = [], seen = new Set<string>();
      for (const path of paths) { this.check(connection, work.signal, false); const checked = await this.inspect(path); this.check(connection, work.signal, false);
        if (!seen.has(checked.root.uri)) { seen.add(checked.root.uri); roots.push({ ...checked.root, path, identity: checked.identity }); } }
      const saved = await this.read(brainId, serverId), selection: Selection = { format: "omni-device-mcp-roots", version: 1, brainId, serverId,
        endpoint: connection.endpoint, configurationSha256: connection.configurationSha256, revision: (saved?.revision ?? 0) + 1, roots };
      this.check(connection, work.signal, false); await this.mutate(brainId, serverId, saved?.revision ?? 0, selection, work.signal); await this.notify(connection);
      if (connection.ready && roots.length) await this.reviewSelection(connection, selection, work.signal);
      return await this.status(brainId, serverId);
    } finally { work.finish(); }
  }
  private async reviewSelection(connection: McpHostConnection, selection: Selection, signal: AbortSignal): Promise<void> {
    const policy = await this.policy(connection, signal); if (policy.level === "off") throw new Error("File access is Off; filesystem root disclosure is disabled.");
    if (selection.configurationSha256 !== connection.configurationSha256 || selection.endpoint !== connection.endpoint) throw new Error("The selected roots belong to a different endpoint/configuration; select them again explicitly.");
    const roots: McpRoot[] = [];
    for (const root of selection.roots) { this.check(connection, signal); roots.push((await this.inspect(root.path, root.identity)).root); this.check(connection, signal); }
    const selectionSha256 = selectionSha(selection);
    const accepted = await this.options.approve(connection, { roots, selectionSha256, revision: selection.revision, policyRevision: policy.revision, disclosureOnly: true, filesystemSandbox: false }, signal);
    this.check(connection, signal); if (!accepted) return;
    const currentPolicy = await this.policy(connection, signal), current = await this.read(connection.brainId, connection.serverId); this.check(connection, signal);
    if (currentPolicy.level === "off" || currentPolicy.revision !== policy.revision || !current || current.revision !== selection.revision || canonical(current) !== canonical(selection)) throw new Error("MCP roots policy or selected paths changed during review; no disclosure grant was installed.");
    for (const root of selection.roots) { await this.inspect(root.path, root.identity); this.check(connection, signal); }
    const finalPolicy = await this.policy(connection, signal);
    if (finalPolicy.level === "off" || finalPolicy.revision !== policy.revision) throw new Error("File policy changed before root disclosure activation.");
    this.live.set(key(connection.brainId, connection.serverId), { connection: structuredClone(connection), revision: selection.revision, policyRevision: policy.revision, selectionSha256 });
    await this.notify(connection);
  }
  async review(brainId: string, serverId: string, signal: AbortSignal): Promise<McpRootsStatus> {
    const connection = this.options.connection(brainId, serverId), work = this.begin(connection, signal);
    try { const saved = await this.read(brainId, serverId); if (!saved?.roots.length) return this.status(brainId, serverId);
      const old = this.revoke(work.id); if (old) await this.notify(old);
      await this.reviewSelection(connection, saved, work.signal); return await this.status(brainId, serverId);
    } finally { work.finish(); }
  }
  async clear(brainId: string, serverId: string, signal: AbortSignal): Promise<McpRootsStatus> {
    const connection = this.options.connection(brainId, serverId), work = this.begin(connection, signal); this.revoke(work.id);
    try { const saved = await this.read(brainId, serverId); if (saved) await this.mutate(brainId, serverId, saved.revision, { ...saved, roots: [], revision: saved.revision + 1 }, work.signal);
      await this.notify(connection); return await this.status(brainId, serverId);
    } finally { work.finish(); }
  }
  /** No server-driven picker/review: unshared, stale or inaccessible roots list empty. */
  async list(origin: McpRequestOrigin, signal: AbortSignal): Promise<{ roots: McpRoot[] }> {
    const connection = this.options.connection(origin.brainId, origin.serverId), id = key(origin.brainId, origin.serverId);
    if (origin.connectionEpoch !== connection.connectionEpoch || origin.endpoint !== connection.endpoint) throw new Error("Stale MCP roots request.");
    const policy = await this.policy(connection, signal), live = this.live.get(id), saved = await this.read(origin.brainId, origin.serverId); this.check(connection, signal);
    if (!live || !saved || policy.level === "off" || live.policyRevision !== policy.revision || live.revision !== saved.revision || saved.configurationSha256 !== connection.configurationSha256 || selectionSha(saved) !== live.selectionSha256 || !this.same(live.connection, connection)) {
      if (this.revoke(id)) await this.notify(connection); return { roots: [] };
    }
    const roots: McpRoot[] = [];
    try { for (const root of saved.roots) { roots.push((await this.inspect(root.path, root.identity)).root); this.check(connection, signal); } }
    catch (error) { this.revoke(id); await this.notify(connection); if (signal.aborted) throw error; return { roots: [] }; }
    const finalPolicy = await this.policy(connection, signal);
    const finalSelection = await this.read(origin.brainId, origin.serverId);
    if (finalPolicy.level === "off" || finalPolicy.revision !== policy.revision || this.live.get(id) !== live || !finalSelection || selectionSha(finalSelection) !== live.selectionSha256) {
      this.revoke(id); await this.notify(connection); return { roots: [] };
    }
    this.check(connection, signal); await this.options.admit(Buffer.byteLength(canonical(roots)) * 8 + 65536, origin.brainId); this.check(connection, signal);
    const returnPolicy = await this.policy(connection, signal);
    if (this.live.get(id) !== live || returnPolicy.level === "off" || returnPolicy.revision !== policy.revision) { this.revoke(id); await this.notify(connection); return { roots: [] }; } return { roots };
  }
  async revokeBrain(brainId: string): Promise<void> {
    for (const work of this.pending.values()) if (work.brainId === brainId) work.controller.abort(new Error("Root disclosure was revoked."));
    for (const [id, live] of this.live) if (live.connection.brainId === brainId) { this.pending.get(id)?.controller.abort(); this.live.delete(id); await this.notify(live.connection); }
  }
  async revokeConnection(brainId: string, serverId: string): Promise<void> {
    const id = key(brainId, serverId); this.pending.get(id)?.controller.abort(new Error("The exact MCP connection retired before root disclosure review completed."));
    const previous = this.revoke(id); if (previous) await this.notify(previous);
  }
  async close(): Promise<void> { this.closed = true; for (const work of this.pending.values()) work.controller.abort(); this.pending.clear(); this.live.clear(); await Promise.all(this.tails.values()); }
}
