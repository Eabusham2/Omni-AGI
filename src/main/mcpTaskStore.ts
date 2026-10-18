import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, opendir, realpath, rename, unlink, link } from "node:fs/promises";
import { dirname, join, resolve, relative, isAbsolute, sep } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { MCP_TASK_WIRE_VERSION, type McpTaskOutcome, type McpTaskRecord, type McpTaskSource } from "../shared/mcpTasks";
import { assertMcpTaskTransition, checkedMcpTaskRecord, mcpTaskObject, mcpTaskSha, MCP_TASK_REF, MCP_TASK_SHA } from "./mcpTaskProtocol";

export const MCP_TASK_DIRECTORY = "operational-mcp-tasks";
const tails = new Map<string, Promise<void>>();
export async function withMcpTaskSnapshot<T>(brainRoot: string, operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  const key = await realpath(brainRoot), previous = tails.get(key) ?? Promise.resolve(); let release!: () => void;
  const done = new Promise<void>(yes => { release = yes; }), next = previous.then(() => done); tails.set(key, next);
  let abort!: () => void;
  const cancelled = new Promise<never>((_resolve, reject) => { abort = () => reject(signal?.reason ?? new Error("MCP task custody wait stopped.")); signal?.addEventListener("abort", abort, { once: true }); if (signal?.aborted) abort(); });
  try { await (signal ? Promise.race([previous, cancelled]) : previous); signal?.throwIfAborted(); return await operation(); }
  finally { signal?.removeEventListener("abort", abort); release(); void next.then(() => { if (tails.get(key) === next) tails.delete(key); }); }
}
export interface McpTaskStoreOptions {
  brainDirectory(brainId: string): Promise<string>;
  admit?(bytes: number, purpose: string): Promise<void>;
  checkDisk?(brainId: string, bytes: number): Promise<void>;
}
export interface McpTaskReadOptions { signal?: AbortSignal; admit?(bytes: number, purpose: string): Promise<void> | void; }
const identity = (stat: { dev: bigint; ino: bigint; size: bigint; mtimeNs: bigint; ctimeNs: bigint }): string => [stat.dev, stat.ino, stat.size, stat.mtimeNs, stat.ctimeNs].join(":");
export async function mcpTaskReadBytes(root: string, local: string, options: McpTaskReadOptions = {}): Promise<{ bytes: Buffer; sha256: string }> {
  options.signal?.throwIfAborted();
  if (!local || local.includes("\\") || local.split("/").some(part => !part || part === "." || part === "..")) throw new Error("Unsafe MCP task custody path.");
  const canonicalRoot = await realpath(root), path = resolve(canonicalRoot, ...local.split("/")), from = relative(canonicalRoot, path);
  if (isAbsolute(from) || from === ".." || from.startsWith(`..${sep}`) || resolve(await realpath(path)) !== path) throw new Error("MCP task custody escaped its exact source root.");
  const before = await lstat(path, { bigint: true }); if (!before.isFile() || before.isSymbolicLink() || before.size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("MCP task custody must be a regular exact-size file.");
  const needed = Number(before.size) * 12 + 65536;
  if (!Number.isSafeInteger(needed)) throw new Error("MCP task metadata exceeds exact allocation.");
  if (options.admit) await options.admit(needed, "MCP task custody JSON");
  else if (needed > freemem() || needed > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error("MCP task custody awaits physical RAM.");
  options.signal?.throwIfAborted(); const file = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  try {
    if (identity(await file.stat({ bigint: true })) !== identity(before)) throw new Error("MCP task custody changed before its read.");
    const bytes = Buffer.alloc(Number(before.size)); let position = 0;
    while (position < bytes.length) { options.signal?.throwIfAborted(); const read = await file.read(bytes, position, Math.min(65536, bytes.length - position), position);
      if (!read.bytesRead) throw new Error("MCP task custody ended before its admitted byte count."); position += read.bytesRead; }
    const extra = Buffer.alloc(1); if ((await file.read(extra, 0, 1, position)).bytesRead) throw new Error("MCP task custody grew past its admitted byte count."); options.signal?.throwIfAborted();
    if (BigInt(bytes.length) !== before.size || identity(await file.stat({ bigint: true })) !== identity(before) || identity(await lstat(path, { bigint: true })) !== identity(before)) throw new Error("MCP task custody changed during its read.");
    return { bytes, sha256: mcpTaskSha(bytes) };
  } finally { await file.close(); }
}
export async function mcpTaskReadJson(root: string, local: string, options: McpTaskReadOptions = {}): Promise<{ value: unknown; bytes: Buffer; sha256: string }> {
  const saved = await mcpTaskReadBytes(root, local, options); return { ...saved, value: JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(saved.bytes)) };
}
async function exists(path: string): Promise<boolean> { try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; } }
async function directory(path: string): Promise<void> { await mkdir(path, { recursive: true, mode: 0o700 }); const stat = await lstat(path); if (!stat.isDirectory() || stat.isSymbolicLink() || resolve(await realpath(path)) !== resolve(path)) throw new Error("MCP task custody has a symbolic/non-directory owner."); }
export async function writeMcpTaskBytes(root: string, local: string, bytes: Buffer, replace = false): Promise<void> {
  if (!local || local.includes("\\") || local.split("/").some(part => !part || part === "." || part === "..")) throw new Error("Unsafe MCP task custody write path.");
  const canonicalRoot = await realpath(root), path = join(canonicalRoot, ...local.split("/")); await directory(dirname(path));
  const temporary = `${path}.${randomUUID()}.next`, output = await open(temporary, "wx", 0o600);
  try { await output.writeFile(bytes); await output.sync(); await output.close(); if (replace) await rename(temporary, path); else await link(temporary, path); }
  finally { await output.close().catch(() => undefined); await unlink(temporary).catch(() => undefined); }
  if (process.platform !== "win32") { const parent = await open(dirname(path), "r"); try { await parent.sync(); } finally { await parent.close(); } }
}

/** Non-executable source-original receipts. Live management grants exist only in McpClient memory. */
export class McpTaskStore {
  constructor(private readonly options: McpTaskStoreOptions) {}
  private async admit(bytes: number, purpose: string): Promise<void> {
    if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("MCP task allocation exceeds exact host planning.");
    if (this.options.admit) await this.options.admit(bytes, purpose);
    else if (bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error("MCP task metadata awaits physical RAM.");
  }
  private async root(brainId: string): Promise<string> {
    const requested = resolve(await this.options.brainDirectory(brainId)), info = await lstat(requested);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("MCP task instance owner must be a real directory."); return realpath(requested);
  }
  private relative(taskRef: string): string { if (!MCP_TASK_REF.test(taskRef)) throw new Error("MCP task reference is invalid."); return `engine/${MCP_TASK_DIRECTORY}/${taskRef}.json`; }
  private read(root: string, local: string): Promise<{ value: unknown; bytes: Buffer; sha256: string }> { return mcpTaskReadJson(root, local, { admit: this.options.admit }); }
  private async write(brainId: string, root: string, local: string, bytes: Buffer, replace = false): Promise<void> {
    await this.admit(bytes.length * 12 + 65536, "MCP task custody write"); await this.options.checkDisk?.(brainId, bytes.length * 3 + 65536); await writeMcpTaskBytes(root, local, bytes, replace);
  }
  async prepare(input: { source: McpTaskSource; requestId: string; actionEventId?: string; chatTurnId?: string; argumentsSha256: string }): Promise<McpTaskRecord> { return (await this.reserveCreation(input)).record; }
  async reserveCreation(input: { source: McpTaskSource; requestId: string; actionEventId?: string; chatTurnId?: string; argumentsSha256: string }): Promise<{ record: McpTaskRecord; created: boolean }> {
    const root = await this.root(input.source.brainId);
    return withMcpTaskSnapshot(root, async () => {
      const seed = mcpTaskSha(JSON.stringify([input.source.brainId, input.source.serverId, input.source.toolId, input.requestId, input.actionEventId ?? null, input.chatTurnId ?? null]));
      const taskRef = `${seed.slice(0, 8)}-${seed.slice(8, 12)}-${seed.slice(12, 16)}-${seed.slice(16, 20)}-${seed.slice(20, 32)}`, local = this.relative(taskRef);
      if (await exists(join(root, local))) {
        const previous = checkedMcpTaskRecord((await this.read(root, local)).value);
        if (previous.argumentsSha256 !== input.argumentsSha256 || JSON.stringify(previous.source) !== JSON.stringify(input.source) || previous.requestId !== input.requestId || previous.actionEventId !== input.actionEventId || previous.chatTurnId !== input.chatTurnId) throw new Error("The original MCP task creation intent changed. Review a new action instead of reusing its identity.");
        return { record: previous, created: false };
      }
      const now = new Date().toISOString(), row = checkedMcpTaskRecord({ format: "omni-mcp-task-record", version: 1, wireVersion: MCP_TASK_WIRE_VERSION,
        taskRef, ...input, createdAt: now, updatedAt: now, creation: "prepared", authority: "historical-not-live" });
      await this.write(input.source.brainId, root, this.relative(row.taskRef), Buffer.from(JSON.stringify(row))); return { record: row, created: true };
    });
  }
  async get(brainId: string, taskRef: string): Promise<McpTaskRecord> {
    const root = await this.root(brainId); return withMcpTaskSnapshot(root, async () => checkedMcpTaskRecord((await this.read(root, this.relative(taskRef))).value));
  }
  async update(brainId: string, taskRef: string, patch: Partial<Pick<McpTaskRecord, "creation" | "remote" | "createResult" | "lastOperation" | "reason">>): Promise<McpTaskRecord> {
    const root = await this.root(brainId);
    return withMcpTaskSnapshot(root, async () => {
      const saved = await this.read(root, this.relative(taskRef)), prior = checkedMcpTaskRecord(saved.value);
      if (prior.source.brainId !== brainId) throw new Error("Inherited MCP task receipts are dormant; source-original ownership is not rewritten.");
      const row = checkedMcpTaskRecord({ ...prior, ...patch, updatedAt: new Date().toISOString() }); if (row.remote) assertMcpTaskTransition(prior.remote, row.remote);
      const history = `engine/${MCP_TASK_DIRECTORY}/history/${saved.sha256}.json`;
      if (!await exists(join(root, history))) await this.write(brainId, root, history, saved.bytes);
      await this.write(brainId, root, this.relative(taskRef), Buffer.from(JSON.stringify(row)), true); return row;
    });
  }
  async putOutcome(brainId: string, taskRef: string, kind: "result" | "rpc-error", payload: Record<string, unknown>): Promise<McpTaskOutcome> {
    const root = await this.root(brainId);
    return withMcpTaskSnapshot(root, async () => {
      const saved = await this.read(root, this.relative(taskRef)), prior = checkedMcpTaskRecord(saved.value);
      if (prior.source.brainId !== brainId || !prior.remote) throw new Error("A final outcome requires its real current-owned accepted task.");
      const bytes = Buffer.from(JSON.stringify(payload)), payloadSha256 = mcpTaskSha(bytes), receiptId = mcpTaskSha(JSON.stringify([prior.source, taskRef, prior.remote.taskId, kind, payloadSha256]));
      if (prior.result && (prior.result.payloadSha256 !== payloadSha256 || prior.result.kind !== kind)) throw new Error("The server returned a conflicting final task outcome. The previous committed outcome was not overwritten.");
      const local = `engine/${MCP_TASK_DIRECTORY}/payloads/${payloadSha256}.json`;
      if (!await exists(join(root, local))) await this.write(brainId, root, local, bytes);
      else if ((await this.read(root, local)).sha256 !== payloadSha256) throw new Error("MCP task result payload checksum changed.");
      if (!prior.result) {
        const history = `engine/${MCP_TASK_DIRECTORY}/history/${saved.sha256}.json`;
        if (!await exists(join(root, history))) await this.write(brainId, root, history, saved.bytes);
        const row = checkedMcpTaskRecord({ ...prior, updatedAt: new Date().toISOString(), result: { receiptId, payloadSha256, kind, observedAt: new Date().toISOString() } });
        await this.write(brainId, root, this.relative(taskRef), Buffer.from(JSON.stringify(row)), true);
      }
      return this.outcome(prior.source, taskRef, { receiptId, payloadSha256, kind }, payload, false);
    });
  }
  private outcome(source: McpTaskSource, taskRef: string, result: { receiptId: string; payloadSha256: string; kind: "result" | "rpc-error" }, payload: Record<string, unknown>, retained: boolean): McpTaskOutcome {
    return { format: "omni-mcp-task-outcome", version: 1, wireVersion: MCP_TASK_WIRE_VERSION, taskRef, source,
      taskResultReceiptId: result.receiptId, kind: result.kind, ...(result.kind === "result" ? { result: payload } : { error: payload as unknown as NonNullable<McpTaskOutcome["error"]> }), payloadSha256: result.payloadSha256, retained, untrustedTaskData: true };
  }
  async retainedOutcome(brainId: string, taskRef: string): Promise<McpTaskOutcome | undefined> {
    const root = await this.root(brainId);
    return withMcpTaskSnapshot(root, async () => {
      const row = checkedMcpTaskRecord((await this.read(root, this.relative(taskRef))).value); if (!row.result) return undefined;
      const saved = await this.read(root, `engine/${MCP_TASK_DIRECTORY}/payloads/${row.result.payloadSha256}.json`);
      if (saved.sha256 !== row.result.payloadSha256) throw new Error("MCP task final payload changed.");
      return this.outcome(row.source, taskRef, row.result, mcpTaskObject(saved.value, "retained task payload"), true);
    });
  }
  async localPage(brainId: string, after?: string, limit = 25): Promise<{ records: Array<{ record: McpTaskRecord; custodyPath: string }>; nextCursor?: string }> {
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200 || after !== undefined && !/^[A-Za-z0-9_-]+$/.test(after)) throw new Error("MCP task metadata paging arguments are invalid.");
    const root = await this.root(brainId), path = join(root, "engine", MCP_TASK_DIRECTORY); if (!await exists(path)) return { records: [] };
    return withMcpTaskSnapshot(root, async () => {
      const cursor = after === undefined ? undefined : Buffer.from(after, "base64url").toString("utf8"), names: string[] = [], pending = [`engine/${MCP_TASK_DIRECTORY}`]; let descriptorBytes = 65536;
      while (pending.length) {
        const prefix = pending.pop()!, current = join(root, prefix), info = await lstat(current);
        if (!info.isDirectory() || info.isSymbolicLink() || await realpath(current) !== current) throw new Error("MCP task page contains an unsafe namespace.");
        const entries = await opendir(current);
        for await (const item of entries) {
          if (item.isDirectory()) { if (item.name === "lineage" || prefix.endsWith("/lineage") && MCP_TASK_SHA.test(item.name)) {
            descriptorBytes += Buffer.byteLength(prefix + item.name) * 4 + 512; await this.admit(descriptorBytes, "MCP task page namespace descriptors"); pending.push(`${prefix}/${item.name}`);
          } continue; }
          if (!item.isFile() || !MCP_TASK_REF.test(item.name.replace(/\.json$/, "")) || !item.name.endsWith(".json")) continue;
          const local = `${prefix}/${item.name}`; if (cursor !== undefined && local <= cursor) continue; names.push(local); names.sort(); if (names.length > limit + 1) names.pop();
        }
      }
      const selected = names.slice(0, limit), records: Array<{ record: McpTaskRecord; custodyPath: string }> = []; let retainedBytes = 0;
      for (const local of selected) { const saved = await this.read(root, local); retainedBytes += saved.bytes.length + Buffer.byteLength(local) + 512;
        const planned = retainedBytes * 16 + 65536;
        await this.admit(planned, "MCP task metadata page and serialization");
        records.push({ record: checkedMcpTaskRecord(saved.value), custodyPath: local }); }
      return { records, ...(names.length > limit ? { nextCursor: Buffer.from(selected.at(-1)!).toString("base64url") } : {}) };
    });
  }
}
