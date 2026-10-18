import { createHash, createHmac, randomBytes, randomUUID, timingSafeEqual, type Hash } from "node:crypto";
import { isUtf8 } from "node:buffer";
import { constants, type BigIntStats, type Dir, type Dirent } from "node:fs";
import { link, lstat, mkdir, open, opendir, realpath, unlink, type FileHandle } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import type { ChatTaskAttachmentReadResult } from "../shared/taskAttachments";
import type { FileToolPage, FileToolSearchMatch, FileToolSearchPage, FileToolSource, TaskFileHandoff } from "../shared/fileTools";
import { withFileHandoffSnapshot } from "./fileToolHandoffCustody";

const BLOCK = 64 * 1024, MAX_PAGE = 1024 * 1024, TTL = 5 * 60_000;
const SHA = /^[a-f0-9]{64}$/;
const digest = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
const canonical = (value: unknown): string => JSON.stringify(value);
function string(value: unknown, label: string, maximum = 32_000): string {
  if (typeof value !== "string" || !value || value.includes("\0") || value.length > maximum) throw new Error(`Invalid ${label}.`);
  return value;
}
function integer(value: unknown, fallback: number, maximum = Number.MAX_SAFE_INTEGER): number {
  const result = value ?? fallback;
  if (typeof result !== "number" || !Number.isSafeInteger(result) || result < 0 || result > maximum) throw new Error("File range/window must use an admitted non-negative safe integer.");
  return result;
}
function identity(info: BigIntStats): string {
  if (info.ino <= 0n || info.size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Stable file identity/size is unavailable.");
  return [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs, info.mode].join(":");
}
function inside(root: string, path: string): boolean {
  const local = relative(root, path);
  return local === "" || local !== ".." && !local.startsWith(`..${sep}`) && !isAbsolute(local);
}
function absolute(value: unknown): string {
  const path = string(value, "absolute file path");
  if (!isAbsolute(path)) throw new Error("File paths must be absolute.");
  return resolve(path);
}
export interface FileToolContext {
  /** Main-owned actual task relation, never accepted from tool arguments. */
  chatTurnId?: string;
  /** Stable main permission/package revisions, not an action/session lease. */
  policyRevision: string;
  assertCurrent(): Promise<void>;
}
export interface FileToolDependencies {
  root(brainId: string): Promise<string>;
  readTask(brainId: string, turnId: string, attachmentId: string, offset: number, bytes: number, signal: AbortSignal): Promise<ChatTaskAttachmentReadResult>;
  attachmentPermission(brainId: string, signal: AbortSignal): Promise<string>;
  assertTaskSource(brainId: string, turnId: string, source: FileToolSource, signal: AbortSignal): Promise<void>;
  checkResources(brainId: string, bytes: number, purpose: string): Promise<void>;
  reserveWrite(root: string, bytes: number): Promise<void>;
  now?(): number;
}
interface SourceReader {
  source: FileToolSource;
  read(offset: number, bytes: number, signal: AbortSignal, context: FileToolContext): Promise<Buffer>;
  assert(signal: AbortSignal, context: FileToolContext): Promise<void>;
  close(): Promise<void>;
}
interface SessionBase {
  id: string; brainId: string; requestId: string; turnId?: string; generation: number; expires: number;
  controller: AbortController; policyRevision: string; busy: boolean;
  timer?: ReturnType<typeof setTimeout>;
  previous?: { cursor: string; requestId: string; argumentsSha256: string; output: unknown };
}
interface ReadSession extends SessionBase { kind: "read"; reader: SourceReader; offset: number; hash: Hash; whole: boolean; pageBytes: number }
interface DirectoryFrame { path: string; identity: string; directory: Dir }
interface SearchSession extends SessionBase {
  kind: "search"; root?: string; rootIdentity?: string; stack: DirectoryFrame[]; initialReader?: SourceReader;
  current?: { reader: SourceReader; metadata: FileToolSearchMatch; offset: number; carry: Buffer; nextMatch: number;
    pending?: { bytes: Buffer; base: number; position: number } };
  matching: FileToolSearchPage["matching"]; needle: Buffer; name: string; recursive: boolean; pageSize: number; scanBytes: number;
  visited: number; scanned: number; list: boolean;
}
type Session = ReadSession | SearchSession;
export class FileToolEffectError extends Error {
  constructor(message: string, readonly receipt: Record<string, unknown>) { super(message); this.name = "FileToolEffectError"; }
}

/** No neural calls. Every source page is authorized again and bounded. */
export class FileToolService {
  private readonly sessions = new Map<string, Session>();
  private readonly key = randomBytes(32);
  private readonly retiring = new Map<string, Set<Promise<void>>>();
  private readonly handoffs = new Map<string, { brainId: string; requestId: string; controller: AbortController; sha256: string; work: Promise<unknown> }>();
  constructor(private readonly dependencies: FileToolDependencies) {}
  private now() { return this.dependencies.now?.() ?? Date.now(); }
  private cursor(session: Session) {
    const body = Buffer.from(canonical([session.id, session.brainId, session.turnId ?? null, session.generation]), "utf8").toString("base64url");
    return `${body}.${createHmac("sha256", this.key).update(body).digest("hex")}`;
  }
  private async guard(brainId: string, signal: AbortSignal, context: FileToolContext): Promise<void> {
    string(context.policyRevision, "main-owned file policy revision", 1024);
    signal.throwIfAborted(); await context.assertCurrent(); signal.throwIfAborted();
    await this.dependencies.root(brainId); signal.throwIfAborted(); await context.assertCurrent(); signal.throwIfAborted();
  }
  private async check(brainId: string, bytes: number, signal: AbortSignal, context: FileToolContext, purpose: string) {
    await this.guard(brainId, signal, context);
    await this.dependencies.checkResources(brainId, bytes, purpose);
    await this.guard(brainId, signal, context);
  }
  private async checkedPath(path: string, root?: string): Promise<BigIntStats> {
    const before = await lstat(path, { bigint: true });
    if (before.isSymbolicLink()) throw new Error("File tools do not follow symbolic links/junctions.");
    const physical = await realpath(path);
    if (resolve(physical) !== resolve(path) || root && !inside(root, physical)) throw new Error("File path crossed its checked root or a symbolic ancestor.");
    const after = await lstat(path, { bigint: true });
    if (identity(before) !== identity(after)) throw new Error("File path changed while being resolved.");
    return after;
  }
  private async hostReader(brainId: string, path: string, signal: AbortSignal, context: FileToolContext, root?: string): Promise<SourceReader> {
    await this.check(brainId, BLOCK * 8, signal, context, "bounded file handle/read scratch");
    const before = await this.checkedPath(path, root);
    if (!before.isFile()) throw new Error("Selected source is not a regular file.");
    const file = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
    const expected = identity(before); let closed = false;
    const assert = async (active: AbortSignal, current: FileToolContext) => {
      active.throwIfAborted(); await this.guard(brainId, active, current);
      if (closed || identity(await file.stat({ bigint: true })) !== expected || identity(await this.checkedPath(path, root)) !== expected)
        throw new Error("Opened file bytes/identity changed; start a new checked read.");
      await this.guard(brainId, active, current);
    };
    const close = async () => { if (!closed) { closed = true; await file.close(); } };
    try {
      await assert(signal, context);
      return { source: { kind: "host-file", path, displayName: basename(path), bytes: Number(before.size), identity: expected }, assert, close,
        read: async (offset, bytes, active, current) => {
          await assert(active, current); await this.check(brainId, bytes * 8 + BLOCK, active, current, "bounded file page and encoding");
          const buffer = Buffer.alloc(Math.min(bytes, Number(before.size) - offset)); let returned = 0;
          while (returned < buffer.length) { active.throwIfAborted(); const read = await file.read(buffer, returned, buffer.length - returned, offset + returned);
            if (!read.bytesRead) throw new Error("File ended before its checked byte coverage."); returned += read.bytesRead; }
          await assert(active, current); return buffer;
        } };
    } catch (error) { await close(); throw error; }
  }
  private async taskReader(brainId: string, attachmentId: string, signal: AbortSignal, context: FileToolContext): Promise<SourceReader> {
    if (!context.chatTurnId) throw new Error("Imported task files require the actual owning/continued task.");
    const turnId = context.chatTurnId;
    const permission = await this.dependencies.attachmentPermission(brainId, signal);
    const first = await this.dependencies.readTask(brainId, turnId, attachmentId, 0, 1, signal);
    const handle = first.attachment, expected = canonical([handle.brainId, handle.id, handle.contentSha256, handle.manifestSha256, handle.bytes, handle.turnId]);
    if (handle.id !== attachmentId) throw new Error("Task-file resolver returned another source.");
    if (!SHA.test(handle.contentSha256) || !SHA.test(handle.manifestSha256) || !Number.isSafeInteger(handle.bytes) || handle.bytes < 0) throw new Error("Task file descriptor has invalid exact-byte identity.");
    let closed = false;
    const source: FileToolSource = { kind: "task-attachment", attachmentId, manifestSha256: handle.manifestSha256, contentSha256: handle.contentSha256,
      displayName: handle.displayName, bytes: handle.bytes, identity: digest(expected) };
    const assert = async (active: AbortSignal, current: FileToolContext) => {
      active.throwIfAborted(); if (closed) throw new Error("Task file cursor is closed."); await this.guard(brainId, active, current);
      if (await this.dependencies.attachmentPermission(brainId, active) !== permission) throw new Error("Task attachment policy changed; review the current file again.");
      await this.dependencies.assertTaskSource(brainId, turnId, source, active);
      await this.guard(brainId, active, current);
    };
    await assert(signal, context);
    return { source, assert, close: async () => { closed = true; }, read: async (offset, bytes, active, currentContext) => {
      await assert(active, currentContext); await this.check(brainId, bytes * 8 + BLOCK, active, currentContext, "bounded imported task-file page");
      const result = await this.dependencies.readTask(brainId, turnId, attachmentId, offset, bytes, active), current = result.attachment;
      if (canonical([current.brainId, current.id, current.contentSha256, current.manifestSha256, current.bytes, current.turnId]) !== expected || result.offset !== offset)
        throw new Error("Imported task file source or scope changed during its page.");
      const buffer = Buffer.from(result.content, result.encoding === "base64" ? "base64" : "utf8");
      if (buffer.length !== result.returnedBytes || buffer.length > Math.min(bytes, source.bytes - offset) ||
          (!buffer.length && offset < source.bytes) || result.totalBytes !== source.bytes ||
          (result.nextOffset !== undefined && result.nextOffset !== offset + buffer.length)) throw new Error("Imported task-file page coverage differs.");
      await assert(active, currentContext); return buffer;
    } };
  }
  private async reader(brainId: string, args: Record<string, unknown>, signal: AbortSignal, context: FileToolContext): Promise<SourceReader> {
    if ((args.path !== undefined) === (args.attachmentId !== undefined)) throw new Error("Choose exactly one absolute path or actual task attachmentId.");
    return args.attachmentId !== undefined ? this.taskReader(brainId, string(args.attachmentId, "attachmentId", 128), signal, context)
      : this.hostReader(brainId, absolute(args.path), signal, context);
  }
  private base(brainId: string, context: FileToolContext, requestId: string): SessionBase {
    return { id: randomUUID(), brainId, requestId, ...(context.chatTurnId ? { turnId: context.chatTurnId } : {}), generation: 0,
      expires: this.now() + TTL, controller: new AbortController(), policyRevision: context.policyRevision, busy: false };
  }
  private async closeSession(session: Session): Promise<void> {
    if (session.timer) clearTimeout(session.timer);
    session.controller.abort(new Error("File cursor retired.")); this.sessions.delete(session.id);
    if (session.kind === "read") await session.reader.close();
    else {
      await session.current?.reader.close(); await session.initialReader?.close();
      await Promise.all(session.stack.splice(0).map(frame => frame.directory.close().catch(() => undefined)));
    }
  }
  cancelBrain(brainId: string, requestId?: string): void {
    for (const handoff of this.handoffs.values()) if (handoff.brainId === brainId && (requestId === undefined || handoff.requestId === requestId)) handoff.controller.abort(new Error("Task file handoff stopped."));
    for (const session of this.sessions.values()) if (session.brainId === brainId && (requestId === undefined || session.requestId === requestId)) {
      const retiring = this.retiring.get(brainId) ?? new Set<Promise<void>>(); this.retiring.set(brainId, retiring);
      const work = this.closeSession(session).catch(() => undefined).finally(() => { retiring.delete(work); if (!retiring.size) this.retiring.delete(brainId); });
      retiring.add(work);
    }
  }
  async settle(brainId: string, requestId?: string): Promise<void> { await Promise.all([...(this.retiring.get(brainId) ?? []), ...[...this.handoffs.values()]
    .filter(work => work.brainId === brainId && (requestId === undefined || work.requestId === requestId)).map(work => work.work.catch(() => undefined))]); }
  async forget(brainId: string): Promise<void> { this.cancelBrain(brainId); await this.settle(brainId); }
  async close(): Promise<void> { for (const id of new Set([...this.sessions.values(), ...this.handoffs.values()].map(session => session.brainId))) await this.forget(id); }
  private async sweep(): Promise<void> { for (const session of this.sessions.values()) if (session.expires <= this.now()) await this.closeSession(session); }
  private retain(session: Session): void {
    if (session.timer) clearTimeout(session.timer);
    session.expires = this.now() + TTL;
    session.timer = setTimeout(() => { void this.closeSession(session).catch(() => undefined); }, TTL);
    session.timer.unref(); this.sessions.set(session.id, session);
  }
  private async session(brainId: string, cursor: unknown, signal: AbortSignal, context: FileToolContext): Promise<Session> {
    const value = string(cursor, "live file cursor", 4096), split = value.lastIndexOf(".");
    const signature = value.slice(split + 1), expected = createHmac("sha256", this.key).update(value.slice(0, split)).digest();
    if (split < 0 || !SHA.test(signature) || !timingSafeEqual(expected, Buffer.from(signature, "hex"))) throw new Error("File cursor is not a current host-owned traversal.");
    const decoded: unknown = JSON.parse(Buffer.from(value.slice(0, split), "base64url").toString("utf8"));
    if (!Array.isArray(decoded) || typeof decoded[0] !== "string") throw new Error("Invalid file cursor.");
    const session = this.sessions.get(decoded[0]);
    if (session && session.brainId === brainId && session.turnId === context.chatTurnId && session.policyRevision !== context.policyRevision) await this.closeSession(session);
    if (!session || session.brainId !== brainId || session.turnId !== context.chatTurnId || session.policyRevision !== context.policyRevision || session.expires <= this.now()) throw new Error("File cursor expired, changed policy/owner/task, or is dormant after restart.");
    await this.guard(brainId, signal, context); session.controller.signal.throwIfAborted();
    return session;
  }
  private async readPage(session: ReadSession, signal: AbortSignal, context: FileToolContext): Promise<FileToolPage> {
    await this.guard(session.brainId, signal, context); session.controller.signal.throwIfAborted();
    const offset = session.offset, buffer = await session.reader.read(offset, session.pageBytes, signal, context);
    session.controller.signal.throwIfAborted(); await this.guard(session.brainId, signal, context);
    session.hash.update(buffer); session.offset += buffer.length; session.generation++;
    const complete = session.offset === session.reader.source.bytes;
    const result: FileToolPage = { format: "omni-file-page", version: 1, source: session.reader.source, offset, bytes: buffer.length,
      encoding: isUtf8(buffer) ? "utf8" : "base64", content: buffer.toString(isUtf8(buffer) ? "utf8" : "base64"), sha256: digest(buffer), complete,
      hasMore: !complete, ...(!complete ? { nextCursor: this.cursor(session), nextOffset: session.offset } : session.whole ? { wholeSha256: session.hash.digest("hex") } : {}) };
    if (complete) await this.closeSession(session); else this.retain(session);
    return result;
  }
  private async directoryFrame(brainId: string, path: string, root: string, signal: AbortSignal, context: FileToolContext): Promise<DirectoryFrame> {
    await this.check(brainId, BLOCK * 8, signal, context, "streamed directory handle");
    const before = await this.checkedPath(path, root); if (!before.isDirectory()) throw new Error("Search root/member is not a directory.");
    const directory = await opendir(path);
    try { const after = await this.checkedPath(path, root); await this.guard(brainId, signal, context);
      if (identity(before) !== identity(after)) throw new Error("Directory changed while opening its checked traversal.");
      return { path, identity: identity(before), directory };
    } catch (error) { await directory.close(); throw error; }
  }
  private async nextFile(session: SearchSession, signal: AbortSignal, context: FileToolContext): Promise<FileToolSearchMatch | undefined> {
    const root = session.root!;
    while (session.stack.length) {
      await this.guard(session.brainId, signal, context); session.controller.signal.throwIfAborted();
      const frame = session.stack.at(-1)!;
      if (identity(await this.checkedPath(root)) !== session.rootIdentity || identity(await this.checkedPath(frame.path, root)) !== frame.identity) throw new Error("Directory changed during this non-snapshot walk; restart its cursor.");
      const entry: Dirent | null = await frame.directory.read();
      await this.guard(session.brainId, signal, context);
      if (identity(await this.checkedPath(frame.path, root)) !== frame.identity) throw new Error("Directory changed while reading its next entry.");
      if (!entry) { session.stack.pop(); await frame.directory.close(); continue; }
      session.visited++; const path = resolve(frame.path, entry.name);
      if (!inside(root, path)) throw new Error("Directory entry escaped its checked root.");
      const info = await lstat(path, { bigint: true }), kind = info.isSymbolicLink() ? "link" : info.isDirectory() ? "directory" : info.isFile() ? "file" : "other";
      if (kind !== "link") await this.checkedPath(path, root);
      if (kind === "directory" && session.recursive) {
        await this.check(session.brainId, BLOCK * 8 + (session.stack.length + 1) * 1024, signal, context, "streamed recursive directory stack");
        session.stack.push(await this.directoryFrame(session.brainId, path, root, signal, context));
      }
      return { path, name: entry.name, kind, bytes: Number(info.size), modifiedAt: info.mtime.toISOString(), fileIdentity: identity(info) };
    }
    return undefined;
  }
  private async searchPage(session: SearchSession, signal: AbortSignal, context: FileToolContext): Promise<FileToolSearchPage> {
    const matches: FileToolSearchMatch[] = [], startScanned = session.scanned, startVisited = session.visited;
    let complete = false, considered = 0;
    while (matches.length < session.pageSize && (session.current?.pending || session.scanned - startScanned < session.scanBytes) && considered < session.pageSize * 8) {
      await this.guard(session.brainId, signal, context); session.controller.signal.throwIfAborted();
      if (!session.current) {
        if (session.initialReader) {
          const reader = session.initialReader; session.initialReader = undefined; session.visited++;
          session.current = { reader, metadata: { attachmentId: reader.source.attachmentId, name: reader.source.displayName,
            kind: "file", bytes: reader.source.bytes, fileIdentity: reader.source.identity }, offset: 0, carry: Buffer.alloc(0), nextMatch: 0 };
        } else {
          const metadata = await this.nextFile(session, signal, context); considered++;
          if (!metadata) { complete = true; break; }
          if (session.matching !== "utf8-literal-bytes") { if (!session.name || metadata.name.toLowerCase().includes(session.name)) matches.push(metadata); continue; }
          if (metadata.kind !== "file" || session.name && !metadata.name.toLowerCase().includes(session.name)) continue;
          session.current = { reader: await this.hostReader(session.brainId, metadata.path!, signal, context, session.root),
            metadata, offset: 0, carry: Buffer.alloc(0), nextMatch: 0 };
        }
      }
      const current = session.current!;
      if (session.matching !== "utf8-literal-bytes") {
        if (!session.name || current.metadata.name.toLowerCase().includes(session.name)) matches.push(current.metadata);
        await current.reader.close(); session.current = undefined; continue;
      }
      await this.check(session.brainId, MAX_PAGE + session.pageSize * 4096 + session.needle.length * 8, signal, context, "bounded file content-search window");
      await current.reader.assert(signal, context);
      if (!current.pending) {
        const buffer = await current.reader.read(current.offset, Math.min(BLOCK, session.scanBytes - (session.scanned - startScanned)), signal, context);
        session.scanned += buffer.length;
        const joined = Buffer.concat([current.carry, buffer]), base = current.offset - current.carry.length;
        current.offset += buffer.length;
        current.pending = { bytes: joined, base, position: Math.max(0, current.nextMatch - base) };
        current.carry = Buffer.from(joined.subarray(Math.max(0, joined.length - session.needle.length + 1)));
        if (!buffer.length && current.offset !== current.reader.source.bytes) throw new Error("Content search made no source progress.");
      }
      const pending = current.pending; let match: number;
      while ((match = pending.bytes.indexOf(session.needle, pending.position)) !== -1) {
        matches.push({ ...current.metadata, byteOffset: pending.base + match, matchedBytesSha256: digest(session.needle) });
        current.nextMatch = pending.base + match + 1; pending.position = match + 1;
        if (matches.length === session.pageSize) break;
      }
      await current.reader.assert(signal, context);
      if (match === -1) current.pending = undefined;
      if (!current.pending && current.offset === current.reader.source.bytes) { await current.reader.close(); session.current = undefined; }
    }
    session.generation++;
    if (Buffer.byteLength(canonical(matches), "utf8") > 8 * MAX_PAGE) throw new Error("File-search result viewport exceeds transport bytes; reduce pageSize. No tail was reported complete.");
    if (!session.current && !session.initialReader && !session.stack.length) complete = true;
    const result: FileToolSearchPage = { format: "omni-file-search-page", version: 1, matching: session.matching, matches,
      scannedBytes: session.scanned - startScanned, visitedEntries: session.visited - startVisited, hasMore: !complete, complete,
      consistency: "checked-open-files-not-an-atomic-tree-snapshot", ...(!complete ? { nextCursor: this.cursor(session) } : {}) };
    if (complete) await this.closeSession(session); else this.retain(session);
    return result;
  }
  private async handoff(brainId: string, args: Record<string, unknown>, requestId: string, signal: AbortSignal, context: FileToolContext) {
    if (args.path !== undefined || !context.chatTurnId) throw new Error("Task handoff accepts only an actual current-task attachmentId, never a supplied destination path.");
    const reader = await this.taskReader(brainId, string(args.attachmentId, "attachmentId", 128), signal, context);
    let file: FileHandle | undefined, temporary: string | undefined;
    let published: Record<string, unknown> | undefined;
    try {
      const root = resolve(await this.dependencies.root(brainId)), turnId = context.chatTurnId;
      const requestSha256 = digest(canonical([brainId, turnId, requestId, reader.source]));
      const requestKey = digest(canonical([brainId, turnId, requestId]));
      const relativePath = `attachments/handoffs/${requestSha256}.data`, receiptPath = join(root, "attachments", "handoffs", `${requestKey}.json`), path = join(root, ...relativePath.split("/"));
      await this.dependencies.reserveWrite(root, reader.source.bytes + BLOCK); await this.guard(brainId, signal, context);
      await mkdir(dirname(path), { recursive: true });
      if (!inside(root, await realpath(dirname(path)))) throw new Error("Task handoff destination escaped its owned workspace.");
      let receipt: TaskFileHandoff | undefined;
      const intentPath = receiptPath + ".intent", intent = { format: "omni-task-file-handoff-intent", version: 1, brainId, turnId, requestId, requestSha256, source: reader.source, relativePath };
      try {
        const info = await this.checkedPath(intentPath, root);
        if (!info.isFile() || info.size !== BigInt(Buffer.byteLength(canonical(intent)))) throw new Error("Task handoff intent has a different bounded shape.");
        await this.check(brainId, Number(info.size) * 8 + BLOCK, signal, context, "task handoff durable intent");
        const input = await open(intentPath, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
        try { const bytes = await input.readFile("utf8"); if (bytes !== canonical(intent) || identity(await input.stat({ bigint: true })) !== identity(info) || identity(await this.checkedPath(intentPath, root)) !== identity(info)) throw new Error("Task handoff requestId has a different or changed durable source binding."); }
        finally { await input.close(); }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        const staged = intentPath + "." + randomUUID() + ".next";
        const output = await open(staged, "wx", 0o600);
        try { await output.writeFile(canonical(intent), "utf8"); await output.sync(); await output.close(); await link(staged, intentPath); }
        finally { await output.close().catch(() => undefined); await unlink(staged).catch(() => undefined); }
      }
      const sourceIndexDirectory = join(root, "attachments", "handoffs", "by-data"), sourceIndex = join(sourceIndexDirectory, `${requestSha256}.json`);
      const sourceBinding = canonical({ requestKey, relativePath, intentSha256: digest(canonical(intent)), authority: "historical-not-live" });
      await mkdir(sourceIndexDirectory, { recursive: true });
      if (resolve(await realpath(sourceIndexDirectory)) !== resolve(sourceIndexDirectory)) throw new Error("Handoff source index has a symbolic ancestor.");
      if (!await lstat(sourceIndex).catch(error => { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; })) {
        const staged = sourceIndex + "." + randomUUID() + ".next", output = await open(staged, "wx", 0o600);
        try { await output.writeFile(sourceBinding, "utf8"); await output.sync(); await output.close(); await link(staged, sourceIndex); }
        finally { await output.close().catch(() => undefined); await unlink(staged).catch(() => undefined); }
      }
      try {
        const info = await this.checkedPath(receiptPath, root);
        if (!info.isFile() || info.size > 1024n * 1024n) throw new Error("Task handoff receipt is not bounded authentic metadata.");
        await this.check(brainId, Number(info.size) * 8 + BLOCK, signal, context, "task handoff receipt");
        const metadata = await open(receiptPath, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
        try { receipt = JSON.parse(await metadata.readFile("utf8")) as TaskFileHandoff;
          if (identity(await metadata.stat({ bigint: true })) !== identity(info) || identity(await this.checkedPath(receiptPath, root)) !== identity(info)) throw new Error("Task handoff receipt changed.");
        } finally { await metadata.close(); }
        if (receipt.format !== "omni-task-file-handoff" || receipt.version !== 1 || receipt.brainId !== brainId || receipt.turnId !== turnId || receipt.requestId !== requestId || receipt.requestSha256 !== requestSha256 || receipt.relativePath !== relativePath || receipt.sha256 !== reader.source.contentSha256 || receipt.bytes !== reader.source.bytes || canonical(receipt.source) !== canonical(reader.source)) throw new Error("Task handoff receipt belongs to another exact source/request.");
      } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
      const existing = await lstat(path).catch(error => { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; });
      if (!existing) {
        temporary = `${path}.${randomUUID()}.next`; file = await open(temporary, "wx", 0o600);
        const hash = createHash("sha256"); let offset = 0;
        while (offset < reader.source.bytes) {
          await this.dependencies.reserveWrite(root, BLOCK); await this.guard(brainId, signal, context);
          const bytes = await reader.read(offset, Math.min(BLOCK, reader.source.bytes - offset), signal, context); hash.update(bytes);
          let written = 0; while (written < bytes.length) { const result = await file.write(bytes, written, bytes.length - written); if (!result.bytesWritten) throw new Error("Task handoff write made no progress."); written += result.bytesWritten; }
          offset += bytes.length;
        }
        const sha = hash.digest("hex");
        if (sha !== reader.source.contentSha256) throw new Error("Task handoff full source SHA differs from the actual attachment.");
        await file.sync(); await file.close(); file = undefined; await reader.assert(signal, context); await this.guard(brainId, signal, context);
        // The deterministic destination is app-owned and is never overwritten.
        const raced = await lstat(path).catch(error => { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; });
        if (raced) throw new Error("Task handoff destination appeared during publication; inspect the existing copy.");
        await link(temporary, path); published = { copyPublished: true, executionStarted: false, brainId, turnId, requestId, requestSha256, source: reader.source, relativePath,
          sha256: sha, bytes: reader.source.bytes, authority: "historical-not-live" };
        await unlink(temporary); temporary = undefined;
      }
      const verified = await this.hashFile(brainId, path, signal, context);
      if (verified.sha256 !== reader.source.contentSha256 || verified.bytes !== reader.source.bytes) throw new Error("Existing task handoff changed; it was not replaced.");
      receipt ??= { format: "omni-task-file-handoff", version: 1, brainId, turnId, requestId, requestSha256, source: reader.source,
        relativePath, bytes: verified.bytes, sha256: verified.sha256, createdAt: new Date(this.now()).toISOString(), authority: "historical-not-live" };
      await this.guard(brainId, signal, context);
      if (!await lstat(receiptPath).catch(error => { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; })) {
        const staged = receiptPath + "." + randomUUID() + ".next", output = await open(staged, "wx", 0o600);
        try { await output.writeFile(canonical(receipt), "utf8"); await output.sync(); await output.close(); await link(staged, receiptPath); }
        finally { await output.close().catch(() => undefined); await unlink(staged).catch(() => undefined); }
      }
      await this.guard(brainId, signal, context);
      return { ...receipt, path, executionStarted: false, instruction: "Use the exact returned path in a separately permissioned shell/code action. This receipt grants no execution or copied authority." };
    } catch (error) { if (published) throw new FileToolEffectError(error instanceof Error ? error.message : String(error), published); throw error; }
    finally { await file?.close(); if (temporary) await unlink(temporary).catch(() => undefined); await reader.close(); }
  }
  async hashFile(brainId: string, path: string, signal: AbortSignal, context: FileToolContext): Promise<{ sha256: string; bytes: number }> {
    const reader = await this.hostReader(brainId, path, signal, context), hash = createHash("sha256"); let offset = 0;
    try { while (offset < reader.source.bytes) { const bytes = await reader.read(offset, BLOCK, signal, context); hash.update(bytes); offset += bytes.length; }
      await reader.assert(signal, context); return { sha256: hash.digest("hex"), bytes: offset };
    } finally { await reader.close(); }
  }
  async execute(brainId: string, action: string, args: Record<string, unknown>, requestId: string, signal: AbortSignal, context: FileToolContext): Promise<unknown> {
    await this.sweep(); await this.guard(brainId, signal, context);
    if (action === "materialize-task") {
      const key = canonical([brainId, context.chatTurnId, string(requestId, "requestId", 128)]), sha256 = digest(canonical(args)), existing = this.handoffs.get(key);
      if (existing) { if (existing.sha256 !== sha256) throw new Error("File handoff requestId has different operands."); const result = await existing.work; await this.guard(brainId, signal, context); return result; }
      const controller = new AbortController(), abort = () => controller.abort(signal.reason);
      signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort();
      const work = (async () => { const root = await this.dependencies.root(brainId); await this.guard(brainId, controller.signal, context);
        return withFileHandoffSnapshot(root, () => this.handoff(brainId, args, requestId, controller.signal, context), controller.signal); })();
      this.handoffs.set(key, { brainId, requestId, controller, sha256, work });
      try { return await work; } finally { signal.removeEventListener("abort", abort); if (this.handoffs.get(key)?.work === work) this.handoffs.delete(key); }
    }
    if (["read-next", "search-next", "list-next", "close"].includes(action)) {
      if (Object.keys(args).some(name => !["cursor", "path"].includes(name))) throw new Error("Continuation takes only its exact cursor/path, not a different filter.");
      const cursor = string(args.cursor, "file cursor", 4096), session = await this.session(brainId, cursor, signal, context);
      if (args.path !== undefined) {
        const actualPath = session.kind === "read" ? session.reader.source.path : session.root ?? session.current?.reader.source.path ?? session.initialReader?.source.path;
        if (absolute(args.path) !== actualPath) throw new Error("Continuation path differs from its original host-owned source.");
      }
      if (session.busy) throw new Error("This exact file cursor already has an in-flight page.");
      if (action === "close") { await this.closeSession(session); return { closed: true }; }
      const argumentsSha256 = digest(canonical([action, args]));
      if (session.previous?.cursor === cursor) {
        if (session.previous.requestId !== requestId || session.previous.argumentsSha256 !== argumentsSha256) throw new Error("Consumed file cursor belongs to a different request.");
        if (session.kind === "read") await session.reader.assert(signal, context); else await session.current?.reader.assert(signal, context);
        return structuredClone(session.previous.output);
      }
      if (cursor !== this.cursor(session) || (action === "read-next") !== (session.kind === "read") ||
          session.kind === "search" && (action === "list-next") !== session.list) throw new Error("File cursor/source/page changed; use its current exact continuation.");
      session.busy = true;
      try { const value = session.kind === "read" ? await this.readPage(session, signal, context) : await this.searchPage(session, signal, context);
        const output = session.kind === "search" && session.list ? { ...value, entries: (value as FileToolSearchPage).matches, returned: (value as FileToolSearchPage).matches.length,
          ...((value as FileToolSearchPage).complete ? { total: session.visited, totalKnown: true } : { totalKnown: false }) } : value;
        session.previous = { cursor, requestId, argumentsSha256, output: structuredClone(output) }; return output;
      } catch (error) { await this.closeSession(session); throw error; } finally { session.busy = false; }
    }
    if (["read", "read-segment", "read-all"].includes(action)) {
      const pageBytes = integer(args.maxBytes, BLOCK, MAX_PAGE); if (!pageBytes) throw new Error("File page must contain at least one admitted byte.");
      const reader = await this.reader(brainId, args, signal, context), offset = integer(args.offset, 0);
      if (offset > reader.source.bytes) { await reader.close(); throw new Error("File offset exceeds the checked source size."); }
      if (args.expectedFileIdentity !== undefined && args.expectedFileIdentity !== reader.source.identity) { await reader.close(); throw new Error("Selected file differs from the requested exact identity."); }
      const session: ReadSession = { ...this.base(brainId, context, requestId), kind: "read", reader, offset, pageBytes, hash: createHash("sha256"), whole: offset === 0 };
      if (action !== "read-segment") this.retain(session);
      try { const result = await this.readPage(session, signal, context);
        if (action === "read-segment") { await this.closeSession(session); delete result.nextCursor; } return result;
      } catch (error) { await this.closeSession(session); throw error; }
    }
    if (action === "search" || action === "list") {
      if (args.mode !== undefined && !["metadata", "name", "content"].includes(String(args.mode)) || args.recursive !== undefined && typeof args.recursive !== "boolean") throw new Error("Invalid file-search mode/recursion option.");
      const matching = action === "list" || args.mode === "metadata" ? "metadata" : args.mode === "content" ? "utf8-literal-bytes" : "name-substring-case-insensitive";
      const pattern = matching === "utf8-literal-bytes" ? string(args.query, "literal UTF-8 query", 4096) : "";
      const session: SearchSession = { ...this.base(brainId, context, requestId), kind: "search", stack: [], matching, needle: Buffer.from(pattern, "utf8"),
        name: matching === "name-substring-case-insensitive" ? string(args.query, "file name query", 4096).toLowerCase() : args.name === undefined || args.name === "" ? "" : string(args.name, "file-name filter", 4096).toLowerCase(),
        recursive: action !== "list" && args.recursive !== false, pageSize: integer(args.pageSize, 64, 256),
        scanBytes: integer(args.scanBytes, MAX_PAGE, 16 * MAX_PAGE), visited: 0, scanned: 0, list: action === "list" };
      if (!session.pageSize || !session.scanBytes) throw new Error("Search windows must be positive; no input is silently clamped.");
      try {
        if (args.attachmentId !== undefined) { if (args.path !== undefined) throw new Error("Choose exactly one search path or current attachment."); session.initialReader = await this.taskReader(brainId, string(args.attachmentId, "attachmentId", 128), signal, context); }
        else {
          const path = absolute(args.path), info = await this.checkedPath(path);
          if (info.isDirectory()) { session.root = path; session.rootIdentity = identity(info); session.stack.push(await this.directoryFrame(brainId, path, path, signal, context)); }
          else session.initialReader = await this.hostReader(brainId, path, signal, context);
        }
        this.retain(session);
        const result = await this.searchPage(session, signal, context);
        return action === "list" ? { ...result, entries: result.matches, returned: result.matches.length, ...(result.complete ? { total: session.visited, totalKnown: true } : { totalKnown: false }) } : result;
      } catch (error) { await this.closeSession(session); throw error; }
    }
    throw new Error("Unknown permissioned file action.");
  }
}
