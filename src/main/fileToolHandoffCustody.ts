/** Exact handoff bytes/history, never portable file/code permissions. */
import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { link, lstat, mkdir, open, opendir, realpath, rename, unlink } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { freemem } from "node:os";
import { requireDiskWrite } from "./diskSpace";
import type { TaskFileHandoff } from "../shared/fileTools";

const SHA = /^[a-f0-9]{64}$/;
const STAGING = /^(?:(?:custody|by-data|interrupted|history)\/)?[a-f0-9]{64}\.(?:data|json(?:\.intent)?)\.[a-f0-9-]{36}\.next$/;
const tails = new Map<string, Promise<void>>();
const canonical = (value: unknown) => JSON.stringify(value);
const sha = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
export interface FileHandoffCustodyOptions { signal?: AbortSignal; admit?(bytes: number, purpose: string): Promise<void> | void; reserve?(root: string, bytes: number): Promise<void> | void; readOnly?: boolean; }
const check = (options: FileHandoffCustodyOptions) => options.signal?.throwIfAborted();
async function admit(options: FileHandoffCustodyOptions, bytes: number) {
  check(options); if (options.admit) await options.admit(bytes, "file-handoff closure page"); else if (bytes > freemem()) throw new Error("File-handoff closure waits for physical memory."); check(options);
}
async function reserve(options: FileHandoffCustodyOptions, root: string, bytes: number) {
  check(options); if (options.reserve) await options.reserve(root, bytes); else await requireDiskWrite(root, { operationWriteBytes: bytes }); check(options);
}
export async function withFileHandoffSnapshot<T>(brainRoot: string, operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  const root = resolve(brainRoot), previous = tails.get(root) ?? Promise.resolve();
  let release!: () => void;
  const done = new Promise<void>(complete => { release = complete; }), next = previous.then(() => done); tails.set(root, next);
  let abort!: () => void;
  const cancelled = new Promise<never>((_, reject) => { abort = () => reject(signal?.reason ?? new Error("File workspace wait cancelled.")); signal?.addEventListener("abort", abort, { once: true }); if (signal?.aborted) abort(); });
  try { await (signal ? Promise.race([previous, cancelled]) : previous); signal?.throwIfAborted(); return await operation(); }
  finally { signal?.removeEventListener("abort", abort); release(); void next.then(() => { if (tails.get(root) === next) tails.delete(root); }); }
}
const fileIdentity = (info: { dev: bigint; ino: bigint; size: bigint; mtimeNs: bigint; ctimeNs: bigint }) => [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].join(":");
async function owned(root: string, relativePath: string) {
  if (!relativePath || relativePath.includes("\\") || relativePath.split("/").some(part => !part || part === "." || part === "..")) throw new Error("Unsafe handoff closure path.");
  const path = resolve(root, ...relativePath.split("/")), local = relative(root, path);
  if (isAbsolute(local) || local === ".." || local.startsWith(`..${sep}`)) throw new Error("Handoff closure escaped its root.");
  const info = await lstat(path, { bigint: true });
  if (!info.isFile() || info.isSymbolicLink() || resolve(await realpath(path)) !== path) throw new Error("Handoff closure contains a symbolic/non-regular source.");
  return { path, info, identity: fileIdentity(info) };
}
async function hashFile(root: string, path: string, options: FileHandoffCustodyOptions) {
  const source = await owned(root, path); await admit(options, 1024 * 1024);
  const input = await open(source.path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  try {
    const hash = createHash("sha256"), buffer = Buffer.alloc(65536); let bytes = 0;
    if (fileIdentity(await input.stat({ bigint: true })) !== source.identity) throw new Error("Handoff source changed before open.");
    while (true) { check(options); const read = await input.read(buffer, 0, buffer.length, bytes); if (!read.bytesRead) break; hash.update(buffer.subarray(0, read.bytesRead)); bytes += read.bytesRead; }
    if (BigInt(bytes) !== source.info.size || fileIdentity(await input.stat({ bigint: true })) !== source.identity || (await owned(root, path)).identity !== source.identity) throw new Error("Handoff source changed during closure hashing.");
    return { sha256: hash.digest("hex"), bytes };
  } finally { await input.close(); }
}
async function json(root: string, path: string, options: FileHandoffCustodyOptions): Promise<{ value: Record<string, any>; raw: Buffer; sha256: string }> {
  const source = await owned(root, path);
  if (source.info.size > 1024n * 1024n) throw new Error("Handoff metadata exceeds its typed transport envelope.");
  await admit(options, Number(source.info.size) * 8 + 65536);
  const input = await open(source.path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  try {
    const raw = await input.readFile();
    if (BigInt(raw.length) !== source.info.size || fileIdentity(await input.stat({ bigint: true })) !== source.identity || (await owned(root, path)).identity !== source.identity) throw new Error("Handoff metadata changed while reading.");
    const value: unknown = JSON.parse(raw.toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Malformed handoff metadata.");
    return { value: value as Record<string, any>, raw, sha256: sha(raw) };
  } finally { await input.close(); }
}
async function exists(path: string): Promise<boolean> { try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; } }
async function write(root: string, path: string, value: unknown, options: FileHandoffCustodyOptions, replace = false) {
  const output = join(root, ...path.split("/")), bytes = Buffer.isBuffer(value) ? value : Buffer.from(canonical(value));
  await reserve(options, root, bytes.length * 2 + 65536); await mkdir(dirname(output), { recursive: true });
  if (resolve(await realpath(dirname(output))) !== resolve(dirname(output))) throw new Error("Handoff custody destination has symbolic ancestors.");
  const temporary = output + "." + randomUUID() + ".next", file = await open(temporary, "wx", 0o600);
  try { await file.writeFile(bytes); await file.sync(); await file.close(); check(options);
    if (replace) await rename(temporary, output); else await link(temporary, output);
  } finally { await file.close().catch(() => undefined); await unlink(temporary).catch(() => undefined); }
}
function intent(value: Record<string, any>, name: string) {
  const source = value.source;
  if (value.format !== "omni-task-file-handoff-intent" || value.version !== 1 || typeof value.brainId !== "string" || !value.brainId ||
      typeof value.turnId !== "string" || !value.turnId || typeof value.requestId !== "string" || !value.requestId || !SHA.test(value.requestSha256) ||
      !source || source.kind !== "task-attachment" || typeof source.attachmentId !== "string" || !SHA.test(source.manifestSha256) || !SHA.test(source.contentSha256) ||
      !Number.isSafeInteger(source.bytes) || source.bytes < 0 || typeof source.identity !== "string" || !source.identity || source.path !== undefined ||
      value.relativePath !== `attachments/handoffs/${value.requestSha256}.data` ||
      name !== sha(canonical([value.brainId, value.turnId, value.requestId])) + ".json.intent" ||
      value.requestSha256 !== sha(canonical([value.brainId, value.turnId, value.requestId, source]))) throw new Error("Handoff intent/source binding failed.");
}
async function* files(root: string, prefix = ""): AsyncGenerator<string> {
  if (!await exists(root)) return;
  const info = await lstat(root); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Unsafe handoff workspace directory.");
  const directory = await opendir(root);
  try { for await (const entry of directory) {
    const name = prefix + entry.name;
    if (entry.isSymbolicLink()) throw new Error("Handoff workspace contains a link.");
    if (entry.isDirectory()) {
      if (!/^(?:custody|by-data|interrupted|history)$/.test(name.split("/")[0]!)) throw new Error("Unknown handoff namespace.");
      yield* files(join(root, entry.name), name + "/");
    } else if (entry.isFile()) yield name; else throw new Error("Handoff workspace contains a non-regular file.");
  } } finally { await directory.close().catch(() => undefined); }
}

/** Call under withFileHandoffSnapshot. Writes only explicit dormant aliases;
 * immutable original receipts/intents/data are never rewritten or regranted. */
export async function prepareFileHandoffCustody(brainRoot: string, ownerBrainId: string, options: FileHandoffCustodyOptions = {}): Promise<boolean> {
  const root = resolve(brainRoot), base = join(root, "attachments", "handoffs");
  if (!await exists(base)) return false;
  for await (const name of files(base)) {
    check(options);
    if (!name.includes("/") && /^[a-f0-9]{64}\.json\.intent$/.test(name)) {
      const saved = await json(root, "attachments/handoffs/" + name, options); intent(saved.value, name);
      const value = saved.value, requestKey = name.slice(0, 64), receiptRelative = `attachments/handoffs/${requestKey}.json`;
      let receiptSha256: string | null = null;
      if (await exists(join(root, receiptRelative))) {
        const receipt = await json(root, receiptRelative, options), body = receipt.value as TaskFileHandoff;
        if (body.format !== "omni-task-file-handoff" || body.version !== 1 || body.brainId !== value.brainId || body.turnId !== value.turnId || body.requestId !== value.requestId ||
            body.requestSha256 !== value.requestSha256 || body.relativePath !== value.relativePath || body.bytes !== value.source.bytes || body.sha256 !== value.source.contentSha256 ||
            canonical(body.source) !== canonical(value.source) || body.authority !== "historical-not-live" || !Number.isFinite(Date.parse(body.createdAt))) throw new Error("Handoff receipt differs from its original source intent.");
        receiptSha256 = receipt.sha256;
      }
      let dataSha256: string | null = null;
      if (await exists(join(root, value.relativePath))) {
        const data = await hashFile(root, value.relativePath, options);
        if (data.sha256 !== value.source.contentSha256 || data.bytes !== value.source.bytes) throw new Error("Handoff copy does not preserve its exact original source bytes.");
        dataSha256 = data.sha256;
      } else if (receiptSha256) throw new Error("Completed handoff is missing its copied bytes.");
      const alias = { format: "omni-task-file-handoff-custody", version: 1, ownerBrainId, sourceBrainId: value.brainId,
        requestKey, intentSha256: saved.sha256, receiptSha256, dataSha256, relativePath: value.relativePath,
        state: receiptSha256 ? "completed-historical" : dataSha256 ? "published-unacknowledged" : "interrupted-unpublished", authority: "historical-not-live" };
      const aliasPath = `attachments/handoffs/custody/${requestKey}.json`;
      if (await exists(join(root, aliasPath))) {
        const previous = await json(root, aliasPath, options);
        if (previous.value.format !== "omni-task-file-handoff-custody" || previous.value.version !== 1 || previous.value.authority !== "historical-not-live" ||
            previous.value.requestKey !== requestKey || previous.value.intentSha256 !== saved.sha256 || previous.value.sourceBrainId !== value.brainId) throw new Error("Existing handoff custody alias lost its exact original intent.");
        if (canonical(previous.value) !== canonical(alias) && !options.readOnly) {
          const archive = `attachments/handoffs/history/${previous.sha256}.json`;
          if (!await exists(join(root, archive))) await write(root, archive, previous.raw, options);
          await write(root, aliasPath, alias, options, true);
        }
      } else if (!options.readOnly) await write(root, aliasPath, alias, options);
      const index = `attachments/handoffs/by-data/${value.requestSha256}.json`, binding = { requestKey, relativePath: value.relativePath, intentSha256: saved.sha256, authority: "historical-not-live" };
      if (!await exists(join(root, index))) { if (!options.readOnly) await write(root, index, binding, options); }
      else if (canonical((await json(root, index, options)).value) !== canonical(binding)) throw new Error("Handoff source index conflict.");
    } else if (STAGING.test(name)) {
      const source = await hashFile(root, `attachments/handoffs/${name}`, options), path = `attachments/handoffs/interrupted/${sha(name)}.json`;
      const note = { format: "omni-interrupted-file-handoff", version: 1, ownerBrainId,
        originalRelativePath: `attachments/handoffs/${name}`, bytes: source.bytes, sha256: source.sha256,
        state: "interrupted-unattributed", usableAsCompletedHandoff: false, authority: "historical-not-live" };
      if (!await exists(join(root, path))) { if (!options.readOnly) await write(root, path, note, options); }
      else { const previous = await json(root, path, options); if (previous.value.format !== "omni-interrupted-file-handoff" || previous.value.sha256 !== source.sha256 || previous.value.bytes !== source.bytes ||
          previous.value.originalRelativePath !== `attachments/handoffs/${name}` || previous.value.authority !== "historical-not-live") throw new Error("Interrupted handoff bytes changed.");
        if (previous.value.ownerBrainId !== ownerBrainId && !options.readOnly) { const archive = `attachments/handoffs/history/${previous.sha256}.json`;
          if (!await exists(join(root, archive))) await write(root, archive, previous.raw, options); await write(root, path, note, options, true); } }
    }
  }
  // Validate every raw file, including orphan/corrupt publication and staging.
  for await (const name of files(base)) {
    check(options);
    if (STAGING.test(name)) {
      if (!options.readOnly && !await exists(join(base, "interrupted", sha(name) + ".json"))) throw new Error("Unclassified interrupted handoff staging.");
      continue;
    }
    if (name.includes("/")) {
      if (!/^(?:custody|by-data|interrupted|history)\/[a-f0-9]{64}\.json$/.test(name)) throw new Error("Unknown nested handoff custody file.");
      const saved = await json(root, "attachments/handoffs/" + name, options);
      if (name.startsWith("history/") && name.slice(8, 72) !== saved.sha256) throw new Error("Historical handoff bytes no longer match their immutable archive hash.");
      if (name.startsWith("custody/")) {
        const key = name.slice(8, 72), source = await json(root, `attachments/handoffs/${key}.json.intent`, options); intent(source.value, `${key}.json.intent`);
        const receiptPath = `attachments/handoffs/${key}.json`, receiptSha = await exists(join(root, receiptPath)) ? (await json(root, receiptPath, options)).sha256 : null;
        const data = await exists(join(root, source.value.relativePath)) ? await hashFile(root, source.value.relativePath, options) : null;
        const expected = { format: "omni-task-file-handoff-custody", version: 1, ownerBrainId, sourceBrainId: source.value.brainId,
          requestKey: key, intentSha256: source.sha256, receiptSha256: receiptSha, dataSha256: data?.sha256 ?? null, relativePath: source.value.relativePath,
          state: receiptSha ? "completed-historical" : data ? "published-unacknowledged" : "interrupted-unpublished", authority: "historical-not-live" };
        if (options.readOnly) {
          const prior = saved.value;
          if (prior.format !== expected.format || prior.version !== 1 || prior.ownerBrainId !== ownerBrainId || prior.sourceBrainId !== expected.sourceBrainId ||
              prior.requestKey !== key || prior.intentSha256 !== expected.intentSha256 || prior.relativePath !== expected.relativePath || prior.authority !== "historical-not-live" ||
              prior.receiptSha256 !== null && prior.receiptSha256 !== expected.receiptSha256 || prior.dataSha256 !== null && prior.dataSha256 !== expected.dataSha256 ||
              !["completed-historical", "published-unacknowledged", "interrupted-unpublished"].includes(prior.state)) throw new Error("Orphan or wrong-owner historical handoff custody alias.");
        } else if (canonical(saved.value) !== canonical(expected)) throw new Error("Orphan, stale or wrong-owner handoff custody alias.");
      }
      if (name.startsWith("interrupted/")) {
        const original = saved.value.originalRelativePath;
        if (typeof original !== "string" || !original.startsWith("attachments/handoffs/") || !STAGING.test(original.slice("attachments/handoffs/".length)) ||
            name.slice(12, 76) !== sha(original.slice("attachments/handoffs/".length))) throw new Error("Orphan interrupted handoff alias.");
        const source = await hashFile(root, original, options);
        if (saved.value.ownerBrainId !== ownerBrainId || saved.value.format !== "omni-interrupted-file-handoff" || saved.value.version !== 1 ||
            saved.value.bytes !== source.bytes || saved.value.sha256 !== source.sha256 || saved.value.state !== "interrupted-unattributed" ||
            saved.value.usableAsCompletedHandoff !== false || saved.value.authority !== "historical-not-live") throw new Error("Interrupted handoff alias lost its exact staging bytes.");
      }
      if (name.startsWith("by-data/")) {
        if (!SHA.test(saved.value.requestKey) || saved.value.authority !== "historical-not-live") throw new Error("Malformed handoff source index.");
        const source = await json(root, `attachments/handoffs/${saved.value.requestKey}.json.intent`, options); intent(source.value, `${saved.value.requestKey}.json.intent`);
        if (source.sha256 !== saved.value.intentSha256 || source.value.requestSha256 !== name.slice(8, 72) || source.value.relativePath !== saved.value.relativePath) throw new Error("Handoff data index has no exact original intent.");
      }
      continue;
    }
    if (/^[a-f0-9]{64}\.data$/.test(name)) {
      const index = `attachments/handoffs/by-data/${name.slice(0, 64)}.json`;
      if (!await exists(join(root, index))) throw new Error("Handoff data has no exact source intent.");
    } else if (/^[a-f0-9]{64}\.json$/.test(name)) {
      if (!await exists(join(base, name + ".intent"))) throw new Error("Handoff receipt has no exact source intent.");
    } else if (/^[a-f0-9]{64}\.json\.intent$/.test(name)) { /* checked above */ }
    else if (/^[a-f0-9]{64}\.(?:data|json(?:\.intent)?)\.[a-f0-9-]{36}\.next$/.test(name)) {
      if (!options.readOnly && !await exists(join(base, "interrupted", sha(name) + ".json"))) throw new Error("Unclassified interrupted handoff staging.");
    } else throw new Error("Unknown handoff file cannot be silently included as completed state.");
  }
  return true;
}
/** Read-only parent closure validation; never annotates/rekeys a source. */
export async function validateFileHandoffCustody(root: string, ownerBrainId: string, options: FileHandoffCustodyOptions = {}): Promise<boolean> {
  return prepareFileHandoffCustody(root, ownerBrainId, { ...options, readOnly: true });
}
export async function rekeyFileHandoffCustody(root: string, targetBrainId: string, options: FileHandoffCustodyOptions = {}): Promise<void> {
  await withFileHandoffSnapshot(root, () => prepareFileHandoffCustody(root, targetBrainId, options).then(() => undefined), options.signal);
}
/** Full merge preserves exact originals and explicitly dormant aliases; no
 * old absolute path, live cursor, credentials or execution grant is adopted. */
export async function mergeFileHandoffCustody(targetRoot: string, targetBrainId: string, sourceRoot: string, sourceBrainId: string, options: FileHandoffCustodyOptions = {}): Promise<void> {
  if (resolve(targetRoot) === resolve(sourceRoot)) throw new Error("Handoff merge source/destination must be distinct owned workspaces.");
  await withFileHandoffSnapshot(sourceRoot, async () => {
    if (!await validateFileHandoffCustody(sourceRoot, sourceBrainId, options)) return;
    await withFileHandoffSnapshot(targetRoot, async () => {
      for await (const name of files(join(sourceRoot, "attachments", "handoffs"))) {
        const sourceRelative = "attachments/handoffs/" + name, binding = await hashFile(sourceRoot, sourceRelative, options);
        const alias = /^(?:custody|interrupted)\/[a-f0-9]{64}\.json$/.test(name);
        const targetRelative = alias ? `attachments/handoffs/history/${binding.sha256}.json` : sourceRelative;
        const destination = join(targetRoot, ...targetRelative.split("/"));
        if (await exists(destination)) {
          const current = await hashFile(targetRoot, targetRelative, options);
          if (current.sha256 !== binding.sha256 || current.bytes !== binding.bytes) throw new Error("Full-merge handoff source conflict; no default winner.");
          continue;
        }
        await reserve(options, targetRoot, binding.bytes + 65536); await mkdir(dirname(destination), { recursive: true });
        if (resolve(await realpath(dirname(destination))) !== resolve(dirname(destination))) throw new Error("Merged handoff destination escaped through a symbolic ancestor.");
        const source = await owned(sourceRoot, sourceRelative), temporary = destination + "." + randomUUID() + ".next";
        const input = await open(source.path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW)); let output;
        try { output = await open(temporary, "wx", 0o600); await admit(options, 1024 * 1024);
          const buffer = Buffer.alloc(65536), hash = createHash("sha256"); let offset = 0;
          while (true) { check(options); const read = await input.read(buffer, 0, buffer.length, offset); if (!read.bytesRead) break;
            await reserve(options, targetRoot, read.bytesRead + 65536); hash.update(buffer.subarray(0, read.bytesRead)); let written = 0;
            while (written < read.bytesRead) { const saved = await output.write(buffer, written, read.bytesRead - written); if (!saved.bytesWritten) throw new Error("Merged handoff copy made no progress."); written += saved.bytesWritten; } offset += read.bytesRead; }
          if (offset !== binding.bytes || hash.digest("hex") !== binding.sha256 || fileIdentity(await input.stat({ bigint: true })) !== source.identity || (await owned(sourceRoot, sourceRelative)).identity !== source.identity) throw new Error("Handoff source changed during merge.");
          await output.sync(); await output.close(); output = undefined; check(options); await link(temporary, destination);
        } finally { await input.close(); await output?.close(); await unlink(temporary).catch(() => undefined); }
      }
      await prepareFileHandoffCustody(targetRoot, targetBrainId, options);
    }, options.signal);
  }, options.signal);
}
