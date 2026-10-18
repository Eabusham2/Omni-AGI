import { createHash, randomUUID } from "node:crypto";
import { constants, type BigIntStats } from "node:fs";
import { lstat, mkdir, open, opendir, realpath, rename, rm } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import type { PortableFile } from "../shared/portableWorkspace";
import type { DiskSpaceTelemetry } from "../shared/types";

export interface PortableIoHooks { signal?: AbortSignal; checkDisk(path: string, bytes: number): Promise<DiskSpaceTelemetry>; admit(bytes: number): void; checkpoint?(): Promise<void>; progress?(file: PortableFile): Promise<void> }
const PAGE = 64 * 1024;
export const portableSha256 = (value: string | Buffer): string => createHash("sha256").update(value).digest("hex");
export function portableRelative(value: string): string {
  if (!value || isAbsolute(value) || value.includes("\\") || value.includes("\0") || value.split("/").some(part => !part || part === "." || part === ".." || part.includes(":"))) throw new Error("Unsafe portable relative path."); return value;
}
export async function portableExists(path: string): Promise<boolean> { try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; } }
function signature(info: BigIntStats): string { return [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].join(":"); }
async function inside(root: string, path: string): Promise<void> {
  const rel = relative(await realpath(root), await realpath(path)); if (!rel || rel.startsWith("..") || isAbsolute(rel)) throw new Error("Portable file escaped its selected root.");
}
export async function* portableFiles(root: string, prefix = ""): AsyncGenerator<string> {
  const directory = join(root, ...prefix.split("/").filter(Boolean)), info = await lstat(directory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Portable source directory must not be a link.");
  for await (const item of await opendir(directory)) {
    const name = prefix ? `${prefix}/${item.name}` : item.name; portableRelative(name);
    if (item.isSymbolicLink() || !item.isFile() && !item.isDirectory()) throw new Error(`Unsupported portable source ${name}.`);
    if (item.isDirectory()) yield* portableFiles(root, name); else yield name;
  }
}
/** Exact-FD streaming custody; neither a symlink nor an after-await replacement is copied as trusted data. */
export async function portableFile(root: string, name: string, hooks: PortableIoHooks, destination?: string): Promise<PortableFile> {
  portableRelative(name); const path = join(root, ...name.split("/")); await inside(root, path);
  const before = await lstat(path, { bigint: true }); if (!before.isFile() || before.isSymbolicLink() || before.size < 0n || before.size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Portable payload is not a regular representable file.");
  const sourceBytes = Number(before.size);
  hooks.signal?.throwIfAborted(); hooks.admit(PAGE * 2); const source = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  let output: Awaited<ReturnType<typeof open>> | undefined;
  try {
    if (signature(await source.stat({ bigint: true })) !== signature(before)) throw new Error("Portable source changed before opening.");
    if (destination) { await mkdir(dirname(destination), { recursive: true, mode: 0o700 }); output = await open(destination, "wx", 0o600); }
    const digest = createHash("sha256"), buffer = Buffer.allocUnsafe(PAGE); let offset = 0;
    for (;;) {
      if (offset % (16 * PAGE) === 0) {
        await hooks.checkpoint?.();
        if (output && offset < sourceBytes) await hooks.checkDisk(destination!, Math.min(16 * PAGE, sourceBytes - offset));
      }
      hooks.signal?.throwIfAborted(); const next = await source.read(buffer, 0, PAGE, offset); if (!next.bytesRead) break;
      if (output) { hooks.signal?.throwIfAborted(); let written = 0;
        while (written < next.bytesRead) { const result = await output.write(buffer, written, next.bytesRead - written, offset + written);
          if (result.bytesWritten < 1) throw new Error("Portable output write made no progress."); written += result.bytesWritten; }
      }
      digest.update(buffer.subarray(0, next.bytesRead)); offset += next.bytesRead;
    }
    await inside(root, path);
    if (offset !== sourceBytes || signature(await source.stat({ bigint: true })) !== signature(before) || signature(await lstat(path, { bigint: true })) !== signature(before)) throw new Error("Portable source identity changed while transferring.");
    await output?.sync(); return { path: name, bytes: offset, sha256: digest.digest("hex") };
  } finally { await output?.close(); await source.close(); }
}
export async function portableReadJson<T>(root: string, name: string, hooks: PortableIoHooks, expected?: PortableFile): Promise<T> {
  const verified = await portableFile(root, name, hooks); if (expected && (verified.sha256 !== expected.sha256 || verified.bytes !== expected.bytes || verified.path !== expected.path)) throw new Error("Portable JSON checksum differs from its manifest.");
  hooks.admit(verified.bytes * 8 + PAGE); const path = join(root, ...portableRelative(name).split("/")), before = await lstat(path, { bigint: true }), handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try { const bytes = await handle.readFile(); await inside(root, path);
    if (signature(await handle.stat({ bigint: true })) !== signature(before) || signature(await lstat(path, { bigint: true })) !== signature(before) || portableSha256(bytes) !== verified.sha256) throw new Error("Portable JSON changed after admission.");
    hooks.signal?.throwIfAborted(); return JSON.parse(bytes.toString("utf8")) as T;
  } finally { await handle.close(); }
}
export async function portableWriteJson(path: string, value: unknown, hooks: PortableIoHooks, immutable = false): Promise<PortableFile> {
  const bytes = Buffer.from(JSON.stringify(value)); hooks.admit(bytes.length * 2 + PAGE); await hooks.checkDisk(dirname(path), bytes.length); hooks.signal?.throwIfAborted();
  await mkdir(dirname(path), { recursive: true, mode: 0o700 }); const temporary = immutable ? path : `${path}.${randomUUID()}.next`;
  try { const handle = await open(temporary, "wx", 0o600); try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
    if (!immutable) await rename(temporary, path);
  } finally { if (!immutable) await rm(temporary, { force: true }); }
  return { path: basename(path), bytes: bytes.length, sha256: portableSha256(bytes) };
}
export async function portableTreeSha256(root: string, hooks: PortableIoHooks): Promise<string> {
  const files: PortableFile[] = []; for await (const name of portableFiles(root)) { hooks.admit((files.length + 1) * 256); files.push(await portableFile(root, name, hooks)); }
  return portableSha256(JSON.stringify(files.sort((a, b) => a.path.localeCompare(b.path))));
}
export interface PortableDirectoryIdentity { device: string; inode: string }
export async function portableDirectoryIdentity(path: string): Promise<PortableDirectoryIdentity> {
  const info = await lstat(path, { bigint: true }); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Portable publication directory is unsafe.");
  return { device: String(info.dev), inode: String(info.ino) };
}
export async function portableOwnsDirectory(path: string, identity: PortableDirectoryIdentity): Promise<boolean> {
  try { const actual = await portableDirectoryIdentity(path); return actual.device === identity.device && actual.inode === identity.inode; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}
/** mkdir is the no-overwrite CAS. The runnable/visible commit file moves last. */
export async function publishPortableDirectory(stage: string, destination: string, commitFile: string, hooks: PortableIoHooks,
  persistOwner: (identity: PortableDirectoryIdentity) => Promise<void>, retained?: PortableDirectoryIdentity, expectedTreeSha256?: string): Promise<PortableDirectoryIdentity> {
  portableRelative(commitFile); if (commitFile.includes("/")) throw new Error("Portable commit must be a top-level file.");
  let owner = retained;
  if (!owner) { await mkdir(destination, { recursive: false, mode: 0o700 }); owner = await portableDirectoryIdentity(destination); await persistOwner(owner); }
  if (!await portableOwnsDirectory(destination, owner)) throw new Error("Portable publication no longer owns its reserved output directory.");
  if (retained) {
    if (!expectedTreeSha256) throw new Error("Interrupted portable publication has no prepared content hash.");
    const combined = new Map<string, PortableFile>();
    for (const root of [stage, destination]) if (await portableExists(root)) for await (const name of portableFiles(root)) {
      if (combined.has(name)) throw new Error("Interrupted portable publication has ambiguous duplicate files.");
      hooks.admit((combined.size + 1) * 256); combined.set(name, await portableFile(root, name, hooks));
    }
    if (portableSha256(JSON.stringify([...combined.values()].sort((a, b) => a.path.localeCompare(b.path)))) !== expectedTreeSha256) throw new Error("Interrupted portable publication differs from its exact prepared content.");
  }
  const names = await portableExists(stage) ? (await opendir(stage)) : undefined;
  const remaining: string[] = []; if (names) for await (const entry of names) remaining.push(entry.name);
  remaining.sort((a, b) => a === commitFile ? 1 : b === commitFile ? -1 : a.localeCompare(b));
  for (const name of remaining) {
    hooks.signal?.throwIfAborted(); await hooks.checkpoint?.();
    if (!await portableOwnsDirectory(destination, owner)) throw new Error("Portable publication directory changed during transfer.");
    if (await portableExists(join(destination, name))) throw new Error("Portable publication refuses to replace an existing file.");
    await rename(join(stage, name), join(destination, name));
  }
  if (!await portableExists(join(destination, commitFile))) throw new Error("Portable publication has no final commit file.");
  return owner;
}
export async function windowsPortableArchitecture(root: string, name: string): Promise<"x64" | "arm64"> {
  portableRelative(name); const path = resolve(root, ...name.split("/")); await inside(root, path); const before = await lstat(path, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink()) throw new Error("Selected Windows runtime executable is unsafe.");
  const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try { const dos = Buffer.alloc(64); if ((await handle.read(dos, 0, 64, 0)).bytesRead !== 64 || dos.readUInt16LE(0) !== 0x5a4d) throw new Error("Selected runtime is not a Windows PE executable.");
    const offset = dos.readUInt32LE(0x3c); if (BigInt(offset) + 6n > before.size) throw new Error("Invalid Windows PE header offset."); const pe = Buffer.alloc(6);
    if ((await handle.read(pe, 0, 6, offset)).bytesRead !== 6 || pe.readUInt32LE(0) !== 0x4550) throw new Error("Invalid Windows PE signature.");
    if (signature(await handle.stat({ bigint: true })) !== signature(before) || signature(await lstat(path, { bigint: true })) !== signature(before)) throw new Error("Windows executable changed during inspection.");
    const machine = pe.readUInt16LE(4); if (machine === 0x8664) return "x64"; if (machine === 0xaa64) return "arm64"; throw new Error("Windows runtime has an unsupported architecture.");
  } finally { await handle.close(); }
}
