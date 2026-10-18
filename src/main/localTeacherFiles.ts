import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, type FileHandle } from "node:fs/promises";
import { basename, isAbsolute } from "node:path";
import type { LocalTeacherFileIdentity, LocalTeacherGgufHeader } from "../shared/localTeacher";
import { taskAttachmentFileIdentity, TaskAttachmentReadVerifier } from "./taskAttachmentReadVerifier";
import { localTeacherHash } from "./localTeacherFormat";

export interface LocalTeacherSelectedFile extends LocalTeacherFileIdentity { path: string; selectedFdIdentity?: string }
/** Exact full-source hash in fixed 64KiB scratch; never loads tensors. */
export async function hashLocalTeacherFile(path: string, signal: AbortSignal,
  admit: (bytes: number) => Promise<void>, progress?: (bytes: number, total: number) => void): Promise<LocalTeacherSelectedFile> {
  if (!isAbsolute(path) || path.includes("\0")) throw new Error("The teacher dependency must come from an explicit absolute host selection.");
  signal.throwIfAborted(); await admit(65536);
  const before = await lstat(path, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink() || before.size < 1n || before.size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Selected teacher dependency is not a supported exact regular file.");
  const file = await open(path, constants.O_RDONLY | (process.platform !== "win32" ? constants.O_NOFOLLOW ?? 0 : 0));
  try {
    const identity = taskAttachmentFileIdentity(before), digest = createHash("sha256"), scratch = Buffer.alloc(65536);
    if (taskAttachmentFileIdentity(await file.stat({ bigint: true })) !== identity) throw new Error("Teacher dependency changed before its selected FD opened.");
    let offset = 0, lastProgress = 0; progress?.(0, Number(before.size));
    while (offset < Number(before.size)) {
      signal.throwIfAborted(); const { bytesRead } = await file.read(scratch, 0, Math.min(scratch.length, Number(before.size) - offset), offset);
      if (!bytesRead) throw new Error("Teacher dependency ended before full SHA coverage.");
      digest.update(scratch.subarray(0, bytesRead)); offset += bytesRead;
      if (Date.now() - lastProgress > 100 || offset === Number(before.size)) { progress?.(offset, Number(before.size)); lastProgress = Date.now(); }
    }
    const after = await lstat(path, { bigint: true });
    if (after.isSymbolicLink() || taskAttachmentFileIdentity(after) !== identity || taskAttachmentFileIdentity(await file.stat({ bigint: true })) !== identity) throw new Error("Selected teacher dependency changed during its full hash.");
    return { path, displayName: basename(path), bytes: Number(before.size), sha256: digest.digest("hex"), selectedFdIdentity: identity };
  } finally { await file.close(); }
}

class GgufReader {
  offset = 0;
  constructor(readonly file: FileHandle, readonly bytes: number, readonly signal: AbortSignal) {}
  async read(length: number): Promise<Buffer> {
    this.signal.throwIfAborted();
    if (!Number.isSafeInteger(length) || length < 0 || this.offset + length > this.bytes) throw new Error("GGUF metadata extends past the exact selected file.");
    const data = Buffer.alloc(length); let done = 0;
    while (done < length) { this.signal.throwIfAborted(); const row = await this.file.read(data, done, length - done, this.offset + done); if (!row.bytesRead) throw new Error("GGUF metadata was truncated."); done += row.bytesRead; }
    this.offset += length; return data;
  }
  async skip(length: bigint): Promise<void> {
    this.signal.throwIfAborted();
    if (length < 0n || BigInt(this.offset) + length > BigInt(this.bytes)) throw new Error("GGUF metadata length exceeds the selected source.");
    this.offset += Number(length);
  }
  async u32(): Promise<number> { return (await this.read(4)).readUInt32LE(); }
  async u64(): Promise<bigint> { return (await this.read(8)).readBigUInt64LE(); }
  async string(retain: boolean, key = false): Promise<string | undefined> {
    const length = await this.u64();
    if (key && length > 65535n) throw new Error("GGUF metadata key exceeds the format's supported key envelope.");
    if (!retain || length > 65535n) { await this.skip(length); return undefined; }
    return new TextDecoder("utf-8", { fatal: true }).decode(await this.read(Number(length)));
  }
  async value(type: number, retain: boolean): Promise<number | string | undefined> {
    const widths: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
    if (type === 8) return this.string(retain);
    if (type === 9) {
      const element = await this.u32(), count = await this.u64();
      if (element === 9 || element > 12) throw new Error("Unsupported nested/unknown GGUF metadata array.");
      if (element === 8) { if (count > BigInt(Math.floor((this.bytes - this.offset) / 8))) throw new Error("Invalid GGUF string-array count.");
        for (let index = 0n; index < count; index++) await this.string(false);
      } else { const width = widths[element]; if (!width) throw new Error("Unsupported GGUF scalar type."); await this.skip(count * BigInt(width)); }
      return undefined;
    }
    const width = widths[type]; if (!width) throw new Error("Unsupported GGUF metadata type.");
    if (!retain) { await this.skip(BigInt(width)); return undefined; }
    const raw = await this.read(width);
    const value = type === 0 || type === 7 ? raw.readUInt8() : type === 1 ? raw.readInt8() : type === 2 ? raw.readUInt16LE()
      : type === 3 ? raw.readInt16LE() : type === 4 ? raw.readUInt32LE() : type === 5 ? raw.readInt32LE() : type === 6 ? raw.readFloatLE()
        : type === 10 ? Number(raw.readBigUInt64LE()) : type === 11 ? Number(raw.readBigInt64LE()) : raw.readDoubleLE();
    if (!Number.isFinite(value) || ([10, 11].includes(type) && !Number.isSafeInteger(value))) return undefined;
    return value;
  }
}

/** Bounded metadata scan only. No assertion that arbitrary tensor payloads are valid. */
export async function readLocalTeacherGguf(file: FileHandle, bytes: number, signal: AbortSignal): Promise<LocalTeacherGgufHeader> {
  const reader = new GgufReader(file, bytes, signal), magic = await reader.read(4), version = await reader.u32();
  if (magic.toString("ascii") !== "GGUF" || (version !== 2 && version !== 3)) throw new Error("Select a supported little-endian GGUF v2/v3 file.");
  const tensors = await reader.u64(), count = await reader.u64();
  if (count > BigInt(Math.floor((bytes - reader.offset) / 12))) throw new Error("GGUF metadata count cannot fit its source.");
  const strings: Record<string, string> = {}, geometry: Record<string, number> = {};
  const wanted = new Set(["general.architecture", "general.name", "general.license", "general.license.link"]);
  const numeric = /\.(?:context_length|embedding_length|block_count|attention\.(?:head_count(?:_kv)?|key_length|value_length))$/;
  for (let index = 0n; index < count; index++) {
    signal.throwIfAborted(); const key = await reader.string(true, true); if (key === undefined) throw new Error("Invalid GGUF metadata key.");
    const type = await reader.u32(), value = await reader.value(type, wanted.has(key) || numeric.test(key));
    if (wanted.has(key) && typeof value === "string") { if (key in strings) throw new Error("Duplicate retained GGUF metadata key."); strings[key] = value; }
    if (numeric.test(key) && typeof value === "number") { if (key in geometry) throw new Error("Duplicate retained GGUF geometry key."); geometry[key] = value; }
    if (Buffer.byteLength(JSON.stringify(geometry)) > 65536) throw new Error("GGUF retained geometry exceeds the admitted header-summary memory envelope; no keys were silently dropped.");
  }
  return { version, tensorCount: tensors.toString(), metadataCount: count.toString(), metadataEnd: reader.offset,
    ...(strings["general.architecture"] ? { architecture: strings["general.architecture"] } : {}),
    ...(strings["general.name"] ? { name: strings["general.name"] } : {}),
    ...(strings["general.license"] ? { declaredLicense: strings["general.license"] } : {}),
    ...(strings["general.license.link"] ? { declaredLicenseLink: strings["general.license.link"] } : {}),
    geometry, scope: "header-metadata-only", tensorPayloadValidated: false };
}
/** Immediate header read after the full selection SHA: same exact inode/change
 * identity, so no second whole-GGUF hash merely to inspect declared metadata. */
export async function inspectSelectedLocalTeacherHeader(selected: LocalTeacherSelectedFile, signal: AbortSignal): Promise<LocalTeacherGgufHeader> {
  if (!selected.selectedFdIdentity) throw new Error("The full selection hash has no current exact-FD identity.");
  signal.throwIfAborted(); const file = await open(selected.path, constants.O_RDONLY | (process.platform !== "win32" ? constants.O_NOFOLLOW ?? 0 : 0));
  const current = async () => { signal.throwIfAborted(); const atPath = await lstat(selected.path, { bigint: true });
    if (atPath.isSymbolicLink() || taskAttachmentFileIdentity(atPath) !== selected.selectedFdIdentity ||
        taskAttachmentFileIdentity(await file.stat({ bigint: true })) !== selected.selectedFdIdentity) throw new Error("GGUF changed between its selection hash and bounded header inspection."); };
  try { await current(); const header = await readLocalTeacherGguf(file, selected.bytes, signal); await current(); return header; }
  finally { await file.close(); }
}

export const localTeacherFileBinding = (owner: string, file: LocalTeacherSelectedFile) => ({
  authorityKey: localTeacherHash(`local-teacher\0${owner}\0${file.sha256}`), path: file.path, contentSha256: file.sha256, bytes: file.bytes
});
export { TaskAttachmentReadVerifier as LocalTeacherFileVerifier };
