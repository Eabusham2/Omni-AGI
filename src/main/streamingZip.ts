import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import {
  mkdir,
  mkdtemp,
  lstat,
  open,
  rename,
  rm,
  stat,
  statfs
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { pipeline } from "node:stream/promises";
import { Transform } from "node:stream";
import { createInflateRaw } from "node:zlib";

const UINT16_MAX = 0xffff;
const UINT32_MAX = 0xffff_ffff;
const ZIP_LOCAL_HEADER = 0x04034b50;
const ZIP_DATA_DESCRIPTOR = 0x08074b50;
const ZIP_CENTRAL_HEADER = 0x02014b50;
const ZIP64_END = 0x06064b50;
const ZIP64_LOCATOR = 0x07064b50;
const ZIP_END = 0x06054b50;
const ZIP64_EXTRA = 0x0001;
const UTF8_AND_DESCRIPTOR_FLAGS = 0x0808;
const UNIX_REGULAR_FILE_ATTRIBUTES = (0o100600 * 0x1_0000) >>> 0;
const MINIMUM_DISK_RESERVE = 256n * 1024n * 1024n;
const MAXIMUM_DISK_RESERVE = 2n * 1024n * 1024n * 1024n;

export interface StreamingZipSource {
  name: string;
  contents?: Uint8Array;
  sourcePath?: string;
}

export interface ExtractedZipEntry {
  name: string;
  path: string;
  compressedBytes: number;
  uncompressedBytes: number;
  crc32: number;
}

export interface ExtractedZipArchive {
  root: string;
  entries: Map<string, ExtractedZipEntry>;
}

interface CentralEntry {
  name: string;
  crc32: number;
  size: number;
  offset: number;
}

interface ParsedCentralEntry {
  name: string;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
  localOffset: number;
  compression: number;
  flags: number;
  externalAttributes: number;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let value = 0; value < table.length; value += 1) {
    let current = value;
    for (let bit = 0; bit < 8; bit += 1) {
      current = (current & 1) !== 0 ? 0xedb88320 ^ (current >>> 1) : current >>> 1;
    }
    table[value] = current >>> 0;
  }
  return table;
})();

class Crc32 {
  private value = 0xffff_ffff;

  update(chunk: Uint8Array): void {
    let current = this.value;
    for (let index = 0; index < chunk.byteLength; index += 1) {
      current = CRC32_TABLE[(current ^ chunk[index]!) & 0xff]! ^ (current >>> 8);
    }
    this.value = current >>> 0;
  }

  digest(): number {
    return (this.value ^ 0xffff_ffff) >>> 0;
  }
}

function checkedNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds this runtime's addressable file range.`);
  }
  return Number(value);
}

function safeArchivePath(path: string): boolean {
  if (
    !path ||
    path.includes("\0") ||
    path.includes("\\") ||
    path.startsWith("/") ||
    /^[a-zA-Z]:/.test(path)
  ) {
    return false;
  }
  return path
    .split("/")
    .every((part) => part !== "" && part !== "." && part !== "..");
}

export function assertSafeArchivePath(path: string): void {
  if (!safeArchivePath(path)) throw new Error(`Unsafe path in .omni bundle: ${path}`);
}

async function writeAll(
  handle: Awaited<ReturnType<typeof open>>,
  value: Uint8Array,
  position: number
): Promise<number> {
  let cursor = 0;
  while (cursor < value.byteLength) {
    const { bytesWritten } = await handle.write(
      value,
      cursor,
      value.byteLength - cursor,
      position + cursor
    );
    if (bytesWritten <= 0) throw new Error("The .omni archive write made no progress.");
    cursor += bytesWritten;
  }
  return position + value.byteLength;
}

async function readExactly(
  handle: Awaited<ReturnType<typeof open>>,
  position: number,
  length: number
): Promise<Buffer> {
  const value = Buffer.allocUnsafe(length);
  let cursor = 0;
  while (cursor < length) {
    const { bytesRead } = await handle.read(value, cursor, length - cursor, position + cursor);
    if (bytesRead <= 0) throw new Error("The .omni ZIP container is truncated.");
    cursor += bytesRead;
  }
  return value;
}

function zip64Extra(values: bigint[]): Buffer {
  const result = Buffer.alloc(4 + values.length * 8);
  result.writeUInt16LE(ZIP64_EXTRA, 0);
  result.writeUInt16LE(values.length * 8, 2);
  values.forEach((value, index) => result.writeBigUInt64LE(value, 4 + index * 8));
  return result;
}

function sourceSize(source: StreamingZipSource): Promise<number> | number {
  if (source.contents) return source.contents.byteLength;
  if (!source.sourcePath) throw new Error(`Archive source ${source.name} has no content.`);
  return lstat(source.sourcePath).then((info) => {
    if (!info.isFile()) throw new Error(`Archive source ${source.name} is not a regular file.`);
    return info.size;
  });
}

export async function ensureDiskReserve(
  path: string,
  additionalBytes: number | bigint = 0
): Promise<void> {
  await mkdir(path, { recursive: true });
  const filesystem = await statfs(path, { bigint: true });
  const available = filesystem.bavail * filesystem.bsize;
  const capacity = filesystem.blocks * filesystem.bsize;
  const reserve = [
    MINIMUM_DISK_RESERVE,
    capacity / 100n,
    MAXIMUM_DISK_RESERVE
  ].sort((left, right) => (left < right ? -1 : left > right ? 1 : 0))[1]!;
  const required =
    typeof additionalBytes === "bigint" ? additionalBytes : BigInt(additionalBytes);
  if (required < 0n || available < required + reserve) {
    throw new Error(
      "The operation paused before crossing the configured free-disk reserve watermark."
    );
  }
}

async function replaceAtomically(temporary: string, destination: string): Promise<void> {
  const backup = `${destination}.${process.pid}.previous`;
  let movedPrevious = false;
  try {
    try {
      await rename(temporary, destination);
      return;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (!["EEXIST", "EPERM", "ENOTEMPTY"].includes(code ?? "")) throw error;
    }
    await rename(destination, backup);
    movedPrevious = true;
    await rename(temporary, destination);
    await rm(backup, { force: true });
    movedPrevious = false;
  } catch (error) {
    if (movedPrevious) {
      try {
        await rename(backup, destination);
      } catch {
        // Preserve the original error; the backup path remains recoverable.
      }
    }
    throw error;
  }
}

/**
 * Write a stored-entry ZIP, upgrading individual fields and the directory to
 * ZIP64 when their physical ZIP limits are crossed. Entry payloads are never
 * accumulated in memory.
 */
export async function writeStreamingZip(
  destination: string,
  sources: StreamingZipSource[]
): Promise<void> {
  const seen = new Set<string>();
  const prepared: Array<StreamingZipSource & { size: number }> = [];
  let payloadBytes = 0n;
  for (const source of sources) {
    assertSafeArchivePath(source.name);
    if (seen.has(source.name)) throw new Error(`Duplicate path in .omni bundle: ${source.name}`);
    seen.add(source.name);
    if (Buffer.byteLength(source.name, "utf8") > UINT16_MAX) {
      throw new Error(`Archive path ${source.name} exceeds the physical ZIP filename field.`);
    }
    const size = await sourceSize(source);
    if (!Number.isSafeInteger(size) || size < 0) {
      throw new Error(`Archive source ${source.name} has an invalid size.`);
    }
    prepared.push({ ...source, size });
    payloadBytes += BigInt(size);
  }

  const parent = dirname(resolve(destination));
  await ensureDiskReserve(parent, payloadBytes + BigInt(prepared.length * 256 + 4096));
  const temporaryRoot = await mkdtemp(join(parent, ".omni-export-"));
  const temporary = join(temporaryRoot, basename(destination));
  const handle = await open(temporary, "wx", 0o600);
  let offset = 0;
  const central: CentralEntry[] = [];
  try {
    for (const source of prepared) {
      const name = Buffer.from(source.name, "utf8");
      const entryOffset = offset;
      const largeEntry = source.size >= UINT32_MAX;
      const localExtra = largeEntry
        ? zip64Extra([BigInt(source.size), BigInt(source.size)])
        : Buffer.alloc(0);
      const local = Buffer.alloc(30);
      local.writeUInt32LE(ZIP_LOCAL_HEADER, 0);
      local.writeUInt16LE(largeEntry ? 45 : 20, 4);
      local.writeUInt16LE(UTF8_AND_DESCRIPTOR_FLAGS, 6);
      local.writeUInt16LE(0, 8);
      local.writeUInt16LE(0, 10);
      local.writeUInt16LE(0x21, 12);
      local.writeUInt32LE(0, 14);
      local.writeUInt32LE(largeEntry ? UINT32_MAX : 0, 18);
      local.writeUInt32LE(largeEntry ? UINT32_MAX : 0, 22);
      local.writeUInt16LE(name.byteLength, 26);
      local.writeUInt16LE(localExtra.byteLength, 28);
      offset = await writeAll(handle, local, offset);
      offset = await writeAll(handle, name, offset);
      offset = await writeAll(handle, localExtra, offset);

      const crc = new Crc32();
      let observed = 0;
      const writeChunk = async (chunk: Uint8Array): Promise<void> => {
        crc.update(chunk);
        observed += chunk.byteLength;
        offset = await writeAll(handle, chunk, offset);
      };
      if (source.contents) {
        await writeChunk(source.contents);
      } else {
        for await (const chunk of createReadStream(source.sourcePath!)) {
          await writeChunk(chunk as Buffer);
        }
      }
      if (observed !== source.size) {
        throw new Error(`Archive source ${source.name} changed while it was being exported.`);
      }

      const descriptor = Buffer.alloc(largeEntry ? 24 : 16);
      descriptor.writeUInt32LE(ZIP_DATA_DESCRIPTOR, 0);
      descriptor.writeUInt32LE(crc.digest(), 4);
      if (largeEntry) {
        descriptor.writeBigUInt64LE(BigInt(observed), 8);
        descriptor.writeBigUInt64LE(BigInt(observed), 16);
      } else {
        descriptor.writeUInt32LE(observed, 8);
        descriptor.writeUInt32LE(observed, 12);
      }
      offset = await writeAll(handle, descriptor, offset);
      central.push({
        name: source.name,
        crc32: crc.digest(),
        size: observed,
        offset: entryOffset
      });
    }

    const centralOffset = offset;
    for (const entry of central) {
      const name = Buffer.from(entry.name, "utf8");
      const largeSize = entry.size >= UINT32_MAX;
      const largeOffset = entry.offset >= UINT32_MAX;
      const extraValues: bigint[] = [];
      if (largeSize) extraValues.push(BigInt(entry.size), BigInt(entry.size));
      if (largeOffset) extraValues.push(BigInt(entry.offset));
      const extra = extraValues.length > 0 ? zip64Extra(extraValues) : Buffer.alloc(0);
      const record = Buffer.alloc(46);
      record.writeUInt32LE(ZIP_CENTRAL_HEADER, 0);
      record.writeUInt16LE((3 << 8) | (extra.length > 0 ? 45 : 20), 4);
      record.writeUInt16LE(extra.length > 0 ? 45 : 20, 6);
      record.writeUInt16LE(UTF8_AND_DESCRIPTOR_FLAGS, 8);
      record.writeUInt16LE(0, 10);
      record.writeUInt16LE(0, 12);
      record.writeUInt16LE(0x21, 14);
      record.writeUInt32LE(entry.crc32, 16);
      record.writeUInt32LE(largeSize ? UINT32_MAX : entry.size, 20);
      record.writeUInt32LE(largeSize ? UINT32_MAX : entry.size, 24);
      record.writeUInt16LE(name.byteLength, 28);
      record.writeUInt16LE(extra.byteLength, 30);
      record.writeUInt16LE(0, 32);
      record.writeUInt16LE(0, 34);
      record.writeUInt16LE(0, 36);
      record.writeUInt32LE(UNIX_REGULAR_FILE_ATTRIBUTES, 38);
      record.writeUInt32LE(largeOffset ? UINT32_MAX : entry.offset, 42);
      offset = await writeAll(handle, record, offset);
      offset = await writeAll(handle, name, offset);
      offset = await writeAll(handle, extra, offset);
    }
    const centralSize = offset - centralOffset;
    const needsZip64 =
      central.length >= UINT16_MAX ||
      centralOffset >= UINT32_MAX ||
      centralSize >= UINT32_MAX ||
      central.some((entry) => entry.size >= UINT32_MAX || entry.offset >= UINT32_MAX);
    if (needsZip64) {
      const zip64Offset = offset;
      const zip64End = Buffer.alloc(56);
      zip64End.writeUInt32LE(ZIP64_END, 0);
      zip64End.writeBigUInt64LE(44n, 4);
      zip64End.writeUInt16LE((3 << 8) | 45, 12);
      zip64End.writeUInt16LE(45, 14);
      zip64End.writeUInt32LE(0, 16);
      zip64End.writeUInt32LE(0, 20);
      zip64End.writeBigUInt64LE(BigInt(central.length), 24);
      zip64End.writeBigUInt64LE(BigInt(central.length), 32);
      zip64End.writeBigUInt64LE(BigInt(centralSize), 40);
      zip64End.writeBigUInt64LE(BigInt(centralOffset), 48);
      offset = await writeAll(handle, zip64End, offset);
      const locator = Buffer.alloc(20);
      locator.writeUInt32LE(ZIP64_LOCATOR, 0);
      locator.writeUInt32LE(0, 4);
      locator.writeBigUInt64LE(BigInt(zip64Offset), 8);
      locator.writeUInt32LE(1, 16);
      offset = await writeAll(handle, locator, offset);
    }
    const end = Buffer.alloc(22);
    end.writeUInt32LE(ZIP_END, 0);
    end.writeUInt16LE(0, 4);
    end.writeUInt16LE(0, 6);
    end.writeUInt16LE(needsZip64 ? UINT16_MAX : central.length, 8);
    end.writeUInt16LE(needsZip64 ? UINT16_MAX : central.length, 10);
    end.writeUInt32LE(needsZip64 ? UINT32_MAX : centralSize, 12);
    end.writeUInt32LE(needsZip64 ? UINT32_MAX : centralOffset, 16);
    end.writeUInt16LE(0, 20);
    offset = await writeAll(handle, end, offset);
    await handle.sync();
    await handle.close();
    await replaceAtomically(temporary, resolve(destination));
  } catch (error) {
    await handle.close().catch(() => undefined);
    throw error;
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

function readZip64Extra(
  extra: Buffer,
  needsUncompressed: boolean,
  needsCompressed: boolean,
  needsOffset: boolean
): { uncompressed?: bigint; compressed?: bigint; offset?: bigint } {
  let cursor = 0;
  while (cursor + 4 <= extra.byteLength) {
    const id = extra.readUInt16LE(cursor);
    const length = extra.readUInt16LE(cursor + 2);
    cursor += 4;
    if (cursor + length > extra.byteLength) throw new Error("The .omni ZIP extra field is invalid.");
    if (id === ZIP64_EXTRA) {
      const end = cursor + length;
      const result: { uncompressed?: bigint; compressed?: bigint; offset?: bigint } = {};
      const readValue = (): bigint => {
        if (cursor + 8 > end) throw new Error("The .omni ZIP64 extra field is truncated.");
        const value = extra.readBigUInt64LE(cursor);
        cursor += 8;
        return value;
      };
      if (needsUncompressed) result.uncompressed = readValue();
      if (needsCompressed) result.compressed = readValue();
      if (needsOffset) result.offset = readValue();
      return result;
    }
    cursor += length;
  }
  throw new Error("The .omni ZIP64 entry is missing its size or offset metadata.");
}

function decodeName(value: Buffer, utf8: boolean): string {
  if (!utf8 && value.some((byte) => byte > 0x7f)) {
    throw new Error("Non-UTF-8 .omni ZIP filenames are not supported.");
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(value);
  } catch {
    throw new Error("The .omni ZIP filename is not valid UTF-8.");
  }
}

async function parseCentralDirectory(
  archivePath: string
): Promise<{ entries: ParsedCentralEntry[]; centralOffset: number }> {
  const archive = await stat(archivePath);
  if (!archive.isFile()) throw new Error("The selected .omni bundle is not a regular file.");
  const handle = await open(archivePath, "r");
  try {
    if (archive.size < 22) throw new Error("The .omni ZIP container is truncated.");
    const tailLength = Math.min(archive.size, 22 + UINT16_MAX);
    const tailOffset = archive.size - tailLength;
    const tail = await readExactly(handle, tailOffset, tailLength);
    let endIndex = -1;
    for (let index = tail.byteLength - 22; index >= 0; index -= 1) {
      if (
        tail.readUInt32LE(index) === ZIP_END &&
        index + 22 + tail.readUInt16LE(index + 20) === tail.byteLength
      ) {
        endIndex = index;
        break;
      }
    }
    if (endIndex < 0) throw new Error("The .omni file is not a supported ZIP container.");
    const endOffset = tailOffset + endIndex;
    const disk = tail.readUInt16LE(endIndex + 4);
    const centralDisk = tail.readUInt16LE(endIndex + 6);
    const entriesOnDisk = tail.readUInt16LE(endIndex + 8);
    const totalEntries = tail.readUInt16LE(endIndex + 10);
    if (disk !== 0 || centralDisk !== 0) {
      throw new Error("Multi-disk .omni ZIP containers are not supported.");
    }
    if (entriesOnDisk !== totalEntries) {
      throw new Error("The .omni ZIP directory has inconsistent per-disk entry counts.");
    }
    let count = BigInt(totalEntries);
    let centralSize = BigInt(tail.readUInt32LE(endIndex + 12));
    let centralOffset = BigInt(tail.readUInt32LE(endIndex + 16));
    if (
      count === BigInt(UINT16_MAX) ||
      centralSize === BigInt(UINT32_MAX) ||
      centralOffset === BigInt(UINT32_MAX)
    ) {
      if (endOffset < 20) throw new Error("The .omni ZIP64 locator is missing.");
      const locator = await readExactly(handle, endOffset - 20, 20);
      if (locator.readUInt32LE(0) !== ZIP64_LOCATOR || locator.readUInt32LE(4) !== 0) {
        throw new Error("The .omni ZIP64 locator is invalid.");
      }
      const zip64Offset = checkedNumber(locator.readBigUInt64LE(8), "ZIP64 directory");
      if (locator.readUInt32LE(16) !== 1) {
        throw new Error("Multi-disk .omni ZIP64 containers are not supported.");
      }
      const zip64 = await readExactly(handle, zip64Offset, 56);
      if (
        zip64.readUInt32LE(0) !== ZIP64_END ||
        zip64.readBigUInt64LE(4) < 44n ||
        zip64.readUInt32LE(16) !== 0 ||
        zip64.readUInt32LE(20) !== 0 ||
        zip64.readBigUInt64LE(24) !== zip64.readBigUInt64LE(32)
      ) {
        throw new Error("The .omni ZIP64 directory is invalid.");
      }
      count = zip64.readBigUInt64LE(32);
      centralSize = zip64.readBigUInt64LE(40);
      centralOffset = zip64.readBigUInt64LE(48);
    }
    const entryCount = checkedNumber(count, "ZIP entry count");
    const centralStart = checkedNumber(centralOffset, "ZIP central-directory offset");
    const centralBytes = checkedNumber(centralSize, "ZIP central-directory length");
    if (
      centralStart < 0 ||
      centralBytes < 0 ||
      centralStart + centralBytes > endOffset
    ) {
      throw new Error("The .omni ZIP directory points outside the archive.");
    }

    const seen = new Set<string>();
    const entries: ParsedCentralEntry[] = [];
    let cursor = centralStart;
    for (let index = 0; index < entryCount; index += 1) {
      const fixed = await readExactly(handle, cursor, 46);
      if (fixed.readUInt32LE(0) !== ZIP_CENTRAL_HEADER) {
        throw new Error("The .omni ZIP central directory is invalid.");
      }
      const flags = fixed.readUInt16LE(8);
      const compression = fixed.readUInt16LE(10);
      if ((flags & 0x1) !== 0) throw new Error("Encrypted .omni entries are not supported.");
      if (compression !== 0 && compression !== 8) {
        throw new Error("The .omni bundle uses an unsupported ZIP compression method.");
      }
      const nameLength = fixed.readUInt16LE(28);
      const extraLength = fixed.readUInt16LE(30);
      const commentLength = fixed.readUInt16LE(32);
      const variable = await readExactly(
        handle,
        cursor + 46,
        nameLength + extraLength + commentLength
      );
      const name = decodeName(variable.subarray(0, nameLength), (flags & 0x0800) !== 0);
      assertSafeArchivePath(name);
      if (seen.has(name)) throw new Error(`Duplicate path in .omni bundle: ${name}`);
      seen.add(name);
      const externalAttributes = fixed.readUInt32LE(38);
      if (fixed.readUInt16LE(34) !== 0) {
        throw new Error("Multi-disk .omni ZIP entries are not supported.");
      }
      if (((externalAttributes >>> 16) & 0o170000) === 0o120000) {
        throw new Error(`Symbolic links are not allowed in .omni bundles: ${name}`);
      }
      const rawCompressed = fixed.readUInt32LE(20);
      const rawUncompressed = fixed.readUInt32LE(24);
      const rawOffset = fixed.readUInt32LE(42);
      const needsCompressed = rawCompressed === UINT32_MAX;
      const needsUncompressed = rawUncompressed === UINT32_MAX;
      const needsOffset = rawOffset === UINT32_MAX;
      const zip64 =
        needsCompressed || needsUncompressed || needsOffset
          ? readZip64Extra(
              variable.subarray(nameLength, nameLength + extraLength),
              needsUncompressed,
              needsCompressed,
              needsOffset
            )
          : {};
      entries.push({
        name,
        crc32: fixed.readUInt32LE(16),
        compressedSize: checkedNumber(
          needsCompressed ? zip64.compressed! : BigInt(rawCompressed),
          `${name} compressed size`
        ),
        uncompressedSize: checkedNumber(
          needsUncompressed ? zip64.uncompressed! : BigInt(rawUncompressed),
          `${name} uncompressed size`
        ),
        localOffset: checkedNumber(
          needsOffset ? zip64.offset! : BigInt(rawOffset),
          `${name} local-header offset`
        ),
        compression,
        flags,
        externalAttributes
      });
      cursor += 46 + variable.byteLength;
    }
    if (cursor !== centralStart + centralBytes) {
      throw new Error("The .omni ZIP central-directory length is inconsistent.");
    }
    const payloadRanges: Array<{ start: number; end: number; name: string }> = [];
    for (const entry of entries) {
      const dataOffset = await entryDataOffset(handle, entry, centralStart);
      payloadRanges.push({
        start: entry.localOffset,
        end: dataOffset + entry.compressedSize,
        name: entry.name
      });
    }
    payloadRanges.sort((left, right) => left.start - right.start);
    for (let index = 1; index < payloadRanges.length; index += 1) {
      if (payloadRanges[index]!.start < payloadRanges[index - 1]!.end) {
        throw new Error(
          `ZIP payloads for ${payloadRanges[index - 1]!.name} and ${payloadRanges[index]!.name} overlap.`
        );
      }
    }
    return { entries, centralOffset: centralStart };
  } finally {
    await handle.close();
  }
}

async function entryDataOffset(
  handle: Awaited<ReturnType<typeof open>>,
  entry: ParsedCentralEntry,
  centralOffset: number
): Promise<number> {
  const local = await readExactly(handle, entry.localOffset, 30);
  if (
    local.readUInt32LE(0) !== ZIP_LOCAL_HEADER ||
    local.readUInt16LE(6) !== entry.flags ||
    local.readUInt16LE(8) !== entry.compression
  ) {
    throw new Error(`The local ZIP header for ${entry.name} is inconsistent.`);
  }
  const nameLength = local.readUInt16LE(26);
  const extraLength = local.readUInt16LE(28);
  const name = decodeName(
    await readExactly(handle, entry.localOffset + 30, nameLength),
    (entry.flags & 0x0800) !== 0
  );
  if (name !== entry.name) throw new Error(`The local ZIP filename for ${entry.name} is inconsistent.`);
  const offset = entry.localOffset + 30 + nameLength + extraLength;
  if (
    offset < 0 ||
    entry.compressedSize < 0 ||
    offset + entry.compressedSize > centralOffset
  ) {
    throw new Error(`The compressed data for ${entry.name} escapes the ZIP payload.`);
  }
  return offset;
}

/**
 * Extract a ZIP/ZIP64 archive to a new temporary directory. Paths, duplicate
 * entries, Unix symlinks, encryption, CRC, declared lengths, and disk reserve
 * are validated before callers inspect any extracted state.
 */
export async function extractStreamingZip(
  archivePath: string,
  temporaryParent = tmpdir()
): Promise<ExtractedZipArchive> {
  const parsed = await parseCentralDirectory(archivePath);
  const total = parsed.entries.reduce(
    (sum, entry) => sum + BigInt(entry.uncompressedSize),
    0n
  );
  await ensureDiskReserve(temporaryParent, total);
  const root = await mkdtemp(join(temporaryParent, ".omni-import-"));
  const archive = await open(archivePath, "r");
  const extracted = new Map<string, ExtractedZipEntry>();
  try {
    for (const entry of parsed.entries) {
      const dataOffset = await entryDataOffset(archive, entry, parsed.centralOffset);
      const destination = resolve(root, ...entry.name.split("/"));
      const relativeDestination = relative(resolve(root), destination);
      if (
        relativeDestination === "" ||
        isAbsolute(relativeDestination) ||
        relativeDestination === ".." ||
        relativeDestination.startsWith(`..${sep}`)
      ) {
        throw new Error(`Unsafe path in .omni bundle: ${entry.name}`);
      }
      await mkdir(dirname(destination), { recursive: true });
      const crc = new Crc32();
      let bytes = 0;
      const meter = new Transform({
        transform(chunk: Buffer, _encoding, callback) {
          bytes += chunk.byteLength;
          if (bytes > entry.uncompressedSize) {
            callback(new Error(`ZIP entry ${entry.name} expands beyond its declared size.`));
            return;
          }
          crc.update(chunk);
          callback(null, chunk);
        }
      });
      if (entry.compressedSize === 0) {
        if (entry.uncompressedSize !== 0) {
          throw new Error(`ZIP entry ${entry.name} has an inconsistent empty payload.`);
        }
        await open(destination, "wx", 0o600).then((file) => file.close());
      } else {
        const input = createReadStream(archivePath, {
          start: dataOffset,
          end: dataOffset + entry.compressedSize - 1
        });
        const output = createWriteStream(destination, {
          flags: "wx",
          mode: 0o600
        });
        if (entry.compression === 8) {
          await pipeline(input, createInflateRaw(), meter, output);
        } else {
          await pipeline(input, meter, output);
        }
      }
      if (bytes !== entry.uncompressedSize || crc.digest() !== entry.crc32) {
        throw new Error(`ZIP entry ${entry.name} failed length or CRC validation.`);
      }
      extracted.set(entry.name, {
        name: entry.name,
        path: destination,
        compressedBytes: entry.compressedSize,
        uncompressedBytes: entry.uncompressedSize,
        crc32: entry.crc32
      });
    }
    return { root, entries: extracted };
  } catch (error) {
    await rm(root, { recursive: true, force: true });
    throw error;
  } finally {
    await archive.close();
  }
}

export async function streamFileSha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}
