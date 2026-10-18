import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { constants, type BigIntStats } from "node:fs";
import { lstat, open, realpath, type FileHandle } from "node:fs/promises";
import { dirname } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";

const HASH_WINDOW_BYTES = 64 * 1024;
const DEFAULT_PROOF_BYTES = 256 * 1024;
const SHA = /^[a-f0-9]{64}$/;

export interface TaskAttachmentReadBinding {
  /** Binds current owner/registry, exact public manifest and private source SHA. */
  authorityKey: string;
  path: string;
  contentSha256: string;
  bytes: number;
}
interface VerificationProof { identity: string; bytes: number }
export interface TaskAttachmentVerifierOptions {
  cacheBytes?: number;
  platform?: NodeJS.Platform;
  /** Testable conservative FS classification, never a renderer setting. */
  strongLocalChangeTimes?(path: string, signal?: AbortSignal): Promise<boolean>;
  reserveMemory?(bytes: number, label: string): void;
}

export function taskAttachmentFileIdentity(info: BigIntStats): string {
  if (!info.isFile() || [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].some(value => typeof value !== "bigint") ||
      info.dev < 0n || info.ino <= 0n || info.size < 0n || info.mtimeNs <= 0n || info.ctimeNs <= 0n) {
    throw new Error("Attachment file identity is unavailable or unsupported.");
  }
  return [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].join(":");
}

async function localApfsChangeTimes(path: string, signal?: AbortSignal): Promise<boolean> {
  if (process.platform !== "darwin") return false;
  signal?.throwIfAborted();
  const directory = await realpath(dirname(path));
  return new Promise(resolve => {
    execFile("/sbin/mount", [], { timeout: 5000, maxBuffer: 256 * 1024, windowsHide: true, signal }, (error, stdout) => {
      if (error) { resolve(false); return; }
      const mounts = stdout.split("\n").flatMap(line => {
        const match = /^.+ on (.+) \(([^)]+)\)$/.exec(line);
        if (!match || directory !== match[1] && !directory.startsWith(`${match[1]!.replace(/\/$/, "")}/`)) return [];
        return [{ path: match[1]!, flags: match[2]!.split(", ") }];
      }).sort((a, b) => b.path.length - a.path.length);
      resolve(mounts[0]?.flags[0] === "apfs" && mounts[0].flags.includes("local"));
    });
  });
}

/** Process-private bounded proof metadata only; no retained FD, source bytes or persisted authority. */
export class TaskAttachmentReadVerifier {
  private readonly proofs = new Map<string, VerificationProof>();
  private proofBytes = 0;
  private readonly platform: NodeJS.Platform;
  private readonly cacheBytes: number;
  readonly statistics = { fullHashAttempts: 0, fullHashes: 0, hashedBytes: 0, cacheHits: 0, proofs: 0, proofBytes: 0 };

  constructor(private readonly options: TaskAttachmentVerifierOptions = {}) {
    this.platform = options.platform ?? process.platform;
    this.cacheBytes = options.cacheBytes ?? DEFAULT_PROOF_BYTES;
    if (!Number.isSafeInteger(this.cacheBytes) || this.cacheBytes < 0) throw new Error("Invalid attachment verification cache byte budget.");
  }

  reserveMemory(bytes: number, label: string): void {
    if (!Number.isSafeInteger(bytes) || bytes < 1) throw new Error("Invalid attachment read memory envelope.");
    if (this.options.reserveMemory) { this.options.reserveMemory(bytes, label); return; }
    if (bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) {
      throw new Error(`${label} waits for available physical read memory.`);
    }
  }

  private key(binding: TaskAttachmentReadBinding): string {
    return createHash("sha256").update(JSON.stringify([binding.authorityKey, binding.path, binding.contentSha256, binding.bytes])).digest("hex");
  }
  private forget(key: string): void {
    const proof = this.proofs.get(key);
    if (proof) { this.proofBytes -= proof.bytes; this.proofs.delete(key); }
    this.statistics.proofs = this.proofs.size; this.statistics.proofBytes = this.proofBytes;
  }
  private remember(key: string, identity: string): void {
    this.forget(key);
    const bytes = 256 + 2 * (key.length + identity.length);
    if (bytes > this.cacheBytes) return;
    while (this.proofBytes + bytes > this.cacheBytes) this.forget(this.proofs.keys().next().value!);
    this.reserveMemory(this.proofBytes + bytes, "Attachment verification proof metadata");
    this.proofs.set(key, { identity, bytes }); this.proofBytes += bytes;
    this.statistics.proofs = this.proofs.size; this.statistics.proofBytes = this.proofBytes;
  }
  private cached(key: string, identity: string): boolean {
    const proof = this.proofs.get(key);
    if (!proof || proof.identity !== identity) { this.forget(key); return false; }
    this.proofs.delete(key); this.proofs.set(key, proof); return true;
  }

  async assertOpened(binding: TaskAttachmentReadBinding, file: FileHandle, identity: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    const [opened, atPath] = await Promise.all([file.stat({ bigint: true }), lstat(binding.path, { bigint: true })]);
    if (!atPath.isFile() || atPath.isSymbolicLink() || taskAttachmentFileIdentity(opened) !== identity ||
        taskAttachmentFileIdentity(atPath) !== identity || opened.size !== BigInt(binding.bytes)) {
      throw new Error("Attachment bytes or pathname changed while an exact opened read was pending.");
    }
    signal?.throwIfAborted();
  }

  /** Public only to permit focused actual-FD instrumentation, not exposed over preload. */
  async hashOpened(binding: TaskAttachmentReadBinding, file: FileHandle, identity: string, signal?: AbortSignal): Promise<string> {
    this.statistics.fullHashAttempts++;
    this.reserveMemory(HASH_WINDOW_BYTES, "Attachment SHA read buffer");
    const buffer = Buffer.alloc(Math.min(HASH_WINDOW_BYTES, Math.max(1, binding.bytes))), digest = createHash("sha256");
    let offset = 0;
    while (offset < binding.bytes) {
      signal?.throwIfAborted();
      const read = await file.read(buffer, 0, Math.min(buffer.length, binding.bytes - offset), offset);
      if (!read.bytesRead) throw new Error("Attachment ended before its verified byte coverage.");
      digest.update(buffer.subarray(0, read.bytesRead)); offset += read.bytesRead;
      await this.assertOpened(binding, file, identity, signal);
    }
    await this.assertOpened(binding, file, identity, signal);
    const result = digest.digest("hex");
    if (result !== binding.contentSha256) throw new Error("Attachment bytes failed content-addressed integrity validation.");
    this.statistics.fullHashes++; this.statistics.hashedBytes += offset;
    return result;
  }

  async withVerifiedFile<T>(binding: TaskAttachmentReadBinding, assertAuthority: () => Promise<void>,
    operation: (file: FileHandle) => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (!SHA.test(binding.contentSha256) || !SHA.test(binding.authorityKey) || !Number.isSafeInteger(binding.bytes) || binding.bytes < 1) {
      throw new Error("Attachment read binding is invalid.");
    }
    signal?.throwIfAborted(); await assertAuthority();
    const atPath = await lstat(binding.path, { bigint: true });
    if (!atPath.isFile() || atPath.isSymbolicLink() || atPath.size !== BigInt(binding.bytes)) throw new Error("Attachment blob is not a safe regular file.");
    const noFollow = this.platform !== "win32" && typeof constants.O_NOFOLLOW === "number" ? constants.O_NOFOLLOW : 0;
    const file = await open(binding.path, constants.O_RDONLY | noFollow), key = this.key(binding);
    let eligible = false;
    try {
      const opened = await file.stat({ bigint: true }), identity = taskAttachmentFileIdentity(opened);
      if (taskAttachmentFileIdentity(atPath) !== identity) throw new Error("Attachment source was replaced before its exact file opened.");
      await this.assertOpened(binding, file, identity, signal);
      const known = noFollow !== 0 && this.platform !== "win32" && this.cached(key, identity);
      if (known) { this.statistics.cacheHits++; eligible = true; }
      else {
        this.forget(key);
        await this.hashOpened(binding, file, identity, signal);
        // Unknown/network/coarse-timestamp filesystems and Windows always
        // rehash. A prior real SHA, not an inode, creates any reusable proof.
        eligible = noFollow !== 0 && this.platform !== "win32" && opened.mtimeNs % 1_000_000_000n !== 0n &&
          opened.ctimeNs % 1_000_000_000n !== 0n &&
          await (this.options.strongLocalChangeTimes ?? localApfsChangeTimes)(binding.path, signal);
      }
      await assertAuthority(); await this.assertOpened(binding, file, identity, signal);
      const result = await operation(file);
      await this.assertOpened(binding, file, identity, signal); await assertAuthority(); signal?.throwIfAborted();
      if (eligible) this.remember(key, identity);
      return result;
    } catch (error) { this.forget(key); throw error; }
    finally { try { await file.close(); } catch (error) { this.forget(key); throw error; } }
  }
}
