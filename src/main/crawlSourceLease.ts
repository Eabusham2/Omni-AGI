import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { hashFile } from "./dataIngestion";

export interface CrawlSourceLease {
  schemaVersion: 1;
  url: string;
  kind: "text" | "image" | "audio" | "video";
  contentHash: string;
  contentType: string;
  file: string;
  bytes: number;
  path: string;
  metadataPath: string;
}

/** Active source bytes are immutable until their neural/repository commit. */
export async function acquireCrawlSourceLease(input: {
  directory: string;
  url: string;
  kind: CrawlSourceLease["kind"];
  contentHash: string;
  contentType: string;
  extension: string;
  text?: string;
  sourcePath?: string;
}): Promise<CrawlSourceLease> {
  const identity = createHash("sha256").update(input.url).digest("hex");
  const metadataPath = join(input.directory, `${identity}.lease.json`);
  await mkdir(input.directory, { recursive: true });
  let stored: Omit<CrawlSourceLease, "path" | "metadataPath">;
  try {
    const metadataInfo = await lstat(metadataPath);
    if (!metadataInfo.isFile() || metadataInfo.size > 16_384) {
      throw new Error("Active crawl source lease metadata is not a safe bounded file.");
    }
    stored = JSON.parse(await readFile(metadataPath, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    if (!/^[a-f0-9]{64}$/u.test(input.contentHash) || !/^\.[a-z0-9]{1,12}$/u.test(input.extension)) {
      throw new Error("Crawl source lease identity is invalid.");
    }
    const file = `${identity}${input.extension}`;
    const path = join(input.directory, file);
    try {
      await lstat(path);
    } catch (missing) {
      if ((missing as NodeJS.ErrnoException).code !== "ENOENT") throw missing;
      if (input.text !== undefined) {
        await writeFile(path, input.text, { encoding: "utf8", flag: "wx", mode: 0o600 });
      } else if (input.sourcePath) {
        // Both paths are brain-owned spools on the same filesystem. Moving
        // keeps the native source snapshot without doubling disk consumption.
        await rename(input.sourcePath, path);
      } else {
        throw new Error("Crawl source lease requires staged input bytes.");
      }
    }
    const sourceInfo = await lstat(path);
    if (!sourceInfo.isFile() || (await hashFile(path)) !== input.contentHash) {
      throw new Error("Crawl source lease bytes do not match their committed identity.");
    }
    stored = {
      schemaVersion: 1, url: input.url, kind: input.kind,
      contentHash: input.contentHash, contentType: input.contentType,
      file, bytes: sourceInfo.size,
    };
    const temporary = `${metadataPath}.tmp-${randomUUID()}`;
    await writeFile(temporary, `${JSON.stringify(stored)}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
    await rename(temporary, metadataPath);
  }
  if (
    stored.schemaVersion !== 1 || stored.url !== input.url ||
    !["text", "image", "audio", "video"].includes(stored.kind) ||
    !/^[a-f0-9]{64}$/u.test(stored.contentHash) ||
    typeof stored.file !== "string" || basename(stored.file) !== stored.file ||
    !stored.file.startsWith(`${identity}.`) ||
    typeof stored.contentType !== "string" || stored.contentType.length > 4_096 ||
    !Number.isSafeInteger(stored.bytes) || stored.bytes < 0
  ) {
    throw new Error("Active crawl source lease contract is invalid.");
  }
  const path = join(input.directory, stored.file);
  const sourceInfo = await lstat(path);
  if (!sourceInfo.isFile() || sourceInfo.size !== stored.bytes || (await hashFile(path)) !== stored.contentHash) {
    throw new Error("Active crawl source lease changed before deterministic resume.");
  }
  return { ...stored, path, metadataPath };
}

export async function releaseCrawlSourceLease(lease: CrawlSourceLease): Promise<void> {
  await rm(lease.metadataPath, { force: true });
  await rm(lease.path, { force: true });
}
