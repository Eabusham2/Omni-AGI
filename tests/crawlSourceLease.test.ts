import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { acquireCrawlSourceLease, releaseCrawlSourceLease } from "../src/main/crawlSourceLease";

const digest = (value: string) => createHash("sha256").update(value).digest("hex");

describe("immutable active crawl input leases (storage only)", () => {
  let directory: string;
  beforeEach(async () => { directory = await mkdtemp(join(tmpdir(), "omni-crawl-lease-")); });
  afterEach(async () => { await rm(directory, { recursive: true, force: true }); });

  it("keeps a failed text source's exact path, bytes, inode, and mtime on retry", async () => {
    const original = "original hash-bound source";
    const input = {
      directory, url: "https://example.com/source", kind: "text" as const,
      contentHash: digest(original), contentType: "text/plain", extension: ".txt", text: original,
    };
    const first = await acquireCrawlSourceLease(input);
    const before = await stat(first.path);
    const retried = await acquireCrawlSourceLease({ ...input, text: "refetched changed input", contentHash: digest("refetched changed input") });
    expect(retried.path).toBe(first.path);
    expect(retried.contentHash).toBe(first.contentHash);
    expect(await readFile(retried.path, "utf8")).toBe(original);
    const after = await stat(retried.path);
    expect([after.ino, after.mtimeMs, after.size]).toEqual([before.ino, before.mtimeMs, before.size]);
    await releaseCrawlSourceLease(retried);
    await expect(stat(retried.path)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(stat(retried.metadataPath)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("moves a media spool into an immutable lease instead of replaying a random new path", async () => {
    const sourcePath = join(directory, "download-a.mp4");
    await writeFile(sourcePath, "original media bytes");
    const before = await stat(sourcePath);
    const first = await acquireCrawlSourceLease({
      directory, url: "https://example.com/media", kind: "video",
      contentHash: digest("original media bytes"), contentType: "video/mp4", extension: ".mp4", sourcePath,
    });
    await expect(stat(sourcePath)).rejects.toMatchObject({ code: "ENOENT" });
    expect((await stat(first.path)).ino).toBe(before.ino);
    const changedPath = join(directory, "download-b.mp4");
    await writeFile(changedPath, "changed remote media");
    const retried = await acquireCrawlSourceLease({
      directory, url: first.url, kind: "video", contentHash: digest("changed remote media"),
      contentType: "video/mp4", extension: ".mp4", sourcePath: changedPath,
    });
    expect(retried.path).toBe(first.path);
    expect(retried.contentHash).toBe(first.contentHash);
    expect(await readFile(retried.path, "utf8")).toBe("original media bytes");
    expect(await readFile(changedPath, "utf8")).toBe("changed remote media");
  });

  it("fails closed on changed active bytes without overwriting them", async () => {
    const input = {
      directory, url: "https://example.com/source", kind: "text" as const,
      contentHash: digest("first"), contentType: "text/plain", extension: ".txt", text: "first",
    };
    const lease = await acquireCrawlSourceLease(input);
    await writeFile(lease.path, "changed");
    await expect(acquireCrawlSourceLease(input)).rejects.toThrow(/changed before deterministic resume/i);
    expect(await readFile(lease.path, "utf8")).toBe("changed");
  });
});
