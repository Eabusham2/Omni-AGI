import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  extractStreamingZip,
  writeStreamingZip,
  type StreamingZipSource
} from "../src/main/streamingZip";

describe("streaming ZIP/ZIP64 container", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
    );
  });

  it("streams more entries than the retired 4,096-file archive ceiling", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-streaming-zip-"));
    roots.push(root);
    const archivePath = join(root, "many.omni");
    const sources: StreamingZipSource[] = Array.from(
      { length: 4_105 },
      (_, index) => ({
        name: `shards/${index.toString().padStart(5, "0")}.json`,
        contents: Buffer.from(`${index}\n`)
      })
    );

    await writeStreamingZip(archivePath, sources);
    const extracted = await extractStreamingZip(
      archivePath,
      join(root, "extract")
    );
    expect(extracted.entries.size).toBe(sources.length);
    await expect(
      readFile(extracted.entries.get("shards/04104.json")!.path, "utf8")
    ).resolves.toBe("4104\n");
  }, 30_000);

  it("streams file payloads and rejects duplicate or symbolic-link-style paths", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-streaming-file-"));
    roots.push(root);
    const sourceDirectory = join(root, "source");
    await mkdir(sourceDirectory);
    const payload = join(sourceDirectory, "payload.bin");
    await writeFile(payload, Buffer.alloc(3 * 1024 * 1024, 0x5a));
    const archivePath = join(root, "payload.omni");

    await writeStreamingZip(archivePath, [
      { name: "tensors/core.safetensors", sourcePath: payload }
    ]);
    const extracted = await extractStreamingZip(
      archivePath,
      join(root, "extract")
    );
    expect(
      extracted.entries.get("tensors/core.safetensors")?.uncompressedBytes
    ).toBe(3 * 1024 * 1024);
    await expect(
      writeStreamingZip(join(root, "duplicate.omni"), [
        { name: "same.bin", contents: Buffer.from([1]) },
        { name: "same.bin", contents: Buffer.from([2]) }
      ])
    ).rejects.toThrow(/Duplicate path/);
    await expect(
      writeStreamingZip(join(root, "traversal.omni"), [
        { name: "../outside.bin", contents: Buffer.from([1]) }
      ])
    ).rejects.toThrow(/Unsafe path/);
  });

  it("reads a ZIP64 end record without materializing a central directory", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-zip64-"));
    roots.push(root);
    const archivePath = join(root, "empty-zip64.omni");
    const zip64End = Buffer.alloc(56);
    zip64End.writeUInt32LE(0x06064b50, 0);
    zip64End.writeBigUInt64LE(44n, 4);
    zip64End.writeUInt16LE(45, 12);
    zip64End.writeUInt16LE(45, 14);
    const locator = Buffer.alloc(20);
    locator.writeUInt32LE(0x07064b50, 0);
    locator.writeBigUInt64LE(0n, 8);
    locator.writeUInt32LE(1, 16);
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(0xffff, 8);
    end.writeUInt16LE(0xffff, 10);
    end.writeUInt32LE(0xffff_ffff, 12);
    end.writeUInt32LE(0xffff_ffff, 16);
    await writeFile(archivePath, Buffer.concat([zip64End, locator, end]));

    const extracted = await extractStreamingZip(
      archivePath,
      join(root, "extract")
    );
    expect(extracted.entries.size).toBe(0);
  });
});
