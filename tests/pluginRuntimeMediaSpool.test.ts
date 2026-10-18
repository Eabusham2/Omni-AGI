import { createHash } from "node:crypto";
import { lstat, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { PluginRuntimeMediaSpool, type PluginRuntimeMediaSpoolOptions } from "../src/main/pluginRuntimeMediaSpool";
import { extractStreamingZip, streamFileSha256 } from "../src/main/streamingZip";

// Each workspace owns its extraction children; never race two recursive removers over the same tree.
const roots: string[] = [];
const sha = (value: Buffer | string): string => createHash("sha256").update(value).digest("hex");

afterEach(async () => {
  await Promise.all(roots.splice(0).map(async root => {
    await rm(root, { recursive: true, force: true });
    await expect(lstat(root)).rejects.toMatchObject({ code: "ENOENT" });
  }));
});

async function fixture(overrides: Partial<PluginRuntimeMediaSpoolOptions> = {}) {
  const workspaceRoot = await mkdtemp(join(tmpdir(), "omni-runtime-media-"));
  roots.push(workspaceRoot);
  const admissions: number[] = [], disk: number[] = [];
  const options: PluginRuntimeMediaSpoolOptions = {
    workspaceRoot,
    identity: { pluginId: "fixture.media", packageSha256: "a".repeat(64), formatId: "media", moduleId: "media",
      moduleSha256: "b".repeat(64), transactionKey: "c".repeat(64), sourceSha256: "d".repeat(64), sourceBytes: 123 },
    allowedMediaTypes: ["image/png", "audio/wav", "video/mp4"],
    maximumChunkBytes: 64 * 1024,
    admitMemory: async bytes => { admissions.push(bytes); return { release: () => undefined }; },
    checkDisk: async (_directory, bytes) => { disk.push(bytes); },
    ...overrides
  };
  return { workspaceRoot, admissions, disk, spool: await PluginRuntimeMediaSpool.create(options) };
}

async function emit(
  spool: PluginRuntimeMediaSpool,
  asset: { assetId: string; kind: "image" | "audio" | "video"; mimeType: string; bytes: Buffer },
  pageBytes = 7
) {
  await spool.start(asset);
  for (let offset = 0; offset < asset.bytes.length; offset += pageBytes) {
    const page = asset.bytes.subarray(offset, Math.min(asset.bytes.length, offset + pageBytes));
    await spool.write({ assetId: asset.assetId, offset, bytesBase64: page.toString("base64") });
  }
  return spool.end({ assetId: asset.assetId, bytes: asset.bytes.length, sha256: sha(asset.bytes) });
}

describe("runtime-stream media asset custody", () => {
  it("commits ordered mixed modality members with exact hashes and no caption/metadata surrogate", async () => {
    const value = await fixture();
    const png = Buffer.from("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489", "hex");
    const wav = Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WAVEfmt "), Buffer.alloc(24)]);
    const mp4 = Buffer.concat([Buffer.from([0, 0, 0, 20]), Buffer.from("ftypisom"), Buffer.alloc(12)]);
    await emit(value.spool, { assetId: "picture", kind: "image", mimeType: "image/png", bytes: png });
    await emit(value.spool, { assetId: "sound", kind: "audio", mimeType: "audio/wav", bytes: wav });
    await emit(value.spool, { assetId: "motion", kind: "video", mimeType: "video/mp4", bytes: mp4 });
    const result = await value.spool.complete();
    expect(result.state).toBe("ready");
    if (result.state !== "ready") throw new Error("Expected a ready media bundle.");
    expect(result.provenance.assets).toEqual([
      expect.objectContaining({ ordinal: 0, assetId: "picture", kind: "image", mimeType: "image/png", memberName: "media/00000000.png", bytes: png.length, sha256: sha(png) }),
      expect.objectContaining({ ordinal: 1, assetId: "sound", kind: "audio", mimeType: "audio/wav", memberName: "media/00000001.wav", bytes: wav.length, sha256: sha(wav) }),
      expect.objectContaining({ ordinal: 2, assetId: "motion", kind: "video", mimeType: "video/mp4", memberName: "media/00000002.mp4", bytes: mp4.length, sha256: sha(mp4) })
    ]);
    expect(result.provenance.bundleSha256).toBe(await streamFileSha256(result.bundlePath));
    const extracted = await extractStreamingZip(result.bundlePath, value.workspaceRoot, { checkDisk: async () => undefined });
    expect(dirname(extracted.root)).toBe(value.workspaceRoot);
    expect([...extracted.entries.keys()]).toEqual(["media/00000000.png", "media/00000001.wav", "media/00000002.mp4"]);
    expect(extracted.entries.has("manifest.json")).toBe(false);
    for (const asset of result.provenance.assets) {
      expect(await streamFileSha256(extracted.entries.get(asset.memberName)!.path)).toBe(asset.sha256);
    }
    expect(value.admissions).toContain(512 * 1024);
    expect(value.disk.some(bytes => bytes >= png.length + wav.length + mp4.length)).toBe(true);
    await result.cleanup();
    await expect(lstat(join(value.workspaceRoot, "plugin-runtime-media", "c".repeat(64))))
      .rejects.toMatchObject({ code: "ENOENT" });
    expect((await lstat(extracted.root)).isDirectory()).toBe(true);
  });

  it("fails the whole source transaction on kind/signature or ordering mismatch and removes partial custody", async () => {
    const value = await fixture();
    await value.spool.start({ assetId: "wrong", kind: "image", mimeType: "image/png" });
    const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0]);
    await value.spool.write({ assetId: "wrong", offset: 0, bytesBase64: jpeg.toString("base64") });
    await expect(value.spool.end({ assetId: "wrong", bytes: jpeg.length, sha256: sha(jpeg) })).rejects.toThrow(/does not match declared image\/png/i);
    await expect(lstat(join(value.workspaceRoot, "plugin-runtime-media", "c".repeat(64)))).rejects.toMatchObject({ code: "ENOENT" });

    const reordered = await fixture();
    await reordered.spool.start({ assetId: "ordered", kind: "audio", mimeType: "audio/wav" });
    await expect(reordered.spool.write({ assetId: "ordered", offset: 1, bytesBase64: Buffer.from("RIFF").toString("base64") }))
      .rejects.toThrow(/out of order/i);
    await expect(lstat(join(reordered.workspaceRoot, "plugin-runtime-media", "c".repeat(64)))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("reports an empty parser result explicitly without constructing a learned-input bundle", async () => {
    const value = await fixture();
    const result = await value.spool.complete();
    expect(result).toMatchObject({ state: "no-assets", assets: [], reason: "no-assets-no-modality-updates" });
    expect("bundlePath" in result).toBe(false);
    await result.cleanup();
  });
});
