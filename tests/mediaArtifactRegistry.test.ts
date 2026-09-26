import { createHash } from "node:crypto";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import {
  ARTIFACT_INDEX_DATABASE,
  ArtifactIndexStore,
  MediaArtifactRegistry,
  serializeArtifactIndex,
  type PersistedGeneratedArtifact
} from "../src/main/mediaArtifactRegistry";
import type { ModalityPreview } from "../src/shared/types";

const roots: string[] = [];
const registries: MediaArtifactRegistry[] = [];

afterEach(async () => {
  registries.splice(0).forEach((registry) => registry.dispose());
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-media-registry-"));
  roots.push(root);
  const brainRoot = join(root, "brain-a");
  const otherRoot = join(root, "brain-b");
  const previewRoot = join(brainRoot, "engine", ".preview-cache", "job");
  const finalRoot = join(brainRoot, "engine", "artifacts");
  await Promise.all([
    mkdir(previewRoot, { recursive: true }),
    mkdir(finalRoot, { recursive: true }),
    mkdir(join(otherRoot, "engine", "artifacts"), { recursive: true })
  ]);
  const bytes = Buffer.concat([
    Buffer.from("89504e470d0a1a0a", "hex"),
    Buffer.from("real decoder bytes")
  ]);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const previewPath = join(previewRoot, `${sha256}.png`);
  const finalPath = join(finalRoot, "final.png");
  await Promise.all([writeFile(previewPath, bytes), writeFile(finalPath, bytes)]);
  const registry = new MediaArtifactRegistry((brainId) =>
    brainId === "brain-a" ? brainRoot : otherRoot
  );
  registries.push(registry);
  return { registry, brainRoot, otherRoot, bytes, sha256, previewPath, finalPath };
}

function preview(path: string, sha256: string, revision: number): ModalityPreview {
  return {
    schemaVersion: 1,
    revision,
    progress: 0.5,
    statusLabel: "Real decoder revision",
    mimeType: "image/png",
    artifactPath: path,
    payloadSha256: sha256,
    producer: "same-brain-decoder",
    actualDecoderOutput: true
  };
}

function wavBytes(formatTag = 1, pcmBytes = 1_024): Buffer {
  const pcm = Buffer.alloc(pcmBytes, 0);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(formatTag, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(16_000, 24);
  header.writeUInt32LE(32_000, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

describe("renderer-safe media artifact registry", () => {
  it("authorizes a packaged-browser-compatible PCM WAV and canonicalizes its MIME", async () => {
    const { registry, brainRoot } = await fixture();
    const bytes = wavBytes();
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const path = join(brainRoot, "engine", "artifacts", `${sha256}.wav`);
    await writeFile(path, bytes);
    const completed = await registry.completeJob("brain-a", "pcm-audio", {
      modality: "audio",
      mimeType: "audio/x-wav",
      path,
      artifactSha256: sha256
    }) as Record<string, unknown>;

    expect(completed.mimeType).toBe("audio/wav");
    expect(completed.mediaUrl).toMatch(/^omni-media:\/\/artifact\//);
    await expect(registry.authorize(String(completed.mediaUrl))).resolves.toMatchObject({
      path: await realpath(path),
      mimeType: "audio/wav",
      size: 1_068,
      sha256
    });
  });

  it("keeps a six-second PCM WAV embedded through final-output sanitization", async () => {
    const { registry, brainRoot } = await fixture();
    const bytes = wavBytes(1, 91_904 * 2);
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const path = join(brainRoot, "engine", "artifacts", `${sha256}.wav`);
    const dataUrl = `data:audio/wav;base64,${bytes.toString("base64")}`;
    await writeFile(path, bytes);

    const completed = await registry.completeJob("brain-a", "auto-audio", {
      modality: "audio",
      mimeType: "audio/wav",
      path,
      artifactSha256: sha256,
      dataUrl
    }) as Record<string, unknown>;

    expect(bytes).toHaveLength(183_852);
    expect(dataUrl.length).toBeGreaterThan(96 * 1024);
    expect(completed.dataUrl).toBe(dataUrl);
    expect(completed.mediaUrl).toMatch(/^omni-media:\/\/artifact\//);
    expect(completed).not.toHaveProperty("path");
  });

  it("rejects a RIFF/WAVE shell whose codec cannot play in packaged Chromium", async () => {
    const { registry, brainRoot } = await fixture();
    const bytes = wavBytes(6);
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const path = join(brainRoot, "engine", "artifacts", `${sha256}.wav`);
    await writeFile(path, bytes);

    await expect(registry.completeJob("brain-a", "bad-wav-codec", {
      modality: "audio",
      mimeType: "audio/wav",
      path,
      artifactSha256: sha256
    })).rejects.toThrow(/signature does not match its MIME type/i);
  });

  it("leases checksum-bound files without exposing paths or large base64", async () => {
    const { registry, previewPath, sha256 } = await fixture();
    const leased = registry.leasePreview("brain-a", "job:brain-a:one", {
      ...preview(previewPath, sha256, 0),
      dataUrl: `data:image/png;base64,${"A".repeat(200_000)}`
    });
    expect(leased?.mediaUrl).toMatch(
      new RegExp(`^omni-media://artifact/[a-f0-9]{48}/${sha256}$`)
    );
    expect(leased?.dataUrl).toBeUndefined();
    expect(leased?.path).toBeUndefined();
    expect(leased?.artifactPath).toBeUndefined();
    const authorized = await registry.authorize(leased!.mediaUrl!);
    expect(authorized).toMatchObject({
      path: await realpath(previewPath),
      sha256,
      mimeType: "image/png"
    });
  });

  it("rejects cross-brain paths, traversal-shaped URLs, and bad checksums", async () => {
    const { registry, otherRoot, previewPath, sha256 } = await fixture();
    expect(registry.leasePreview(
      "brain-b",
      "job:brain-b:cross",
      preview(previewPath, sha256, 0)
    )).toBeUndefined();
    expect(await registry.authorize(
      `omni-media://artifact/../../${sha256}`
    )).toBeUndefined();

    const wrong = registry.leasePreview(
      "brain-a",
      "job:brain-a:wrong",
      preview(previewPath, "f".repeat(64), 0)
    );
    expect(wrong).toBeUndefined();
    expect(otherRoot).not.toBe("");
  });

  it("revokes stale revisions and replaces previews with a final lease", async () => {
    const { registry, previewPath, finalPath, sha256 } = await fixture();
    const first = registry.leasePreview(
      "brain-a", "job:brain-a:replace", preview(previewPath, sha256, 0)
    )!;
    const second = registry.leasePreview(
      "brain-a", "job:brain-a:replace", preview(previewPath, sha256, 1)
    )!;
    expect(registry.leasePreview(
      "brain-a", "job:brain-a:replace", preview(previewPath, sha256, 1)
    )).toBeUndefined();
    expect(await registry.authorize(first.mediaUrl!)).toBeUndefined();
    expect(await registry.authorize(second.mediaUrl!)).toBeDefined();

    const final = await registry.completeJob("brain-a", "replace", {
      modality: "image",
      mimeType: "image/png",
      path: await realpath(finalPath),
      artifactSha256: sha256,
      dataUrl: `data:image/png;base64,${"A".repeat(200_000)}`
    }) as Record<string, unknown>;
    expect(final.path).toBeUndefined();
    expect(final.dataUrl).toBeUndefined();
    expect(final.mediaUrl).toMatch(/^omni-media:\/\/artifact\//);
    expect(await registry.authorize(second.mediaUrl!)).toBeUndefined();
    expect(await registry.authorize(String(final.mediaUrl))).toMatchObject({
      path: await realpath(finalPath),
      sha256
    });
  });

  it("rejects a file changed after lease registration", async () => {
    const { registry, previewPath, sha256 } = await fixture();
    const leased = registry.leasePreview(
      "brain-a", "job:brain-a:mutated", preview(previewPath, sha256, 0)
    )!;
    await writeFile(previewPath, Buffer.from("changed after registration"));
    await expect(registry.authorize(leased.mediaUrl!)).rejects.toThrow(
      /changed after its lease|signature does not match|checksum does not match/i
    );
    expect(await registry.authorize(leased.mediaUrl!)).toBeUndefined();
  });

  it("reissues a verified short-lived lease from the persisted index after restart", async () => {
    const { registry, brainRoot, otherRoot, finalPath, sha256 } = await fixture();
    await registry.completeJob("brain-a", "persisted-job", {
      modality: "image",
      mimeType: "image/png",
      path: finalPath,
      artifactSha256: sha256,
      seed: 41,
      initialization: "locally-trained"
    });
    registry.dispose();
    const restarted = new MediaArtifactRegistry((brainId) =>
      brainId === "brain-a" ? brainRoot : otherRoot
    );
    registries.push(restarted);
    const artifacts = await restarted.listArtifacts("brain-a");
    expect(artifacts.artifacts).toHaveLength(1);
    expect(artifacts.artifacts[0]).toMatchObject({
      brainId: "brain-a",
      sha256,
      seed: 41,
      initialization: "locally-trained",
      available: true
    });
    expect(artifacts.artifacts[0]?.mediaUrl).toMatch(/^omni-media:\/\/artifact\//);
    expect(await restarted.authorize(artifacts.artifacts[0]!.mediaUrl!)).toMatchObject({
      sha256
    });

    await rm(finalPath, { force: true });
    restarted.dispose();
    const afterDelete = new MediaArtifactRegistry((brainId) =>
      brainId === "brain-a" ? brainRoot : otherRoot
    );
    registries.push(afterDelete);
    expect((await afterDelete.listArtifacts("brain-a")).artifacts).toMatchObject([
      { available: false, unavailableReason: "missing" }
    ]);
  });

  it("binds keyset cursors and rows to the append-only artifact hash chain", async () => {
    const { brainRoot, finalPath, sha256, bytes } = await fixture();
    const directory = join(brainRoot, "engine", "artifacts");
    const artifacts: PersistedGeneratedArtifact[] = Array.from({ length: 3 }, (_, index) => ({
      id: createHash("sha256").update(`chained-${index}`).digest("hex"),
      modality: "image",
      mimeType: "image/png",
      sha256,
      bytes: bytes.length,
      relativePath: `artifacts/${finalPath.split("/").at(-1)}`,
      createdAt: new Date(1_700_000_000_000 + index).toISOString()
    }));
    await ArtifactIndexStore.replace(directory, "brain-a", artifacts);
    const store = await ArtifactIndexStore.open(directory, "brain-a");
    const first = store.page(undefined, 1);
    expect(first.nextCursor).toMatch(/^[1-9][0-9]*\.[a-f0-9]{64}$/);
    const badCursor = `${first.nextCursor!.slice(0, -1)}${
      first.nextCursor!.endsWith("0") ? "1" : "0"
    }`;
    expect(() => store.page(badCursor, 1)).toThrow(/cursor checksum/i);
    store.close();

    const database = new DatabaseSync(join(directory, ARTIFACT_INDEX_DATABASE));
    database.prepare("UPDATE artifacts SET payload_json=? WHERE sequence=3").run(
      JSON.stringify({ ...artifacts[2], qualityNote: "tampered" })
    );
    database.close();
    const tampered = await ArtifactIndexStore.open(directory, "brain-a");
    try {
      expect(() => tampered.page(undefined, 1)).toThrow(/row checksum/i);
    } finally {
      tampered.close();
    }
  });

  it("atomically migrates and keyset-pages 10k+ artifacts with bounded row memory", async () => {
    const { registry, brainRoot, finalPath, sha256, bytes } = await fixture();
    const artifacts: PersistedGeneratedArtifact[] = Array.from(
      { length: 10_050 },
      (_, index) => ({
        id: createHash("sha256").update(`artifact-${index}`).digest("hex"),
        modality: "image",
        mimeType: "image/png",
        sha256,
        bytes: bytes.length,
        relativePath: `artifacts/${finalPath.split("/").at(-1)}`,
        createdAt: new Date(1_700_000_000_000 + index).toISOString()
      })
    );
    await writeFile(
      join(brainRoot, "engine", "artifacts", "index.json"),
      serializeArtifactIndex("brain-a", artifacts)
    );
    const store = await ArtifactIndexStore.open(
      join(brainRoot, "engine", "artifacts"),
      "brain-a"
    );
    let cursor: string | undefined;
    let received = 0;
    let peakRowsRead = 0;
    let peakPageBytes = 0;
    const heapBeforePaging = process.memoryUsage().heapUsed;
    do {
      const page = store.page(cursor, 100);
      peakRowsRead = Math.max(peakRowsRead, page.rowsRead);
      peakPageBytes = Math.max(
        peakPageBytes,
        Buffer.byteLength(JSON.stringify(page))
      );
      received += page.artifacts.length;
      cursor = page.nextCursor;
    } while (cursor);
    const heapGrowth = process.memoryUsage().heapUsed - heapBeforePaging;
    store.close();
    expect(received).toBe(10_050);
    expect(peakRowsRead).toBe(101);
    expect(peakPageBytes).toBeLessThan(128 * 1024);
    expect(heapGrowth).toBeLessThan(64 * 1024 * 1024);
    await expect(
      realpath(join(brainRoot, "engine", "artifacts", ARTIFACT_INDEX_DATABASE))
    ).resolves.toContain(ARTIFACT_INDEX_DATABASE);
    await expect(
      realpath(join(brainRoot, "engine", "artifacts", "index.json"))
    ).rejects.toMatchObject({ code: "ENOENT" });

    const rendererPage = await registry.listArtifacts("brain-a", undefined, 100);
    expect(rendererPage.artifacts).toHaveLength(100);
    expect(await registry.authorize(rendererPage.artifacts[0]!.mediaUrl!)).toBeDefined();
    expect(await registry.authorize(rendererPage.artifacts.at(-1)!.mediaUrl!)).toBeDefined();
  }, 30_000);
});
