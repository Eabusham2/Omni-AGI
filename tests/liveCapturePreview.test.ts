import { describe, expect, it, vi } from "vitest";
import { attachOwnedCapturePreview, createAcceptedSnapshotPreview } from "../src/renderer/src/capturePreviewOwnership";

describe("borrowed capture presentation", () => {
  it("uses the exact owned stream, detaches only the view, and never stops a track", async () => {
    const stop = vi.fn(), stream = { getTracks: () => [{ stop }] } as unknown as MediaStream;
    const video = { srcObject: null, muted: false, playsInline: false, play: vi.fn(async () => {}), pause: vi.fn() } as unknown as HTMLVideoElement;
    const failed = vi.fn(), release = attachOwnedCapturePreview(video, stream, failed);
    expect(video.srcObject).toBe(stream);
    expect(video.muted).toBe(true);
    expect(video.playsInline).toBe(true);
    release();
    expect(video.srcObject).toBeNull();
    expect(stop).not.toHaveBeenCalled();
    expect(failed).not.toHaveBeenCalled();
  });

  it("suppresses a late play failure after detach and does not detach a newer owner's stream", async () => {
    let reject!: (error: Error) => void;
    const stream = {} as MediaStream, newer = {} as MediaStream;
    const video = { srcObject: null, play: vi.fn(() => new Promise((_, no) => { reject = no; })), pause: vi.fn() } as unknown as HTMLVideoElement;
    const failed = vi.fn(), release = attachOwnedCapturePreview(video, stream, failed);
    video.srcObject = newer;
    release();
    reject(new Error("not allowed"));
    await Promise.resolve();
    expect(failed).not.toHaveBeenCalled();
    expect(video.srcObject).toBe(newer);
    expect(video.pause).not.toHaveBeenCalled();
  });

  it("creates one exact accepted still URL and revokes it idempotently", async () => {
    const blobs: Blob[] = [], urls = { createObjectURL: vi.fn((blob: Blob) => { blobs.push(blob); return "blob:owned-frame"; }), revokeObjectURL: vi.fn() };
    const packet = { modality: "video" as const, timestampMs: 42, mimeType: "image/jpeg", data: new Uint8Array([1, 2, 3]) };
    const preview = createAcceptedSnapshotPreview(packet, urls);
    expect(preview.url).toBe("blob:owned-frame");
    expect(new Uint8Array(await blobs[0]!.arrayBuffer())).toEqual(packet.data);
    expect(blobs[0]!.type).toBe("image/jpeg");
    preview.release(); preview.release();
    expect(urls.revokeObjectURL).toHaveBeenCalledExactlyOnceWith("blob:owned-frame");
    expect(() => createAcceptedSnapshotPreview({ ...packet, mimeType: "image/svg+xml" }, urls)).toThrow("supported still-image");
    expect(urls.createObjectURL).toHaveBeenCalledOnce();
  });
});
