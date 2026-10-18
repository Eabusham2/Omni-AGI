import type { LivePerceptionCapturePacket } from "./livePerceptionController";

/** Borrow the capture owner's stream. Detaching a view must never stop its tracks. */
export function attachOwnedCapturePreview(
  video: HTMLVideoElement,
  stream: MediaStream,
  onError: (message: string) => void
): () => void {
  let attached = true;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  const failed = (): void => {
    if (attached) onError("The live preview could not play. Capture remains active; Stop closes the source.");
  };
  try { void Promise.resolve(video.play()).catch(failed); }
  catch { failed(); }
  return () => {
    attached = false;
    if (video.srcObject !== stream) return;
    video.pause();
    video.srcObject = null;
  };
}

/** One latest accepted frame only; URL lifetime belongs to the mounted view. */
export function createAcceptedSnapshotPreview(
  packet: LivePerceptionCapturePacket,
  urls: Pick<typeof URL, "createObjectURL" | "revokeObjectURL"> = URL
): { url: string; release: () => void } {
  if (packet.modality !== "video" || !["image/jpeg", "image/png", "image/webp"].includes(packet.mimeType)) {
    throw new Error("This capture runtime does not supply a supported still-image preview.");
  }
  const url = urls.createObjectURL(new Blob([new Uint8Array(packet.data)], { type: packet.mimeType }));
  let released = false;
  return { url, release: () => { if (!released) { released = true; urls.revokeObjectURL(url); } } };
}
