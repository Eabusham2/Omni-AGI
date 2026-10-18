import { useEffect, useRef, useState } from "react";
import type { LivePerceptionCapturePacket, LivePerceptionPreview, LivePerceptionState } from "./livePerceptionController";
import { attachOwnedCapturePreview, createAcceptedSnapshotPreview } from "./capturePreviewOwnership";
import { Icon } from "./icons";

export function LiveCapturePreview({ preview, state, loaded }: {
  preview: Readonly<LivePerceptionPreview>;
  state: Readonly<LivePerceptionState>;
  loaded: boolean;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [playError, setPlayError] = useState("");
  const [snapshotSource, setSnapshotSource] = useState<{ packet: LivePerceptionCapturePacket; url: string }>();
  const [snapshotError, setSnapshotError] = useState("");
  const { stream, snapshot } = preview;
  const source = state.source === "screen" ? "Screen" : "Camera";

  useEffect(() => {
    setPlayError("");
    if (!stream || !video.current) return;
    return attachOwnedCapturePreview(video.current, stream, setPlayError);
  }, [stream]);

  useEffect(() => {
    setSnapshotError("");
    setSnapshotSource(undefined);
    if (!snapshot) return;
    try {
      const owned = createAcceptedSnapshotPreview(snapshot);
      setSnapshotSource({ packet: snapshot, url: owned.url });
      return owned.release;
    } catch (error) {
      setSnapshotError(error instanceof Error ? error.message : "Snapshot preview is unavailable.");
    }
  }, [snapshot]);

  const still = snapshotSource?.packet === snapshot ? snapshotSource : undefined;
  return <div className="camera-screen-preview">
    <figure className="camera-screen-preview__live">
      {stream ? <video ref={video} autoPlay muted playsInline aria-label={`${source} live preview`} /> :
        <div className="camera-screen-preview__empty">
          <Icon name="video" size={24} />
          <span>{!loaded ? "Load this instance to share a source." : state.phase === "requesting"
            ? "Choose a source in the system permission dialog."
            : state.enabled ? "Live preview is unavailable in this capture runtime." : "Camera and screen are off."}</span>
        </div>}
      {stream ? <figcaption><i aria-hidden="true" /> {source} live preview · {state.delivery === "snapshots" ? "snapshots shared on request" : "muted"}</figcaption> : null}
    </figure>
    {playError ? <p role="alert">{playError}</p> : null}
    {snapshot ? <details className="camera-screen-preview__snapshot">
      <summary>Latest accepted snapshot{snapshot?.settings?.width && snapshot.settings.height ? ` · ${snapshot.settings.width}×${snapshot.settings.height}` : ""}</summary>
      {still ? <img src={still.url} alt={`${source} snapshot accepted by this observation session`} /> : null}
    </details> : null}
    {snapshotError ? <p role="status">{snapshotError}</p> : null}
  </div>;
}
