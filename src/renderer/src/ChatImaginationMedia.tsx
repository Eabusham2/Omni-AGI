import { useState } from "react";
import type { ActionEvent } from "../../shared/types";
import {
  actionImaginationMedia,
  imaginationRevisionCopy,
  type ImaginationMediaPresentation
} from "./imaginationPresentation";

export function chatImaginationMediaLabel(
  presentation: ImaginationMediaPresentation,
  imagination = true
): { title: string; mediaLabel: string; animatedApng: boolean } {
  const animatedApng = presentation.modality === "video" &&
    presentation.mimeType === "image/apng";
  if (animatedApng) return {
    title: imagination ? "Video imagination" : "Video artifact",
    mediaLabel: imagination ? "Animated APNG video imagination; loops automatically" : "Animated APNG video artifact; loops automatically",
    animatedApng: true
  };
  if (presentation.mimeType.startsWith("audio/") || presentation.modality === "audio") {
    return { title: imagination ? "Sound imagination" : "Audio artifact", mediaLabel: imagination ? "Imagination audio player" : "Tool audio artifact player", animatedApng: false };
  }
  if (presentation.mimeType.startsWith("video/") || presentation.modality === "video") {
    return { title: imagination ? "Video imagination" : "Video artifact", mediaLabel: imagination ? "Imagination video player" : "Tool video artifact player", animatedApng: false };
  }
  return { title: imagination ? "Image imagination" : "Image artifact",
    mediaLabel: imagination ? "Image created by the local imagination action" : "Image artifact returned by this tool action", animatedApng: false };
}

export function chatImaginationUnavailableCopy(
  presentation: ImaginationMediaPresentation,
  playbackFailed: boolean
): string | undefined {
  const artifactPath = presentation.finalArtifactPath || presentation.artifactPath;
  if (presentation.artifactRetention === "ephemeral-spontaneous" && playbackFailed) {
    return "Temporary spontaneous output playback is unavailable. It was not saved; use Record to keep future output.";
  }
  if (playbackFailed) {
    if (artifactPath) return "Inline playback is unavailable. The artifact remains available with this action.";
    if (presentation.downloadUrl) return "Inline playback is unavailable. The actual artifact remains available for download.";
    return "Inline playback is unavailable. No playable saved artifact is available.";
  }
  if (!presentation.sourceUrl && artifactPath) {
    return "Inline preview is unavailable. The artifact remains available with this action.";
  }
  return undefined;
}

export function chatImaginationRetentionCopy(
  state: ActionEvent["state"],
  presentation: ImaginationMediaPresentation
): string {
  if (presentation.artifactRetention === "ephemeral-spontaneous") {
    return "Temporary spontaneous output · Use Record to keep it";
  }
  if (presentation.hasFinalArtifact) return "Final artifact available";
  if (state === "stopped" || state === "failed") return "Last delivered preview";
  return "Delivered preview";
}

export function chatImaginationRevisionLabel(
  state: ActionEvent["state"],
  presentation: ImaginationMediaPresentation
): string {
  if ((state === "stopped" || state === "failed") && presentation.isLiveRevision) {
    const frame = presentation.revision === undefined
      ? "decoder output"
      : `decoder r${presentation.revision}`;
    const reason = state === "failed" ? "Failed" : "Stopped";
    return `${reason} · last delivered ${frame} · ${presentation.statusLabel}`;
  }
  return imaginationRevisionCopy(presentation);
}

export function ChatImaginationMedia({ event, placement = "timeline" }: {
  event: ActionEvent;
  placement?: "timeline" | "audit";
}) {
  const presentation = actionImaginationMedia(event);
  const [failedSource, setFailedSource] = useState<string | null>(null);
  const isImagination = event.action.kind === "imagine" ||
    event.action.toolId === "modality.imagine";
  const artifactPath = presentation.finalArtifactPath || presentation.artifactPath;
  const hasActualOutput = Boolean(presentation.sourceUrl || artifactPath);

  if ((placement === "timeline" ? !isImagination : isImagination) || !hasActualOutput) return null;
  const labels = chatImaginationMediaLabel(presentation, isImagination);
  const source = presentation.sourceUrl;
  const playbackFailed = Boolean(source) && failedSource === source;
  const unavailable = chatImaginationUnavailableCopy(presentation, playbackFailed);
  const media = !playbackFailed && source && presentation.mimeType.startsWith("image/") ? (
    <img src={source} alt={labels.mediaLabel} onError={() => setFailedSource(source)} />
  ) : !playbackFailed && source && presentation.mimeType.startsWith("audio/") ? (
    <audio src={source} controls preload="metadata" aria-label={labels.mediaLabel}
      onError={() => setFailedSource(source)} />
  ) : !playbackFailed && source && presentation.mimeType.startsWith("video/") ? (
    <video src={source} controls preload="metadata" aria-label={labels.mediaLabel}
      onError={() => setFailedSource(source)} />
  ) : null;

  return (
    <section className={`chat-imagination-media chat-imagination-media--${placement}`} aria-label={labels.title}
      data-imagination-state={event.state === "stopped" || event.state === "failed"
        ? event.state : presentation.isFinal ? "final" : "live"}>
      <header>
        <strong>{labels.title}</strong>
        <small aria-live="polite">{chatImaginationRevisionLabel(event.state, presentation)}</small>
      </header>
      {media}
      {labels.animatedApng && media ? (
        <small className="chat-imagination-media__animation">Animated APNG video · loops automatically</small>
      ) : null}
      {unavailable ? <p role="status">{unavailable}</p> : null}
      {placement === "timeline" ? (
        <small className="chat-imagination-media__artifact">
          {chatImaginationRetentionCopy(event.state, presentation)}
        </small>
      ) : artifactPath ? (
        <small className="chat-imagination-media__artifact">
          {presentation.finalArtifactPath ? "Final artifact" : "Artifact"} · {artifactPath}
        </small>
      ) : null}
    </section>
  );
}
