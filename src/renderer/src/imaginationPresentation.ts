import type {
  ActionEvent,
  GeneratedArtifact,
  ModalityKind,
  ModalityPreview,
  RuntimeJob
} from "../../shared/types";

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function mediaFields(value: Record<string, unknown> | null): {
  sourceUrl: string | null;
  downloadUrl: string | null;
  mimeType: string;
  artifactPath: string;
} {
  const mimeType = typeof value?.mimeType === "string" ? value.mimeType : "";
  const candidate = typeof value?.dataUrl === "string" ? value.dataUrl : "";
  const dataUrl =
    /^(?:image|audio|video)\/[a-z0-9.+-]+$/i.test(mimeType) &&
    candidate.startsWith(`data:${mimeType};base64,`)
      ? candidate
      : null;
  const mediaUrl = typeof value?.mediaUrl === "string" &&
    /^omni-media:\/\/artifact\/[a-f0-9]{48}\/[a-f0-9]{64}$/.test(value.mediaUrl)
      ? value.mediaUrl
      : null;
  const artifactPath =
    typeof value?.path === "string"
      ? value.path
      : typeof value?.artifactPath === "string"
        ? value.artifactPath
        : "";
  return {
    // Embedded media bypasses Electron's custom-scheme audio/video data pipe.
    // Keep the opaque capability as the preferred download for persisted files.
    sourceUrl: dataUrl ?? mediaUrl,
    downloadUrl: mediaUrl ?? dataUrl,
    mimeType,
    artifactPath
  };
}

export interface ImaginationMediaPresentation {
  sourceUrl: string | null;
  downloadUrl: string | null;
  mimeType: string;
  artifactPath: string;
  finalArtifactPath: string;
  isFinal: boolean;
  isLiveRevision: boolean;
  revision?: number;
  progress?: number;
  statusLabel: string;
  modality?: Exclude<ModalityKind, "vision">;
}

export interface PersistedArtifactMediaPresentation {
  kind: "image" | "animated-video" | "audio" | "video" | "unavailable";
  ariaLabel: string;
  animationCopy?: string;
}

/**
 * Describe persisted media from its verified modality and MIME identity.
 * APNG is the honest fallback for a locally generated video when MP4 encoding
 * is unavailable: Chromium renders it through an image element, but its acTL
 * timeline loops automatically and must still be announced as animation.
 */
export function persistedArtifactMediaPresentation(
  artifact: Pick<GeneratedArtifact, "available" | "mimeType" | "modality">
): PersistedArtifactMediaPresentation {
  if (!artifact.available) {
    return {
      kind: "unavailable",
      ariaLabel: `Saved ${artifact.modality} imagination is unavailable`
    };
  }
  if (artifact.modality === "video" && artifact.mimeType === "image/apng") {
    return {
      kind: "animated-video",
      ariaLabel: "Animated APNG video imagination; loops automatically",
      animationCopy: "Animated APNG · loops automatically"
    };
  }
  if (artifact.mimeType.startsWith("audio/")) {
    return {
      kind: "audio",
      ariaLabel: "Saved audio imagination player"
    };
  }
  if (artifact.mimeType.startsWith("video/")) {
    return {
      kind: "video",
      ariaLabel: "Saved video imagination player"
    };
  }
  if (artifact.mimeType.startsWith("image/")) {
    return {
      kind: "image",
      ariaLabel: `Saved ${artifact.modality} imagination`
    };
  }
  return {
    kind: "unavailable",
    ariaLabel: `Saved ${artifact.modality} imagination is unavailable`
  };
}

/**
 * Prefer the completed artifact, while retaining the last real decoder image
 * if a large final file is path-only. No synthetic placeholder is promoted to
 * media and no revision is treated as authoritative before job completion.
 */
export function imaginationMediaPresentation(
  outputValue: unknown,
  preview: ModalityPreview | undefined,
  complete: boolean,
  fallbackModality?: ModalityKind
): ImaginationMediaPresentation {
  const output = record(outputValue);
  const finalMedia = mediaFields(output);
  const previewMedia = mediaFields(preview ? { ...preview } : null);
  const useFinalMedia = complete && Boolean(
    finalMedia.sourceUrl || finalMedia.artifactPath
  );
  const selected = useFinalMedia ? finalMedia : previewMedia;
  const displayed = selected.sourceUrl
    ? selected
    : complete && previewMedia.sourceUrl
      ? previewMedia
      : selected;
  const outputModality = output?.modality;
  const modality =
    ["image", "audio", "video"].includes(String(outputModality))
      ? outputModality as Exclude<ModalityKind, "vision">
      : preview?.modality ?? (
          fallbackModality && fallbackModality !== "vision"
            ? fallbackModality
            : undefined
        );
  return {
    sourceUrl: displayed.sourceUrl,
    downloadUrl: finalMedia.downloadUrl ?? displayed.downloadUrl,
    mimeType: displayed.mimeType,
    artifactPath: selected.artifactPath,
    finalArtifactPath: complete ? finalMedia.artifactPath : "",
    isFinal: complete,
    isLiveRevision: Boolean(preview) && !complete,
    revision: preview?.revision,
    progress: complete ? 1 : preview?.progress,
    statusLabel: complete
      ? "Final neural artifact"
      : preview?.statusLabel ?? "Waiting for the first decoder revision",
    modality
  };
}

export function actionImaginationMedia(
  event: ActionEvent
): ImaginationMediaPresentation {
  return imaginationMediaPresentation(
    event.execution?.output,
    event.preview,
    event.state === "complete",
    event.action.arguments.modality as ModalityKind | undefined
  );
}

export function jobImaginationMedia(
  job: RuntimeJob | null,
  fallbackModality: ModalityKind
): ImaginationMediaPresentation {
  return imaginationMediaPresentation(
    job?.output,
    job?.preview,
    job?.state === "complete",
    fallbackModality
  );
}

export function imaginationRevisionCopy(
  presentation: ImaginationMediaPresentation
): string {
  if (presentation.isFinal) {
    return presentation.revision === undefined
      ? "Final artifact"
      : `Final artifact · replaced decoder r${presentation.revision}`;
  }
  if (presentation.revision === undefined) return presentation.statusLabel;
  const progress = presentation.progress === undefined
    ? ""
    : ` · ${Math.round(presentation.progress * 100)}%`;
  return `Live decoder r${presentation.revision}${progress} · ${presentation.statusLabel}`;
}

/** Use a finite count only when it came from current worker/UI state. */
export function imaginationSeedCopy(
  outputValue: unknown,
  preview: ModalityPreview | undefined,
  measuredWorkingMemoryCount?: number
): string {
  const output = record(outputValue);
  const ideaSeed = record(output?.ideaSeed);
  const measured = [
    ideaSeed?.activeAssemblyCount,
    preview?.activeAssemblyCount,
    measuredWorkingMemoryCount
  ].find(
    (value) => typeof value === "number" && Number.isSafeInteger(value) && value >= 0
  );
  const source =
    typeof ideaSeed?.source === "string"
      ? ideaSeed.source
      : preview?.ideaSource;
  if (typeof measured === "number" && measured > 0) {
    return `Seeded from ${measured} active neural ${measured === 1 ? "assembly" : "assemblies"}`;
  }
  if (measured === 0 && source === "intrinsic-neural-cold-start") {
    return "No active assemblies · intrinsic neural seed";
  }
  return "Seeded from active neural state";
}
