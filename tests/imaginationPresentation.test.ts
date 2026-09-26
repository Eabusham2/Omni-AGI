import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ActionEvent, GeneratedArtifact, RuntimeJob } from "../src/shared/types";
import {
  actionImaginationMedia,
  imaginationRevisionCopy,
  imaginationSeedCopy,
  jobImaginationMedia,
  persistedArtifactMediaPresentation
} from "../src/renderer/src/imaginationPresentation";

function job(overrides: Partial<RuntimeJob> = {}): RuntimeJob {
  return {
    id: "job-1",
    brainId: "brain-1",
    kind: "image",
    state: "running",
    progress: 0.5,
    label: "Generating image",
    createdAt: "2026-09-07T00:00:00.000Z",
    updatedAt: "2026-09-07T00:00:01.000Z",
    ...overrides
  };
}

describe("progressive imagination presentation", () => {
  it("wires real revisions without fake sweep animation or invented idea counts", () => {
    const app = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
      "utf8"
    );
    const styles = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/styles.css"),
      "utf8"
    );
    expect(app).toContain("jobImaginationMedia(job, mode)");
    expect(app).toContain("Progressive imagination · decoder r");
    expect(app).toContain("actionEvents.length");
    expect(app).toContain("Shared neural image · audio · video · vision pathways");
    expect(app).not.toContain("Seeded from 6 active ideas");
    expect(app).not.toContain("4 built-in baselines");
    expect(styles).not.toContain("animation: sweep");
  });

  it("shows the newest real revision and replaces it with the final artifact", () => {
    const preview = {
      schemaVersion: 1 as const,
      revision: 2,
      progress: 0.7,
      statusLabel: "Diffusion/VQ decoder revision 3/4",
      modality: "image" as const,
      stage: "diffusion-vq-decode" as const,
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,cHJldmlldw==",
      actualDecoderOutput: true as const,
      spatialResolutionReduced: false as const
    };
    const live = jobImaginationMedia(job({ preview }), "image");
    expect(live).toMatchObject({
      sourceUrl: preview.dataUrl,
      isLiveRevision: true,
      isFinal: false,
      revision: 2,
      progress: 0.7
    });
    expect(imaginationRevisionCopy(live)).toContain("Live decoder r2");

    const completed = jobImaginationMedia(job({
      state: "complete",
      progress: 1,
      preview,
      output: {
        modality: "image",
        mimeType: "image/png",
        dataUrl: "data:image/png;base64,ZmluYWw=",
        path: "/brain/artifacts/final.png"
      }
    }), "image");
    expect(completed).toMatchObject({
      sourceUrl: "data:image/png;base64,ZmluYWw=",
      finalArtifactPath: "/brain/artifacts/final.png",
      isFinal: true,
      isLiveRevision: false
    });
    expect(imaginationRevisionCopy(completed)).toBe(
      "Final artifact · replaced decoder r2"
    );
  });

  it("prefers a validated embedded WAV over a capability URL for playback", () => {
    const dataUrl = `data:audio/wav;base64,${Buffer.alloc(183_852).toString("base64")}`;
    const mediaUrl = `omni-media://artifact/${"a".repeat(48)}/${"b".repeat(64)}`;
    const presentation = jobImaginationMedia(job({
      kind: "audio",
      state: "complete",
      progress: 1,
      output: {
        modality: "audio",
        mimeType: "audio/wav",
        dataUrl,
        mediaUrl
      }
    }), "audio");

    expect(dataUrl.length).toBeGreaterThan(96 * 1024);
    expect(presentation.sourceUrl).toBe(dataUrl);
    expect(presentation.downloadUrl).toBe(mediaUrl);
  });

  it("keeps the last decoder frame visible when a large final is path-only", () => {
    const presentation = jobImaginationMedia(job({
      kind: "video",
      state: "complete",
      progress: 1,
      preview: {
        revision: 1,
        mimeType: "image/apng",
        dataUrl: "data:image/apng;base64,cHJldmlldw=="
      },
      output: {
        modality: "video",
        mimeType: "video/mp4",
        path: "/brain/artifacts/final.mp4"
      }
    }), "video");
    expect(presentation.sourceUrl).toBe("data:image/apng;base64,cHJldmlldw==");
    expect(presentation.mimeType).toBe("image/apng");
    expect(presentation.finalArtifactPath).toBe("/brain/artifacts/final.mp4");
    expect(presentation.isFinal).toBe(true);
  });

  it("uses measured seed counts and never invents a fixed idea total", () => {
    expect(imaginationSeedCopy({
      ideaSeed: { activeAssemblyCount: 3, source: "active-assemblies" }
    }, undefined, 9)).toBe("Seeded from 3 active neural assemblies");
    expect(imaginationSeedCopy({
      ideaSeed: {
        activeAssemblyCount: 0,
        source: "intrinsic-neural-cold-start"
      }
    }, undefined)).toBe("No active assemblies · intrinsic neural seed");
    expect(imaginationSeedCopy(undefined, undefined)).toBe(
      "Seeded from active neural state"
    );
  });

  it("uses final action output in the natural chat card presentation", () => {
    const event: ActionEvent = {
      id: "action-1",
      brainId: "brain-1",
      action: {
        kind: "imagine",
        source: "brain",
        toolId: "modality.imagine",
        action: "generate",
        arguments: { modality: "audio" }
      },
      state: "complete",
      createdAt: "2026-09-07T00:00:00.000Z",
      updatedAt: "2026-09-07T00:00:02.000Z",
      preview: {
        revision: 1,
        mimeType: "audio/wav",
        dataUrl: "data:audio/wav;base64,cHJldmlldw=="
      },
      execution: {
        id: "execution-1",
        toolId: "modality.imagine",
        action: "generate",
        state: "complete",
        startedAt: "2026-09-07T00:00:00.100Z",
        finishedAt: "2026-09-07T00:00:02.000Z",
        output: {
          mimeType: "audio/wav",
          dataUrl: "data:audio/wav;base64,ZmluYWw="
        }
      }
    };
    expect(actionImaginationMedia(event).sourceUrl).toBe(
      "data:audio/wav;base64,ZmluYWw="
    );
  });

  it("opens a selected-brain gallery and pages the existing artifact API", () => {
    const app = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
      "utf8"
    );
    expect(app).toContain('const [galleryOpen, setGalleryOpen] = useState(false)');
    expect(app).toContain('onClick={() => {\n              setGalleryOpen(true);');
    expect(app).toContain('role="dialog"');
    expect(app).toContain('{brain.name} imagination gallery');
    expect(app).toContain('window.omni.modality.artifacts(brain.id, artifactCursor, 48)');
    expect(app).toContain('aria-label="Close imagination gallery"');
    expect(app).toContain("<PersistedArtifactMedia artifact={artifact} />");
  });

  it("names a working WAV player without presenting AX fallback text as an error", () => {
    const presentation = persistedArtifactMediaPresentation({
      available: true,
      modality: "audio",
      mimeType: "audio/wav"
    });

    expect(presentation).toEqual({
      kind: "audio",
      ariaLabel: "Saved audio imagination player"
    });
    const app = readFileSync(
      resolve(import.meta.dirname, "../src/renderer/src/App.tsx"),
      "utf8"
    );
    expect(app).toContain("aria-label={presentation.ariaLabel}");
    expect(app).not.toContain("Unable to play media");
  });

  it("announces a video APNG as automatically looping animation", () => {
    const artifact = {
      available: true,
      modality: "video" as const,
      mimeType: "image/apng"
    } satisfies Pick<GeneratedArtifact, "available" | "mimeType" | "modality">;

    expect(persistedArtifactMediaPresentation(artifact)).toEqual({
      kind: "animated-video",
      ariaLabel: "Animated APNG video imagination; loops automatically",
      animationCopy: "Animated APNG · loops automatically"
    });
  });
});
