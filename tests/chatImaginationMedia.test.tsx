import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ActionEvent } from "../src/shared/types";
import {
  ChatImaginationMedia,
  chatImaginationMediaLabel,
  chatImaginationRevisionLabel,
  chatImaginationRetentionCopy,
  chatImaginationUnavailableCopy
} from "../src/renderer/src/ChatImaginationMedia";
import { actionImaginationMedia } from "../src/renderer/src/imaginationPresentation";

function imaginationEvent(overrides: Partial<ActionEvent> = {}): ActionEvent {
  return {
    id: "imagination-action",
    brainId: "brain-1",
    action: {
      kind: "imagine",
      source: "organic",
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image" }
    },
    state: "complete",
    createdAt: "2026-10-13T00:00:00.000Z",
    updatedAt: "2026-10-13T00:00:01.000Z",
    execution: {
      id: "execution-1",
      toolId: "modality.imagine",
      action: "generate",
      state: "complete",
      startedAt: "2026-10-13T00:00:00.100Z",
      finishedAt: "2026-10-13T00:00:01.000Z",
      output: {
        modality: "image",
        mimeType: "image/png",
        dataUrl: "data:image/png;base64,aW1hZ2U=",
        path: "/brain/artifacts/final.png"
      }
    },
    ...overrides
  };
}

describe("visible chat imagination media", () => {
  it("mounts validated media before the still-collapsed audit details", () => {
    const markup = renderToStaticMarkup(<ChatImaginationMedia event={imaginationEvent()} />);
    expect(markup).toContain('class="chat-imagination-media chat-imagination-media--timeline"');
    expect(markup).toContain("Final artifact");
    expect(markup).toContain("Final artifact available");
    expect(markup).not.toContain("/brain/artifacts/final.png");
    expect(markup).toContain("<img");

    const source = readFileSync(resolve(import.meta.dirname, "../src/renderer/src/App.tsx"), "utf8");
    const card = source.slice(source.indexOf("function ChatActionCard"), source.indexOf("function AttachmentLearningCard"));
    expect(card.indexOf("<ChatImaginationMedia event={event} />")).toBeLessThan(card.indexOf("<details"));
    expect(card).not.toContain("<details open");
    expect(card.slice(card.indexOf("<details"))).not.toMatch(/<(?:img|audio|video)\b/);
  });

  it("renders video APNG as visible animation rather than calling it a still image", () => {
    const event = imaginationEvent({
      state: "running",
      execution: undefined,
      action: {
        kind: "imagine",
        source: "organic",
        toolId: "modality.imagine",
        action: "generate",
        arguments: { modality: "video" }
      },
      preview: {
        revision: 3,
        progress: 0.75,
        statusLabel: "Temporal decoder revision",
        modality: "video",
        mimeType: "image/apng",
        dataUrl: "data:image/apng;base64,YW5pbWF0aW9u"
      }
    });
    const presentation = actionImaginationMedia(event);
    expect(chatImaginationMediaLabel(presentation)).toMatchObject({
      title: "Video imagination",
      animatedApng: true
    });
    const markup = renderToStaticMarkup(<ChatImaginationMedia event={event} />);
    expect(markup).toContain("Animated APNG video imagination; loops automatically");
    expect(markup).toContain("Animated APNG video · loops automatically");
    expect(markup).toContain("<img");
    expect(markup).not.toContain("<video");
  });

  it("keeps audio user-controlled and preserves path-only and playback-failure fallbacks", () => {
    const audio = imaginationEvent({ execution: {
      id: "execution-audio", toolId: "modality.imagine", action: "generate", state: "complete",
      startedAt: "2026-10-13T00:00:00.100Z", finishedAt: "2026-10-13T00:00:01.000Z",
      output: { modality: "audio", mimeType: "audio/wav", dataUrl: "data:audio/wav;base64,YXVkaW8=" }
    }, action: { kind: "imagine", source: "human", toolId: "modality.imagine", action: "generate", arguments: { modality: "audio" } } });
    const audioMarkup = renderToStaticMarkup(<ChatImaginationMedia event={audio} />);
    expect(audioMarkup).toContain("<audio"); expect(audioMarkup).toContain('controls=""');
    expect(audioMarkup).toContain('preload="metadata"'); expect(audioMarkup).not.toContain("autoplay");

    const pathOnly = imaginationEvent({ execution: {
      id: "execution-video", toolId: "modality.imagine", action: "generate", state: "complete",
      startedAt: "2026-10-13T00:00:00.100Z", finishedAt: "2026-10-13T00:00:01.000Z",
      output: { modality: "video", mimeType: "video/mp4", path: "/brain/artifacts/final.mp4" }
    }, action: { kind: "imagine", source: "brain", toolId: "modality.imagine", action: "generate", arguments: { modality: "video" } } });
    const presentation = actionImaginationMedia(pathOnly);
    expect(chatImaginationUnavailableCopy(presentation, false)).toContain("Inline preview is unavailable");
    expect(chatImaginationUnavailableCopy(presentation, true)).toContain("artifact remains available");
    expect(renderToStaticMarkup(<ChatImaginationMedia event={pathOnly} />)).toContain("Final artifact available");
  });

  it("does not promote ordinary tool media output into the imagination surface", () => {
    const event = imaginationEvent({ action: { kind: "tool", source: "brain", toolId: "web.fetch", action: "fetch", arguments: {} } });
    expect(renderToStaticMarkup(<ChatImaginationMedia event={event} />)).toBe("");
  });

  it("labels a terminated retained frame as last delivered rather than live", () => {
    for (const state of ["stopped", "failed"] as const) {
      const event = imaginationEvent({ state, execution: undefined, preview: {
        revision: 7, progress: 0.6, statusLabel: "Temporal decoder revision",
        modality: "video", mimeType: "image/apng", dataUrl: "data:image/apng;base64,bGFzdA=="
      }, action: { kind: "imagine", source: "organic", toolId: "modality.imagine", action: "generate", arguments: { modality: "video" } } });
      const copy = chatImaginationRevisionLabel(state, actionImaginationMedia(event));
      expect(copy).toContain(`${state === "failed" ? "Failed" : "Stopped"} · last delivered decoder r7`);
      expect(copy).not.toContain("Live decoder");
      expect(renderToStaticMarkup(<ChatImaginationMedia event={event} />)).toContain("last delivered decoder r7");
    }
  });

  it("keeps ordinary tool media inside audit placement and reports terminal media state honestly", () => {
    const tool = imaginationEvent({ action: { kind: "tool", source: "brain", toolId: "web.fetch", action: "fetch", arguments: {} } });
    const audited = renderToStaticMarkup(<ChatImaginationMedia event={tool} placement="audit" />);
    expect(audited).toContain("Image artifact"); expect(audited).toContain("<img");
    expect(renderToStaticMarkup(<ChatImaginationMedia event={tool} />)).toBe("");

    const stopped = imaginationEvent({ state: "stopped", execution: undefined, preview: {
      revision: 8, statusLabel: "Last renderer frame", modality: "image", mimeType: "image/png", dataUrl: "data:image/png;base64,bGFzdA=="
    } });
    expect(renderToStaticMarkup(<ChatImaginationMedia event={stopped} />)).toContain('data-imagination-state="stopped"');
  });

  it("keeps managed paths in expanded audit while basic-chat captions stay readable", () => {
    const event = imaginationEvent();
    const timeline = renderToStaticMarkup(<ChatImaginationMedia event={event} />);
    const auditedTool = renderToStaticMarkup(<ChatImaginationMedia event={{ ...event,
      action: { kind: "tool", source: "brain", toolId: "code.execute", action: "run", arguments: {} } }} placement="audit" />);
    expect(timeline).toContain("Final artifact available"); expect(timeline).not.toContain("/brain/artifacts/final.png");
    expect(auditedTool).toContain("/brain/artifacts/final.png");
    const styles = readFileSync(resolve(import.meta.dirname, "../src/renderer/src/styles.css"), "utf8");
    expect(styles).toMatch(/\.chat-imagination-media__artifact\s*\{[^}]*font-size:\s*12px/s);
  });

  it("labels a terminal ephemeral final from its authoritative retention marker", () => {
    const mediaUrl = `omni-media://artifact/${"a".repeat(48)}/${"b".repeat(64)}`;
    const event = imaginationEvent({ action: { kind: "imagine", source: "organic", toolId: "modality.imagine", action: "generate", arguments: { modality: "image" } },
      execution: { id: "execution-ephemeral", toolId: "modality.imagine", action: "generate", state: "complete",
        startedAt: "2026-10-13T00:00:00.100Z", finishedAt: "2026-10-13T00:00:01.000Z",
        output: { modality: "image", mimeType: "image/png", mediaUrl, artifactRetention: "ephemeral-spontaneous" } } });
    const presentation = actionImaginationMedia(event), markup = renderToStaticMarkup(<ChatImaginationMedia event={event} />);
    expect(presentation).toMatchObject({ sourceUrl: mediaUrl, hasFinalArtifact: true, artifactRetention: "ephemeral-spontaneous" });
    expect(chatImaginationRetentionCopy(event.state, presentation)).toBe("Temporary spontaneous output · Use Record to keep it");
    expect(markup).toContain(mediaUrl); expect(markup).toContain("Temporary spontaneous output"); expect(markup).toContain("Use Record to keep it");
    expect(markup).not.toContain("Saved in Imagination");
  });
});
