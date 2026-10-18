import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { NativeAudioInputSummary } from "../src/renderer/src/NativeAudioInputSummary";
it("shows only real source/coverage fields, not an invented transcript or model prompt", () => {
  const html = renderToStaticMarkup(<NativeAudioInputSummary input={{ kind: "native-audio", inputSha256: "a".repeat(64), attachmentId: "audio",
    sourceBrainId: "brain", sourceTurnId: "turn", utteranceId: "utterance", sequence: 1, capturedAt: "2026-10-11T12:00:00Z",
    mimeType: "audio/wav", assemblyId: "assembly", sourceBytes: 3184, sourceSha256: "b".repeat(64), decodedWindows: 2,
    modelSamples: 8000, modelSampleRate: 16000, wholeInputTraversed: true, nativeSensory: true, transcriptUsed: false }} />);
  expect(html).toContain("Voice message"); expect(html).toContain("without a platform transcript");
  expect(html).toContain("3,184"); expect(html).toContain("bbbbbbbb");
  expect(html).not.toContain("relativePath"); expect(html).not.toContain("permissionRevision");
});
