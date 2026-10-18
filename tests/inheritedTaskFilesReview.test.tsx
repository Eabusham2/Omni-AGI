import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { InheritedTaskFilesReview } from "../src/renderer/src/InheritedTaskFilesReview";
import type { InheritedTaskFileReview } from "../src/shared/inheritedTaskFiles";

const review: InheritedTaskFileReview = {
  format: "omni-inherited-task-file-review", version: 1, reviewId: "review", brainId: "merged", executionId: "execution",
  rootTurnId: "root", sourceBrainId: "parent", sourceTurnId: "original", frontier: {
    checkpointId: "checkpoint", turnId: "root", inputSha256: "a".repeat(64), lastStreamSequence: 2,
    sampledTokens: 1, publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true
  }, inputSha256: "a".repeat(64),
  rootAttentionEpoch: 2, sourceAttentionEpoch: 1, ownerBindingSha256: "b".repeat(64), sourceClosureSha256: "c".repeat(64),
  permission: { toolId: "task.attachments", level: "ask", revisionSha256: "d".repeat(64) },
  files: [{ attachmentId: "file", displayName: "Original task appendix.pdf", kind: "file", mimeType: "application/pdf", bytes: 1234,
    contentSha256: "e".repeat(64), manifestSha256: "f".repeat(64), sourceSha256: "a".repeat(64), originalBinding: {
      brainId: "parent", turnId: "original", attachmentId: "source-file", manifestSha256: "f".repeat(64), sourceSha256: "a".repeat(64) } }],
  createdAt: "2026-10-13T12:00:00Z", reviewSha256: "b".repeat(64), state: "review-required"
};
const props = { review, input: "Compare the supplied appendix", busy: false, paused: false,
  onApprove: () => {}, onPause: () => {}, onReviewAgain: () => {} };
describe("exact merged-task file review presentation", () => {
  it("shows actual task/files and separates file adoption from parent grants", () => {
    const html = renderToStaticMarkup(<InheritedTaskFilesReview {...props} />);
    expect(html).toContain(props.input); expect(html).toContain(review.files[0]!.displayName);
    expect(html).toContain(review.files[0]!.contentSha256); expect(html).toContain("parent credentials and device grants are not inherited");
    expect(html).toContain("Use these files and continue"); expect(html).toContain("Leave paused");
  });
  it("allows exact saved-copy approval while Off, but not after leaving paused", () => {
    const off = renderToStaticMarkup(<InheritedTaskFilesReview {...props} review={{ ...review, permission: { ...review.permission, level: "off" } }} />);
    expect(off).not.toMatch(/disabled=""[^>]*>Use these files and continue/);
    expect(off).toContain("Original computer paths stay blocked");
    const paused = renderToStaticMarkup(<InheritedTaskFilesReview {...props} paused />);
    expect(paused).toContain("Saved task remains paused"); expect(paused).toContain("Review again");
    expect(paused).not.toContain("Use these files and continue");
  });
});
