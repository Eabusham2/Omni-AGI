import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { InheritedTaskFilesReview } from "../../src/renderer/src/InheritedTaskFilesReview";
import type { InheritedTaskFileReview } from "../../src/shared/inheritedTaskFiles";
import "../../src/renderer/src/styles.css";
import "../../src/renderer/src/pluginManager.css";
import "../../src/renderer/src/maps.css";

const review: InheritedTaskFileReview = {
  format: "omni-inherited-task-file-review", version: 1, reviewId: "synthetic-review", brainId: "synthetic-owner",
  executionId: "a".repeat(64), rootTurnId: "synthetic-turn", sourceBrainId: "synthetic-parent", sourceTurnId: "source-turn",
  frontier: { checkpointId: "checkpoint", turnId: "synthetic-turn", inputSha256: "b".repeat(64), lastStreamSequence: 2,
    sampledTokens: 1, publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true },
  inputSha256: "b".repeat(64), rootAttentionEpoch: 2, sourceAttentionEpoch: 1, ownerBindingSha256: "c".repeat(64),
  sourceClosureSha256: "d".repeat(64), permission: { toolId: "task.attachments", level: "off", revisionSha256: "e".repeat(64) },
  files: [{ attachmentId: "synthetic-file", displayName: "Retained research appendix with a long descriptive filename for wrapping checks.pdf",
    kind: "file", mimeType: "application/pdf", bytes: 1234567, contentSha256: "f".repeat(64), manifestSha256: "a".repeat(64),
    sourceSha256: "b".repeat(64), originalBinding: { brainId: "synthetic-parent", turnId: "source-turn", attachmentId: "source-file",
      manifestSha256: "a".repeat(64), sourceSha256: "b".repeat(64) } }],
  createdAt: "2026-10-17T00:00:00Z", reviewSha256: "a".repeat(64), state: "review-required"
};
function Fixture() {
  const [compact, setCompact] = useState(false), [dark, setDark] = useState(false), [busy, setBusy] = useState(false),
    [paused, setPaused] = useState(false), [receipt, setReceipt] = useState("No simulated approval");
  return <main style={{ maxWidth: compact ? 360 : 880, padding: 16, margin: "0 auto", minWidth: 0, boxSizing: "border-box" }}>
    <h1>Saved file review</h1><p>UI-only fixture. No files are opened, permissions granted, or brains loaded.</p>
    <div className="map-connections__actions">
      <button className="button" onClick={() => { setDark(!dark); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setCompact(!compact)}>Compact content</button>
      <button className="button" onClick={() => setBusy(!busy)}>Toggle busy</button>
    </div>
    <InheritedTaskFilesReview review={review} input={"Compare the retained appendix with the earlier report. ".repeat(8)}
      busy={busy} paused={paused} onApprove={() => setReceipt("Simulated explicit saved-copy approval only")}
      onPause={() => setPaused(true)} onReviewAgain={() => setPaused(false)} />
    <p role="status">{receipt}</p>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
