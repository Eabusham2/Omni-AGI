import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { InheritedTaskFilesReview } from "../../src/renderer/src/InheritedTaskFilesReview";
import type { InheritedTaskFileReview } from "../../src/shared/inheritedTaskFiles";
import "../../src/renderer/src/styles.css";

const review: InheritedTaskFileReview = {
  format: "omni-inherited-task-file-review", version: 1, reviewId: "synthetic-review", brainId: "synthetic-merged", executionId: "synthetic-execution",
  rootTurnId: "root", sourceBrainId: "synthetic-parent", sourceTurnId: "original", frontier: {
    checkpointId: "synthetic-checkpoint", turnId: "root", inputSha256: "a".repeat(64), lastStreamSequence: 2,
    sampledTokens: 1, publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true }, inputSha256: "a".repeat(64),
  rootAttentionEpoch: 2, sourceAttentionEpoch: 1, ownerBindingSha256: "b".repeat(64), sourceClosureSha256: "c".repeat(64),
  permission: { toolId: "task.attachments", level: "ask", revisionSha256: "d".repeat(64) },
  files: [{ attachmentId: "synthetic-file", displayName: "The complete original task appendix with a long filename.pdf", kind: "file", mimeType: "application/pdf", bytes: 123456,
    contentSha256: "e".repeat(64), manifestSha256: "f".repeat(64), sourceSha256: "a".repeat(64), originalBinding: {
      brainId: "synthetic-parent", turnId: "original", attachmentId: "original-file", manifestSha256: "f".repeat(64), sourceSha256: "a".repeat(64) } }],
  createdAt: "2026-10-13T12:00:00Z", reviewSha256: "b".repeat(64), state: "review-required"
};
function Fixture() {
  const [paused, setPaused] = useState(false), [busy, setBusy] = useState(false), [narrow, setNarrow] = useState(false), [notice, setNotice] = useState("");
  return <main style={{ padding: 18, maxWidth: 950, margin: "0 auto", height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Existing saved-task file review · UI check</h1>
    <p>Real review component, synthetic disclosure/callback only. No brain, model, files, permission grant, adoption or resume runs.</p>
    <div className="automation-buttons">
      <button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full surface" : "320px surface"}</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      <button className="button" onClick={() => setBusy(value => !value)}>{busy ? "End synthetic verification" : "Simulate verifying"}</button>
    </div>
    <div style={{ width: narrow ? 320 : "100%", maxWidth: "100%", marginTop: 14 }}>
      <InheritedTaskFilesReview review={review} input="Compare the full supplied appendix against both drafts. Continue the same saved task without replaying completed actions."
        busy={busy} paused={paused} onApprove={() => setNotice("Synthetic click only; no file adoption or neural resume performed.")}
        onPause={() => { setPaused(true); setNotice("Synthetic task remains paused; no resume callback ran."); }}
        onReviewAgain={() => { setPaused(false); setNotice(""); }} />
      <p role="status">{notice}</p>
      <label>Existing unsent composer<textarea style={{ width: "100%", minHeight: 75 }} defaultValue="Keep this unsent draft" /></label>
      <p>Neural submissions: 0. File grants: 0.</p>
    </div>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
