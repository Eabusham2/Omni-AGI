import type { InheritedTaskFileReview } from "../../shared/inheritedTaskFiles";

/** A file-only review, not permission to inherit the parent's tools or grants. */
export function InheritedTaskFilesReview({ review, input, busy, error, paused, onApprove, onPause, onReviewAgain }: {
  review: InheritedTaskFileReview; input: string; busy: boolean; error?: string; paused: boolean;
  onApprove(): void; onPause(): void; onReviewAgain(): void;
}) {
  return <section className="plugin-approval-review" aria-label="Review files for saved merged task">
    <strong>{paused ? "Saved task remains paused" : "Review files before continuing the saved task"}</strong>
    <p>Only these saved copies become available to this task after approval, even with file access Off. Original computer paths stay blocked; parent credentials and device grants are not inherited.</p>
    <details><summary>Saved task</summary><pre tabIndex={0} style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: 160, overflow: "auto" }}>{input}</pre></details>
    <ul style={{ maxHeight: 180, overflow: "auto", overflowWrap: "anywhere", paddingInlineStart: 20 }}>
      {review.files.map(file => <li key={file.attachmentId}><strong>{file.displayName}</strong>
        <small> · {file.bytes.toLocaleString()} bytes · {file.mimeType}</small>
        <details><summary>Exact file identity</summary><code>{file.contentSha256}</code></details>
      </li>)}
    </ul>
    <details><summary>Review identity and current access</summary>
      <p>File access: {review.permission.level}</p><code style={{ overflowWrap: "anywhere" }}>{review.reviewSha256}</code>
    </details>
    <div className="plugin-manager__actions">
      {paused ? <button type="button" className="button" disabled={busy} onClick={onReviewAgain}>Review again</button> : <>
        <button type="button" className="button button--primary" disabled={busy} onClick={onApprove}>{busy ? "Verifying exact files…" : "Use these files and continue"}</button>
        <button type="button" className="button" disabled={busy} onClick={onPause}>Leave paused</button>
      </>}
    </div>
    {error ? <p role="alert">{error}</p> : null}
  </section>;
}
