# Task files across warm Steer

This source slice implements task-file visibility across an actual warm Steer. It does not train, load, generate with, or establish intelligence quality for a brain.

## Owned boundary

`ChatActionController.setWarmTaskAttachmentTransfer` is private main composition. The callback runs after the actual predecessor's neural-boundary promise resolves, only when that predecessor genuinely yielded, names this exact successor, and has the same measured attention epoch. It binds normalized text or the original typed audio input SHA; renderer `replacesTurnId` alone cannot create a scope.

The existing attachment registry's SQLite database contains three prefixed tables:

- `task_continuation_scopes`: immutable original record/SHA, distinct local scope identity, current owner/turn alias, state/revision, and optional authenticated-resume receipt pointer.
- `task_continuation_files`: ordered original-file aliases and exact rekey/merge custody steps.
- `task_continuation_resumes`: immutable native-frontier proof records, separate from original input/audio identity.

Each original file reference retains its owning turn, content SHA, full public manifest, private source JSON/SHA, byte count, MIME, kind and timestamps. Private paths/capture provenance remain in main-owned storage, not chat prompt text or renderer-safe handles. This is an unsanitized operational archive, not a privacy-cleaned export.

`TaskContinuationAttachments` is a separate visibility facade. Original `TaskAttachmentStore`, `TaskAttachmentService` and exact-FD verifier semantics remain unchanged. Staging, binding and native audio source validation still use the original service. The facade:

- Selects only the yielded task's bound files plus its already-verified inherited scope. Unsent selections and unrelated chat history are excluded.
- Keeps successor-selected files independently bound to their actual successor.
- Reads inherited files through the base service with the **original owning turn**, never by rewriting descriptors, rebinding old handles or copying blobs during Steer.
- Checks actual attention epoch and persisted scope revision before and after awaited reads; cancellation, Fresh and revocation withhold bytes.
- Leaves every normal ToolExecutor permission and Ask decision in force. A scope is not permission, a microphone grant, or an automatic effect replay.
- Rejects a continued turn ID reused with a different current input SHA before native dispatch.
- Projects verified inherited IDs onto accepted/final task presentation while preserving each original handle/manifest. The default projector still rejects unrelated mixed-turn handles.

## Restart and portability

The prefixed tables travel atomically with the existing attachment-registry SQLite snapshot. Workspace validation verifies original record hashes, every source/manifest, ordered aliases and custody chains, current-owner bindings, and resume proof hashes.

Duplicate, import and recovery adoption retain immutable originals but make continuation authority dormant. Same-owner recovery does not alter source manifests or add fictitious rekey steps. No Load click, ordinary new turn or renderer-selected ID activates dormant scopes.

Main service activation happens only after it authenticates a real retained native execution. `verifiedTaskContinuationResume` additionally checks the current runtime owner, complete checkpoint/token frontier, exact input and SHA, committed accepted-input turn, and real receipt attention epoch. Inherited execution/source-turn identity is separate from the destination alias. The actual resumed proof is persisted. Fresh retires older-epoch scopes rather than reviving them.

Full merge preserves distinct source bindings as deterministic, dormant destination aliases; identical original records are not enough to deduplicate distinct destination task bindings. Existing merge attachment policy retains parent files as retired custody, so these merged scopes do not become ordinary bound-file grants. A subsequent distinct [reviewed inherited-task file reader](inherited-task-files.md) now verifies the immutable original closure and exact current-main adoption without blind rebinding. Its focused host evidence is recorded separately; native/model and mounted UI acceptance remain open.

Original accepted audio input/cursor/ledger hashes are not rewritten. File visibility does not re-admit raw microphone data or create a capture lease. Pending imported authority does not execute or replay any completed effect.

## Focused evidence

On 2026-10-13, fourteen new actual small SQLite/file and synthetic-controller cases passed in 1.076 seconds of tests (1.86 seconds total). A subsequent distinct six-case selection passed in 145 milliseconds (1.64 seconds total): two additional scope permission/audio cases plus four pure exact-frontier verifier cases. The earlier fourteen were skipped, not repeated, in that second run. These are **20 distinct host cases**, not a real neural-model test.

Covered: original descriptors/blob identity, unrelated/unsent exclusion, successor-owned files, transitive scope, lost ACK/idempotence, mismatched input rejection, genuine warm and zero-token yield, normal-completion/forged-ID refusal, Fresh and same-epoch revocation during pending reads, safe mixed-owner presentation, actual workspace export/adoption, dormant exact-resume activation, same-owner recovery, distinct merge custody, alias tamper rejection, current ToolExecutor Off policy and immutable native-audio source identity without capture regrant.

Files: `tests/taskContinuationAttachments.test.ts`, `tests/taskContinuationResume.test.ts`.

Not performed: model loading/training/generation, real microphone input, live neural task-file use after Steer, packaged-platform acceptance, CI, commit or publication. A helper/store fixture verifies the main-only resume contract; it does not substitute for a real native saved-cursor/attachment end-to-end run. The larger attachment and Send/Queue/Steer feature rows therefore remain open.
