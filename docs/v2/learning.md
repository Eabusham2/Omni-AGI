# Durable learning and evidence workspaces

## Implemented contract

`src/shared/learning.ts`, `learningTaskStore.ts` and `learningCoordinator.ts`
provide typed, owner-bound Specific Learn and selected-claim Fact Check jobs.
The preload/IPC and `ToolExecutor` expose start, status, pause, resume, stop,
steer, record and explicit correction acceptance. A task can use pasted text,
host-selected file/folder manifests and attributed web sources. A generated
filesystem path is not a file grant. Dataset labels use verified source names
and full file counts. UI initiation remains subject to the instance's Off
setting; autonomous calls also pass the normal tool approval pipeline.

SQLite records operational jobs, evidence and receipts, not a cue/answer neural
memory. Source ingestion uses the existing native learning path. An empty
learned Learn action binds the actual admitted turn and its existing replay
job, and distinguishes retained activity, queued replay and committed replay.
It never manufactures an empty successful learning job. Source receipts and
finished progress do not establish comprehension or recall quality.

The controller binds native task turns to exact task/claim identities.
User-requested task data is visible conversation input. It is not a hidden
behavioral prompt, a retrieved answer, or a publisher-truth assertion.
Actual terminal web search/fetch results retain URL, hash, tool receipt,
declared date evidence and the applicable scope revision. Repeated captures
of identical content do not create a new semantic evidence item or an endless
assessment loop; their additional receipts remain in the operational event
ledger. A captured result does not itself become a truth finding. Only an
actual authorized record operation creates a model finding, labelled as an
evidence-based self-report. Unresolved claims remain unresolved.

`webLearningPolicy.ts` enforces domain/date scope before dispatch, on fetch
redirects and browser requests, and when committing evidence. Unknown dates
are not invented. User steering updates the live policy; generated dependent
jobs inherit it. A suspended task retains its original admitted policy hash,
so finishing an older native turn cannot claim newer instructions complete.
Conflicting findings are retained. A correction enters ordinary learning only
after explicit user acceptance.

## Capability learning

`learningCapabilityPack.ts` defines a separate versioned project-authored
schema/example pack. Its 17 records cover the new learning/checking actions
and the current-experience Learn route. Declared placeholders are not treated
as observed successful host outcomes. Frozen origin-v3 examples, declarations
and receipts are unchanged.

`supplementary_capabilities.py` updates actual packed native action, route and
argument heads. It uses `CapabilityRollbackSnapshot`, admitted numeric-basis
retention anchors, per-record losses and grounded-owner protection. Decline,
cancellation or save failure restores the candidate. There is no floating
learned master, reward model or readiness quiz. Initial Build invokes this
supplementary head teaching after selected initial sources and before opening
chat. An honestly declined candidate retains prior heads and does not block
chat on answering a quiz. Text-only curriculum exposure is not labelled
successful action-head training. Learned use remains unqualified.

## Unload, restart and portability

The task's native turn/input hash and source cursor are durable. Source work
pauses at a committed record/page boundary. A native task already parked by
Unload is not awaited again by the learning boundary, which would deadlock
the same Load-dependent turn promise. Explicit Load resumes that promise or
reattaches the exact saved input/frontier under its task/claim binding. User
Pause/Stop does not gain automatic restart authority. Quit first requires an
acknowledged native unload, then closes task ledgers, then tears down the
worker. A named unsupported boundary prevents an untruthful saved-state quit.

Recovery detaches the old host attachment after acknowledged Unload, without
native cancellation. Learning then closes its cached SQLite handle before
workspace replacement. Question timers and pending delivery are drained
before replacing their records. This does not establish that an OS cannot
terminate the application externally.

`learningWorkspacePortability.ts` packages task/source/question history and
exact selected source bytes using checked content-addressed blobs. Mutable
SQLite snapshots include committed WAL pages; immutable archives retain their
exact bytes. Fork, all export modes, import and recovery include this workspace.
Copies are dormant, owner-rebound and carry no inherited credential/effect
authority. Imported canonical source paths do not become external filesystem
grants. Missing original bytes are reported, not silently skipped. An origin
export with a newer learned prefix preserves the task but blocks its cursor
when the selected neural generation does not match.

`source_relocation.py` verifies byte-identical owned files and rebinds native
source identities while preserving committed record/window position and the
learning schedule. It does not replay old records. `fullMergeLearningWorkspace.ts`
projects parent tasks, manifests, questions and source/crawl cursors into the
new owner with namespaced operational IDs and preserved native transaction
IDs. Exact parent-turn IDs bind to the declared regional execution queue;
missing frontiers are explicitly blocked rather than restarted. Original
parent workspaces remain in their complete snapshots. Resumed generated task
IDs can resolve only through the controller's matching ancestral alias.

## Recorded scoped evidence

- `tests/learningCoordinator.test.ts`: source idempotency/restart, retained
  cursors, serialized work, native unload/Load, independent selected claims,
  explicit corrections and steering. Twelve cases passed in the October 11 run,
  including direct-UI Off enforcement, direct native Resume's single actual
  task receipt, and detaching a completed reply parked at the unload gate.
- `tests/learningShutdown.test.ts`: two cases passed; checkpoint acknowledgement
  precedes teardown and unsupported boundaries refuse teardown.
- `tests/learningWorkspacePortability.test.ts`: four cases passed; committed
  WAL tasks and exact sources survive two owner rebinds, web leases/frontiers
  rekey, mismatched substrate generations block even with identical core bytes,
  and two-parent projection preserves source bytes and exact declared
  native-turn aliases. This uses synthetic queue metadata, not a model-quality
  or actual inherited-neural-execution proof. Interrupted inventories cannot
  masquerade as complete empty workspaces; merge retry rebuilds only that
  private, uncommitted output.
- Four newly affected repository cases passed: fork isolation, recovery,
  unsanitized all-mode export/import, and inert ancestral intent retention.
- Nine Build-coordinator and seven full-merge-transaction cases passed after
  integrating the new teaching phase and complete parent workspace inventory.
- Forty affected dataset/crawl cases, eleven question cases, and five question
  portability cases passed. These are host/source fixtures, not broad trained
  brain comprehension tests.
- `engine/tests/test_source_relocation.py`: one real tiny native ingestion
  cursor commit/rebind/save/reload case passed in 33.050 seconds. It verifies
  zero record replay and no parameter delta during source relocation, plus
  corrupt replacement rejection. No initial curriculum/corpus was run.
- Earlier unchanged component evidence: five supplementary packed-head cases
  passed in 1.095 seconds (update, retention rollback, cancellation/save failure,
  grounded protection and malformed declaration). The actual 17-record pack
  hash/schema passed cross-language validation. Five tool-policy/correlation
  and two capability-pack/terminal-receipt cases passed.

## Acceptance still distinct

The lifecycle lane owns the actual inherited regional native/controller
continuation proof and monotonic completed external-effect recovery. The
learning projection test does not substitute for that proof. Selecting an
older neural checkpoint must not pretend that newer completed result-learning
receipts are present in those older weights, nor repeat an external action.
Broader learned comprehension, natural tool use, cleared-context recall,
external-provider interoperability and full-folder training remain separate
acceptance work. No CI, release or long user training was performed by this
implementation lane.
