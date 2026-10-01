# Agreement implementation closeout

This continues the chronological comparison in
[AGREEMENT_REMAINDER_2026_09_30.md](AGREEMENT_REMAINDER_2026_09_30.md).
The user then requested implementation of the surviving source differences.
Later decisions remain authoritative; canceled settings and closed historical
tests are not restored. This document records implementation and scoped code
verification, not a new trained-brain qualification.

The [current full agreement ledger](CURRENT_AGREEMENT_LEDGER.md) reconciles the
whole grouped inventory, canceled choices, historical bug closures and deferred
qualification, rather than only R1–R12. Its bounded follow-through records the
additional distributed rehearsal, packed-change reporting and agent fixes.

## Implemented corrections

- [x] **R1: ongoing input and memory handoff.** Received input now learns into
  packed assemblies/STDP before reply decoding and has its own atomic observed-
  input receipt. Stop/failure preserves an admitted human entry without an
  invented assistant message. Exact retries reuse the receipt. Older raw activity
  is protected while its cortical work remains pending. Cortical replay commits
  one exhaustive UTF-8 window at a time, with causal overlap, resizeable physical
  windows, restart cursors, fair service between ongoing chats and idle work.
  Paused dataset record/window positions remain fixed while separately observed
  experiences change neural state. Source: `brain.py`, `online_replay.py`,
  `recent_token_activity.py`, `chatInputReceipt.ts`, worker/controller/presentation.
- [x] **R2: actual action results survive failures.** Exact observed outcomes
  are spooled and learned through permanent native ingestion receipts. Stop,
  lost ACK and post-effect audit errors cannot cause tool execution to repeat.
  Failed dispatched attempts can inform an unfinished reply and later memory;
  they do not become positive route-success examples. No-dispatch denials do
  not pretend an external effect happened. Dormant identities stay dormant.
  Source: `completedActionEvidence.ts`, `brainService.ts`, `toolExecutor.ts`,
  `chatActionController.ts`, `chat_tool_observation.py`, worker/native receipt.
- [x] **R3: independent media lifetime.** Committed text returns without waiting
  for an image/video/audio preview. Unfinished artifact claims return not-ready
  instead of blocking the serial neural dispatcher. Source: `worker.py` and
  the host artifact-status/control transport.
- [x] **R4: Ask imagination during writing.** Exact permission plus durable
  pre-effect intent authorizes a flags-only warm control. The chat thread starts
  its owned snapshot at a safe boundary; the stdin reader never touches neural
  state. Source: worker, supervisor, tool executor and action controller.
- [x] **R5: natural and UI geometry access.** Learned schemas, exact forwarding
  and the evolution form expose supported width/head changes and registered
  real holdouts. Source: `brainService.ts`, `chatActionController.ts`,
  `EvolutionWorkspace.tsx`, `evolutionView.ts`, controller/worker.
- [x] **R6: supported improvement and organic recursion.** Protected, streamed
  paired observations for every native candidate require evidence of benefit in an agreed objective with
  other domains preserved. Ties/insufficient observations are not benefit.
  Promotion rechecks the actual paired files. Real success/cost measurements
  permit parent/child process comparison; no meta-improvement is fabricated.
  A promoted experiment is an observed experience followed by ordinary idle
  reassessment: the neural head chooses new legal work or none. The host no
  longer repeats a stale absolute target, doubles dimensions, or silently
  substitutes expert growth. Source: `paired_geometry_statistics.py`, native
  evaluator/evolution, controller and `index.ts`. Source candidates separately
  require a demonstrated repaired immutable functional check, with the other
  required checks preserved. A nonempty all-green diff or a single faster timing
  is not source benefit. This deterministic check is not labeled neural quality
  or statistical speed improvement. Native evaluator/evidence source is also
  protected from candidate edits by the hash-bound evaluator policy.
- [x] **R7: middle capability rehearsal for long single records.** Actual
  committed in-record progress can trigger middle rehearsal when the finite
  record-count midpoint is absent. Resume state prevents repeat rehearsal.
  Source: `online_replay.py`, ingestion checkpoints in `brain.py`, and the
  distributed committed-window rehearsal/cursor publication follow-through.
- [x] **R8: owned evaluation storage transactions.** Copies reserve aggregate
  spill before writing, retain committed files and clean uncommitted copies
  after quota/hash/cancel/save failures. Source: `registered_geometry_holdouts.py`
  and exact snapshot resource accounting.
- [x] **R9: complete portable continuation.** New safe internal names preserve
  original provenance. Existing nonportable names use deterministic physical
  aliases while original logical manifests/origin hashes remain unchanged.
  Fork/recovery/export/import preserve pending action evidence, native evaluator
  score/baseline/candidate files and inert evolution archives. Imports do not
  authorize actions or automatically run dormant learning.
- [x] **R10: truthful activity graphics.** Fixed Library/header bars were
  removed; queried counters and measured map activity remain. Source: `App.tsx`
  and its appearance CSS.
- [x] **R11: block-sparse recurrent state.** Fresh recurrence allocates no
  logical full-matrix buffers. Dynamic blocks hold actual ternary/control state,
  page cold storage and preserve precise STDP, decay and rollback. Dense native
  state has byte-exact migration proofs; cursor bindings are verified in the
  original layout before the same cursor is rebound. Canonical sparse ordering
  preserves replay/load seals. Normal integrity/export visits allocated state,
  not every implicit pair. Fresh planning/inventory reflects sparse storage;
  original native descriptor hashes remain supported. Source: sparse router,
  paging, packing/diagnostics, native inventory and brain load/budget hooks.
- [x] **R12: the user's selected allocation-budget behavior.** The actual reply
  chooses reserve/monitor/reclaim/spill/pause, with Minecraft-like allocation.
  The selected amount is a persistent budget filled on demand, not physical
  page pinning. Fresh admission and active-operation sampling check actual app
  family usage, with reclaim and recoverable pauses at safe boundaries. Existing
  OS containment is reported only when verified; universal hard RSS is not
  claimed. Source: `managed_process_memory.py`, `offload.py`, worker/brain guards.
- [x] **Own experiences participate.** The all-experience intent now includes
  its own observed speech and performed Ponder activity with self provenance.
  Speech follows fast learning and resumable cortical replay. Ponder's actual
  neural working mixture enters shared assemblies/STDP and latent replay;
  no synthetic thought text or independent factual endorsement is invented.
  The assistant-created unconditional exclusion is removed.
- [x] **Planned discrete audio codes.** Actual residual-VQ A/B IDs train the
  existing ternary generator through masked code prediction and drive seeded
  progressive waveform generation. The useful continuous residual is retained;
  no external audio model or new full floating learned master is installed.
  Source: `modalities.py`.

These corrections preserve one own neural identity, mandatory packed ternary
learned weights, no answer-key/sequence replay, no hidden behavioral prompt,
initial training before chat without a quiz, automatic video runtime, whole-data
coverage, user permissions, the largest shared storage pool and unsanitized
saved content. They do not restore Starter/Blank/personality choices or manual
consolidation controls.

## Verification and remaining qualification boundary

Verification is limited to changed code, protocols, pure math, storage and
controller fixtures. No app, CI, packaging or release run is requested here.
The source walkthrough and chronological comparison are complete; a source
path/test result is not a biological or intelligence claim.

Completed focused checks:

- 50 Node controller/protocol/arithmetic fixtures passed across nine files:
  completed action evidence and tool transport, received chat input, saved
  evolution continuation, geometry surface, failed tool observations, and
  sparse-aware native architecture arithmetic.
- 78 Python constructor-free primitive/storage fixtures passed: received input,
  exhaustive replay windows, action-result receipts, sparse router state/replay,
  audio-code protocol, geometry storage/statistics, permissioned inline control,
  failed tool observations, and managed RAM ceilings.
- Three additional constructor-free worker-recovery fixtures passed. They verify
  that an admitted human checkpoint survives unfinished generation, without
  fabricating assistant text or repeating input learning.
- Seven final source-improvement policy/controller fixtures passed, including
  repaired-check gain, all-green/no-gain rejection, exact approval/rollback and
  evaluator drift. Thirteen unrelated controller cases were deliberately not
  rerun. This run contains the four source-policy fixtures; earlier overlapping
  runs are not added again.
- The universal native-promotion follow-through passed three new constructor-
  free Python fixtures and 14 selected Node surface/controller fixtures. It
  covers all non-geometry native routes, missing registration before loading,
  real protected storage and tampered public-promotion refusal. The Node run
  overlaps earlier surface checks and is not added to a claimed unique total.
- Both TypeScript projects passed after the UI/evaluation follow-through.
  Affected Python compilation and whitespace checks also passed.
- The later whole-inventory follow-through passed three constructor-free
  distributed-window storage/controller checks, two packed reporting checks
  and five new agent host-stub checks. This closes distributed single-record
  middle rehearsal, truthful packed module reports and ordinary permissioned
  child actions without synthetic answer/count/task clipping. Existing tests
  and deferred native qualification were not restarted.

These counts are separate selected runs, not a full-suite or trained-brain
result. Earlier overlapping checks are not added again to inflate coverage.
The changed React controls were reviewed with the functional-update/transient-
ref checklist. Candidate dimensions start from the identity's saved architecture;
new fields use theme colors, wrapping and visible keyboard focus instead of
unstyled nested form controls.

One agent accidentally included `test_state_offload`, which constructs temporary
brain fixtures. That command was interrupted and is not passing evidence.
The root checked that no matching test process remained. No supplied-corpus
training or new user-saved model was initiated by this closeout.

The user-deferred run still includes trained native conversation and cleared-
context recall, useful natural tools/imagination, neural speech quality,
beneficial recursive generations and physical throughput/peak measurements.
Those are not newly reopened chores or substituted by old Falcon/answer-replay
results. Source implementation completion must not be presented as their proof.
