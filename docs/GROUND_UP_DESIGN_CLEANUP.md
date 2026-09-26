# Ground-up OmniCortex design cleanup

Updated: 2026-09-23

> Superseded for the current goal by
> [Ground-up design correction checklist](GROUND_UP_DESIGN_CORRECTION_CHECKLIST.md).
> In particular, the older acceptance of resident FP32 master weights below
> was explicitly rejected by the user's later ternary-only correction. Keep
> this page as a dated audit record, not the active implementation spec.

This is the active, narrowed closeout requested by the user. Do **not** create
or train a new brain, run another Falcon memory proof, or begin the next
ground-up training goal until the user sends it. The old Falcon instances are
legacy test data, not evidence for the intended architecture.

## User intent checked against the full project log

At the 2026-09-23 reread checkpoint, 571 user-authored text entries in the
current project session were read in order, including repeated corrections;
the linked continuity transcripts were
also checked. This is a decision ledger, **not** a claim that the implementation
already satisfies every request:

- BitNet, snnTorch, LLaMA-like code, papers, and the original source folders are
  **research references**. Preserve those materials. A new Build must create
  one project-authored OmniCortex from random weights, never wrap, convert, or
  silently fall back to Falcon or another pretrained runtime.
- Inference uses exact ternary eligible weights; higher-precision master and
  optimizer state are learning machinery within that same brain, not a second
  language model. Training and conversation must change genuine neural state.
- An ordinary utterance or multimodal experience enters transient attention,
  working/recurrent activity, a fast whole-experience/temporal pattern,
  adaptive assemblies and synapses, and later weighted slow replay. These
  states overlap rather than forming a rigid promotion ladder; weak activity
  can lose access and recur, and generated replies do not teach themselves
  without independent evidence. No `remember this` grammar, fact parser, cue-to-answer
  table, source-text prompt injection, or automatic verbatim answer replay may
  stand in for learning. Reuse, salience, interference, recurrence, stability,
  and decay should affect accessibility continuously; the UI must not present
  manual consolidation or fixed stages as the model's memory mechanism.
- Tool use, Ponder, imagination, agents, and improvement are learned neural
  action choices available inside the continuous chat, not slash-command or
  keyword shortcuts. Permission enforcement and observable external effects
  stay outside the model. A tiny initial capability curriculum is not proof of
  broad fluency, useful autonomy, or human-like understanding.
- Preserve the existing good modalities, ingestion, full-dataset coverage,
  offload, copy/export, UI, and recovery work. This cleanup removes substitute
  paths surgically; it does not trigger another long Build or training run.
- The five named Falcon instances have separate explicit deletion approval,
  but their saved state is untouched until the exact deletion action. The
  user's `/Downloads/ai train` folder and research references are out of scope
  for deletion.

## Steps — 2 of 4 source-verified

1. [x] Re-read the authored project history and lock the intended design:
   one from-random OmniCortex; temporary working activity; organic settling;
   experiences changing neural synapses/vectors immediately and dense weights
   through durable replay; no answer-key memory, hidden behavioral prompt, or
   legacy model standing in for a new Build.
2. [x] Remove non-ground-up product/code compatibility and substitute paths.
   New Build and accepted `.omni` imports must require native OmniCortex.
   Remove Falcon selectors, foundation fallbacks, exact/statistical
   cue-to-answer storage, fixed keyword action overrides, and UI options for
   legacy identities. Preserve real ternary, SNN/STDP, liquid, VSA, modalities,
   data coverage, permission enforcement, origin, and recovery behavior.
3. [ ] Audit and correct the genuine future-build mechanisms. Reopened after
   finding silent host coercion of brain-selected evolution candidates. Check normal chat and every
   supported ingestion path for whole-experience activity, adaptive sparse
   neural state, working/afterimage/recurrent state, durable queued cortical
   learning, and typed learned tool/Ponder/imagination/agent selection. Any
   untrained or unavailable capability must fail honestly, not return a canned
   result or claim a learned skill from metadata alone.
4. [ ] Run focused source tests and inspect the final diff/compliance matrix;
   permanently delete only the five exact app-managed Falcon instances after
   action-time confirmation; document residual unproved capability. Do not
   run a new brain Build, long dataset training, or Falcon memory test.

Focused source-only checks on the current tree passed after the adaptive
large-data assembly, neural-action-argument, honest-preview, and distinct
transient-episode corrections. A full app Build, saved native brain training,
and live capability proof remain deferred by the user's instruction. The five
exact saved Falcon instances were revalidated read-only and remain present;
no deletion should be inferred from code removal.

## Step 3 source observations (not a trained-brain pass)

- Native creation/import guards are in the worker, config, and repository.
  The old Falcon/Starter runtime modules and cue-to-answer module are absent;
  surviving old-format field checks reject files instead of loading them.
- A normal chat turn sends current/recent tokens through the decoder, routes
  recalled distributed activity as a neural bias, and admits the full human
  utterance to the substrate and STDP state. Fast changes occur when the turn
  commits; a durable background job can later update dense cortical weights.
  The visible conversation ledger is evidence for that replay, not an
  automatic long-term text prompt.
- If a chat or feedback checkpoint fails, the worker now discards its mutable
  in-RAM object and reloads only the committed generation. Chat recovery
  reconciles the exact turn receipt and hash-chained native conversation head;
  uncommitted rows are removed. The separate append-only diagnostic event log
  may retain a failed pre-save mutation event, so diagnostics are not claimed
  to be a perfectly atomic journal.
- Continuous memory settling remeasures salience, recurrence, interference,
  reuse, stability, and decay; afterimages may cool or reactivate without
  deleting their learned assemblies. This is source behavior, not proof of
  reliable post-Fresh recall from a trained native brain.
- Effective eligible forward projections and recurrent sparse synapses use
  exact ternary values; floating master weights/optimizer moments remain
  learning state and remain resident. Packed bytes and version-bound scales
  are reused for unchanged forwards, but rows are unpacked for integer math,
  not executed by a fused 1.58-bit kernel. The runtime-card gate now detects
  unexpected bare dense linear layers as well as convolution blockers. Small
  source tests check this; no 1.58-bit resident-RAM claim or biological
  equivalence is made.
- Packed-shard cold-load verification now validates every checksum, decoded
  tensor, and coverage entry while retaining only the dynamic synapse tensor
  needed by native load. Distributed promotion's manifest-only checks retain
  none of the decoded tensors. Tiny corruption/duplicate fixtures remain green;
  this removes cumulative decoded-pack retention, not the live FP32 masters,
  full CPU core checkpoint, optimizer, or possible repair copies.
- Large-dataset ingestion visits valid records and now grows multiple neural
  fields under resource guards rather than one field per source. Related rows
  can reinforce an assembly across sources, while each row has transient
  whole-experience activity. Source tests do not prove full-dataset semantic
  retention or useful natural recall.
- Mid-stream Parquet/JSONL traversal errors and a short Parquet batch iterator
  now keep coverage incomplete; the ingestion controller gates final 100%
  progress and completion receipts before promoting a prefix-only pass. The
  desktop cursor remains paused/resumable. The same prefix-failure rule now
  covers ZIP/TAR/WebDataset inner members, Office archives, and compressed
  text; four focused archive tests include a normal 513-member fixture. The
  full user corpus remains unproved.
- Tool/action kinds and routes are trainable. A new ternary argument head can
  form a bounded typed idle query or objective from internal neural activity,
  but only after real host-confirmed training for that route; a syntax-only
  initial curriculum does not enable autonomous competence. Explicit chat
  values for other tools still use post-selection literal parsing and host
  schema/permission checks. Specific normal-chat tool selection now uses the
  same trained internal neural route as idle cognition; the text-route head is
  no longer a normal-chat fallback. The same checkpointed argument head now
  has a structural browser-operation target trained from successful host
  outcomes, separate from any selector or typed-text target. It requires a
  grounded no-step example and selected-kind example before prose can add one
  browser step; idle additionally requires grounded full arguments. Source-only
  checks cover these gates, but useful live tool behavior, natural browser
  semantics/generalization, and prompt-free action remain **open**.
- Browser automation no longer converts prose regex matches into click, type,
  press, wait, or screenshot actions. A negated phrase alone cannot create a
  browser step; whether a trained model correctly selects no-step for unseen
  negations remains unproved. Explicit structured multi-step JSON after neural
  route selection receives recursive kind/field/bound checks. A supported
  single prose step (click, press, wait, extract, or screenshot) requires a
  confident, host-grounded neural operation choice, then only a bounded
  explicit operand is copied; typing and navigation steps still require
  structured JSON. Uncertain choices leave URL-only browsing. The host does
  not silently clear typed fields or capture an unrequested final screenshot.
  Natural multi-step browser planning and live
  semantic generalization are **not** claimed.
- A structured chat evolution request can preserve an explicit neural, data,
  substrate, or architecture candidate kind rather than having the brain
  replace it with `substrate`. Untyped prose still reaches the host's
  substrate default; autonomous candidate strategy is unproved.
- Browser design preview no longer fabricates a brain reply or plasticity.
  An untrained modality pack likewise cannot claim organic imagination quality.
- Different experiences may now share a semantic assembly without collapsing
  into one working/afterimage vector. Near-identical repeats can merge; cold
  pages are inspected before removal so a distinct episode is not lost under
  disk-reserve pressure.
- Cold-page lookup now uses the existing SQLite assembly index in bounded
  parameter batches instead of scanning every stored page. Pure tests compare
  its selected IDs with the former full-scan behavior; live throughput remains
  unmeasured.
- In desktop transactional chat, current input conditions generation through
  temporary activity; fast synapses commit after successful decode and before
  turn completion, while dense cortical learning is queued. A completed reply
  alone does not prove new cortical weights or post-Fresh verbal recall.
- Durable latent replay no longer has a fixed importance cutoff that excluded
  all ordinary 0.30–0.58 document chunks. A deterministic, nonzero weighted
  admission uses current substrate/focus retention signals; the decision is
  made before lifecycle settling so a disk-reserve pause keeps the former
  mutation boundary. This is sampled replay, not proof that every passage is
  mastered or that the slow cortex has finished training.
- Finite distributed corpus training now schedules one capability rehearsal
  at the midpoint of committed waves, even when its periodic interval is not
  reached, and still performs final rehearsal before promotion. The midpoint
  receipt survives resume without replaying that event. This changes the
  source schedule, not evidence of an actual trained native model.
- Slow-training microbatches now append their already-selected latent replay
  vectors through one reserve preflight and one SQLite transaction/checkpoint
  per batch instead of per vector. Inserts roll back together; a failed
  post-commit checkpoint is recorded without reporting committed rows as
  retryable. This is source-level SSD-write reduction, not a measured long-run
  throughput or wear result.
- Replay checkpoint verification now validates every committed and pending
  SQLite row in a bounded 32-row read batch instead of materializing the full
  replay corpus in Python RAM. Checksum/high-water and corruption detection
  remain compatible. It still performs O(total replay bytes) I/O per full
  checkpoint; no long-corpus throughput result is claimed.
- Full replay iteration now pages by the monotonically increasing sequence
  key rather than an increasingly expensive SQL OFFSET. Sparse/gapped rows,
  concurrent append, and late-row corruption remain covered by pure tests;
  this is not a measured end-to-end training speed claim.
- Modality-pack checksums and snapshot checksums now stream checkpoint file
  bytes rather than loading whole packs or concatenating full tensor files in
  RAM. Pure fixtures confirm the old checksum formula byte-for-byte; tensor
  loading and snapshot copying remain separate resource paths.
- Portable export now scans selected current/origin substrate JSON shards for
  recognized credentials or private paths before publication and rejects a
  contaminated archive without changing content-addressed neural records.
  Replay SQLite is structurally and checksum-validated before export and
  import promotion, rather than accepting any file named `replay.sqlite3`.
  Export stages current and origin SQLite through a WAL-aware snapshot before
  hashing/ZIP so a committed or pending replay row is not omitted with the
  sidecar. Focused fixtures keep writers open to verify both cases.
  Regex scanning cannot certify arbitrary binary learned weights secret-free.
- Sanitized `.omni` still omits the conversation ledger, queued chat replay,
  active source-bound ingestion cursors, and cold temporary working-memory
  pages; the exported page checkpoint is
  explicitly reset and nonempty imported claims are rejected. The format
  documentation calls out the omissions instead of describing it as an exact
  current continuation. Whether
  to add a sensitive exact-private export awaits the user's privacy choice.
- Focused constructor-free checks on the affected final source passed: 35
  neural route/schema/action tests, 16 replay-policy/batch/checkpoint/iterator
  tests, four distributed-rehearsal schedule tests, one promotion-guard
  fixture, and three streaming-hash tests. These sets overlap and are not a
  full-suite pass. Python compilation and `git diff --check` passed; the last
  focused TypeScript run passed 28 tests and typechecking.
- The subsequent portable-state slice passed 53 focused Node tests, including
  WAL-only committed/pending replay, current/origin projection, safe staging,
  replay-row checksum, and invalid-page/cursor claims. Typecheck and diff
  checks passed. These are source fixtures, not a trained-brain reload.
- The browser/evolution-action boundary passed 12 focused Python tests (with 11
  subtests); the combined neural-action slice passed 39 Python tests and 40
  subtests. Thirty-two targeted Node tests passed, including five
  mocked-Electron browser checks for no implicit final screenshot, explicit
  clearing, one requested screenshot artifact, and malformed-step rejection.
  Typecheck and diff checks passed. A real packaged browser task with a
  trained native brain remains unverified.
- A tiny grounded browser no-operation fixture updates real checkpointed
  `ActionArgumentHead` weights and reloads the resulting weights and route
  marker. This is parameter-training evidence for that head only, not proof
  of whole-brain semantic recall or browser competence.
- Two older state-offload fixtures stop before paging because they omit the
  now-required native-ground-up initialization flag. A separate older
  constructor-based lifecycle suite was invoked once in error and paused at
  the host's real disk reserve. None is counted as a passing regression; no
  reserve was weakened to force them through.
- The stale preview E2E source was changed to assert engine-unavailable truth;
  Queue/Steer receipts are asserted in the real Electron E2E instead. Both
  changed specs typecheck, but neither was run because this cleanup does not
  authorize an app or brain Build. A later E2E run remains required.

## Explicit boundary

This cleanup cannot prove intelligence, fluent conversation, reliable
post-Fresh neural recall, media quality, or autonomous self-improvement for a
new ground-up brain without building and training one. Those acceptance tests
belong to the next goal, only when the user requests it. Do not mark those
capabilities as passed from source code, small fixtures, or legacy Falcon
history.
