# Remaining Live Acceptance Gates — No-Repeat Closeout

> Historical checklist, superseded on 2026-09-22 by
> [Ground-up OmniCortex design cleanup](GROUND_UP_DESIGN_CLEANUP.md). The user
> explicitly deferred new brain Builds/training and stopped Falcon memory
> testing. Do not execute the gates below during the current cleanup goal.

Updated: 2026-09-22

## Current correction steps — 1 of 4 proved on the saved brain

1. [ ] **Real weight learning:** complete a chat-derived slow training job and
   verify a module-only weight checksum changes, `training_steps` increases,
   and the change survives reload. At last inspection the saved instance had
   eight pending jobs and zero completed; the prior preflight failed its 20 GiB
   reserve. After the approved cache cleanup, APFS reported 25.9 GB physically
   free, but replay completion has not been retested or proved.
2. [ ] **Reliable scheduling:** give queued training the idle worker, skip
   identities with no pending job, keep failures visible, and retry with
   bounded backoff. Focused source tests pass; saved-brain completion is still
   unproved.
3. [ ] **Usable recall:** default chat no longer reads exact or statistical
   `NeuralSequenceMemory` answer cues. A paraphrased, nonverbatim answer from
   changed shared cortical weights remains an open acceptance test.
4. [x] **Truthful live UI:** the refreshed app displayed fast synaptic state
   separately from trainable-module updates, 8 pending / 0 completed replay
   jobs, module weights unverified, and the actual storage-reserve error on
   2026-09-22. The separate later wording fix for a stale `Idle / ready` label
   remains source-only and does not turn the other three gates green.

These four steps are the current parameter-learning correction, not a claim
that the full Omni AGI Studio goal is complete. The app's plan-widget update
control is unavailable in this resumed task, so this checklist is the visible
workspace counter until that control returns.

Separate architecture gaps remain outside this four-check counter: a fresh
ground-up Build has only a 68-record tool/action curriculum and no broad
language pretraining. A trainable ternary head now selects generic tool/action
routes in source, while literal argument parsing and specialized
imagination/agent/evolution defaults remain; unseen-phrase reliability is not
proved. Native Ponder now has a source-level pre-speech path, but its focused
runtime tests and a live trained-brain check are deferred. External tool
results still arrive after a reply commits. None of these is counted as a
passed capability merely because the code path exists.

This is the temporary authoritative checklist for the rest of Step 12. It contains only unfinished evidence or behavior directly affected by a later fix.

Delete a gate from this file immediately when it passes. Delete this entire file when no gates remain.

The current hands-on target is `Nova Exhaustive Proof copy` (`5565790c-1808-41e6-9d1e-1db35e17ad49`), a legacy hybrid used only as a regression fixture. It does not count as the final from-random OmniCortex architecture. Full ground-up training is explicitly deferred below.

## Mandatory no-repeat rule

Before starting **every** gate:

1. Search this document and `.runtime/live-acceptance/results.md`.
2. Search the full Codex task history and saved receipts for the exact feature, marker, and affected code path.
3. Inspect current persisted app/brain state when it can prove the result without rerunning anything.
4. If current evidence already proves the same behavior and later edits did not affect it, delete the gate from this file and do not execute it again.
5. Rerun only a failed, unproved, or directly affected path. Use a new unique marker when a changed path needs fresh evidence.
6. A gate passes on behavior and evidence, not merely because no exception appeared.
7. Neural mutations for one identity remain serialized. Independent read-only source tests may run in parallel.
8. Record every new pass, failure, cancellation, and fix in `.runtime/live-acceptance/results.md` immediately.

## Locked memory behavior

The confirmed design is already substantially implemented. The closeout must fix and verify it, not replace it with a rigid stage machine.

- Memory settling is automatic and continuous. There is no user-facing Consolidate button, mode, or command.
- The normal UI does not teach a fixed three-stage model.
- Fleeting token/sensory activity, recurrent working focus, fading scratch-like afterimages, fast episodic/temporal synapses, spreading semantic assemblies, slower cortical parameters, and cold disk-backed state overlap rather than behaving like boxes on a conveyor belt.
- Every user experience enters generic whole-experience activity and fast substrate/STDP learning. No regex, fact parser, `remember` wording, or source type decides whether it may be learned; default chat does not create a separate cue-to-answer sequence entry.
- Related co-activation reinforces a whole experience. Unrelated competition creates interference.
- Recurrence, reuse, salience, stability, prediction error, rehearsal, activation, interference, and decay continuously change accessibility and replay strength.
- Weak or redundant one-off activity may lose influence and become difficult to access. Learned structure is not arbitrarily deleted, and later related activity can reactivate and strengthen it.
- Replay gradually changes slower cortical parameters. A casual turn must not synchronously block chat on a full slow-cortex backward pass.
- Wrong, truncated, or same-turn generated output is not self-reinforced without independent evidence.
- Fresh attention clears recent tokens, active focus, scratch/afterimages, spike/liquid/recurrent activity, and temporary workspace state. It preserves learned fast episodes, assemblies, synapses, slow parameters, visible history, and immutable origin.
- Visible history, tool audits, and outward Ponder messages remain available through the permissioned history tool. They are never silently inserted into generation as hidden prompt text.
- Hardware-tested Auto/Extended/Manual context planning, suitability coloring, RAM-first hot-state residency, shared storage spill, reactivation, reserve checks, and measured slowdown warnings remain part of the product contract.

## Remaining gates

### [ ] 1. Human-like text conversation, memory transfer, and cleared-context recall

Preflight: do not repeat the old TOPAZ test. This gate is required only because the memory-transfer implementation changed.

- Start a natural conversation by mentioning one new unique experience without saying `remember`, formatting it as Q/A, or immediately asking for it back. Use a fresh marker; the older `MINT-2746` run is historical diagnostic evidence and must not be repeated as the changed-path proof.
- Continue for several coherent, unrelated turns so the experience leaves immediate focus and exercises working context, afterimages, temporal links, interference, and recurrence.
- During those turns, combine Gate 7's Queue/Steer/navigation checks rather than creating a second conversation run.
- Prove the reply becomes available after fast whole-experience learning, while slow replay is background, visible, preemptible, priority-scaled, and restart-safe.
- Verify fast substrate/STDP/afterimage changes occur without a synchronous full slow-cortex wait and without a sequence answer-key write.
- Verify weak-access fading changes influence, not stored structure, and related recurrence can warm accessibility again.
- Start Fresh attention, close the app/worker, and reopen it.
- Confirm recent token context, active focus, afterimages, liquid/recurrent state, and temporary workspace are empty while learned neural state and visible history remain.
- Ask naturally about the original experience without including the answer.
- Require the response to contain the marker.
- Require trace evidence of zero prior-dialogue prompt tokens, no source-text retrieval, no hidden behavioral prompt, no sequence-answer lookup, and neural assembly activation.

### [ ] 2. Resume the existing JSONL training manifest

Preflight: reuse the existing interrupted manifest; do not select the file again or create another duplicate manifest.

- Resume the existing `JUNIPER-8051` JSONL job with manifest ID
  `1740d0ea-e56f-450b-a63a-8a2cb3b2a5ac` through the visible Resume control.
  Do not create another manifest; the older `01499197-d4a6-4340-a322-ddc38bb9ef3c`
  attempt remains historical evidence only.
- Reach true 1/1 file and record traversal with 71/71 bytes committed.
- Verify honest phase, rate, ETA, pause/cancel semantics, and job-scoped coverage.
- Verify source-ledger addition plus parameter, synapse, concept/neuron, experience, and training-step deltas.
- Verify no skipped or double-committed record after the earlier cancellations.

### [ ] 3. Chat attachment learning

- Use the Conversation attachment picker for `.runtime/live-acceptance/chat-attachment.txt` (`ONYX-4732`).
- Verify immediate visible attachment state and one neural-learning transaction.
- Continue the conversation before recall; do not ask immediately in the same context.
- Use Fresh + worker restart, then require natural recall with no raw attachment text injected into the prompt.

### [ ] 7. Queue, Steer, navigation, and receipt persistence

Combine this with Gate 1.

- During loading/generation, ordinary Enter queues and Cmd/Ctrl+Enter Steers.
- Steer changes direction only at a safe boundary, never overwrites the original or queued message, and keeps the warm worker.
- Navigate to Data, Brain Map, or Tools and back while generation continues.
- After visible reply generation completes, the composer offers normal Send and
  Stop—never Queue or Steer—while any brief atomic commit remains serialized
  internally.
- Reopen the app and verify human/brain messages, queue/steer receipts, actions, token count, and committed status appear exactly once with no stuck cursor or `responding` state.

### [ ] Final combined zero-context recall sweep

- After Gates 1, 2, and 3 have committed, start Fresh attention and fully
  restart the worker once. Include a crawl marker only if a changed-path
  two-page crawl was actually committed; do not invent or repeat Gate 5.
- Naturally ask about each newly committed source in separate turns: the new
  conversation marker, `JUNIPER-8051`, and `ONYX-4732`. `MARIGOLD-6194` and
  `CYPRESS-2468` require verified provenance before inclusion and must not be
  treated as completed merely because an old checklist named them.
- Require zero prior-dialogue prompt tokens, no hidden history/source-text
  injection, and neural episodic/assembly evidence for each answer.
- Do this immediately before Gate 12 so the restart also proves final receipt
  persistence.

### [ ] 12. Final affected GUI/runtime sweep

- Live-test connected-path lookup using the new indexed path; no full-shard scan or orphan inspection worker.
- Recheck only affected controls: Resume, job coverage/cancel, worker lifecycle,
  warm-turn reuse, and the generation Queue/Steer versus post-reply Send composer.
- Smoke every still-unproved visible button/function from the final inventory. Search the ledger/history before each click.
- Do not run a full test suite, package build, or CI in this closeout. Use the
  focused evidence already produced by each repair lane plus the final GUI audit.

## Deferred to the next training goal

- Full `/Users/eyadabushama/Downloads/ai train` traversal, including 10BT and the full multimodal corpus.
- Full from-random ground-up OmniCortex training on that corpus.
- Broad general-coherence/intelligence qualification after the long training run.
- Frontier-quality image/audio/video training quality.

The current goal still includes all other gates in this document.

## Closeout rule

Step 12 is complete only when every remaining gate is either:

- live-passed with recorded evidence,
- explicitly identified as the deferred full-training scope above.

An implementation-only result, a unit test without required live evidence, or a `no error` result is not sufficient.
