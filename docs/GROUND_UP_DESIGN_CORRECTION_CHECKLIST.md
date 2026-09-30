# Ground-up OmniCortex: corrected design and remaining work

Reconciled through 2026-09-30 against
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). The scoped
native-only source/GUI/CI/package goal concluded with published **v1.1.0**;
this document must not reopen that goal or claim a trained brain passed live
acceptance. It preserves the earlier 26 September checklist as a dated
checkpoint and separates its design requirements from the later release
closure and subsequent source corrections. Later user corrections
take precedence over the older v1 plans wherever they differ.

The audit records 27 release assets and successful all-target run
`36495422025`, package-source tag `e96f580`, release-workflow commit `bd3ac8e`,
and only `main` locally/remotely at that historical audit. Full native corpus/
quality testing remains user-deferred. Application corrections were committed
as `5b15e22`; the repeat comparison now requests a local commit, not GitHub CI
or publication. Neither source checks nor a commit prove useful native recall.

The [30 September repeat recheck](AGREEMENT_RECHECK_2026_09_30.md) and
[implementation status](IMPLEMENTATION_STATUS.md) govern current source
disposition. The dated boxes and evidence below are historical; they must not
override newer code findings or become another model-testing checklist.

## What the user actually wants

1. **One original brain, not a disguised second model.** A new Build creates
   OmniCortex from its own initial weights, then trains that same brain. Falcon,
   Llama, BitNet checkpoints, teacher APIs, and key/value lookup must not be
   hidden runtime origins or answer fallbacks. External work is a research
   reference or explicitly selected training source, not the resident brain.
   The user rejected the Falcon hybrid after discovering it (thread entries
   274–283, 331–332, 547–551).
2. **The resident learned synapses are ternary.** Eligible cortical and sparse
   synapses have effective values exactly `-1`, `0`, or `+1`, stored as packed
   codes and updated as the brain learns. The later explicit "only ternary"
   correction (432–437, 572, 577) supersedes the older plans' allowance for a
   *full resident FP32/BF16 master copy* (7, 42). Floating-point activations,
   temporary calculations, and bounded transient eligibility may be necessary;
   those must not be misrepresented as a second full model. "Learning weight"
   is an implementation term, not a user-requested kind of memory.
3. **Experience changes the brain, not an answer table.** Ordinary chat,
   reading, media, and tool experience may change attention and temporary
   activity first; relevant parts can remain active, fade, recur, and become
   durable changes to the same neural parameters/synapses used by generation.
   Saying "remember this" must not switch to a special fact parser. No
   question/answer key, exact-answer injection, hidden retrieved passage, or
   separately authoritative idea database may be called neural learning
   (360–370, 455–463, 508–509, 525–536, 541–546).
4. **Continuous, brain-like memory dynamics.** Working context, transient
   afterimages, recurrent activity, activation, salience, interference, reuse,
   stability, decay, and reactivation interact while the brain runs. It should
   not wait for a manual "Consolidate" operation, nor expose a rigid
   three-stage diagram to ordinary users. The live context may be cleared;
   durable recall must then come from changed neural state, not pasted chat
   history or a raw-source retrieval path. Basic Build should not ask users to
   choose a memory "recipe" or manually trigger consolidation; exact source
   archiving, if offered, is separate from what the brain remembers
   (106–122, 463, 536).
5. **Organic capabilities.** Pondering, imagination, tool use, agents, web
   research, and recursive improvement are available inside the same chat and
   neural identity. They may be requested or chosen by the brain, without a
   backend personality instruction, slash-command dependency, mandatory
   pondering, or a separate "code AI". Tool permissions still gate external
   actions. Tool meanings should be learned in initial and ongoing training,
   not merely injected as prose for each turn (47–48, 79, 94–99, 116).
6. **Real initial and ongoing training.** Building should visibly complete
   initial training before normal chat. Later training should visit every valid
   record in selected full datasets—including Parquet, web crawls, text/code,
   image, audio, video, folders, and chat experience—with coverage and truthful
   progress; it must not stop at a small arbitrary idea/chunk count. A small
   tool curriculum is not a broadly capable language model. The user-provided
   supplied Downloads training folder is reserved for the later full
   training run, not silently consumed now (73–77, 99, 170–173, 539, 547–551).
7. **Resources and interface must match the real brain.** RAM allocation,
   packed residency, storage spill, training cursor, working-memory size,
   parameter count, speed/ETA, and neural traces should report actual state.
   Avoid unnecessary RAM/disk spikes, false queue/generating states, hidden
   failures, and hardcoded small limits. Keep the UI simple and correct on
   Windows, macOS, and Linux; mobile may be local or companion (94–95,
   125–170, 179–209, 426–438, 469–507).

## Closed source/release goal and current correction boundary

The earlier **source/design cleanup for future ground-up Builds** scope
(entries 545–551, 572–587) deferred brain creation/training and hands-on model
proof. Later explicit release instructions completed the scoped v1.1.0
code/GUI/CI/package and repository closure. Neither that closure nor the new
requirements audit erases the product vision or authorizes another model run.

Current work follows the transcript audit's correction contract. Preserve
historical accepted checks as closed, retain deferred native quality gates,
and distinguish each new source fix from publication. The latest clarifications
are device-theoretical context capacity with physical validation, initial
training → chat without a blocking quiz, automatic reviewed video-runtime
setup, and an exact unsanitized saved-instance export.

### Historical cleanup order (retained checkpoint, not a fresh action list)

1. Read the full authored chat log, then keep this request list and status
   current. After a context compaction, reread the raw user messages before
   making further design decisions. Do not claim that reading proves the code
   works.
2. Remove fake/legacy production paths and make the current ground-up
   implementation follow the design below. Preserve good unrelated work and
   the user's research references. No Falcon, converted checkpoint, external
   model, cue→answer lookup, raw-memory answer injection, or separate “code
   AI” may silently stand in for OmniCortex.
3. Check the source and GUI for truthful behavior and wording. Correct
   remaining design gaps; do not replace the user's design with an invented
   tool quiz, a new memory type, or a personality switch.
4. **Only after cleanup**, perform code-only verification and CI. Report what
   passed and failed separately. Do not build/train a brain, quiz a model,
   launch a hands-on Falcon memory test, or repeat completed feature tests in
   this goal. The user will initiate the later ground-up training/live goal.

## Historical 26 September checklist snapshot

The recorded boxes below describe that earlier checkpoint, not today's
release disposition. Their unchecked broad design items must not be converted
into new release/CI/deletion chores. Published v1.1.0 and user-closed engineering
checks are recorded above; current substantive gaps and uncommitted corrections
are tracked separately in the transcript audit. In particular, an old unchecked
source step is not proof that native-only packaging is still unpublished.

- [x] **1. Recover the request.** Read all 595 user-authored text entries in
  the raw project log, including repeats and the final corrections. This pass
  included the original prompt and both v1 plans. The linked screenshots are
  not interpreted as proof of their contents. No implementation pass follows
  merely from reading.
- [ ] **2. Correct the production brain path.** A new Build must start with
  project-authored OmniCortex, not Falcon or another model. Packed ternary
  synapses/parameters must be the authoritative persistent weights; audit all
  learned embeddings, codebooks, recurrent controls, sparse links, optimizer
  mirrors, imports, and growth. Preserve temporary computation needed to run
  the model, but do not call a second full FP32/BF16 weight copy “1.58-bit.”
  Ordinary experiences—not just “remember” or direct questions—must move
  through active/working/afterimage dynamics and change the brain used for
  future generation. Remove cue→answer/fact tables, verbatim answer replay,
  hidden long-term text injection, and hardcoded capability shortcuts.
- [ ] **3. Correct source/UI claims without redesigning unrelated features.**
  Keep Ponder, imagination, tools, web, agents, and recursive improvement
  available organically in one chat and one brain, with external permission
  checks. Tool meaning is initial/ongoing neural learning, not a prose persona
  prompt. The ordinary GUI should avoid manual consolidation, rigid visible
  “three stages,” curiosity/personality sliders, legacy brain-origin choices,
  unexplained “learning weights,” and false precision/resource claims.
  Preserve data ingestion, multimodal paths, storage spill, recovery, and
  research references. Record any source gap honestly rather than inventing a
  substitute.
- [ ] **4. Verify only in the authorized order.** After Steps 2–3 are
  source-clean, run compilation/typechecking/static or code-only checks, then
  CI. No new brain Build, training run, model quiz, hands-on recall/feature
  pass, or release promotion in this goal. Give a file-and-evidence-backed
  gap report. Defer full-folder training and live capability proof until the
  user sends the next goal.

## Historical source evidence from the earlier cleanup

- The five named app-managed Falcon instances were separately approved for
  permanent deletion (550, 553) and were removed after exact-target checks.
  Their chats/snapshots/neural state are not recoverable through the app.
- The supplied Downloads training folder and `.runtime/bitnet-src` are the
  user's data and research references. They are excluded from cleanup.
- Current source uses packed-authoritative linear/convolutional/embedding,
  normalization-gain, residual-gain, expert-route, spiking, and sparse-link
  weights. Obsolete floating-master projection classes were removed, and the
  native audit now rejects a new floating trainable parameter. This is
  source enforcement, not proof of natural recall or a conversational
  ground-up Build. The scoped v1.1.0 code/package CI has completed; newer
  corrections are not yet packaged or released. Existing floating-control
  checkpoints are intentionally rejected by the strict changed state schema.
- The current worker previews an ordinary chat experience before generation,
  but commits fast synapse/working-state changes only after the turn completes;
  dense cortical learning is queued later. This protects cancelled turns, but
  does not establish same-turn parameter updates or cleared-context recall.
  The production prompt uses current and bounded recent dialogue tokens,
  while long-term recall supplies neural vectors, not source passages. The
  substrate still stores semantic labels/fingerprints for inspection; those
  are not evidence of an answer index or of useful understanding.
- The learned action head exists, but fixed confidence/permission/schema
  gates remain outside it and its initial curriculum does not verify natural
  tool competence. We must not relabel any future fact/key-value lookup as
  “learned recall” in the GUI.
- **Strict unification is source-progress, not accepted.** The new v3 VSA
  source stores adaptive neuron/assembly vectors as one shared packed ternary
  row per identity and rejects old floating-vector states. Transient reads are
  normalized for activation math. This removes the separate learned float
  vector authority in new state, but full source-path verification, disk paging,
  and actual learned recall are still open; do not call the brain fully
  unified or 1.58-bit end-to-end from these edits alone.
- Multi-rank packed training currently fails closed rather than claiming
  synchronized mutations. This is an honest limitation for the later
  distributed training goal, not a completed distributed capability.
- Current packed projections implement checkpointed **uint8 output-row
  resistance** through `_PackedRowMetaplasticity`, applied during direct
  ternary updates. This is a bounded row-level retention mechanism, not a
  dense FP32 master, Fisher matrix or differentiable weight-anchor penalty.
  The remaining legacy floating-anchor helpers have no native packed weight
  parameters to stabilize. Their presence must not be described as the current
  cortical mechanism. Useful long-run retention/non-forgetting remains
  deferred native proof; source resistance counters alone do not establish it.
- **Large-source idea growth is not yet accepted.** The source-size threshold
  that silently sent large datasets into a few shared statistical fields has
  been removed for new transactions: detailed assemblies are selected and
  admission is estimated per bounded checkpoint window, with a recoverable
  resource pause instead of representation downgrade. Existing active v2
  cursors keep their frozen schedule. This is a correctness change, not proof
  that a huge dataset will finish: the live pager, v3 resumable microbatch
  schedule, and cold-load metadata path still need end-to-end integration.
- Paged packed vectors, assembly metadata, exact paged similarity, and a
  bounded v3 substrate writer have source-level storage checks. The new-file
  ingestion path now selects detailed learning with a joint `brain.json`
  checkpoint and checks source and shard identity on resume. This is not yet
  a completed large-dataset acceptance result:
  neuron metadata and packed vectors have paged v3 source paths, but some
  idle/inspection/filter operations still materialize total-dependent state,
  and the writer scans all neuron/assembly records per checkpoint despite
  bounded buffers. Source-only storage checks do not establish low-I/O
  full-corpus throughput or learned recall.
- No claim of consciousness, biological equivalence, guaranteed AGI, perfect
  memory, or fluency from random weights is justified here.

## Later corrections and native quality proof

The newer source adds disk-floor enforcement, parser leases/admission,
episode-linked raw-token cooling, exact unsanitized saved-state export/recovery,
shared learned-memory parameter accounting, paged core/attention, RAM-first
Auto, synchronized packed updates, incremental checkpoints, cortical inspection,
speech and mid-generation action routes. The separate vetted codec catalog is
published. The old claims above that multi-rank is blocked or the codec catalog
is empty are historical, not current source findings.

The repeat recheck still identifies real limits: dense recurrent state,
cooperative rather than hard aggregate RAM limits, incomplete shared actual
spill-quota accounting, native whole-value allocations, compatible rather than
arbitrary width/head migration and total-dependent integrity work. Native
fluency/recall, media quality and beneficial autonomous improvement remain
deferred proof. Do not mark those limits fixed merely because a source mechanism
exists, nor call already corrected source paths missing from the current tree.

The broader requirements below are retained, but existing engineering checks
are **not being reopened** and useful native quality is **not claimed now**:

- One continuous chat/identity per brain; duplicate, fork, merge preview,
  immutable origin, recovery, `.omni` import/export, and clearly authorized
  deletion. No second hidden language model or separate code assistant.
- Full streaming training from user-selected files/folders, PDF/Office/EPUB,
  JSONL/CSV/Parquet/Arrow/WebDataset, code repositories and web crawling;
  every valid record visited, deduplicated, attributed, resumable, and shown
  with honest progress/coverage/ETA. The user's `Downloads/ai train` corpus is
  for the next user-initiated run, not this cleanup.
- Shared-brain image, audio, and video input, learning, and imagination;
  progressive output, cancellation, hardware-scaled live voice/video/device
  access, snapshots, and chat attachments. Media quality must be measured,
  not inferred from the existence of a generator.
- Natural in-chat Ponder, tools, web, coding, agents, and recursive source/neural
  improvement with visible permission gates; no slash-only dependency or
  compulsory pondering. Train capability knowledge into the same brain.
- Hardware-based RAM allocation, adjustable working context, storage spill,
  reserve/pressure checks, packed weight accounting, realistic low-end and
  high-end scaling, and truthful token/parameter/synapse counters.
- Readable Windows/macOS/Linux GUI, auto light/dark and distinct appearance
  layouts, accessible controls, Queue/Steer/Stop, honest activity state, and
  mobile local/companion support. No confusing origin/memory-mode/personality
  choices in the ordinary Build flow.
- A later user-initiated native run must establish natural learned recall after
  clearing temporary context, competence/tool use, meaningful multimodal
  quality and real resource behavior. The already completed v1.1.0 release is
  separate evidence; new source fixes are not its published contents. Neither
  code/worker-health CI nor the legacy Falcon/answer-key runs satisfy native
  quality gates. Do not repeat historical MCP/media/recovery/device/crawl/UI
  checks merely because these future model-quality requirements remain.
