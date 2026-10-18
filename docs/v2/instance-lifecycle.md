# Explicit instance load, unload and supported resume

## Latest native proof decision

The user deferred the current-format inherited native host/Worker resume proof
on October 13. It is not required for this goal and is not marked passed.
No replacement microfixture, neural run, model creation or training is next.
The source implementation, focused checks and historical attempt records below
remain distinct from that deferred integration proof.

## Approved contract

The current authority is PLAN.md lines112–120 and the literal request's Load /
Unload / Auto-load paragraphs. Auto-load is off by default. Load preflights the
saved shape and current resource envelope. Unload is not Stop, Cancel, input
replay or a new generation. It preserves supported execution state and resumes
the same turn/job without repeating tokens or external effects. Unrecreatable
external handles require explicit reconnection.

## Current implementation boundary

- The product uses `suspend_chat` / `suspend_instance`, not the old internal
  teardown `unload`, for explicit instance suspension. The flags-only reader
  requests a supported boundary; native serialization and release stay serial.
- Native `resume_chat` continues the authenticated token/cache/RNG/UTF-8
  frontier and existing action ledger. It does not call input-bearing `chat`,
  repeat input learning, or redo private candidate selection.
- A real short native worker close/reload/resume proof passed (126.494 seconds).
  This is exact continuation evidence, not a trained-language capability claim.
- Private Best Work search and latent Ponder currently wait for the next
  supported visible-generation boundary. They are not falsely reported as
  independently checkpointable Python generator frames.
- Chat's input admission and completed effects already have idempotent durable
  receipts. Preserve those; do not create a duplicate human turn on resume.
- Dataset traversal already has committed cursors. Connect cooperative pause
  and resume; do not recrawl/retrain committed rows.

## Implementation gates

1. Typed native execution cursor and safe, bounded tensor/cache persistence;
   validate ownership, checkpoint binding, exact token frontier and RNG.
2. Brain chat continuation covering current conditioning, action ledger,
   diagnostics, stream sequence and finalization, without rerunning admission
   or private candidate selection. Unsupported phases wait for a named safe
   boundary rather than being falsely reported unloaded.
3. Worker flags-only suspension control, durable pause receipt, learned-runtime
   release, and same-turn resume. Separate external handle reconnection.
4. Main lifecycle service and operation gate; integrate existing jobs and
   queues without cancellation, preserve complete resume state in portability.
5. Default-off Auto-load, live fit/load dialog, editable settings and Unload UI.
6. Tiny deterministic uninterrupted-vs-resumed fixtures, no repeated effects,
   recovery/corruption/pressure cases, and rendered UI contract. No user brain,
   long corpus training or learned-capability claim belongs to these checks.

## Host and renderer implementation boundary

`InstanceLifecycleController` stores only the per-instance Auto-load preference
in the installation-local `.instance-lifecycle` directory. Imported model data
does not acquire Auto-load authority. `BrainService.chat` no longer sends an
implicit `load`. An explicit Load preflights the saved shape and actual current
context occupancy. Offline settings save the host configuration without an
`update_config` call or a neural `_get`; the next explicit Load supplies the
reviewed resource configuration. Current execution occupancy is a lower bound
on editable context, not a guessed fixed floor.

The host treats `suspend_chat` as a flags-only request. A validated native
suspension frontier ends one worker segment and releases `withBrainWrite`,
while the outer chat promise and action controller stay alive. `suspend_instance`
must acknowledge release before the host reports unloaded. Queued jobs stay
queued; unsupported running jobs and external handles wait for a named safe
boundary instead of being cancelled. Load gives the retained turn priority over
other queued learning, which cannot invalidate its checkpoint. Resume uses
`resume_chat` with checkpoint ownership and exact pending observations, never
another input-bearing `chat` call. Explicit Stop alone calls
`stop_suspended_chat` and requires its exact tombstone acknowledgement.

Cold host reattachment restores the authenticated published prefix before
starting the same turn ID. Earlier action proposals seed the controller's
deduplication set; they are not redispatched as new effects. Session-scoped
camera/microphone/screen handles must end before unload and require explicit
reconnection. Existing host effect promises are preserved in-process. Broader
external-process recovery still requires its own durable permission/receipt
reconciliation; it is not represented as an exactly restored OS handle.

The standalone Load dialog is mounted when opening Library instances and in
the workspace header, with a Device & runtime review entry point. Auto-load is
off by default. A blocked Load can wait for live fit; choosing Edit settings or
closing the dialog pauses that waiting loop. The chat workspace remains mounted
through suspension, and new neural actions are gated while unloaded. Stop is
still a distinct control.

### Scoped evidence, not native completion

`tests/instanceLifecycle.test.ts` uses tiny filesystem and protocol mocks to
cover default-off preference, fit rejection/recovery, same outer promise,
released writer/offline save, exact resume/Stop separation, stale-status and
command-ordering races, frontier priority, cold-action deduplication, and a host
effect paused behind Load. A focused EngineSupervisor test confirms status and
suspension controls bypass a pending chat without resolving it or killing the
worker; existing scoped Stop/Steer cases remain separate regression evidence.

The rendered Electron fixture verifies the real React controls with synthetic
receipts: off-by-default Auto-load, fit waiting/edit pause, no early unloaded
state, same-turn checkpoint display, reconnection disclosure, and dark/light
narrow layouts. No model is loaded by that fixture.

The opt-in `test_native_instance_lifecycle` proof uses private copies of a
previously initialized genuine native origin: one real token frontier is
saved, the learned owner and lease are released (weak reference collected), a
new Worker reads the published prefix without loading, explicit Load restores
it, and the remaining text/counts match the uninterrupted fixture. Stream
sequences are unique, the input is learned once, and normal own-output
observation remains active. Lost final acknowledgement emits no tokens/actions
again. A second real turn is suspended/unloaded and receives a separate Stop
tombstone; its admitted input is retained. Source and immutable origin metadata
remain unchanged. Action choice alone is fixed to Talk in this structural
fixture; it does not claim organic tool choice or fluency.

## Open acceptance and portability work

`executionCheckpointPortability.ts` validates sealed activity, diagnostic
SQLite/tensors, generation bindings and full effect receipts. It can rekey the
mutable instance identity without reserializing native numeric controls or
changing user text, neural bytes and prior action IDs. Import/fork permissions
remain destination-owned; old schema grants cannot authorize inline media.
The host metadata fixtures are separate from actual repository/native proof.
The production `BrainService` / `EngineSupervisor` / subprocess-worker fixture
passed in 320.306 seconds: a real paused turn was forked, exported/imported with
identity collision rekeying, resumed through the original parked outer
promise, and restored from a recovery point. Each remaining suffix matched;
the initialized source fixture remained unchanged. It did not retrain an
initial curriculum or claim trained conversation quality.

That already-running worker used execution schema 1. Schema 2 subsequently
added explicit global CPU/backend RNG, distinct from the private token sampler,
for stochastic post-reply learning and borrowed regional finalization. The
old full-path proof is not schema-2 evidence. Current-format codec/RNG and host
payload/header validation have focused component coverage; current-format
regional/host integration acceptance remains open until its actual fixture
finishes. The importer now validates activity safetensor header geometry as
well as file hashes, before publishing a destination.

An inherited parent cursor cannot be rebound directly to a newly connected
decoder: its cache width and sampling computation differ. Executable merged
continuation requires a borrowed regional owner, exact per-region runtime
finalization and one root-owned pending execution queue, with interfering root
mutations held until those frontiers drain. Preserving ancestor snapshots is
necessary but is not sufficient evidence for executable merged pending work.
The host status and admission gate also preserve that pending queue: explicit
Load alone does not release unrelated background mutations while an inherited
frontier remains. The borrowed regional runner and its root-owned completion
integration are still separate acceptance gates.

## October 11 active-time, runtime-seal and recovery additions

`ActivityClock` owns elapsed cognitive activity and the three decay/idle
cadence markers. Native token suspension freezes and persists its exact state;
the same owner resumes it at native work, excluding unloaded and cold-load
time. Real audit dates, user clocks/timers, provenance and neuron timestamps
are not rewritten. Borrowed regional owners hydrate separate source clocks;
root finalization translates only those three cadence markers into root clock
coordinates. The real decay/idle code has a fake-clock equal-active-time proof
including a simulated week unloaded. This is not a claim that independently
timed executions have identical wall-duration or learned capability.

New native execution manifests seal the Python/PyTorch build and libraries,
packed CPU/Metal kernels, compute source, CPU/accelerator hardware, backend,
threading and deterministic-math settings. Restore checks that declaration
and the exact paused clock before binding any saved cache. A backend string
alone is insufficient. `current()` remains archival-readable so complete old
exports and explicit Stop preserve historical state; adoption of a new
executable frontier uses `require_supported_frontier()`. Missing old seals
are explicitly unsupported, never inferred or fabricated. Matching seals are
not independent evidence of cross-ISA or cross-accelerator bitwise parity;
qualified actual continuation evidence remains scoped to the tested host.

Read-only status and the Load review disclose an incompatible saved frontier.
They do not try to fix a software/hardware mismatch with larger RAM settings.
The explicit `stopSavedTurn` action can end an unattached saved frontier without
loading or readmitting its input; it requires the exact turn/checkpoint or
inherited-execution identity and a native tombstone acknowledgement. It does
not erase learned state or old snapshot bytes. Attached turns retain their
existing chat Stop control. Unload remains a separate suspension action.

The main-only `queryNativeTurnStatus` API distinguishes ledger-bound committed,
retained, active, accepted-unfinished and absent input ownership for scheduled
work. Missing/corrupt evidence and conflicting input refuse blind replay.
Old durable human input remains conservatively accepted-unfinished when a
bounded receipt tail or an identity-safe copy no longer provides a completed
receipt. This is operational effect/admission bookkeeping, not a model-facing
Q/A lookup, hidden prompt or claimed semantic memory.

Current distinct scoped evidence:

- Eleven native clock/runtime/RNG/worker-protocol cases passed in 2.377 seconds,
  including exact metadata-only clock qualification before execution adoption.
- The affected real native selected-Best-Work replay case passed in 55.606
  seconds with a newly captured clock/runtime seal, without repeating private
  candidate selection. It is not a cold subprocess portability proof.
- Thirty-one affected lifecycle, scheduling-status and execution-portability
  host cases passed. The new narrow Electron unsupported-frontier/explicit
  Stop/live-fit interaction passed in 9.1 seconds; its light 390px screenshot
  was visually inspected, with no horizontal overflow or hidden bottom
  controls. No model ran in that UI. Main and renderer typechecks passed after
  the integration; whitespace checks passed.
- The old 320.306-second production host/subprocess portability proof remains
  schema-1 evidence. The earlier regional helper cold-load proof required a
  disclosed private-fixture metadata repair and is not a clean current-tree
  end-to-end pass.

The new authentic token capture eventually finished in 1,023.892 seconds on
its retained private clone, with prefix `e`, unchanged learned checksum and
unchanged original source. It did not restore an old cursor, rerun a curriculum
or construct another neural model. That latency fails the bounded expectation;
it is not a performance pass for the subsequently optimized exporter.

The current sealed inherited Worker gate remains open. Its one authorized
attempt ended at 113.192 seconds: authenticity was verified at 1.166 seconds,
but the first explicit Load was not ready at the 110-second cooperative
boundary request. The loader acknowledged safe cancellation before the
120-second hard watchdog. No reference, decoding, resuspension, second Worker
or new training occurred. Loading latency is failed/unqualified; exact current
regional Worker continuation is still unproven. No retry was started, and the
temporary executable-source freeze was explicitly released.

After that terminal attempt, read-only verification confirmed the authentic
captured checkpoint is still suspended, the root queue entry is still queued,
all six source-runtime file seals and neural tensor seals match, and the
immutable connected origin verifies. Prefix `e` is unchanged. No independent
reference or new resuspension checkpoint exists. This check loaded no model.

The opt-in test requires an independently restored current-frontier reference
before actual Worker continuation; historical `eddeesa` is not an assumed
target for the fresh computation. The reference can reuse the first explicit
Load without final learning/effects or root publication. The retained source
checkpoint, its actual prefix and its provenance remain available for a future
authorized acceptance after the load bottleneck is fixed. The large user
training folder, original brains and original sealed fixture are not used for
new training or modified by these checks.

## October 13 cold working page recency correction

The remaining suspended-time defect was in `PagedWorkingMemory`, not the
already-correct global decay clock or cycle-based afterimages. Its retrieval
and eviction priority included `exp(-(time.time()-updated_at)/3600)`. A long
Unload could therefore erase the recency contribution and change which cold
page was retrieved or removed, even though the global activity clock was
paused. The ranking formula and salience/reuse/retention contributions remain
unchanged; production cold-page recency now uses the owning paused clock.

New and touched pages store `activityUpdatedAt` separately in their temporary
metadata. `working_pages.updated_at` and IO audit timestamps stay real wall
time. An older unstamped page uses one persisted `working_activity_clock`
baseline derived from the authenticated checkpoint's saved activity coordinate
and `updated_at` provenance date. Reopening does not replace that baseline
with a later checkpoint, file mtime, copy time or current wall time. This keeps
the best-known committed age; it does not retroactively undo unknown pauses in
old software. If historical age cannot be established, that page receives no
invented recency advantage until actual use gives it a new activity stamp.

Binding the clock during Load is read-only. Baseline creation occurs only at
an actual mutable page write or owned checkpoint save; it does not eagerly
rewrite old rows or tensor payloads. Normal owners bind their own clock, and a
borrowed regional owner binds its separate source clock in its newly isolated
runtime copy. Frozen source/origin databases remain read-only. Whole SQLite
custody preserves the baseline and stamps through copies. Full merge and
inherited completion translate known page age into the destination's logical
coordinate while retaining original wall audit values. Future stamps or clock
rollback produce zero age rather than negative decay; unknown age stays
explicitly unknown. The changed cold-page math is included in the runtime
source seal, with no seal retrofitted onto an older frontier.

Eight new `test_cold_working_activity.py` checks passed without constructing an
`AdaptiveBrain`: six SQLite/numeric cases in 0.098 seconds and two separate
copy/source-wiring cases in 0.340 seconds. They cover the former rank flip,
equal retrieval/trim scores after a simulated unloaded week, read-only binding
and unchanged database bytes, persisted legacy baselines, unknown age,
transactional touch failure with unchanged payload, actual page-adoption age
including future timestamps, corrupted coordinates, isolated SQLite copying,
and non-resuming production bindings. Changed Python sources compile and
scoped whitespace checks pass. The already-closed clock/lifecycle tests were
not rerun.

This closes the confirmed cold-page time-domain implementation gap. It does
not establish post-learning equivalence for a complete native Worker run,
inherited cold resume, cross-hardware continuation or learned memory quality.
Those native acceptance gates remain open. No model Load, initialization,
training, user corpus, download or packaged execution belongs to these checks.

The matching host custody validator is `portableWorkingActivity.ts`, called
from `BrainRepository`'s existing saved-working-page validation before
publication. It checks an optional baseline's exact table schema, singleton
row, format/version and finite nonnegative activity/nullable wall coordinates;
baseline triggers are refused. The existing page traversal also validates
finite signed activity timestamps and an exact `true` unknown marker that
cannot coexist with a timestamp. Old native rows without a baseline remain
compatible without invented freshness. Five new tiny host SQLite cases passed
in 14 ms, including unchanged read-only file bytes. These are separate from
the eight Python clock checks; no old portability suite or native model ran.

## October 13 inherited completion reconciliation and host proof preflight

The main-only inherited lost-acknowledgement gap is source-fixed. Previously,
`BrainService.chat` excluded inherited turns from durable reconciliation, and
public Resume required a still-pending queue entry even after the native
completion committed. `inheritedChatReconciliation.ts` now binds the completed
root queue, stable native receipt, adopted original human ID/input hash/epoch,
and current root reply/trace. A repeated completed Resume reads that receipt
without loading or redispatching the native turn. One new constructor-free
host/SQLite case passed in 112 ms, including a lost final RPC acknowledgement,
original human epoch 3 versus root epoch 9, unchanged native ledger bytes and
no second native dispatch. This is host reconciliation evidence, not an actual
current-format native continuation pass.

The separately approved opt-in `currentInheritedNativeHost.test.ts` invocation
exited before any Worker or Load. Its selected-file preflight used an obsolete
flat `state/replay.sqlite3` path. The externally capped runner exited 1 after
2.279 seconds, with no native stages or worker PIDs; sampled family RSS was
524,730,368 bytes and minimum free disk was 34,215,182,336 bytes. Its logs and
`bounded-attempt.json` remain in the private temporary directory
`omni-connected-current-seal-host-JHhuUP`. No native continuation clause was
exercised, so this is a harness-preflight failure rather than a native failure
or pass. The runner and test processes exited, and no second invocation ran.

The source-only inventory correction requires sixteen selected files and
resolves the authenticated connected descriptor's actual parent aliases. It
checks parent owner, metadata/core/plasticity hashes and each selected active
mutable/substrate generation-manifest hash. Those selected manifests do not
stand in for hashing every referenced payload; the capture helper's full
source-a inventory is a separate assertion. Preflight cleanup is inside the
host fixture's `try/finally`.

Read-only checks of the retained D9 and F8 fixtures found that both descriptors
bind distinct parent-b region `r493e9fbd47a9573abecb5d20d1341f52` to metadata hash
`07470379b0acadfde0a99db8dada5eaf9204306fa593b7eb57c44c066bb653df`.
That metadata is absent from all four native resolver aliases. D9's original
`b/engine` paths and the known `/tmp/omni-verified-curriculum.vQScVn` parent-b
copy do not provide it either. The surviving substrate-only graph metadata is
not an authenticated substitute. No surviving snapshot/export closure was
identified within these already-known fixture roots; user data was not broadly
searched. The checked fixtures therefore cannot supply this full host proof
without recovering the genuine dependency into a new private copy. Original
fixtures, old cursors and their hashes were not rewritten or resealed.

The current-format inherited host/Worker gate remains open. The intended
proof also has no completed real external effect in its retained source;
Talk-only stream/idempotency checks cannot establish broad effect replay or
learned language quality. No initialization, curriculum, training, corpus,
slow replay or extra neural attempt was used to bypass the missing dependency.

## Read only source qualification and user deferral

`current_inherited_fixture_source.py` imports only the Python standard library
and verifies declared generation/payload hashes without loading tensors or a
brain. It refuses to promote a current checkpoint into an immutable training
origin. Three new custody-only helper checks passed in 15 ms; no earlier tests
or native run were repeated.

The new A-only preflight also refused the retained source. Its committed
mutable generation references the missing core blob
`state/blobs/e7136763f71aa1cc57d44f73926350a8bf78cee59e6c8b6270e57468b9ea3baa.safetensors`.
The surviving active core file does not satisfy the host snapshot collector's
required declared blob. The origin separately lacks `brain.json`,
`core.safetensors`, state/substrate pointer manifests and the packed manifest
with its SHA sidecar. Native `brain.py` origin validation and
`full_merge.py`'s parent curriculum verification require those actual origin
files and training-receipt/packed hashes. Neither embedded current metadata nor
a substrate-only graph record substitutes for them.

The CLI exited 2 with `qualified:false`,
`currentLearnedClosureVerified:false` and `neuralExecution:false`. These are
retained-fixture custody failures, not a newly established production resume
defect. The user then deferred this native proof; it is excluded from current
goal testing rather than treated as complete. No source repair, origin
promotion, parameterized replacement harness or microfixture initialization
was performed. Future native qualification requires a new explicit request.
