# Reviewed App data: native admission and task attribution

This is an implemented, bounded native learning boundary, not a model-quality
or general MCP Apps acceptance claim. It does not use a Q/A lookup, fabricate a
tool result, insert hidden source prose into a prompt, or borrow another model.

## Ownership and protocol

- The main process owns exact reviewed `recordJson` and `dataJson` bytes,
  immutable captures, the full audit history, review IDs and FIFO queue claims.
- `app_observation_frontier` and `admit_app_observation` run only on the existing
  loaded neural worker's serialized writer. They do not implicitly Load a brain
  or run on the flags-only reader. Active, suspended and inherited decoder
  frontiers block admission; they are not changed to accept App updates.
- `src/shared/nativeAppObservation.ts` defines the input/frontier/head/result.
  `src/main/nativeAppObservation.ts` defines exact raw UTF-8 and cross-language
  ordered frontier hashes. Empty frontier is 64 zero characters.
- Exact record/data hashes, content byte length, current source owner, method,
  review/stream identities and source predecessor must match before learning.
  Source labels remain attribution, not behavioral instructions.
- Every same-stream record is exactly the preceding ordinal plus one. A new
  stream starts at ordinal 1/zero source predecessor and must compare-and-swap
  the actual prior native frontier. Every new head's `baseFrontierSha256` binds
  that exact prior native head, independently of its source-record predecessor.

## Real learning and atomicity

An observed record invokes the existing
`learn_experience(text, kind='experience', source='mcp-app', steps=0)` path.
Normal neural assembly formation, ternary contact state, working/scratch
activity, automatic settling and replay admission remain in that path.
`steps=0` does not claim a slow cortical optimizer update or trained fluency.

The original text slice preserves text, structured JSON and inline text resource
order. Its new ordinary image/audio/reference extension is documented in
`native-app-observation-media.md`, with genuine same-brain sensory code and a
replayable current-ACK file. Source/codec contract evidence is distinguished
from actual packed media proof. It does not learn textual base64 and call that
sensory learning; private sampling alone is not ordinary learning evidence.

A superseded reviewed record advances only metadata, with
`state='advanced-without-observation'` and `learning=null`. It does not invoke
learning or sensory decoding. Main must deliver these records in FIFO order;
there is no opaque source-chain skip.

Only one bounded native head is persisted with `brain.save()`. It contains
hashes, the disposition and actual experience count/assembly/afterimage
receipt, not original passages or an unbounded native record list. Paused data
training uses the existing observed-neural-boundary save, so its cursor does
not retain a stale neural checksum after independent authorized experience.

An exact retry of the current committed head returns `duplicate=true` without
learning or saving again. The host's durable claim/audit remains authority for
older streams and never automatically reuses a completed stream ID. A lost
ACK must be resolved before the host dispatches the next record.

If mutation starts but publication fails, the warm owner is fenced and released;
retry requires explicit Load of the last atomic checkpoint. No floating whole
brain rollback copy is created. If the exact head is already in atomic
`brain.json` when a later cleanup fails, its durable ACK wins; `reloadRequired`
is returned and the warm owner is still released. This distinguishes committed
learning from a failed warm attempt instead of replaying both.

The release is explicit to main: a published result includes
`reloadRequired=true,nativeOwnerReleased=true`; an unpublished failure is RPC
`-32031` with current brain ownership, `nativeAppObservation=true`,
`committed=false` and the same release flags. A best-effort
`native-owner-released` event carries the typed `omni-native-owner-release`
receipt. Main correlates the actual result/error before closing instance
admission; a late unowned event is not authority. It must not preflight-autoload
a task follow-up. The two affected no-model release fixtures passed again in
0.051 s after adding the structured transport receipt.

## Ordinary App messages

`ui/message` may continue as an ordinary task only through main's internal
`NativeAppChatInput` proof. Native checks the exact reviewed record/data and
original completed-call capture bytes, current observed frontier and projected
plain task SHA. The initial follow-up supports text blocks only. Source labels,
tool schemas and capture text are never added as a hidden prompt prefix.

The decoder still receives its canonical user-role task. Actual authorship is
stored separately: `input_provenance` on the ledger message and
`inputProvenance` on accepted/completed input receipts. It records the App,
review, capture, original call owner, observation and source hashes. Copies do
not rewrite original authorship as a new human observation.

The already admitted experience and actual afterimage are reused, not admitted
again as a human episode. Slow replay is explicitly `evidenceRole='mcp-app'`
with matching persisted provenance; it is not mislabeled human evidence. A
previously accepted exact task may retry against its saved source receipt even
after the current App head moves; a new task must bind the current native head.
Suspension retains only bounded provenance/accepted state, not the raw proof
object in its generation context.

## Scoped evidence and remaining acceptance

- 13 constructor-free/synthetic Python cases passed in 0.064 s: exact bytes and
  owner rejection, complete projections, explicit unsupported media, FIFO,
  superseded records, same-head retry, restart metadata, new-stream CAS,
  pre/post-publication failure, paused training save routing, active/retained
  writer guards, source proof and no duplicate episode helper.
- Two additional constructor-free worker cleanup cases passed in 0.020 s:
  pre-publication failure releases/fences the owner without implicit reload or
  retry; post-publication cleanup failure returns the exact durable ACK after
  releasing the warm owner. Python compilation, Node TypeScript and the diff
  whitespace check passed after the source hooks.
- One tiny actual native fixture passed in 19.069 s under a hard 30-second
  subprocess timeout. It used width 8, VSA dimension 16 and 4 router neurons;
  no curriculum, provider, user corpus or saved user instance was involved.
  Actual steps=0 experience increased native state/experience count; save/reload
  preserved the head. Its chat accepted-input boundary persisted genuine App
  provenance and an App replay record while a second `learn_experience` and all
  decoder generation were explicitly forbidden. No answer was generated.
- An affected strengthening checks packed `uint8` neuron-vector bytes/counts
  before admission, after admission, after the accepted task and after reload.
  That run reached the hard 30-second timeout and was killed. Its saved fixture
  contains the actual observed head, seven neurons/42 synapses and the accepted
  App task/replay receipt. The before/after packed assertions precede that task
  in the test, but the complete strengthened reload assertion did not finish.
  Therefore this strengthening is not reported as passed, and no automatic
  additional native run was started. The original 19.069-second full mechanism
  check remains separate. Future bounded runs now flush their measured packed
  hash evidence before reload rather than losing all partial telemetry.
- Node/Python cross-language ordered frontier hashing has a fixed Unicode
  fixture in `tests/nativeAppObservation.test.ts`; all three hash cases passed
  in 5 ms. Two existing constructor-free ordinary replay-index cases passed
  in 0.005 s. No broad inference/replay suite was run.

The connected App host, reviewed durable queue, bounded native admission,
source-bound text follow-up and UI attribution now each have focused source or
mechanism evidence. One complete production provider→App→review→native→chat
flow plus packaged restart remains distinct from those slices. Ordinary App
media code has its own scoped evidence; full live-adapter proof and coherent
reply/tool competence are not established by this text run. Copy/dormancy,
permissions and artifact custody retain their separate evidence.
