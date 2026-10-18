# Instance-owned speech routing

## Exact default

The 200B boundary applies to **output**, not microphone input:

- Auto output uses platform TTS through **200,000,000,000** measured logical
  neural parameters, inclusive. Strictly above that count it requests the same
  brain's audio region.
- Auto input remains platform speech recognition. Input and output each retain
  an independent `auto`, `platform`, or `native` choice.
- Parameter count is not an audio-quality certificate. Unavailable requested
  native output remains unavailable; the application does not silently switch
  to platform TTS. Learned speech quality is reported separately.
- Native input means a genuine direct same-brain audio conversation route.
  Assisted audio observation plus a platform transcript must not be advertised
  as that route. Until the direct route reports `nativeAudioChatInput: true`,
  native input remains explicitly unavailable. It is never silently replaced
  by STT. When a genuine route is present, platform STT is not a prerequisite.

## Parameter-count provenance

The host reads `runtime_card.parameterAccounting.totalNeuralParameters`, the
measurement produced by the native brain's `parameter_accounting()` method.
That method counts distinct packed model owners, actual sparse synapses and
packed substrate elements/repeated contacts; assembly aliases are not counted
twice. Packed bytes, temporary activation buffers and advertised architecture
labels are not the speech threshold's parameter count.

`readCommittedVoiceParameterEvidence` checks native ownership/origin, exact
integer accounting and its additive invariant, the source shard coverage,
current substrate and mutable manifest hashes, their joint learned-state
binding, both active pointers, and unchanged metadata file identity. Connected
counts cover all measured distinct regional memories, not merely the outer
root's graph cardinality. Metadata reads are admitted against transfer RAM.

The returned evidence contains the metadata SHA, source/mutable generation
SHAs, native learned checksum, count basis and measurement timestamp. This is
a checked **last committed native measurement**, not a new neural model Load,
an independent recount of every tensor payload, or an estimate reconstructed
from build width, model name, parameter advertising or checkpoint file size.
An unavailable or currently publishing count leaves Auto output unresolved.
Manual choices remain possible; no implicit default size is invented.

## Persistence and migration

Presentation settings live separately from learned state at
`voice/workspace.sqlite3`. Each instance owns its own revision and
compare-and-set updates. A later stale neural/host document save cannot replace
a newer voice choice. Reads do not initialize a missing legacy record.

New native and new merged identities initialize Auto **before** their first
discoverable host document. External imports without saved voice preferences
also initialize Auto before identity publication. They never inherit an old
renderer-global preference from another instance or installation.

Only a pre-existing local identity without an instance record is eligible for
one-time legacy migration. A present valid global v1 record maps both stored
booleans, including `false`, to explicit manual native/platform choices. Pace
and live/buffered delivery are retained. Migration never overwrites an existing
record, including a fresh Auto record. Invalid/absent legacy storage must not
be treated as a newly saved manual selection.

The existing portable workspace inventory now validates and includes the exact
voice database for snapshot, trusted copy/fork, export/import and recovery.
Copies have independent mutable files/owners. Full merge keeps parent variants
as inert idempotent settings history; it does not let the last parent silently
choose the new identity's route or replace a separately reviewed destination
setting. No microphone grant, running call, native capability or credential is
inherited with these presentation preferences.

## Typed API and renderer resolution

`window.omni.voice` exposes:

- `preferences(brainId)` → instance settings/revision plus parameter evidence.
- `update({ brainId, expectedRevision, preferences })` → CAS save and snapshot.
- `migrateLegacy({ brainId, legacy })` → one-time eligible migration/snapshot.

`InstanceVoiceSettings` is `{ schemaVersion: 1, input, output, deliveryMode,
pace }`. The shared `resolveInstanceVoice` combines those settings, complete
native count evidence and actual runtime capabilities. It returns independent
route/reason/availability details and the existing controller's v1 boolean
presentation preferences. `controllerPreferences: null` means Auto output
could not be determined: the caller must not start a voice session by silently
assuming platform output.

IPC remains restricted to the trusted Studio sender. These user settings do
not enter prompts, model training, learned personality or tool-permission
authority. Existing live/buffered pacing and barge-in state-machine behavior
were not rewritten by the preference backend.

## Focused evidence and limits

Twenty-one new cases passed across three files: nine pure routing/validation
cases, eight actual host SQLite/CAS/migration/portable-envelope cases and four
synthetic native metadata/manifest evidence cases. Test execution was **2.07s**
inside a **10.54s** single-worker runner session, including transform/startup.

Coverage includes exact below/equal/above200B boundaries; independent manual
overrides; no native fallback; unknown/incomplete evidence; assisted-versus-
direct input capability; fresh Auto and one-time legacy behavior; stale CAS/
document isolation; snapshot/copy/recovery ownership and unchanged source;
missing-setting imports; retained conflicting parent preferences; wrong owner,
oversized typed settings and unknown portable files; and manifest/pointer/
arithmetic tampering. Synthetic count fixtures instantiate no neural model and
do not establish that a 200B brain or trained speech capability was exercised.

The root Call integration now uses per-instance preferences and current
metadata-only capabilities before Start, with stale instance/Unload guards.
Four new presentation/strict-legacy cases passed. A bounded synthetic Electron
interaction check covers independent A/B choices, unavailable native output,
missing-count Auto, active control locks and320px light reflow; see
[voice-attachment-ui-visual.md](voice-attachment-ui-visual.md). This is not a
real capture or voice acceptance claim. Genuine direct audio conversation,
packaged capture/playback and service/platform-dependent behavior remain
separate integration gates. No real capture, voice, model, corpus training,
CI or release run was performed for these preference/UI slices.
