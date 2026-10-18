# Same-brain raw audio conversation input

This is a source/contract implementation, not a claim of trained speech
understanding. No native model was constructed, trained or run for these
checks. Paired audio/language training and real-device qualification remain
separate acceptance work.

## Input is audio, not an STT transcript

When Native input is selected, `LiveVoiceController` never starts the platform
recognizer. Real microphone PCM is segmented at sound/silence boundaries and
encoded as a complete WAV utterance; those boundaries do not infer words.
No caption, generated transcript, behavioral prompt or foreign model is used.
Platform STT remains the unchanged Auto/default input choice. An unavailable
Native route fails explicitly instead of falling back to STT.

`readyCapabilities()` refreshes current loaded-only metadata, without opening
a microphone or loading a brain. A failed constructor probe can be refreshed
after explicit Load. Start reuses the completed probe. Stop/dispose cannot
revive a pending start. Active input/output routes cannot change mid-call;
pace and live/buffered presentation can still change. Existing platform voice,
live/buffered output and barge-in behavior are kept.

## Trusted host boundary

Public `chat.sendAudio` accepts the typed `omni-native-audio-chat` request:
capture session, utterance ID, sequence, MIME, canonical raw base64, capture
time and optional source sample-rate/channel metadata. It accepts no pathname,
grant, model identity, transcript or hidden instructions.

An actual current main-owned microphone session is required. Its mind, state,
session grant, worker PID, modality and physical transport envelope are checked;
an arbitrary renderer session string or merely having OS microphone permission
does not create a lease. Revoked OS access is rejected on supported hosts.
Exact utterance sequence/hash retries are correlated; a different body or turn
cannot replace an existing claim.

Raw bytes use the same immutable `TaskAttachmentService` as ordinary file
attachments: stage, bind to the exact turn and resolve its verified descriptor.
There is no second blob or permission store. Staging alone does not learn or
launch dataset ingestion. The actual turn is the explicit sensory experience.
The internal `omni-native-audio-input` hashes its exact typed attachment and
capture metadata. It is not `sha256("")` or the hash of a display label.

Private paths and lease data remain main/native-owned. Renderer messages and
accepted receipts receive only `NativeAudioMessageInfo`: source/hash/MIME/byte
and measured coverage metadata. The original raw audio stays an owned task
attachment. Public saved-continuation presentation is also path-free.

## Same learned sensory and language owners

`native_audio_chat.py` verifies the immutable regular file, its full hash,
inode/size/timing and typed MIME, then uses the existing admitted audio codec
windows. Every source sample is visited; model-rate resampling and final padding
report actual sample/window counts. Resource admission and cancellation remain
active; unsupported containers fail rather than being labeled learned.

Each window enters this brain's real `modalities.perception_embedding`, its
ternary sensory lift and a bounded position-bound VSA accumulator. These are
transient activity tensors, not floating learned weight copies. One ordinary
`_admit_sensory_embedding` forms the sensory assembly, updates STDP/working
activity, automatic settling and replay. Connected-region observation shares
the ordinary transaction and atomic root save. There is no selector optimizer
update per packet or fabricated text target.

The normal `AdaptiveBrain.chat` action/language route uses that sensory cue,
real recalled neural activity and working state. The current text input is
truly empty and permitted only with the trusted typed audio identity. Its
language prompt contains actual recent tokens and structural role boundaries,
not a manufactured "Audio message" sentence. Action selection, Ponder and
tools use the ordinary learned route. Output may be primitive until trained.

Fast input learning and its complete sensory receipt commit together before
the reply. Normal prior-experience/self replay remains normal; raw audio does
not create a nonexistent transcript for slow language training. Exact accepted
retries read the admitted packed sensory assembly, not the raw audio or a
stored answer. Completed retries preserve ordinary no-effect-replay behavior.

## Queue, Steer, Stop and continuation

- Audio uses the existing serialized chat/action writer and stream IDs.
  The current lease and immutable attachment are revalidated after queue wait
  before new admission. A real prior accepted/completed receipt is not treated
  as a new microphone observation.
- Warm Steer uses the existing native boundary, not process replacement.
  An early unadmitted audio predecessor is carried as a private typed
  `TemporarySteeringContext.audioInputs` source, never empty-caption prose.
  Its current authority is rechecked and its audio enters temporary sensory
  conditioning. It is not a fabricated text lesson or blind duplicate admission.
  Fresh discards retired carry. Existing text carry is unchanged.
- Stop cancels the exact pending IO stage and/or ordinary turn, awaiting their
  owned boundaries. It does not erase an accepted input or terminate another
  writer. Stop during attachment binding cannot later start a neural call.
  Late OS/capture authorization results close their owned handles; an already
  stopped start cannot open another permission request.
- Accepted audio identity is validated through native suspension and resumed
  finalization. Saved conditioning/cursor resume without another microphone
  observation, source re-read, transcript or permission grant. Existing active
  external microphone policy still requires ending capture before Unload;
  its live device handle is not implicitly restored later.
- Copy/merge history preserves the original sealed audio identity and source
  owner. Current-owner aliases are separate. Already accepted cursor/ledger
  hashes must not be rewritten to a newly rekeyed public attachment manifest.
  Unadmitted copied work cannot inherit a live capture grant.

## Scoped evidence

Across the targeted runs, not a broad suite or CI:

- Nine new constructor-free Python cases passed: actual small WAV decoding with
  an explicitly synthetic encoder/admission, full sample coverage, source
  tamper/resource/cancel rejection, one ordinary admission contract, accepted
  packed-cue retry, exact original-owner suspension identity and typed
  audio/Fresh handoff. The original eight ran in 0.273 s; the added handoff
  case passed separately.
- Twenty-eight new host/capture/control cases passed across targeted runs:
  typed staging/hash/receipt/public projection, current per-brain lease,
  no-recognizer input, refresh/route locking, real supplied PCM/WAV encoding,
  late permission cleanup, pending-stage Stop, normal Queue/Steer/Stop and
  safe saved-continuation projection. Host encoders/worker results in these
  fixtures are explicitly synthetic, not claimed neural output.
- All 32 affected existing browser/controller cases passed across runs. Three
  obsolete expectations that Native still used assisted STT were updated to
  the genuine raw-input or explicit-unavailable contract; platform cases were
  not rewritten to pretend Native worked.
- Four existing private host Steer cases passed. Eight Python text-handoff
  cases passed across the initial run and one surgical assertion update:
  the trace now correctly lives in the shared continuation finisher, rather
  than being required as source text inside `chat` itself.
- Node and renderer TypeScript, Python compilation and affected whitespace
  checks passed. No model run, real microphone capture, user corpus, provider
  call, CI, commit or release was performed in this lane.

This does not establish trained speech comprehension, coherent audio-conditioned
answers, real-device/package permissions/playback, model-level exact continuation
under audio input or paired training quality. Those remain explicit acceptance
boundaries. Earlier native media proof evidence in
`native-app-observation-media.md` remains separate and is not relabeled as
an audio-chat inference proof.

Durable compact Steer presentation is also a separate task: existing
`turnMetadata` was transient ActiveTurn/stream/result state, not an authoritative
history record. Audio input receipts do not falsely claim to implement that UI
history requirement.
