# Rolling imagination capture and delivered timeline replay

Verified spontaneous imagination has a rolling window of delivered media, retrospective recording, and optional always-save behavior. User-requested outputs keep their existing saved-artifact behavior. A recording preserves immutable delivered fragments. An explicit Make replay operation can compose those fragments into a playable WAV or MP4 without generating new imagination or changing the source archive.

This is the host capture and storage portion of the imagination requirement in `REQUEST_VERBATIM.md` line 191. Natural native-chat intent remains unclassified, so this slice does not complete that requirement or demonstrate trained imagination quality.

## Trusted origin and native output ownership

The main process classifies actual structured action provenance, never prompt keywords:

| Origin | Evidence and retention |
| --- | --- |
| User requested | A main UI request or a main human action. Existing saved final outputs remain unchanged. |
| Spontaneous | A genuine main-owned organic action. Only this origin can use rolling capture and a temporary native output owner. |
| Unclassified | Native chat `source:brain` or missing main provenance. Existing final saving remains unchanged; the recorder does not guess the intent. |

Before spontaneous generation, the main process obtains a one-use native output lease bound to the exact brain, job, and organic action. The loaded worker creates a fresh owned directory under `engine/.spontaneous-imagination/` after disk admission. The decoder writes into this directory through an explicit owned-output parameter. Renderer arguments and model text cannot select the destination, mint a lease, or change retention. Authorization does not implicitly load a brain.

The media registry verifies actual delivered bytes and their hashes. Supported native preview fragments and final artifacts can enter the rolling window; a status event alone cannot become a fragment. A spontaneous final receives an expiring media lease rather than the ordinary permanent artifact index. Its inline image data is removed from the retained result, so an effect receipt does not silently become a second permanent pixel archive. Requested and unclassified finals keep their existing paths and retention.

Inline chat presentation preserves the exact `ephemeral-spontaneous` marker and
labels that media **Temporary spontaneous output · Use Record to keep it**.
It does not infer persistence from action source, keywords, a legacy path, or
the presence of a media URL. Untyped finals are only called available and
decoder previews are only called delivered; the permanent Imagination index is
the evidence that an artifact was actually retained.

Cleanup is limited to verified owned scratch and checks file identity before removal. Existing `engine/artifacts` outputs are never removed by this mechanism. An altered ownership marker, unsafe link, or unexpected scratch payload is preserved and reported instead of recursively deleted. Native unload cleans acknowledged temporary output owners; it does not delete committed recordings.

## Window and recording controls

`window.omni.modality.capture` exposes settings, status, record, stop, cancel, resume, paged recordings and fragments, and removable event subscriptions. It exposes no frame-injection, provenance-authority, native-lease, prompt, learning, or generator endpoint.

The default window is 300 seconds and always-save is Off. Settings support the requested 15, 30, 60, 180 and 300 second choices, plus positive custom values up to 300 seconds with millisecond precision. Delivery lifetime uses a monotonic clock; audit timestamps remain wall time.

Record saves only the requested lookback that is still available, then captures future eligible fragments until Stop. Repeating the same record request ID returns the same recording. A lookback larger than the selected window is refused; expired fragments are not recreated. Each fragment preserves its actual origin, stream, timestamp, content hash, byte count, media type, and available measured frame metadata.

Stop and cancel end future capture while preserving committed fragments. Frame arrivals advance the data revision, not the control revision, so a current Stop remains valid while frames arrive. A stale command from before an explicit Resume is rejected. Dormant or resource-paused recordings require an explicit Resume. Stop, cancel, and resource refusal pause automatic recording; applying settings explicitly can re-arm it.

Always-save starts on a new eligible delivered event, not by replaying previous output. A refused fragment produces a real gap and pauses capture without inventing bytes or stopping native work. Committed archives survive that refusal. Transfer memory, pending IO, and disk admission remain checked at their actual storage boundaries.

Recordings use the representation `verified-delivered-fragment-timeline` and the label **Delivered fragments — not a completed video**. A timeline may include a genuinely completed final artifact, but collecting previews does not turn them into a playable MP4.

Each new supported byte-encoder delivery can carry a strict `mediaTimeline`
descriptor: snapshot, replacement-from-zero, or complete-from-zero; actual
image, sample or frame count; timebase; and measured container audio where
present. A stable prefix is recorded only when the producer supplies a genuine
stability guarantee. Main records its first verified delivery anchor separately
from the media timebase. Neither anchor is described as the moment a thought
began. Stop and pause preserve the measured monotonic recording end, including
time after the final delivery.

## Pause and complete custody

The versioned workspace is `imagination/workspace.sqlite3`. Recorded bytes are content-addressed under `imagination/blobs/`. Still-live window bytes are checkpointed under `imagination/window-blobs/` with their remaining lifetime. Immutable source databases retained during adoption live under `imagination/archives/`.

The complete workspace pipeline preserves active recording state, admitted pending fragment IO, the still-live window, preferences, fragment metadata, and committed archives. Snapshot registration occurs before asynchronous owner initialization, and export waits for the same serial queue; the first admitted event cannot be silently omitted during that initialization boundary.

- Unload drains admitted IO, checkpoints the live window with its clock paused, and makes an active recording dormant. It then removes only this process's unrecorded rolling copies. Explicit Load can activate the retained window clock; it does not silently resume the recording.
- Fresh clears the current unrecorded window deliberately. It leaves committed recordings and older user outputs intact.
- Fork and import preserve complete unsanitized content. Current ownership is rekeyed, while original fragment provenance and immutable source archives remain intact. Copied pending recordings and preferences are dormant; they do not automatically save, execute, learn, or recreate grants.
- Full merge retains source archives, deduplicates identical shared content, and preserves distinct recorded outcomes or window state. Recovery retains later recording history and window custody rather than erasing them with an older snapshot. Repeated adoption does not duplicate identical histories.

Workspace adoption checks owner metadata, declared database objects, regular-file paths, and required content hashes. Unsafe links, undeclared payloads, or changed archives are refused. Expired unrecorded scratch is disposable; an admitted live window or an active recording is not an export sanitization exclusion.

## Playable delivered timeline replay

The additive `window.omni.modality.capture.assembly` API prepares a fixed
recording revision, makes a replay, returns paged history and manifest views,
and exposes exact Stop and dormant Resume controls. Preparation has its own
request ID, sequenced runtime-byte/source/spooling progress, and Stop
acknowledgement. Concurrent exact retries share the same preparation; a ready
receipt reopens its sealed plan. A cancelled, failed or interrupted preparation
does not restart under its old ID. It requires a new explicit request.

Every source revision remains in custody. During replay, a new revision replaces
that stream's displayed state at its recorded delivery offset. Overlapping
prefixes are not concatenated as new content. Available media time advances
within the selected revision rather than restarting earlier audio samples.
Multiple visual streams use a declared fixed grid; independent audio streams
use declared delivery-timed, equal-weight mixing. Missing media is declared
black or silence, and exhausted visual media can hold its last verified frame.
These are presentation decisions, not newly imagined frames or proof of native
cross-stream synchronization.

A retrospective recording preserves a known original stream anchor. If its
start is truncated and source timing is unknown, replay reports a gap unless
the user explicitly supplies the source-media offset. Historical recordings
without a measured end need an explicit presentation tail. Output quantization
and revisions superseded within one output frame are reported; their raw
fragments are retained even when the output clock cannot display both. The
original final media remains separately accessible.

Auto uses actual decoded source dimensions, native timebases, delivered cadence
and the host presentation refresh. Explicit dimensions and FPS are exact;
resource refusal does not silently substitute a smaller request. Audio-only
recordings produce a genuine PCM WAV, not a video with an invented canvas.
Visual replays produce an MP4 with measured output duration and frame count.
The label remains **Delivered timeline replay — presentation timing, not native
synchronization**. The controls stay inside existing recording details and the
chat Record drawer; see [the replay controls evidence](imagination-replay-ui.md).

The host runs the existing provisioned FFmpeg executable with shell-free
arguments and a local-file protocol whitelist. It checks actual required
encoders and binds reviewed runtimes to their binary, artifact, source and
notice identity. The current bundled six-target manifest is GPL-2.0-or-later,
with libx264 enabled; it is not labelled LGPL-only or legally certified by this
implementation. An explicitly configured external runtime is labelled as
external rather than distribution-reviewed. Actual codec work has resource,
disk, process-family cancellation and bounded progress-buffer checks.

## Replay storage and pending work

`imagination-assemblies/` contains sealed source/manifest SQLite files, an
incrementally written and hashed complete JSON manifest, content-addressed
playable output and receipts, durable preparation requests, and append-only
job state. Production planning and encoding iterate bounded rows and pages;
they do not load a recording-sized fragment or segment array. The five-minute
limit is retrospective window length, not an always-save recording limit.
Ten-second encoding checkpoints bound individual work spans without capping
the total recording duration.

Each committed segment advances the SQLite job cursor atomically. A snapshot
waits for admitted preparation and parks an encoder at its committed boundary,
copies complete custody, then may continue the original owner's work. A copied,
imported or merged pending job is dormant. Explicit Resume validates its exact
revision, source seal and retained segments; it does not rerun committed
segments. Failed and cancelled jobs require a new explicit assembly request.
Stop retains original fragments, committed segments and useful partial output.

Complete workspace snapshots include current preparation receipts, sealed
plans, committed cursor/history, pending-job state, source revisions and
completed outputs. Fork/import rekey current host ownership without rewriting
immutable original recording identities. Merge and recovery preserve distinct
histories and deduplicate identical content. A main-owned `currentRecordingId`
alias supports the selected-recording UI while the original manifest identity
stays unchanged. No copy grants execution, loads a model or restarts an encode.

## Scoped verification

Twenty-one distinct new backend cases passed in short targeted runs:

| File | Cases and boundary |
| --- | --- |
| `tests/imaginationCapture.test.ts` | 8: window settings, origin filtering, lookback and future capture, idempotent request, Stop and cancel, resource gaps, dormant copying, control revisions, and verified registry retention. |
| `tests/imaginationCaptureCustody.test.ts` | 3: complete active and live-window snapshot, pending IO and Fresh behavior, merge deduplication, immutable raw archives, and reexport/import. |
| `tests/imaginationCaptureContext.test.ts` | 4: structured origin, exact native lease handshake, untrusted flag removal, and foreign acknowledgement refusal. |
| `tests/imaginationCapturePreload.test.ts` | 1: typed controls and read views without frame or lease authority; subscription cleanup. |
| `engine/tests/test_spontaneous_output.py` | 5: fresh one-use owners, unchanged user outputs, untrusted provenance refusal, bounded unload cleanup, and preserved unsafe or altered scratch evidence. |

The native source compile and scoped whitespace checks passed. These cases used tiny real media files, actual host storage, and synthetic authorized native events; they did not run a model or decoder. The complete repository fork, recovery, export, import, and merge entry points are source-wired to the tested shared workspace helpers, not a newly repeated full repository workflow. Renderer controls and their separate presentation checks are documented by the UI integration owner.

The later playable-replay slice adds fifteen distinct cases, separate from
those closed twenty-one:

| File | Cases and boundary |
| --- | --- |
| `tests/imaginationReplayPlan.test.ts` | 6: strict options, measured end, source anchors, replacement prefixes, unknown timing and honest presentation states. |
| `engine/tests/test_media_timeline.py` | 3: byte-encoder counts/timebases, stable-prefix validation and actual container-audio descriptors; no neural imports or generation. |
| `tests/imaginationReplaySpool.test.ts` | 2: all 2,001 revisions retained with bounded pages and one content inspection, seal/tamper refusal, and a one-hour timeline without a five-minute recording ceiling. |
| `tests/imaginationReplayCodec.test.ts` | 4: actual two-revision PNG-to-MP4 with distinct decoded frames, genuine audio-only WAV without old-prefix replay, pending export/copy/explicit resume from a committed segment, and exact runtime-preparation progress/cancellation/nonrestart. |

The preparation case passed alone in 92 ms with the three codec cases skipped.
The three affected codec/custody cases then passed in 1.377 seconds after the
explicit concat-clock repair, with preparation skipped. That repair fixed a
real single-frame intermediate-container fault: the final concat had inferred
1,000 FPS and produced three frames in 3 ms instead of the declared 300 ms.
Per-segment durations and the exact final output clock now preserve the
declared presentation; the output validator was not weakened.

The separate SQLite replay-job helper has eight additional small cases covering
atomic segment/head publication, exact retry, CAS, immutable source identity,
corruption, bounded iteration, cancellation, read-only custody, dormant rekey
and schema admission. These cases are component evidence, not an end-to-end
portable-package test. Full-history validation runs at custody boundaries;
ordinary status opens inspect the indexed current head, not every historic row.

Real media fixtures used the already installed macOS ARM64 imageio FFmpeg 7.1
on tiny PNG and WAV inputs. They did not download or exercise the packaged
six-platform runtime. Their resource callbacks were fixture adapters, so the
playback results are not a production memory-pressure qualification. No model,
provider, corpus, camera or microphone ran. Existing capture tests and the
separate SQLite helper cases were not repeated for the final codec repair.

## Remaining acceptance boundaries

Native-chat `source:brain` does not establish whether an image or other output was requested by the user or arose spontaneously. A genuine learned intent signal and its end-to-end behavior remain open; keyword classification or deleting unknown outputs would not satisfy that boundary.

Playable delivered-timeline assembly is implemented, but native synchronized
imagination is not established by it. Actual progressive decoder delivery,
learned image/audio/video quality, cross-platform packaged encoder execution,
long-recording throughput and production pressure behavior remain separate
acceptance boundaries. The tiny custody fixture exercises the shared complete
workspace pipeline, not every repository recovery/merge/package UI workflow.
Preparation cancellation used a controlled owned runtime transfer, not a live
download. Native source changes also require a new authentic supported
continuation frontier before a fresh exact-resume claim; old seals must not be
retrofitted.
