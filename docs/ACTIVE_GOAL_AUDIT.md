# Active Goal Completion Audit

> Historical broad-goal audit. The user's latest narrowed source-cleanup goal
> and newer ternary-only correction are tracked in
> [Ground-up design correction checklist](GROUND_UP_DESIGN_CORRECTION_CHECKLIST.md).
> Rows below are not proof that the corrected design is complete.

This matrix is the authoritative checklist for the post-v1 implementation goal.
An item is complete only when current source, automated tests, and (where
applicable) a real packaged runtime exercise provide direct evidence.

Status values: `implemented`, `partial`, `missing`, `unverified`, `blocked`.

## Hardware-safe working memory and scratch disk

- Build/start uses the same live-device preflight with a device-adaptive
  free-space floor (up to the 20 GiB desktop recommendation), checkpoint
  headroom, native model learning state, and the
  configured recurrent/paged memory spill.
- Auto is hardware-suitable; Extended is visibly higher-cost; Manual has a
  live theoretical-safe slider maximum and accepts larger typed values only
  when the physical preflight passes.
- Storage use and a bounded cached slowdown estimate are always visible.
- Auto keeps the configured logical optimizer target exact, choosing the
  largest safe physical divisor up to four and deriving accumulation as the
  quotient. A first uncommitted allocator refusal retries `4 → 2 → 1`; the
  first successful schedule is hash-bound and frozen.
- Full mutable-state generations normally commit every 512 learned source
  records plus the final partial group, rather than writing optimizer state on
  every step. Emergency scratch remains sequential and storage-rate-limited.
- RAM continually prioritizes firing, frequently used, rooted/stable, and
  unfinished state; cold scratch, replay, optimizer, and inactive patterns
  spill first into restart-safe non-pickle storage.
- Evidence: `src/main/resourcePlanner.ts`,
  `docs/RESOURCE_MEMORY_PLANNER.md`, `tests/resourcePlanner.test.ts`, and
  `engine/tests/test_state_offload.py`.

## Model and brain outcome

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| A stable trained brain answers coherently instead of producing gibberish | Ground-up construction/readiness is implemented, but its narrow capability curriculum is not a general-language corpus | missing | Explicitly train the ground-up brain; pass semantic quality probes; preserve and reload it |
| Ground-up knowledge plus continual self-learning | Seeded native initialization, local capability training, and online plasticity exist; useful broad capability is not demonstrated | partial | Provenance-bearing user-data training receipts plus held-out quality and retention results before/after continual learning |
| Training survives RAM pressure through updateable disk-backed state | Live preflight reserves model/checkpoint/memory space; replay, optimizer/activation pressure scratch, and cold working patterns use transactional restart-safe disk state; a platform/device-derived OS and recovery floor remains outside neural use | implemented | Keep `tests/resourcePlanner.test.ts`, `tests/diskSpace.test.ts`, and `engine/tests/test_state_offload.py` green in the final sweep |
| Parameter/synapse memory instead of hidden prompt retrieval | Worker uses bounded dialogue plus neural state; prompt-injection invariants have tests | implemented | Keep invariant tests green during all changes |
| Working memory resembles a bounded global workspace | Token context, recurrent/paged activity, liquid state, salience admission, interference, rehearsal, decay, and eviction interact continuously | implemented | Live Runtime Card and restart tests on the trained brain |
| Neuroplasticity, STDP, ternary weights, fuzziness, traceability | LIF/STDP, ternary projections, liquid dynamics, seeds and operational traces exist | implemented | Live mutation evidence on the trained brain; exact ternary coverage test stays green |
| Duplicate as a recovery copy and prove divergence isolation | Copy-on-write duplicate exists and repository tests cover provenance | partial | Duplicate trained brain, train the copy, prove original checksum/output remains unchanged, reload both |
| Recursive self-improvement | Candidate/evaluation/promotion/rollback controller exists | partial | Run a candidate on the trained instance; show objective gain, promotion record, rollback and evaluator protection |

## Conversation, activity, voice, and imagination

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| Human message appears immediately | Root cause confirmed; renderer-ephemeral optimistic turn patch is in progress | partial | Deterministic delayed-worker E2E plus cancellation/failure/reconciliation checks |
| Always-active capped cognition can speak/work organically | Prompt-free idle cycle exists only while app is running, with fixed scheduler cadence | partial | User-facing active lifecycle/speed caps, spontaneous action/talk live test, restart and cancellation behavior |
| Interruptible live voice with forked utterance chunks | Explicit audio-only microphone permission, continuous platform Web Speech input/TTS output, typed live/buffered delivery, persisted pacing, bounded incremental speech backpressure, exact-turn barge/Stop cancellation, late-reply rejection, and focused tests are implemented. The optional neural-listening path now streams the waveform into the same brain's persistent assemblies/STDP while Web Speech remains only the literal transcript boundary. Generic audio generation is explicitly not advertised as intelligible neural voice | partial | Exercise a physical microphone and platform recognizer in packaged Windows/macOS/Linux builds; a future neural-voice capability requires a verified intelligible speech pack rather than relabelled generic audio |
| Natural on-demand and spontaneous imagination | Neural action head and idle imagination exist; quality and natural behavior are unproven | partial | Coherent trained action selection and visible spontaneous/requested image, audio and video runs |
| Brain may open a creativity menu | The always-available local `studio.ui/open-creativity` capability is encoded through the neural schema channel, executes through the same visible/audited action-card path, and opens Imagination for requested or organic activity. It is local and reversible, so it never widens file, process, network, or source authority; prose and slash commands remain inert | implemented | `engine/omni_core/brain.py`; `src/main/brainService.ts`; `src/main/toolExecutor.ts`; `src/renderer/src/studioUiActions.ts`; `src/renderer/src/App.tsx`; `engine/tests/test_tool_schemas.py`; `tests/actionProtocol.test.ts`; `tests/studioUiActions.test.ts`; `tests/toolExecutor.test.ts` |
| Realtime image, audio and video generation | Tiny research baselines and progressive preview exist | partial | Useful quality packs, latency/throughput targets, cancellation, progressive artifacts and live runtime evidence |
| Video generation/training supports optional synchronized audio | Video ingestion trains visual frames and embedded audio into the same neural identity. When both trained decoders are available, generation decodes video and abstract sound from the exact same internal idea, aligns sound to frame duration, and muxes H.264/AAC MP4; silent/disabled/untrained/APNG fallbacks are explicit and never called speech. Automated MP4 audio-track and inline-imagination tests pass | partial | Exercise synchronized playback on packaged Windows/macOS/Linux builds and record hands-on artifact evidence |

## Data, tools, integrations, and training

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| All supported text/data types stream over the full dataset | Streaming manifests/cursors and broad format coverage exist | partial | Large fixtures and 100% valid-record coverage report in a live training run |
| Image/audio/video files, folders and crawls train neural state | Ingest/backprop paths exist; generic media experience can also enter automatic replay and slow-weight learning | partial | Per-modality parameter-delta tests and full-dataset live coverage |
| Web crawl can train continuously and resume | Persistent crawler exists | implemented | Live crawl pause/resume/cancel/coverage evidence |
| Natural tool/coding/file/MCP use directly in chat | Streamable HTTP and local stdio MCP registry, discovery, namespaced permissions, structural neural schema encoding, natural action materialization, audited calls and mocked protocol tests are implemented | partial | Exercise a connected MCP tool through natural chat on the final created/trained model |
| Simplified System Access and explicit Full Authority controls | Per-tool Off/Ask/Auto/Full controls exist | partial | Consolidated basic toggle/master grant without weakening per-action audit or credential boundaries |
| Brain can inspect available capability settings and request changes | `studio.settings` is a durable typed capability with read-only access inspection and local permission-workspace routing; its result explicitly proves no grants changed | implemented | `engine/omni_core/ground_up.py`; `src/main/toolExecutor.ts`; `src/renderer/src/studioUiActions.ts`; `tests/toolExecutor.test.ts`; `tests/studioUiActions.test.ts` |
| Approval waits 30 seconds by default and is configurable | Persisted 1–3,600 second preference, exact expiry timestamp, visible chat countdown, single-use argument binding and expiry cleanup are implemented | implemented | `src/main/toolPreferences.ts`; `src/main/toolExecutor.ts`; `src/renderer/src/App.tsx`; `tests/toolExecutor.test.ts` |
| Teacher-API training for OpenAI, Anthropic and Gemini | Current provider request adapters, explicit trajectory jobs, cancellation/progress, no-system/no-RLHF metadata, hashed provenance and neural-ingest mutation are implemented with mocked tests for all three providers | partial | Run an explicitly authorized live provider request on the final created model; cost/rate choice remains user-controlled through the submitted prompt list |
| Credentials never enter model memory, logs or `.omni` exports | OS-encrypted app-level storage is outside brain roots; hosts without secure encryption use process-memory only. Tests prove provider/MCP secrets are absent from learned trajectories and persisted declarative registries | implemented | `src/main/secureSecretStore.ts`; `src/main/teacherTraining.ts`; `src/main/mcpClient.ts`; `tests/integrationLearning.test.ts`; existing export secret rejection tests |

## Product experience and distribution

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| Independent color and layout choices | Fixed dark theme only | missing | System/light/dark plus Standard, Classic Blocky, Colorful and Liquid Glass packs; persistence and screenshot tests |
| Responsive all screen sizes/resolutions | Narrow CSS exists but Electron minimum width makes it unreachable; no mobile matrix | partial | 390x844 through ultrawide matrix, touch/safe-area/short-height tests and accessible inspector drawer |
| Chat/media UI remains performant | Duplicate full action events, base64 previews and per-token scroll/rerender are confirmed hot paths | partial | Single ordered event path, lightweight artifact handles, buffered tokens, windowed history and performance tests |
| Android APK | Native Kotlin client, reproducible Gradle project, same-brain gateway, streamed attachment learning, local ARM64 emulator UI run, 2/2 device tests, dedicated CI, and stable-release jobs exist; the release gate now hash-binds both the installable debug-signed and signing-ready unsigned release APKs to emulator evidence | implemented | Re-run the Android workflow from the final merged commit and publish the verifier-approved artifacts |
| iOS custom-signing IPA | Native SwiftUI iPhone/iPad client, Keychain pairing, streamed chat/upload, Xcode project, host-backed protocol smoke, iOS SDK typecheck, simulator unit/UI tests, unsigned/custom-signed IPA script, dedicated CI, and stable-release aggregation exist; this Mac still requires administrator acceptance of the Xcode license before local `simctl` can launch | partial | Run the simulator workflow from the final commit and publish its verifier-approved unsigned IPA; physical-device custom signing still requires user credentials |
| Portable Windows/macOS/Linux, including Unix | Desktop installers/archives and OS/architecture workflows exist | implemented | Preserve packaged smoke tests and re-run after shared-runtime changes |
| Cross-platform CI and release artifacts | Desktop workflows plus dedicated Android APK/emulator and iOS IPA/simulator workflows are present; the stable tag workflow waits for and verifies all desktop/mobile artifact sets before publication | partial | All required desktop/mobile jobs green and checksummed artifacts published from verified main |

## Ground-up model decision record

The earlier proposal to wrap or distill an external pretrained model is
superseded. Every new Build now initializes the native OmniCortex master weights
from its recorded seed and trains that same mutable core. Falcon, Llama,
Hugging Face checkpoints, installed model caches, and API models are not
new-Build origins or runtime fallbacks. Whole-brain Import likewise accepts
only exact-schema native ground-up OmniCortex state; frozen-foundation,
Starter, Blank, Falcon, and other legacy files are rejected without deletion or
relabeling.

The native architecture remains a small research system, not a useful general
language foundation merely because it is ground-up. Its exact Auto parameter
counts are versioned in `architecture/omnicortex-ground-up-v1.json`; they range
from 339,768 trainable master elements on Micro to 9,527,552 on Workstation.
These numbers and the narrow local capability curriculum cannot satisfy a
coherent/non-gibberish capability gate without substantial attributable
training and evaluation.

Available training folders may contain, for example:

- all 14 FineWeb-Edu 10BT Parquet shards (9,672,101 rows and approximately
  9.97 billion source-token metadata count);
- TinyStories, filtered UltraChat SFT material, and a SmolLM2-135M Base teacher;
- the user-provided decompressed Pile shard `00.jsonl` (about 34.4 GB).

None of these assets is consumed merely because it exists on disk. The user
must explicitly select training data and its content fields. Training then
streams without creating
a second tokenized corpus copy, preserve a deterministic cursor, and keep at
the measured device-adaptive disk reserve free for checkpoints and mutable neural state. Python-Edu
metadata is not code content and must be rejected until its referenced blobs are
resolved with valid provenance.

The only acceptable new-Build route is the project-owned native ternary cortex
with Omni's STDP/CfC/VSA substrate, action head, working memory, modality paths,
and continual-learning machinery. An API teacher is permitted only as an
explicit, provenance-recorded user training job; it is never loaded into the
runtime and never invoked as an inference fallback.

Promotion requires exact ternary and packed-kernel parity, complete provenance,
no hidden prompt or raw-memory injection, 100 seeded non-gibberish probes,
held-out capability/assistant-span improvement, plastic recall after restart,
no more than two percentage points of baseline retention loss, action accuracy,
resource-reserve compliance, and duplicate-isolation evidence.

## Required live end-to-end sequence

The final acceptance run must, in order:

1. Build and open the actual packaged application.
2. Create a stable ground-up brain and complete its documented local curriculum
   plus any explicitly selected initial training.
3. Train it on text plus image, audio, video-with-audio, and crawled material.
   The text run includes the supplied `/Users/eyadabushama/Downloads/ai train/`
   FineWeb-Edu shards and complementary provenance-checked datasets.
4. Pass semantic, non-gibberish, retention, action-selection and tool-use probes.
5. Exercise requested and spontaneous imagination, active cognition, and live voice interruption.
6. Duplicate it; mutate/train only the copy; prove origin and duplicate isolation.
7. Restart and prove both brains, working memory policy, long-term neural state,
   artifacts, traces, permissions and provenance persist correctly.
8. Run responsive/theme/accessibility/performance checks in the real UI.
9. Build and smoke desktop portable packages, Android APK and iOS simulator/custom-sign archive.
10. Run CI and compare every row above with direct final evidence.
