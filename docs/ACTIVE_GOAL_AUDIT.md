# Active Goal Completion Audit

This matrix is the authoritative checklist for the post-v1 implementation goal.
An item is complete only when current source, automated tests, and (where
applicable) a real packaged runtime exercise provide direct evidence.

Status values: `implemented`, `partial`, `missing`, `unverified`, `blocked`.

## Model and brain outcome

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| A stable trained brain answers coherently instead of producing gibberish | Existing beta Vera replies are single-character/gibberish; bundled Starter has only a tiny fixture corpus | missing | Train/import a stable-format brain; pass semantic quality probes; preserve and reload it |
| Starter knowledge plus continual self-learning | Starter/bootstrap and online plasticity exist, but useful pretrained capability is not demonstrated | partial | Provenance-bearing starter checkpoint; held-out quality and retention results before/after continual learning |
| Training survives RAM pressure through updateable disk-backed state | Checkpoints/replay can be stored on disk, but optimizer/activation pressure does not yet trigger a transactional spill/resume path | missing | RAM-watermark test spills optimizer/replay/activation shards, continues updating them, restarts without skipped commits, and preserves reserve space |
| Parameter/synapse memory instead of hidden prompt retrieval | Worker uses bounded dialogue plus neural state; prompt-injection invariants have tests | implemented | Keep invariant tests green during all changes |
| Working memory resembles a bounded global workspace | Token context, latent slots, liquid state, admission/eviction and consolidation exist | implemented | Live Runtime Card and restart tests on the trained brain |
| Neuroplasticity, STDP, ternary weights, fuzziness, traceability | LIF/STDP, ternary projections, liquid dynamics, seeds and operational traces exist | implemented | Live mutation evidence on the trained brain; exact ternary coverage test stays green |
| Duplicate as a recovery copy and prove divergence isolation | Copy-on-write duplicate exists and repository tests cover provenance | partial | Duplicate trained brain, train the copy, prove original checksum/output remains unchanged, reload both |
| Recursive self-improvement | Candidate/evaluation/promotion/rollback controller exists | partial | Run a candidate on the trained instance; show objective gain, promotion record, rollback and evaluator protection |

## Conversation, activity, voice, and imagination

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| Human message appears immediately | Root cause confirmed; renderer-ephemeral optimistic turn patch is in progress | partial | Deterministic delayed-worker E2E plus cancellation/failure/reconciliation checks |
| Always-active capped cognition can speak/work organically | Prompt-free idle cycle exists only while app is running, with fixed scheduler cadence | partial | User-facing active lifecycle/speed caps, spontaneous action/talk live test, restart and cancellation behavior |
| Interruptible live voice with forked utterance chunks | No microphone, ASR, TTS, duplex loop or barge-in implementation | missing | Mic capture, local/provider ASR/TTS, chunk-fork state, cutoff test, device permission UI |
| Natural on-demand and spontaneous imagination | Neural action head and idle imagination exist; quality and natural behavior are unproven | partial | Coherent trained action selection and visible spontaneous/requested image, audio and video runs |
| Brain may open a creativity menu | No allowlisted studio UI action; gallery button is inert | missing | Typed `studio.ui/open-creativity` action, durable gallery/drawer and natural-language/organic tests |
| Realtime image, audio and video generation | Tiny research baselines and progressive preview exist | partial | Useful quality packs, latency/throughput targets, cancellation, progressive artifacts and live runtime evidence |
| Video generation/training supports optional synchronized audio | Video paths currently discard audio | missing | Demux/fused representation, audio-preserving ingest, synchronized generation and round-trip tests |

## Data, tools, integrations, and training

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| All supported text/data types stream over the full dataset | Streaming manifests/cursors and broad format coverage exist | partial | Large fixtures and 100% valid-record coverage report in a live training run |
| Image/audio/video files, folders and crawls train neural state | Ingest/backprop paths exist, but generic media-only slow training can fall back to consolidation | partial | Per-modality parameter-delta tests and full-dataset live coverage |
| Web crawl can train continuously and resume | Persistent crawler exists | implemented | Live crawl pause/resume/cancel/coverage evidence |
| Natural tool/coding/file/MCP use directly in chat | Typed built-in tools exist; generic MCP integration is absent | partial | MCP server registry/schema encoder/action cards plus code/file live tests |
| Simplified System Access and explicit Full Authority controls | Per-tool Off/Ask/Auto/Full controls exist | partial | Consolidated basic toggle/master grant without weakening per-action audit or credential boundaries |
| Brain can inspect available capability settings and request changes | Tool schemas expose enabled capabilities, but no safe settings-inspection protocol | missing | Read-only settings capability embedding and typed request-to-open/change-settings flow |
| Approval waits 30 seconds by default and is configurable | No user-facing adjustable approval countdown | missing | Persisted timeout preference, visible countdown, wait/deny/continue tests |
| Teacher-API training for OpenAI, Anthropic and Gemini | Not implemented | missing | Credential-safe provider adapters, dataset/trajectory capture, distillation objectives, provenance, rate/cost controls and mock/live opt-in tests |
| Credentials never enter model memory, logs or `.omni` exports | Existing exports exclude ordinary credentials; new provider credentials do not yet exist | unverified | OS credential-store integration and redaction/export tests for every provider |

## Product experience and distribution

| Requirement | Current evidence | Status | Completion evidence required |
| --- | --- | --- | --- |
| Independent color and layout choices | Fixed dark theme only | missing | System/light/dark plus Standard, Classic Blocky, Colorful and Liquid Glass packs; persistence and screenshot tests |
| Responsive all screen sizes/resolutions | Narrow CSS exists but Electron minimum width makes it unreachable; no mobile matrix | partial | 390x844 through ultrawide matrix, touch/safe-area/short-height tests and accessible inspector drawer |
| Chat/media UI remains performant | Duplicate full action events, base64 previews and per-token scroll/rerender are confirmed hot paths | partial | Single ordered event path, lightweight artifact handles, buffered tokens, windowed history and performance tests |
| Android APK | No Android project/workflow | missing | Reproducible APK, emulator create/train/chat/tool/media smoke and green CI |
| iOS custom-signing IPA | No iOS project/workflow | missing | Simulator tests, unsigned/custom-sign archive path, credential-gated IPA export and green CI |
| Portable Windows/macOS/Linux, including Unix | Desktop installers/archives and OS/architecture workflows exist | implemented | Preserve packaged smoke tests and re-run after shared-runtime changes |
| Cross-platform CI and release artifacts | Existing desktop CI is present; mobile jobs are absent | partial | All required desktop/mobile jobs green and checksummed artifacts published from verified main |

## Trained-model decision record

The existing hardware-resolved GPU profile is not a useful foundation model: its
complete current neural stack is about 2.05 million parameters, its decoder is
about 618 thousand parameters, it uses a byte tokenizer, it has no KV cache,
and Apple Silicon currently falls back to CPU. The bundled starter sees only 12
project-authored passages. It therefore cannot satisfy the coherent/non-gibberish
gate by simply running longer on the supplied corpus.

The supplied training folder currently contains:

- all 14 FineWeb-Edu 10BT Parquet shards (9,672,101 rows and approximately
  9.97 billion source-token metadata count);
- TinyStories, filtered UltraChat SFT material, and a SmolLM2-135M Base teacher;
- the user-provided decompressed Pile shard `00.jsonl` (about 34.4 GB).

Training must select dataset content fields explicitly, stream without creating
a second tokenized corpus copy, preserve a deterministic cursor, and keep at
least 20 GiB of disk free for checkpoints and mutable neural state. Python-Edu
metadata is not code content and must be rejected until its referenced blobs are
resolved with valid provenance.

The shortest acceptable route is a verified native-ternary **base** cortex with
Omni's STDP/CfC/VSA substrate, action head, working memory, modality adapters,
and continual-learning adapters around it. Falcon-E-1B-Base is a technical
candidate, but its Falcon LLM license and exact no-preference-training provenance
must pass review before it can be a distributed default. Microsoft's BitNet
2B4T checkpoint is excluded from the clean default because its model card records
SFT and DPO. If no compatible native checkpoint passes those gates, the fallback
is distillation from the downloaded SmolLM2 **Base** teacher into a ternary
student; the dense teacher is never shipped as the runtime brain.

Promotion requires exact ternary and packed-kernel parity, complete provenance,
no hidden prompt or raw-memory injection, 100 seeded non-gibberish probes,
held-out capability/assistant-span improvement, plastic recall after restart,
no more than two percentage points of baseline retention loss, action accuracy,
resource-reserve compliance, and duplicate-isolation evidence.

## Required live end-to-end sequence

The final acceptance run must, in order:

1. Build and open the actual packaged application.
2. Create a stable brain from a useful Starter or a documented trained origin.
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
