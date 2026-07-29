# Omni AGI Studio stable-v1 completion audit

## Audit boundary

This is a source-state audit of the current stable-v1 implementation. It is
not evidence that an unpublished commit passed native CI, was signed or
notarized, was merged to `main`, or was published as `v1.0.0`.

OmniCortex is a brain-inspired research system. This document does not claim
consciousness, biological equivalence, human-level intelligence, frontier
model quality, guaranteed understanding, perfect memory, or proven AGI. The
bundled Omni Starter is a small project-authored trained baseline. It is not a
large externally published foundation checkpoint, and a Blank Brain remains
primitive until trained.

The detailed implementation evidence is in
[REQUIREMENTS.md](REQUIREMENTS.md). The original-request, line-by-line audit is
in [ORIGINAL_REQUEST_COMPLIANCE.md](ORIGINAL_REQUEST_COMPLIANCE.md). Research
sources and independent implementation boundaries are recorded in
[../RESEARCH.md](../RESEARCH.md).

## Current implementation map

“Implemented” below means that current-revision code and a named automated
fixture exist. It does not replace the final clean test, package, and native-CI
gates.

| Area | Current implementation | Evidence |
| --- | --- | --- |
| Custom local model | `OmniDecoder` is a project-authored PyTorch decoder with RMSNorm, rotary causal attention, a gated feed-forward path, a whole-input global workspace, and no hosted or third-party chat runtime | `engine/omni_core/model.py`; `engine/tests/test_model.py`; `tests/projectIntegrity.test.ts` |
| Mandatory ternary forward path | Eligible linear, convolutional, spiking, liquid, workspace, action, expert, and modality projections use effective `{-1, 0, +1}` weights. Higher-precision master weights, scales, activations, optimizer state, liquid state, and STDP traces remain explicit | `engine/omni_core/model.py`; `engine/omni_core/spiking.py`; `engine/omni_core/liquid.py`; `engine/tests/test_model.py`; `engine/tests/test_dynamics.py` |
| Packed ternary inference state | Current and immutable-origin generations export deterministic two-bit shards; strict verification rejects dense eligible paths, missing tensors, reserved codes, noncanonical padding, or checksum changes | `engine/omni_core/ternary_packing.py`; `engine/tests/test_ternary_packing.py`; `engine/omni_core/brain.py`; `tests/brainRepository.test.ts` |
| Unified associative memory | `NeuralSubstrate` is the authoritative store for concept neurons, distributed idea assemblies, and plastic ternary synapses. Desktop concepts and ideas are derived inspection views, not a second memory database | `engine/omni_core/vsa.py`; `engine/omni_core/brain.py`; `engine/tests/test_memory_modalities.py`; `tests/stableBrainInspection.test.ts` |
| Neuromorphic and liquid dynamics | Stateful LIF activity, causal and anti-causal STDP, metaplastic stability, exact ternary recurrent synapses, CfC control, and experimental LTC dynamics are implemented and persistent | `engine/omni_core/spiking.py`; `engine/omni_core/liquid.py`; `engine/tests/test_dynamics.py` |
| Working memory and whole-input integration | A hardware-sized temporary token window, recurrent liquid state, salience-managed latent slots, rehearsal, decay, eviction, and bidirectional whole-input latent integration feed causal decoding without pasting long-term source passages into the prompt | `engine/omni_core/model.py`; `engine/omni_core/brain.py`; `engine/tests/test_brain.py`; `tests/stableBrainInspection.test.ts` |
| Organic behavior | Curiosity, compute demand, pondering, branching, and generation noise are calculated from prediction error, uncertainty, novelty, learning progress, liquid state, and activity pressure. Stable Build exposes no personality, curiosity, noise, plasticity, neuron-cap, or parallel-thought slider | `engine/omni_core/brain.py`; `engine/omni_core/config.py`; `src/shared/types.ts`; `src/renderer/src/App.tsx`; `engine/tests/test_config.py`; `engine/tests/test_brain.py`; `tests/projectIntegrity.test.ts` |
| Initial and continual learning | Build offers a locally trained Omni Starter or a truly random Blank Brain. Conversation, files, datasets, web experiences, tools, and media can update fast STDP state and queued slow parameters without RLHF, DPO, a reward model, or a hidden persona prompt | `engine/omni_core/starter.py`; `engine/omni_core/brain.py`; `src/main/brainService.ts`; `engine/tests/test_brain.py`; `tests/brainService.test.ts` |
| Whole-dataset traversal | Deterministic streaming manifests, cursors, epochs, coverage, pause/resume, incremental hashing, and transactional commits replace the former file, byte, character, row, and chunk product ceilings | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `tests/dataIngestion.test.ts`; `engine/tests/test_datasets.py` |
| Data formats | Readers cover text/source, PDF, EPUB, Office/OpenDocument text, HTML, CSV/TSV, JSON/JSONL, Parquet, Arrow IPC, SQLite, ZIP/TAR/WebDataset, local Hugging Face-style manifests, remote manifest shards, images, audio, and video. Invalid records are reported rather than silently counted as learned | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `src/main/brainService.ts`; `engine/tests/test_datasets.py`; `engine/tests/test_release_gates.py` |
| Web learning | A brain-local SQLite frontier supports same-site continuous crawling, bounded parallel fetches, persistent resume, deduplication, pacing/backoff, robots controls, external-link opt-in, quarantine, cancellation, provenance, and linked image/audio/video learning | `src/main/dataIngestion.ts`; `src/main/brainService.ts`; `tests/dataIngestion.test.ts` |
| Uploads in all user surfaces | Build, Data Studio, and Run/chat expose native selectors for general files and datasets, images, audio, video, and folders. Run also accepts drag-and-drop and displays a neural-learning receipt | `src/shared/uploadSupport.ts`; `src/main/ipc.ts`; `src/preload/index.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Multimodal learning and imagination | Compact trainable vision, VQ/image diffusion, RVQ/audio, and liquid-gated factorized video baselines share the idea space. Ingestion trains the matching packs; generation can start from active assemblies | `engine/omni_core/modalities.py`; `engine/omni_core/brain.py`; `engine/tests/test_memory_modalities.py`; `engine/tests/test_release_gates.py` |
| Natural typed actions | A learned neural action head scores `talk`, `tool`, `imagine`, `agent`, `ponder`, `learn`, `evolve`, or `stop`. Only the dedicated typed action channel is executable; response prose, slash commands, and tagged text are ignored. Generic neural tool ID/argument materialization is provisional pending its final bridge test | `engine/omni_core/model.py`; `engine/omni_core/brain.py`; `src/main/actionProtocol.ts`; `src/main/chatActionController.ts`; `tests/actionProtocol.test.ts` |
| Real-time imagination | Image, audio, and video generators publish ordered intermediate media previews while a chat turn remains active. Final artifacts and visible tool results re-enter the continuous experience only after completion | `engine/omni_core/modalities.py`; `engine/omni_core/brain.py`; `engine/worker.py`; `src/main/chatActionController.ts`; `tests/actionProtocol.test.ts`; `engine/tests/test_worker.py` |
| Tools directly in chat | The chat controller accepts permission-checked file, PowerShell, code, web, guarded browser, imagination, agent, and source-evolution typed actions with visible cards and cancellation. Imagination, agent, and evolution have an end-to-end neural materializer; the generic neural tool bridge is still a provisional current-revision integration | `tools/catalog.json`; `engine/omni_core/brain.py`; `src/main/toolExecutor.ts`; `src/main/chatActionController.ts`; `engine/tests/test_tool_schemas.py`; `tests/toolExecutor.test.ts` |
| Agents and identity forks | Each build has one continuous identity and chat. Duplicate and fork create copy-on-write neural identities; reviewed merges import assemblies, evidence, artifacts, files, and replay examples without averaging whole identities | `src/main/brainRepository.ts`; `src/main/brainService.ts`; `src/renderer/src/App.tsx`; `tests/brainRepository.test.ts`; `tests/brainService.test.ts`; `tests/toolWorkflows.test.ts` |
| Recursive improvement | Neural and data candidates autonomously train in isolated safe-tensor overlays with immutable capability, retention, integrity, ternary, and resource checks, plus promotion lineage and rollback. Source candidates get an isolated Git worktree and evaluator, but the source proposal does not itself edit code | `src/main/evolutionController.ts`; `engine/omni_core/evolution.py`; `engine/worker.py`; `tests/evolutionController.test.ts`; `engine/tests/test_neural_evolution.py` |
| Storage, export, and sharing | Stable `.omni` bundles include checksum-bound current/origin state and packed ternary generations without executable pickle. Portable and content-addressed local-reference exports, explicit beta rejection/deletion, catalog recipes, and modality packs are implemented | `src/main/brainRepository.ts`; `src/main/catalogInstaller.ts`; `engine/omni_core/ternary_packing.py`; `tests/brainRepository.test.ts`; `tests/catalogInstaller.test.ts` |
| Streamlined UI and inspection | Build is four stages with checkboxes rather than behavior sliders. Run keeps chat, learning, live actions, uploads, Data Studio, a paged multiresolution substrate map, trace, journal, tools, agents, imagination, evolution, duplicate, and export in one workspace | `src/renderer/src/App.tsx`; `src/renderer/src/styles.css`; `tests/projectIntegrity.test.ts`; `tests/stableBrainInspection.test.ts`; `tests/uploadSupport.test.ts` |
| Cross-platform packaging definitions | Windows x64/ARM64, macOS Intel/Apple Silicon, and Linux x64/ARM64 build/package/smoke workflows and guarded release publication are defined. This row does not assert those native jobs are green for the current revision | `package.json`; `.github/workflows/windows.yml`; `.github/workflows/macos.yml`; `.github/workflows/linux.yml`; `.github/workflows/release.yml`; `tests/releasePackaging.test.ts` |
| Source and license cleanup | The BitNet, snnTorch, and NCPS research snapshots are removed from the application source tree; preserved upstream license texts are packaged from `licenses/`, and the notice identifies them as research references rather than runtime imports | `licenses/BitNet-MIT.txt`; `licenses/snnTorch-MIT.txt`; `licenses/NCPS-Apache-2.0.txt`; `THIRD_PARTY_NOTICES.md`; `package.json` |

## Verification status for this revision

The final release gate must use the final merged commit. Earlier local passes
or prior beta Actions runs do not prove the present source tree.

| Gate | Current audit status |
| --- | --- |
| Python neural, dataset, modality, packed-ternary, worker, recovery, and neural-evolution suites | Named fixtures exist and the suite has a recent local pass; rerun after all concurrent stable-v1 edits before release |
| Node/Electron unit, repository, upload, action, dataset, tool, agent, evolution, integrity, and release-verifier suites | Named fixtures exist; final clean full-suite rerun is required |
| TypeScript typecheck and production Electron build | Defined; final clean rerun is required |
| Built Electron UI flow | Defined in `tests/e2e/electron.spec.ts`; final current-revision run is required |
| Windows x64 and ARM64 packages | Workflow and smoke gates are defined; no current-revision native-green claim is made here |
| macOS Intel and Apple Silicon packages | Workflow and smoke gates are defined; no current-revision native-green claim is made here |
| Linux x64 and ARM64 packages | Workflow and smoke gates are defined; no current-revision native-green claim is made here |
| Signing and notarization | Conditional on repository credentials; no signed or notarized artifact is claimed |
| `main`, `v1.0.0`, GitHub Release, and only-branch cleanup | Not complete at this audit point |
| Public Omni Starter checkpoint, dataset ledger, and external loss curves | Not published; the bundled starter is only the small local project-authored baseline |
| Generic file/web/code/browser action materialization from the learned neural `tool` class | Provisional integration; must pass a real worker-to-executor fixture before it is counted complete |
| Evolution candidate review, approval, and rollback controls in the Run renderer | Not present at this audit point; preload/IPC/controller APIs exist, but the renderer does not call them |
| Pressure/convergence-governed generation branching and spreading activation with no implementation-defined step count | Partial: user-facing controls and fixed top-k recall are removed, but chat still uses tier branch budgets and spreading activation still has a four-hop loop |
| Full-Authority source improvement installs a side-by-side rebuilt desktop binary and restarts into it | Not implemented: verified source diffs can be promoted to Git, but the running packaged binary is not replaced or restarted |
| Source evolution autonomously writes its proposed code change | Not implemented: the proposal creates an isolated worktree and task record; another explicitly authorized editor must make the candidate diff |
| Idle cognition can initiate a spontaneous conversational question/message | Not implemented: idle cycles rehearse and can propose typed external actions, but they do not append a `talk` message to the chat |

## Important engineering limits

- “1.58-bit” applies to eligible effective forward weights. It does not mean
  every byte of state is ternary; master weights, activations, normalization,
  optimizer state, liquid state, and plasticity traces require higher
  precision.
- `NeuralSubstrate` unifies authoritative associative memory, but cortical
  master tensors and sparse synapse metadata are different physical data
  structures. The implementation does not pretend every scalar is one
  biological synapse.
- Sparse neurons, assemblies, synapses, and residual experts can grow until a
  RAM/disk reserve pauses them. The live dense decoder does not resize its
  tensor shapes in place, and stable v1 rejects architecture-shape evolution.
- Source evolution promotes a verified Git change. It does not currently build
  and install a replacement desktop binary or restart the running app into that
  binary.
- Source evolution creates and guards an isolated Git worktree, but the
  proposal action does not itself author a code diff. Autonomous recursive
  source editing is therefore incomplete even though neural/data candidate
  training is real.
- Generation branch count is currently bounded by a hardware-tier budget, and
  spreading activation currently uses a four-hop propagation loop. These are
  internal implementation ceilings even though neither is exposed as a user
  personality control.
- Dataset traversal can prove that each valid manifested record was visited;
  it cannot prove that a small model understood or perfectly remembered every
  record.
- “Unlimited” file, dataset, crawl, idea, or growth behavior means no
  product-defined cardinality ceiling. Filesystems, RAM, remote servers,
  decoder availability, cancellation, robots policy, and configured resource
  reserves remain real boundaries.
- Common image, audio, and video formats are supported through installed local
  decoders. “All forms” cannot be guaranteed for corrupt, encrypted,
  proprietary, DRM-protected, or decoder-unsupported media.
- The guarded browser protocol renders a script-disabled public-page snapshot
  and returns text, links, and a screenshot. It is not interactive signed-in
  browser automation.
- Exact source recall requires Total Recall. Human Consolidation and Synapses
  Only are intentionally lossy.
- Operational traces expose seeds, activations, routes, action events, and
  parameter deltas. They are not asserted to be a faithful private
  chain-of-thought transcript.
- No deception, obedience, refusal, personality, or “unaligned” objective is
  trained. There is no hidden behavioral prompt or preference model, but
  external actions still obey explicit host permissions.
- The model may produce uncertain, incorrect, incoherent, or repetitive
  output, especially from a Blank Brain or the small bundled starter.

## Release blockers

Stable `v1.0.0` is not complete until the final source tree passes every named
local gate, all six native package jobs and their packaged-app smoke tests,
artifact-set verification, merge-to-`main`, exact tagging, release
publication, and the requested branch cleanup. Credential-dependent signing
and notarization must be reported as either verified or unavailable, never
implied.
