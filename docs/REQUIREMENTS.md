# Stable-v1 requirements and evidence

> Current source-cleanup scope and open design gaps are tracked in
> [Ground-up design correction checklist](GROUND_UP_DESIGN_CORRECTION_CHECKLIST.md).
> This older matrix records code and past fixtures, not a current trained
> ground-up brain, post-cleanup CI pass, or proof of natural recall.

## How to read this matrix

- **Implemented** means current-revision code and named test coverage exist.
- **Baseline** means a real trainable path exists, but research quality or
  biological equivalence is not claimed.
- **Partial** means a bounded part is implemented and the remaining limitation
  is stated.
- **Open release gate** means implementation exists but final current-revision
  native evidence or publication is still required.

Passing an individual fixture is not the same as passing the final release
matrix. See [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md) for that distinction and
[ORIGINAL_REQUEST_COMPLIANCE.md](ORIGINAL_REQUEST_COMPLIANCE.md) for the
line-by-line user-request audit.

## Brain substrate and computation

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Project-authored model; no third-party chat backend | Implemented | `engine/omni_core/model.py`; `tests/projectIntegrity.test.ts` |
| Decoder with causal attention, RMSNorm, rotary positions, gated feed-forward blocks, and a whole-input global workspace | Implemented | `engine/omni_core/model.py`; `engine/tests/test_model.py` |
| Packed ternary weights are authoritative for native eligible projections and sparse synapses | Source-implemented; full parameter audit open | Current native constructors use packed mutable codes and export now rejects a dense embedding or legacy float-master projection. Small learned normalization/gain/prototype controls still need exact classification against the user's later ternary-only request. `engine/omni_core/model.py`; `engine/omni_core/spiking.py`; `engine/omni_core/brain.py` |
| Temporary higher-precision computation is distinguished from resident learned weights | Partial | Activations and bounded update scratch are not a second saved brain. Do not claim every learned parameter is ternary until the remaining small controls are audited and converted or disclosed. `engine/omni_core/model.py`; `docs/MODEL_CARD.md` |
| Packed two-bit current and origin inference generations with strict coverage and integrity checks | Implemented | Verification includes dynamic-synapse count, order, and exact values, so reload refreshes a stale pack rather than accepting structurally valid but incorrect sparse weights. `engine/omni_core/ternary_packing.py`; `engine/tests/test_ternary_packing.py`; `engine/tests/test_brain.py`; `tests/brainRepository.test.ts` |
| One authoritative associative substrate for concept neurons, distributed idea assemblies, and typed plastic synapses | Implemented | `engine/omni_core/vsa.py`; `engine/omni_core/brain.py`; `engine/tests/test_memory_modalities.py` |
| LIF spiking and causal/anti-causal STDP with metaplasticity | Implemented | `engine/omni_core/spiking.py`; `engine/tests/test_dynamics.py` |
| CfC default temporal controller and experimental LTC path | Implemented | `engine/omni_core/liquid.py`; `engine/tests/test_dynamics.py` |
| VSA/HDC bind, bundle, permutation, approximate recall, and recurrent spreading activation | Implemented | `engine/omni_core/vsa.py`; `engine/tests/test_memory_modalities.py` |
| Recurrent spreading uses exact signed ternary edges | Source-implemented | The contribution is source activation times `-1` or `+1`, not a separate floating synapse magnitude. Negative edges suppress competing assemblies; a live ground-up recall outcome remains unverified. `engine/omni_core/vsa.py`; `engine/tests/test_memory_modalities.py`; `docs/SUBSTRATE_PERSISTENCE.md` |
| Spreading activation continues solely until convergence, interference, workspace pressure, or resource pressure | Implemented | `NeuralSubstrate.recall_vector()` has no hop counter; fan-in-normalized damped recurrent changes form a contraction and settle against an adaptive workspace-pressure floor. `engine/tests/test_memory_modalities.py` proves activation beyond four hops |
| Dynamic sparse growth without a product-defined neuron, assembly, synapse, or expert count | Partial | No fixed global count, but current large-source ingestion can choose shared statistical fields instead of per-record assemblies; the vectors/records need paging and indexed recall before unbounded large-corpus growth is honest. `engine/omni_core/vsa.py`; `engine/omni_core/brain.py`; `docs/GROUND_UP_DESIGN_CORRECTION_CHECKLIST.md` |
| Graceful growth pause before the configured RAM/disk reserve | Implemented | `engine/omni_core/vsa.py`; `engine/omni_core/brain.py`; `tests/dataIngestion.test.ts` |
| Growable substrate persists as bounded content-addressed shards rather than one unbounded tensor | Implemented | Atomic generation manifests reference hash-bucketed neuron/assembly/synapse JSON and safe-tensor shards; unchanged shards are reused and corrupt references fail closed. `engine/omni_core/vsa.py`; `engine/omni_core/persistence.py`; `engine/tests/test_memory_modalities.py`; `docs/SUBSTRATE_PERSISTENCE.md` |
| Dense base architecture can reshape itself while running | Partial | Safe architecture candidates can add load-compatible, initially zero-residual ternary experts with exact rollback. Arbitrary width/depth/router/modality tensor-shape migration remains intentionally rejected |

## Working memory, organic dynamics, and trace

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Temporary token/sensory context plus a limited latent global workspace | Implemented | `engine/omni_core/model.py`; `engine/omni_core/brain.py`; `engine/tests/test_brain.py` |
| Persistent liquid recurrent state and active assemblies | Implemented | `engine/omni_core/brain.py`; `engine/omni_core/liquid.py` |
| Salience admission, interference, rehearsal, decay, eviction, and inspectable occupancy | Implemented | `engine/omni_core/brain.py`; `tests/stableBrainInspection.test.ts` |
| Hardware-sized context with only an Extended working memory checkbox in basic Build | Implemented | New ground-up Builds derive separate active-token and recurrent/paged-memory capacities from the live resource envelope. The planner reports the resulting exact native parameter count and offload plan; compatible imports retain their recorded model/context shape. `architecture/omnicortex-ground-up-v1.json`; `src/main/resourcePlanner.ts`; `src/renderer/src/App.tsx`; `tests/omniArchitectureProfile.test.ts`; `engine/tests/test_ground_up_architecture_contract.py` |
| Long-term source prose is not silently appended to the generation prompt | Implemented | `engine/omni_core/brain.py`; `engine/tests/test_brain.py`; `engine/tests/test_tool_schemas.py` |
| Curiosity, pondering, branch count, and fuzziness emerge from measured internal state instead of sliders | Implemented | `engine/omni_core/brain.py`; `engine/omni_core/config.py`; `src/renderer/src/App.tsx`; `engine/tests/test_brain.py`; `tests/projectIntegrity.test.ts` |
| Branch/ponder computation has no fixed implementation-defined thought count | Implemented | Chat has no tier branch ceiling or user control; computation settles from measured score convergence, organic neural energy, working-memory scale, and disk/RAM reserve pressure |
| Prompt-free idle cognition can rehearse and propose typed actions | Implemented | `engine/omni_core/brain.py`; `src/main/idleCognitionScheduler.ts`; `engine/tests/test_brain.py`; `tests/idleCognitionScheduler.test.ts`; `tests/neuralFeedbackIdle.test.ts` |
| Idle cognition can spontaneously ask the user something in the continuous chat | Implemented | A confident prompt-free `talk` choice decodes from active recurrent state using only language-boundary tokens, persists to the continuous chat, and publishes a visible organic action event. `engine/tests/test_brain.py`; `tests/neuralFeedbackIdle.test.ts` |
| Trace records seeds, activations, recurrence, routes, mutations, and external actions | Implemented | `engine/omni_core/brain.py`; `src/main/chatActionController.ts`; `tests/actionProtocol.test.ts` |
| Trace is a guaranteed faithful chain-of-thought explanation | Not claimed | The product records operational evidence and labels generated explanations as self-report |

## Initial and continual learning

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Default locally initialized OmniCortex | Baseline | New Build creates seeded project-owned weights, trains the transparent local tool/action curriculum, and verifies mutation/coverage/packing before Chat. It is a small local baseline. `engine/omni_core/ground_up.py`; `engine/omni_core/brain.py`; `engine/worker.py`; `tests/brainService.test.ts`; `engine/tests/test_ground_up_architecture_contract.py` |
| Starter, Blank, Falcon/foundation, hybrid, and other legacy imports | Rejected by design | Build and Import expose no compatibility selector or relabeling path. Rejection leaves the selected file untouched. `engine/omni_core/brain.py`; `src/main/brainService.ts`; `src/main/brainRepository.ts`; `tests/brainRepository.test.ts`; `tests/brainService.test.ts` |
| Verified native ground-up `.omni` imports without randomizing imported tensors | Implemented | Accepted packages retain their native tensors and recorded ground-up provenance; both current state and immutable origin must satisfy the exact schema and no-foundation boundary. `src/main/brainService.ts`; `src/main/brainRepository.ts`; `tests/brainRepository.test.ts`; `tests/brainService.test.ts` |
| Dynamic whole-experience learning across working activity, fast temporal/episodic synapses, spreading assemblies, replay, and slow self-supervised weights | Source-implemented; outcome unverified | Ordinary experience can alter packed fast synapses after a turn commits; slow cortical updates are queued. Live native cleared-context recall and broad retention have not passed. `engine/omni_core/brain.py`; `engine/omni_core/spiking.py`; `docs/GROUND_UP_DESIGN_CORRECTION_CHECKLIST.md` |
| Whole-experience, dialogue, temporal, workspace, modality, replay, stability, and action-policy objectives | Implemented | `engine/omni_core/brain.py`; `engine/omni_core/ground_up.py`; `engine/tests/test_model.py`; `engine/tests/test_memory_modalities.py` |
| Auto corpus training preserves an exact logical optimizer target under RAM pressure | Implemented | Auto chooses the largest safe physical divisor no larger than four, derives accumulation as the exact quotient, retries the first uncommitted OOM through `4 → 2 → 1`, and freezes the first successful source-free schedule. `engine/omni_core/brain.py`; `engine/omni_core/offload.py`; `engine/tests/test_allocator_oom_recovery.py`; `engine/tests/test_record_checkpoint_resume.py`; `docs/TRAINING.md`; `docs/RESOURCE_MEMORY_PLANNER.md` |
| Corpus checkpoints are restart-safe without per-step SSD writes | Implemented | The default atomic mutable-state cadence is 512 committed source records plus the final partial group. Parameters, typed safe-tensor Adam state, stability state, replay, coverage, and the hash-bound cursor/schedule commit together; source text and token IDs do not. `engine/omni_core/brain.py`; `engine/omni_core/persistence.py`; `engine/tests/test_record_checkpoint_resume.py` |
| No RLHF, DPO, reward model, preference labels, or hidden persona prompt | Implemented | `engine/omni_core/model.py`; `engine/omni_core/ground_up.py`; `engine/tests/test_brain.py`; `tests/projectIntegrity.test.ts` |
| Conversation can alter learned vocabulary/style associations, including a measured slang response | Baseline | `engine/tests/test_model.py`. This demonstrates adaptation, not a stable personality |
| Candidate training, replay, metaplastic stability, rollback, and interrupted-worker recovery | Implemented | `engine/omni_core/brain.py`; `engine/omni_core/evolution.py`; `engine/tests/test_neural_evolution.py`; `engine/tests/test_interrupted_recovery.py` |
| Perfect retention of every learned experience | Not claimed | Human Consolidation and Synapses Only are lossy; Total Recall retains exact source bytes separately from dynamic neural learning |

## Files, datasets, crawling, and uploads

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| General files and datasets can be selected during Build | Implemented | `src/shared/uploadSupport.ts`; `src/main/ipc.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Separate image, audio, and video selectors during Build | Implemented | `src/shared/uploadSupport.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Data Studio selects general files, images, audio, video, and folders | Implemented | `src/main/ipc.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Run/chat selects general files, images, audio, video, and folders and accepts drag-and-drop | Implemented | `src/main/ipc.ts`; `src/preload/index.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Chat attachments are learned into neural state and return visible receipts | Implemented | `src/renderer/src/App.tsx`; `src/main/brainService.ts`; `tests/uploadSupport.test.ts` |
| PDF, EPUB, Office/OpenDocument text, HTML, Markdown, text, source, CSV/TSV, JSON/JSONL, Parquet, Arrow IPC, SQLite, archives, WebDataset, and local Hugging Face-style manifests | Implemented | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `engine/tests/test_datasets.py`; `tests/dataIngestion.test.ts` |
| Image, audio, and video folders and archive members train matching modality paths | Implemented | `engine/omni_core/brain.py`; `src/main/brainService.ts`; `engine/tests/test_brain.py`; `engine/tests/test_release_gates.py` |
| No former 128 MiB, 2,000-file, 16-million-character, or 64-chunk product cutoff | Implemented | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `tests/dataIngestion.test.ts`; `engine/tests/test_datasets.py` |
| Every valid record in a committed deterministic manifest/source snapshot is visited in every requested epoch | Implemented | Fatal neural, resource, or worker failures preserve incomplete coverage and a resumable cursor rather than claiming completion. `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `tests/dataIngestion.test.ts`; `engine/tests/test_datasets.py` |
| Coverage explicitly reports discovered, processed, rejected, bytes, records, modalities, and errors | Implemented | `src/shared/types.ts`; `src/main/dataIngestion.ts`; `tests/dataIngestion.test.ts` |
| Interrupted jobs resume from a deterministic committed cursor | Implemented | The source-free learning schedule is hash-bound to the same atomic neural generation at the 512-record cadence; committed physical/accumulation/window choices cannot drift on resume. `engine/omni_core/brain.py`; `src/main/dataIngestion.ts`; `engine/tests/test_record_checkpoint_resume.py`; `tests/dataIngestion.test.ts` |
| Same-site continuous crawl, persistent SQLite frontier, automatic parallelism, deduplication, pacing/backoff, robots control, external links, quarantine, stop, and resume | Implemented | `src/main/dataIngestion.ts`; `tests/dataIngestion.test.ts` |
| Crawled image, audio, and video responses are trained | Implemented | `src/main/brainService.ts`; `tests/dataIngestion.test.ts` |
| Literally infinite crawling or support for every media/container format | Not claimed | Crawling ends when stopped, the frontier empties, policy rejects a page, or resources pause; decoders and valid input formats remain real constraints |

## Multimodal learning and progressive local imagination

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Vision encoder maps images into the shared idea space | Baseline | `engine/omni_core/modalities.py`; `engine/tests/test_memory_modalities.py`; `engine/tests/test_release_gates.py` |
| Image VQ autoencoder and ternary latent diffusion transformer train and generate | Baseline | `engine/omni_core/modalities.py`; `engine/tests/test_memory_modalities.py` |
| Audio residual vector quantizer and ternary token generator train and generate | Baseline | `engine/omni_core/modalities.py`; `engine/tests/test_memory_modalities.py`; `engine/tests/test_release_gates.py` |
| Factorized spatial/temporal video model with liquid gating trains and generates local video | Baseline | `engine/omni_core/modalities.py`; `engine/tests/test_memory_modalities.py`; `engine/tests/test_release_gates.py` |
| Image, audio, and video generation can be cued directly by internal assemblies | Implemented | `engine/omni_core/brain.py`; `tools/catalog.json`; `engine/tests/test_brain.py` |
| Organic or requested imagination is a typed chat action, not a slash command or prose tag | Implemented | `engine/omni_core/brain.py`; `src/main/actionProtocol.ts`; `src/main/chatActionController.ts`; `tests/actionProtocol.test.ts` |
| Ordered intermediate image, audio, and video previews stream while the turn is active | Implemented | Cadence is bounded by local hardware/model size and is not token- or frame-synchronous. `engine/omni_core/modalities.py`; `engine/worker.py`; `src/main/chatActionController.ts`; `engine/tests/test_worker.py`; `tests/actionProtocol.test.ts` |
| Trained video and audio decoders can create synchronized media from one internal idea | Implemented baseline | H.264/AAC output length-aligns same-idea neural sound; disabled, untrained, encoder, and APNG fallbacks are explicit, and generic sound is never called speech. `engine/omni_core/brain.py`; `engine/worker.py`; `engine/tests/test_brain.py`; `engine/tests/test_live_observation.py` |
| Frontier-quality image, audio, or video | Not claimed | All modality packs are intentionally small trainable baselines |

## Natural actions, tools, agents, and evolution

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Learned structured action head scores `talk`, `tool`, `imagine`, `agent`, `ponder`, `learn`, `evolve`, and `stop` | Implemented | `engine/omni_core/model.py`; `engine/omni_core/brain.py`; `engine/omni_core/ground_up.py` |
| Prose, slash commands, and tagged text cannot manufacture executable actions | Implemented | `src/main/actionProtocol.ts`; `tests/actionProtocol.test.ts` |
| Action, progress, preview, result, approval, cancellation, and failure cards appear directly in chat for received typed actions | Implemented | `src/main/chatActionController.ts`; `src/renderer/src/App.tsx`; `tests/actionProtocol.test.ts`; `tests/toolExecutor.test.ts` |
| Tool capabilities enter through learned structural schema embeddings and add no behavioral prompt text | Implemented | The neural boundary retains names, primitive field types, and required sets without a fixed schema-count ceiling; untrusted descriptions are excluded. `engine/omni_core/brain.py`; `engine/tests/test_tool_schemas.py` |
| Tool capabilities and newly installed MCP schemas become durable neural experience rather than per-chat prose | Implemented | Built-in semantics are part of the transparent ground-up tool/action curriculum; MCP names and structural JSON Schemas pass through the same fast-synapse, assembly, replay, and slow-weight learning path. Remote descriptions are ignored. `engine/omni_core/ground_up.py`; `engine/omni_core/brain.py`; `src/main/mcpClient.ts`; `src/main/brainService.ts`; `tests/integrationLearning.test.ts`; `engine/tests/test_tool_schemas.py` |
| System files, native shell, code, web search/fetch, guarded browser, imagination, agent, and source-evolution protocols | Implemented | Ground-up schemas are platform-neutral; the host adapter selects noninteractive PowerShell on Windows or `/bin/sh` on macOS/Linux. `engine/omni_core/ground_up.py`; `tools/catalog.json`; `src/main/toolExecutor.ts`; `tests/platformNeutralSystemTools.test.ts`; `tests/toolExecutor.test.ts` |
| Streamable HTTP and installed local stdio MCP servers can be connected from the desktop UI | Implemented | The client negotiates MCP `2025-06-18`, paginates `tools/list`, namespaces typed calls, uses no shell for stdio, and places every discovered tool at Ask permission. HTTPS is required except loopback HTTP. `src/main/mcpClient.ts`; `src/renderer/src/App.tsx`; `tests/integrationLearning.test.ts` |
| External API-teacher learning from OpenAI, Anthropic, or Gemini | Implemented mechanism; live-provider gate remains | Each explicit user-authored line is one disclosed provider request. The response becomes parameter/synapse training data through structured neural ingestion, with no system prompt, RLHF, reward model, or chat-time retrieval. Mocked wire-contract and mutation tests pass; opt-in credentialed live calls remain a Step 12 acceptance gate. `src/main/teacherTraining.ts`; `src/main/brainService.ts`; `src/renderer/src/App.tsx`; `tests/integrationLearning.test.ts` |
| API and MCP credentials never enter a brain, trace, journal, prompt, or `.omni` export | Implemented | Secrets are application-level and OS-encrypted when a secure protector is available; otherwise they are session-memory-only. Only configured state and a short hint return to the renderer. `src/main/secureSecretStore.ts`; `src/main/teacherTraining.ts`; `src/main/mcpClient.ts`; `tests/integrationLearning.test.ts` |
| The brain can inspect access settings and request the permissions surface without changing grants | Implemented | `studio.settings` is a typed learned capability. Inspection returns the current matrix and configurable Ask timeout; opening the surface is a completed local UI action with `grantsChanged: false`. `engine/omni_core/ground_up.py`; `engine/omni_core/brain.py`; `src/main/toolExecutor.ts`; `src/renderer/src/studioUiActions.ts`; `tests/toolExecutor.test.ts`; `tests/studioUiActions.test.ts` |
| Ask-level permission pauses are visible and configurable | Implemented | Tokens default to 30 seconds, expose an exact expiry/countdown, fail closed on expiry, and never silently elevate a grant. The user can configure 1–3,600 seconds. `src/main/toolPreferences.ts`; `src/main/toolExecutor.ts`; `src/renderer/src/App.tsx`; `tests/toolExecutor.test.ts` |
| Public web research remains available without a private search-provider key | Implemented baseline | A configured SearXNG instance is preferred; otherwise the executor parses the public Bing RSS search response, filters to HTTPS results, retains normal network validation, and audits the call. `src/main/toolExecutor.ts`; `tests/toolExecutor.test.ts` |
| Interactive browser automation with navigation, clicks, typing, and signed-in sessions | Implemented | `browser.automation` uses a persistent per-brain sandbox and typed navigate/click/type/press/wait/extract/screenshot steps while validating all public-network requests and denying downloads, popups, browser permissions, and private-network targets |
| Learned generic `tool` choice materializes a valid enabled tool ID, action, and typed arguments end to end | Implemented | The learned head selects the generic channel; deterministic explicit-argument extraction plus VSA/schema evidence ranks enabled System shell, System file, code, search, fetch, and browser candidates without parsing response prose or guessing missing values. `engine/tests/test_ground_up_architecture_contract.py`; `engine/tests/test_tool_schemas.py` |
| Off, Ask, Auto, and Full grants are enforced outside candidate-writable neural state | Implemented | `src/main/toolExecutor.ts`; `src/main/evolutionController.ts`; `tests/toolExecutor.test.ts`; `tests/evolutionController.test.ts` |
| Subagent forks have isolated neural state and reviewed overlay merges | Implemented | The worker hashes both brains' config, parameter checksum, substrate records/vectors, and replay state; Electron binds that digest plus reviewed file hashes, and the merge RPC recomputes it before mutation. Stale source or target state is rejected and whole checkpoints are never averaged. `engine/omni_core/brain.py`; `engine/worker.py`; `src/main/brainService.ts`; `engine/tests/test_worker.py`; `tests/brainService.test.ts`; `tests/toolWorkflows.test.ts` |
| Source evolution creates/tests an isolated Git worktree with diff-bound promotion and rollback | Implemented | `src/main/toolExecutor.ts`; `src/main/evolutionController.ts`; `tests/toolWorkflows.test.ts`; `tests/evolutionController.test.ts` |
| A typed source-evolution proposal authors its declared candidate diff | Implemented | Exact path/content/parent-hash edits are applied atomically inside the new worktree and archived with before/after and diff hashes. Traversal, links, protected evaluators, package/setup scripts, binaries, stale hashes, oversized edits, and empty candidates fail closed. `src/main/toolExecutor.ts`; `src/main/evolutionController.ts`; `tests/toolWorkflows.test.ts`; `tests/actionProtocol.test.ts` |
| The narrow native curriculum reliably discovers and synthesizes arbitrary source-code improvements without source evidence | Not claimed | The safe authoring mechanism accepts only an exact typed edit; it does not pretend that a small local model has general coding intelligence, and generated prose is never executed |
| Full Authority builds, installs, and restarts into a promoted source/binary candidate | Implemented | A passing exact diff is packaged through the protected host lifecycle, stored and hash-verified side by side with snapshot/lineage metadata, merged, then scheduled for delayed packaged-app relaunch. Development/test hosts defer relaunch, Ask/Auto do not activate, and the current executable is never overwritten. `src/main/sourceRuntimeContract.ts`; `src/main/sourceRuntimeLifecycle.ts`; `src/main/toolExecutor.ts`; `tests/toolWorkflows.test.ts` |
| Neural and data evolution use isolated safe-tensor candidates and immutable evaluation | Implemented | `src/main/evolutionController.ts`; `engine/omni_core/evolution.py`; `engine/tests/test_neural_evolution.py`; `tests/evolutionController.test.ts` |
| Recursive generations can reassess the improvement process after promotion | Implemented | `src/main/evolutionController.ts`; `tests/evolutionController.test.ts` |
| Run UI lists evolution candidates and exposes review, approve, stop, and rollback | Implemented | `src/renderer/src/EvolutionWorkspace.tsx`; `src/renderer/src/evolutionView.ts`; `tests/evolutionRenderer.test.ts` |
| Arbitrary architecture/tensor-shape self-rewrite in stable v1 | Not implemented | Resource-checked residual-expert growth is implemented, but arbitrary incompatible shape migration still fails closed because safe state migration is not available |
| “Unaligned” action without host permissions | Not implemented by design | No behavioral alignment objective is trained, but external side effects still require the build’s explicit grant |

## Identity, storage, sharing, and stable interfaces

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| One durable chat and identity per brain | Implemented | `src/main/brainRepository.ts`; `src/main/brainService.ts`; `tests/brainRepository.test.ts` |
| Duplicate button creates an independent copy-on-write identity | Implemented | `src/main/brainRepository.ts`; `src/renderer/src/App.tsx`; `tests/brainRepository.test.ts`; `tests/projectIntegrity.test.ts` |
| Immutable origin, snapshots, restore, fork lineage, and journal | Implemented | Each neural copy includes the exact committed content-addressed substrate generation, not only the monolithic tensor files. `engine/omni_core/persistence.py`; `src/main/brainRepository.ts`; `engine/tests/test_memory_modalities.py`; `tests/brainRepository.test.ts` |
| Stable `BrainConfig` omits beta behavior sliders and neural caps | Implemented | `src/shared/types.ts`; `engine/omni_core/config.py`; `engine/tests/test_config.py`; `tests/brainRepository.test.ts` |
| Stable preload groups include brain, chat, train, data, modality, trace, tool, agent, catalog, evolution, and window operations | Implemented | `src/shared/ipc.ts`; `src/preload/index.ts`; `src/shared/types.ts` |
| Dataset, workspace, substrate, action, and evolution stable-v1 types exist | Implemented | `src/shared/types.ts` |
| JSON-RPC worker supports progress, cancellation, correlation, ordered streaming, and recovery | Implemented | `src/main/engineSupervisor.ts`; `engine/worker.py`; `docs/STREAMING_PROTOCOL.md`; `tests/engineSupervisor.test.ts`; `engine/tests/test_worker.py` |
| Stable `.omni` carries current/origin state, exact substrate generations, packed ternary shards, lineage, provenance, and checksums without pickle | Implemented | `src/main/brainRepository.ts`; `engine/omni_core/persistence.py`; `tests/brainRepository.test.ts`; `tests/streamingZip.test.ts`; `docs/OMNI_FORMAT.md` |
| `.omni` import/export streams ZIP/ZIP64 without fixed archive-byte, expanded-byte, per-entry, or entry-count caps | Implemented | Disk reserve, filesystem limits, safe ZIP structure, CRC, hashes, and declared lengths remain enforced. Fixtures cover 4,105 entries, a streamed multi-megabyte file, ZIP64 metadata, duplicate/traversal rejection, failed-import cleanup, private blobs, and current/origin substrate restoration. `src/main/streamingZip.ts`; `src/main/brainRepository.ts`; `tests/streamingZip.test.ts`; `tests/brainRepository.test.ts`; `docs/OMNI_FORMAT.md` |
| Portable and lightweight local-reference exports | Implemented | `src/main/brainRepository.ts`; `tests/brainRepository.test.ts` |
| Explicit beta-format rejection and confirmed deletion of only app-managed beta paths | Implemented | `src/main/brainRepository.ts`; `tests/brainRepository.test.ts` |
| GitHub/catalog imports are data-only and never auto-run repository setup scripts | Implemented | `src/main/catalogInstaller.ts`; `docs/CATALOG_FORMATS.md`; `tests/catalogInstaller.test.ts` |

## Product UI

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Four-stage Build flow: identity/origin, learning/storage, senses/data, permissions/review | Implemented | `src/renderer/src/App.tsx`; `tests/projectIntegrity.test.ts` |
| Basic flow uses checkboxes and permission segments, not behavior sliders or neural caps | Implemented | `src/renderer/src/App.tsx`; `src/shared/types.ts`; `tests/projectIntegrity.test.ts` |
| Initial resources are removable from the build without deleting the originals | Implemented | `src/main/ipc.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Run keeps chat, uploads, live actions, Data Studio, substrate map, trace, journal, tools, agents, imagination, duplicate, import, and export together | Implemented | `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts`; `tests/stableBrainInspection.test.ts` |
| Run exposes the full evolution-candidate lifecycle rather than only starting a chat evolution action | Implemented | The Evolution workspace lists lineage/evaluations and exposes start, stop, Ask review/promotion, recursive reassessment, and exact rollback through the stable preload API |
| Brain Map is cursor-paged and multiresolution rather than a fixed small mirror | Implemented | `engine/omni_core/brain.py`; `src/main/brainService.ts`; `src/renderer/src/App.tsx`; `tests/stableBrainInspection.test.ts` |
| Runtime card exposes working context/workspace and confirms no hidden prompt or raw long-term text injection | Implemented | `engine/omni_core/brain.py`; `src/renderer/src/App.tsx`; `tests/stableBrainInspection.test.ts` |

## Cross-platform package and release

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Product and lockfile identify stable `v1.1.0`, with no “v2” product naming | Implemented | `package.json`; `package-lock.json`; `scripts/verify-release.mjs` |
| Windows x64 and ARM64 NSIS/ZIP definitions | Implemented | `package.json`; `.github/workflows/windows.yml`; `scripts/package-windows.ps1` |
| Windows ARM64 shell discloses its x64 worker while required native dependency wheels remain incomplete | Implemented | `.github/workflows/windows.yml`; `docs/RELEASE.md` |
| macOS Intel/Apple Silicon DMG/ZIP definitions | Implemented | `package.json`; `.github/workflows/macos.yml`; `scripts/package-posix.sh` |
| Linux x64/ARM64 AppImage/DEB/tarball definitions | Implemented | `package.json`; `.github/workflows/linux.yml`; `scripts/package-posix.sh` |
| Release requires exact tag/version/main ancestry, all six package jobs, smoke records, and artifact checksum verification | Implemented | `.github/workflows/release.yml`; `scripts/verify-release.mjs`; `scripts/verify-release-artifacts.mjs`; `tests/releasePackaging.test.ts` |
| Current revision is green on all six native hosts | Open release gate | Must be established by Actions for the final commit |
| Current revision is merged to `main`, tagged `v1.1.0`, and published | Open release gate | `v1.0.0` is already bound to an earlier main commit; no current-revision release is claimed by these docs |
| Repository has only `main` after feature-branch deletion | Open release gate | Perform only after verified merge and publication |
| Signed/notarized release artifacts | Open release gate | Conditional on configured credentials; otherwise artifacts must be labeled unsigned |

## Licensing and provenance

| Requirement | Status | Evidence and boundary |
| --- | --- | --- |
| Project code uses PolyForm Noncommercial plus a separate commercial license | Implemented | `LICENSE.md`; `COMMERCIAL_LICENSE.md`; `package.json` |
| BitNet, snnTorch, and NCPS remain documented research references while their license texts stay packaged | Implemented | An ignored `.runtime/bitnet-src` developer checkout may be preserved but is never a runtime/Build/Import/package input. `licenses/BitNet-MIT.txt`; `licenses/snnTorch-MIT.txt`; `licenses/NCPS-Apache-2.0.txt`; `RESEARCH.md`; `THIRD_PARTY_NOTICES.md`; `package.json` |
| Research sources and independent implementation boundaries are recorded | Implemented | `RESEARCH.md`; `THIRD_PARTY_NOTICES.md` |
| Imported packs and brains require provenance, checksums, architecture compatibility, and license labels | Implemented | `src/main/catalogInstaller.ts`; `src/main/brainRepository.ts`; `tests/catalogInstaller.test.ts`; `tests/brainRepository.test.ts` |
| Auditable new-Build origin and training record | Implemented | `engine/omni_core/ground_up.py`; `engine/omni_core/brain.py`; `architecture/omnicortex-ground-up-v1.json`; `docs/GROUND_UP_OMNICORTEX.md`; `engine/tests/test_ground_up_architecture_contract.py`. Retired Starter/foundation records remain documentation only and cannot enter Build or Import |

## Required final verification

The final commit must pass these named gates without relying on a previous
revision’s totals:

1. Python neural, model, dynamics, dataset, modality, recovery, packed-ternary,
   worker, and neural-evolution suites.
2. Node/Electron repository, upload, dataset, action, tool, agent, evolution,
   integrity, and release-verifier suites.
3. TypeScript typecheck and production Electron build.
4. Built Electron UI create/train/chat/upload/imagine/duplicate/restart/export
   flow.
5. Windows x64/ARM64, macOS Intel/Apple Silicon, and Linux x64/ARM64 package
   smoke jobs.
6. Exact cross-platform artifact-set and checksum verification.
7. `main` merge, exact `v1.1.0` tag, GitHub Release publication, and requested
   branch cleanup.
