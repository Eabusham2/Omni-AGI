# Original-request compliance matrix

## Purpose and status vocabulary

This matrix breaks the user’s original and follow-up requests into discrete,
auditable statements.

- **Implemented**: current source and a named fixture provide the requested
  mechanism.
- **Baseline**: the mechanism is real and trainable, but quality is deliberately
  limited.
- **Partial**: a safe bounded version exists; the exact wording would overstate
  the implementation.
- **Open**: final native verification, publication, or repository operation has
  not happened yet.
- **Not claimed**: the request is a subjective or scientifically unsupported
  outcome rather than a testable software feature.

This is not a consciousness, human-equivalence, “unaligned AGI,” or
frontier-quality claim.

## Architecture and brain behavior

| User request | Status | What stable v1 actually provides | Evidence |
| --- | --- | --- | --- |
| Build a normal local AI from scratch rather than only wrapping another model | Implemented | Project-authored PyTorch decoder and trainer; no hosted or third-party chat runtime | `engine/omni_core/model.py`; `engine/omni_core/brain.py`; `tests/projectIntegrity.test.ts` |
| Use LLaMA-like modern decoder ideas where useful | Implemented | Causal decoder, RMSNorm, rotary positions, and a gated feed-forward path, independently implemented | `engine/omni_core/model.py`; `RESEARCH.md` |
| Use Microsoft 1.58-bit/BitNet as the starting point | Implemented | Every eligible effective forward projection is ternary; Microsoft code/weights are not the runtime | `engine/omni_core/model.py`; `engine/omni_core/ternary_packing.py`; `engine/tests/test_model.py` |
| Ensure it really uses 1.58-bit rather than silently using 16-bit forward weights | Implemented | Strict ternary audit and packed two-bit shards fail closed on dense eligible projections or mismatched dynamic synapses. Reload verifies dynamic-synapse count/order/values, and sparse recurrent recall uses the exact signed effective weight rather than the higher-precision master magnitude | `engine/omni_core/ternary_packing.py`; `engine/omni_core/vsa.py`; `engine/tests/test_ternary_packing.py`; `engine/tests/test_brain.py`; `engine/tests/test_memory_modalities.py` |
| Make all state 1.58-bit | Partial | Eligible forward weights are ternary; master weights, activations, scales, optimizer state, liquid state, and STDP traces remain higher precision because training requires them | `engine/omni_core/model.py`; `docs/MODEL_CARD.md` |
| Use software neuromorphic computing and SNNs | Implemented | Stateful leaky integrate-and-fire routing and recurrent spike activity | `engine/omni_core/spiking.py`; `engine/tests/test_dynamics.py` |
| Use real STDP | Implemented | Tested causal potentiation, anti-causal depression, decay, metaplasticity, and persistent fast synapses | `engine/omni_core/spiking.py`; `engine/tests/test_dynamics.py` |
| Use liquid neural networks | Implemented | CfC is the stable default and LTC is an experimental engine path | `engine/omni_core/liquid.py`; `engine/tests/test_dynamics.py` |
| Use vector-symbolic architectures | Implemented | VSA/HDC binding, bundling, permutation, distributed assembly formation, and signed recurrent spreading through the authoritative substrate | `engine/omni_core/vsa.py`; `engine/tests/test_memory_modalities.py` |
| Ideas should connect to parts of ideas, not exist only as token strings | Implemented | Atomic and compositional neurons form distributed assemblies joined by typed ternary synapses | `engine/omni_core/vsa.py`; `engine/tests/test_memory_modalities.py` |
| Ideas, parameters, and synapses should be “one thing” | Partial | Associative memory has one authority: `NeuralSubstrate`. Cortical master tensors and dynamic synapse records remain separate physical structures and are not falsely described as identical | `engine/omni_core/vsa.py`; `engine/omni_core/brain.py` |
| Understand an experience as a whole, not only token by token | Baseline | A bidirectional global workspace distills the complete available input into shared latents before causal boundary decoding | `engine/omni_core/model.py`; `engine/tests/test_brain.py` |
| Make behavior fuzzy, noisy, and not perfectly precise | Implemented | State-dependent spike timing, uncertainty, competing branches, organic generation noise, and liquid dynamics | `engine/omni_core/brain.py`; `engine/omni_core/spiking.py` |
| Let it trace its steps | Implemented | Operational traces record seeds, activations, recalls, routing, branches, actions, and parameter deltas | `engine/omni_core/brain.py`; `tests/actionProtocol.test.ts` |
| Expose a faithful hidden chain of thought | Not claimed | Operational evidence is inspectable; generated explanations are not represented as guaranteed private reasoning transcripts | `docs/MODEL_CARD.md`; `engine/omni_core/brain.py` |
| Let synapses strengthen with use and weight some pathways more | Implemented | Latent weight, ternary effective weight, eligibility, stability, use count, timing, rehearsal, and metaplasticity are persistent | `engine/omni_core/vsa.py`; `engine/omni_core/spiking.py` |
| Let neurons, ideas, experts, and synapses grow without arbitrary caps | Implemented | No model-defined cardinality ceiling for sparse structures; residual experts grow after sustained error | `engine/omni_core/vsa.py`; `engine/omni_core/brain.py`; `engine/tests/test_brain.py` |
| Let growth continue literally without limit | Partial | Growth continues until cancellation or a real RAM/disk reserve pauses it; dense base tensor shapes do not grow in place | `engine/omni_core/brain.py`; `engine/omni_core/vsa.py` |
| Remove curiosity, noise, plasticity, neuron-cap, and parallel-thought sliders | Implemented | Stable Build has no behavior sliders; hardware and measured neural state resolve these internals | `src/renderer/src/App.tsx`; `src/shared/types.ts`; `engine/omni_core/config.py`; `tests/projectIntegrity.test.ts` |
| Remove fixed parallel-thought and recurrent-recall counts, not only their UI controls | Implemented | Chat computation settles from neural energy, score convergence, workspace size, and host reserves; recurrent spreading has no hop counter and settles exact ternary excitation/inhibition through fan-in-normalized damping and interference pressure | `engine/omni_core/brain.py`; `engine/omni_core/vsa.py`; `engine/tests/test_memory_modalities.py`; `docs/SUBSTRATE_PERSISTENCE.md` |
| Make curiosity, pondering, imagination, and motivation organic and unprompted | Implemented | Prediction error, novelty, uncertainty, learning progress, spike/liquid state, and unfinished activity drive prompt-free idle cycles and typed proposals; an emitted `ponder` action runs another real recurrent/rehearsal cycle instead of completing as a no-op | `engine/omni_core/brain.py`; `src/main/chatActionController.ts`; `src/main/idleCognitionScheduler.ts`; `engine/tests/test_brain.py`; `tests/actionProtocol.test.ts`; `tests/neuralFeedbackIdle.test.ts` |
| Let idle cognition organically ask the user a question | Implemented | A learned high-confidence idle `talk` choice decodes from recurrent neural state with no behavioral/user prompt, persists in the continuous chat, and emits a visible action event | `engine/omni_core/brain.py`; `src/main/brainService.ts`; `tests/neuralFeedbackIdle.test.ts` |
| Let it develop slang and ways of speaking through adaptation | Baseline | Continuing dialogue training can measurably strengthen a target slang response; no fixed personality is claimed | `engine/tests/test_model.py` |

## Memory, identity, and initial knowledge

| User request | Status | What stable v1 actually provides | Evidence |
| --- | --- | --- | --- |
| Data should become parameters/synapses rather than only temporary context | Implemented | Ingestion mutates fast synapses and slow weights; parameter-only mode does not append retrieved source prose | `engine/omni_core/brain.py`; `engine/tests/test_brain.py` |
| Remember everything | Partial | Total Recall can retain exact source bytes; parametric memory is lossy and cannot guarantee perfect recall | `engine/omni_core/brain.py`; `engine/tests/test_release_gates.py` |
| Add a real working memory/context window | Implemented | Hardware-sized recent role-token/sensory context is separate from recurrent latent assembly slots, liquid state, active assemblies, rehearsal, decay, and eviction. Newly materialized Blank and bundled Starter Micro/Personal/GPU/Workstation builds resolve to 256/1,024/2,048/4,096 context tokens and 128/256/512/1,024 recurrent latent assembly slots; Extended doubles both. Compatible imported checkpoints retain their recorded context/model shape, which the Runtime Card reports. A smaller hardware- and organic-state-derived response budget remains independent of context capacity | `engine/omni_core/config.py`; `engine/omni_core/brain.py`; `engine/omni_core/model.py`; `engine/worker.py`; `engine/tests/test_stable_config.py`; `tests/stableBrainInspection.test.ts` |
| Support short and long timescales | Implemented | Transient workspace, immediate STDP/fast weights, assembly consolidation, replay, and slow-weight training | `engine/omni_core/brain.py`; `engine/omni_core/spiking.py` |
| Do not paste long-term memory into a hidden prompt | Implemented | Runtime trace asserts no textual memory or tool-schema text injection | `engine/omni_core/brain.py`; `engine/tests/test_brain.py`; `engine/tests/test_tool_schemas.py` |
| The brain should include initial training, not only what the user later feeds it | Baseline | Omni Starter trains a small project-authored seed corpus and structured action examples before the immutable origin snapshot | `engine/omni_core/starter.py`; `engine/omni_core/brain.py`; `engine/tests/test_brain.py` |
| Keep an advanced genuinely blank option | Implemented | Blank Brain starts from recorded random initialization and is labeled primitive | `src/renderer/src/App.tsx`; `engine/omni_core/brain.py` |
| One chat and one persistent identity per build | Implemented | Each brain owns one durable conversation, lineage, current state, and immutable origin | `src/main/brainRepository.ts`; `src/main/brainService.ts`; `tests/brainRepository.test.ts` |
| Fork or copy an identity | Implemented | Fork and Duplicate use copy-on-write neural storage and independent lineage | `src/main/brainRepository.ts`; `tests/brainRepository.test.ts` |
| Put a Duplicate button in the interface | Implemented | Duplicate is available in the Brain Library and active Run workspace | `src/renderer/src/App.tsx`; `tests/projectIntegrity.test.ts` |
| Export/share the current brain and its original copy | Implemented | Portable and local-reference `.omni` bundles stream checksum-bound current/origin tensors, exact sharded substrate generations, and packed ternary generations through ZIP/ZIP64 without a product-defined archive byte or entry ceiling | `src/main/brainRepository.ts`; `src/main/streamingZip.ts`; `tests/brainRepository.test.ts`; `tests/streamingZip.test.ts`; `docs/OMNI_FORMAT.md` |
| Remove fixed whole-brain archive limits | Implemented | The former 512 MiB/1 GiB/4,096-entry `.omni` cutoffs are removed. Streaming import/export retains structural and checksum validation, is exercised with 4,105 entries and ZIP64 metadata, and pauses at the real disk reserve | `src/main/streamingZip.ts`; `src/main/brainRepository.ts`; `tests/streamingZip.test.ts`; `docs/OMNI_FORMAT.md` |
| Restore snapshots and the immutable origin | Implemented | Snapshot, restore, origin preservation, exact substrate-generation copying, and corrupt-shard rejection are covered | `engine/omni_core/persistence.py`; `src/main/brainRepository.ts`; `engine/tests/test_memory_modalities.py`; `tests/brainRepository.test.ts` |
| Download compatible premade brains from GitHub | Implemented | Direct data-only `.omni`, recipe, and `.omnipack` assets can be installed after checksum, architecture, provenance, and license validation | `src/main/catalogInstaller.ts`; `docs/CATALOG_FORMATS.md`; `tests/catalogInstaller.test.ts` |
| Execute arbitrary GitHub setup scripts to install a brain | Not implemented by design | GitHub is transport only; repository scripts are never auto-executed | `src/main/catalogInstaller.ts`; `docs/CATALOG_FORMATS.md` |

## Training data, uploads, and crawling

| User request | Status | What stable v1 actually provides | Evidence |
| --- | --- | --- | --- |
| Upload a PDF or programming-language reference and train on it | Implemented | PDF and source-code ingestion update neural state and retain provenance according to the selected memory recipe | `src/main/brainService.ts`; `engine/tests/test_release_gates.py` |
| Upload ordinary files and datasets | Implemented | General file/dataset pickers exist in Build, Data Studio, and Run/chat | `src/shared/uploadSupport.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Upload images | Implemented | Dedicated image selection exists in Build, Data Studio, and Run/chat | `src/shared/uploadSupport.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Upload audio | Implemented | Dedicated audio selection exists in Build, Data Studio, and Run/chat | `src/shared/uploadSupport.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Upload video | Implemented | Dedicated video selection exists in Build, Data Studio, and Run/chat | `src/shared/uploadSupport.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Upload folders and drag files into chat | Implemented | Folder selectors and chat drag-and-drop feed the same neural ingestion service | `src/main/ipc.ts`; `src/renderer/src/App.tsx`; `tests/uploadSupport.test.ts` |
| Remove file-upload limits | Implemented | Former product cutoffs for file bytes, file count, extracted characters, rows, and chunks are removed; traversal streams instead of reading an entire corpus into memory | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `tests/dataIngestion.test.ts` |
| Train on the entire requested dataset | Implemented | Every valid record in the committed deterministic manifest/source snapshot is visited in every requested epoch, with a committed cursor and explicit rejected-record report. Fatal neural, resource, or worker failures leave the run incomplete and resumable instead of claiming completion | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py`; `tests/dataIngestion.test.ts`; `engine/tests/test_datasets.py` |
| Guarantee the model understands every dataset record | Not claimed | Coverage proves traversal and neural updates, not semantic mastery or perfect retention | — |
| Support Parquet | Implemented | PyArrow-backed streaming Parquet batches are tested when the optional decoder is present | `engine/omni_core/datasets.py`; `engine/tests/test_datasets.py` |
| Support standard dataset and OS file forms | Implemented | Text/source, PDF, EPUB, Office/OpenDocument text, HTML, CSV/TSV, JSON/JSONL, Parquet, Arrow IPC, SQLite, ZIP/TAR/WebDataset, and local Hugging Face-style manifests | `src/main/dataIngestion.ts`; `engine/omni_core/datasets.py` |
| Train on images, audio, and video in many common forms | Implemented | Common selector extensions and local decoder paths train the matching modality packs | `src/shared/uploadSupport.ts`; `engine/omni_core/brain.py`; `engine/tests/test_release_gates.py` |
| Train every imaginable proprietary or damaged media form | Not claimed | Corrupt, encrypted, DRM-protected, proprietary, or unavailable-decoder formats can be explicitly rejected | — |
| Scrape and train from the web | Implemented | Brain-local persistent crawler, provenance, quarantine, and neural ingestion | `src/main/dataIngestion.ts`; `src/main/brainService.ts`; `tests/dataIngestion.test.ts` |
| Crawl continuously until Stop and use parallel fetches | Implemented | Same-site continuous frontier with automatic bounded concurrency, pause/resume/cancel, pacing, and resource pause | `src/main/dataIngestion.ts`; `tests/dataIngestion.test.ts` |
| Optionally follow the wider web and control robots/quarantine | Implemented | External-link, robots, and quarantine checkboxes map to crawler policy | `src/renderer/src/App.tsx`; `src/main/dataIngestion.ts`; `tests/dataIngestion.test.ts` |
| Crawl and train linked images, audio, and video | Implemented | Linked media responses are discovered, classified, and trained | `src/main/brainService.ts`; `tests/dataIngestion.test.ts` |
| Never silently skip data | Implemented for committed manifests | Every discovered unit is committed as processed or explicitly rejected. Fatal job errors preserve incomplete coverage and the resume cursor; traversal accounting does not claim that every record was understood or retained | `src/main/dataIngestion.ts`; `tests/dataIngestion.test.ts` |

## Chat actions, tools, imagination, and agents

| User request | Status | What stable v1 actually provides | Evidence |
| --- | --- | --- | --- |
| Use tools directly from chat | Implemented | Typed action and artifact cards execute from the continuous conversation | `src/main/chatActionController.ts`; `src/renderer/src/App.tsx`; `tests/actionProtocol.test.ts` |
| Do not require `/` commands or tagged JSON | Implemented | Only a dedicated typed worker channel is executable; prose, slash text, and `<omni-tool>` text are ignored | `src/main/actionProtocol.ts`; `tests/actionProtocol.test.ts` |
| Let the brain naturally choose whether to talk, use a tool, imagine, agent, ponder, learn, evolve, or stop | Implemented | The neural head scores all eight choices; its dedicated typed channel materializes internal, imagination, agent, evolution, stop, and enabled generic tool actions. Organic edit-free evolution selects a viable substrate/latent-replay overlay rather than an empty source fork | `engine/omni_core/model.py`; `engine/omni_core/brain.py`; `engine/omni_core/starter.py`; `src/main/chatActionController.ts`; `engine/tests/test_tool_schemas.py`; `tests/evolutionController.test.ts` |
| Let the user ask naturally for imagination or an agent | Implemented | Natural input reaches the same learned typed action policy; the UI does not depend on slash parsing | `engine/omni_core/brain.py`; `src/main/chatActionController.ts`; `tests/actionProtocol.test.ts` |
| File, PowerShell, coding, web, and browser tools | Implemented | The learned head first selects the generic `tool` channel; deterministic explicit-argument extraction and VSA/schema ranking then materialize only enabled typed calls. Missing required arguments are not guessed, and execution, progress, cancellation, grants, and audit cards remain outside neural state | `engine/omni_core/brain.py`; `tools/catalog.json`; `src/main/toolExecutor.ts`; `engine/tests/test_tool_schemas.py` |
| Full interactive browser automation | Implemented | A sandboxed persistent per-brain session supports validated navigation, click, type, key, wait, extract, screenshot, and user-visible sign-in while rejecting browser permissions, downloads, popups, and private-network requests | `src/main/toolExecutor.ts`; `tools/catalog.json` |
| Image generation and input | Baseline | Trainable vision and image imagination paths operate locally from the shared idea space | `engine/omni_core/modalities.py`; `engine/tests/test_memory_modalities.py` |
| Audio generation and input | Baseline | Trainable RVQ audio codec/generator and common local decoding paths | `engine/omni_core/modalities.py`; `engine/tests/test_release_gates.py` |
| Video generation and input | Baseline | Trainable factorized liquid-gated video model and local video decoding/encoding | `engine/omni_core/modalities.py`; `engine/tests/test_release_gates.py` |
| Imagine from an internal idea rather than a hidden text prompt | Implemented | Active assembly identifiers cue modality generation directly | `engine/omni_core/brain.py`; `tools/catalog.json` |
| Imagine organically while writing/talking | Implemented as progressive local generation | Organic typed imagination can begin mid-turn and ordered image/audio/video preview revisions stream while generation runs. Cadence is hardware/model bounded, not token- or frame-synchronous, and not guaranteed instantaneous | `engine/omni_core/brain.py`; `engine/omni_core/modalities.py`; `engine/worker.py`; `tests/actionProtocol.test.ts`; `engine/tests/test_worker.py` |
| Make preview generation instantaneous or frontier quality | Not claimed | Preview cadence is bounded by the selected local hardware profile and tiny baseline models | — |
| Fork subagents, let them work separately, then merge | Implemented | Isolated copy-on-write branches return reviewed assemblies, evidence, files, artifacts, and replay examples without whole-model averaging. A worker-generated digest covers both brains' authoritative config, parameters, substrate/vectors, and replay; Electron binds it into review and the merge RPC rejects either side changing afterward | `engine/omni_core/brain.py`; `engine/worker.py`; `src/main/brainService.ts`; `engine/tests/test_worker.py`; `tests/brainService.test.ts` |
| Give Full Authority if the user selects it | Implemented | Full can skip per-action confirmation, while the trusted executor still validates and audits the action | `src/main/toolExecutor.ts`; `tests/toolExecutor.test.ts` |
| No host-side rules of any kind | Partial | No hidden behavioral persona or preference prompt is used, but file/process/network permissions, validation, and rollback remain outside the model to protect the user and host | `engine/omni_core/brain.py`; `src/main/toolExecutor.ts` |
| Train the brain to lie, refuse, obey, or invent ethics | Not implemented by design | None is a training objective. Outputs may still be uncertain or wrong; no moral agency is claimed | `engine/omni_core/starter.py`; `tests/projectIntegrity.test.ts` |
| Make it a real person | Not claimed | One persistent conversational identity is implemented; personhood or consciousness is not asserted | — |

## Recursive self-improvement

| User request | Status | What stable v1 actually provides | Evidence |
| --- | --- | --- | --- |
| Detect limitations and form improvement candidates | Implemented | Controller derives evidence from prediction error, failed actions, regressions, resource pressure, substrate uncertainty, traces, and prior evaluations | `src/main/evolutionController.ts`; `tests/evolutionController.test.ts` |
| Improve neural parameters in isolation | Implemented | Neural and data candidates train in safe-tensor overlays separate from live state | `engine/omni_core/evolution.py`; `engine/tests/test_neural_evolution.py` |
| Improve source code in isolation | Implemented mechanism | Source candidates use Git worktrees, typed atomic compare-and-write edits, before/after/diff hashes, allowlisted checks, promotion, and rollback. Unsafe, stale, protected, binary/setup, oversized, and empty edits fail closed | `src/main/toolExecutor.ts`; `src/main/evolutionController.ts`; `tests/toolWorkflows.test.ts`; `tests/actionProtocol.test.ts` |
| Reliably invent arbitrary source improvements without having exact source evidence | Not claimed | A small local brain is not represented as a general coding oracle. It must emit exact typed paths, contents, and parent hashes; response prose never becomes code | — |
| Under Full Authority, rebuild/install a passing source candidate and restart into it | Implemented | The host snapshots the brain, packages only an exact nonempty candidate that passed immutable evaluation, verifies the native side-by-side executable and complete artifact tree against durable lineage, merges it, and schedules delayed packaged-app relaunch. Ask/Auto never activate it, development/test relaunch is deferred, and the running executable is never overwritten | `src/main/sourceRuntimeContract.ts`; `src/main/sourceRuntimeLifecycle.ts`; `src/main/toolExecutor.ts`; `src/main/evolutionController.ts`; `tests/toolWorkflows.test.ts` |
| Test candidates against immutable evaluation | Implemented | Capability, retention, integrity, ternary, resource, and evaluator-identity checks are outside candidate-writable state | `engine/omni_core/evolution.py`; `src/main/evolutionController.ts`; `engine/tests/test_neural_evolution.py`; `tests/evolutionController.test.ts` |
| Keep an evolutionary lineage and roll back | Implemented | Parentage, hashes, evaluations, promotions, recursion, and rollback points are durable | `src/main/evolutionController.ts`; `engine/omni_core/evolution.py` |
| Recursively improve the improvement process | Implemented | A promoted candidate can parent a reassessment generation under the same immutable evaluator | `src/main/evolutionController.ts`; `tests/evolutionController.test.ts` |
| Review, approve, stop, and roll back evolution from the Run interface | Implemented | The Run Evolution workspace exposes lineage, evaluations, start/stop, Ask review/promotion, recursive reassessment, and exact rollback | `src/renderer/src/EvolutionWorkspace.tsx`; `src/renderer/src/evolutionView.ts`; `tests/evolutionRenderer.test.ts` |
| Automatically rewrite arbitrary architecture and migrate incompatible tensor shapes | Partial | Full Authority may promote a passing resource-checked architecture candidate that grows compatible zero-residual ternary experts; arbitrary incompatible width/depth/router/modality shape migration remains rejected | `src/main/evolutionController.ts`; `engine/omni_core/evolution.py`; `engine/tests/test_neural_evolution.py` |
| Guarantee monotonic intelligence improvement | Not claimed | Candidates can regress or fail; the controller rejects/rolls back measured failures but benchmarks are necessarily incomplete | — |

## Interface, platform, distribution, and cleanup

| User request | Status | What stable v1 actually provides | Evidence |
| --- | --- | --- | --- |
| Native-packaged Windows 11 app with Build and Run | Implemented | Electron main/preload isolation, Windows styling, four-stage Build, and one-chat Run workspace | `src/main/index.ts`; `src/preload/index.ts`; `src/renderer/src/App.tsx` |
| Make the interface easier for basic users | Implemented | Four stages, checkbox choices, automatic hardware sizing, collapsed diagnostics, and no neural personality controls | `src/renderer/src/App.tsx`; `tests/projectIntegrity.test.ts` |
| Keep advanced inspection without exposing personality controls | Implemented | Research diagnostics and paged substrate views show measured state | `src/renderer/src/App.tsx`; `tests/stableBrainInspection.test.ts` |
| Make the Brain Map large enough for the learned substrate | Implemented | Cursor-paged multiresolution clustering avoids a fixed displayed-node ceiling | `engine/omni_core/brain.py`; `src/renderer/src/App.tsx`; `tests/stableBrainInspection.test.ts` |
| Add macOS support | Implemented packaging; open native gate | Intel and Apple Silicon DMG/ZIP workflows exist; final current-revision native-green evidence is pending | `package.json`; `.github/workflows/macos.yml` |
| Add Linux support | Implemented packaging; open native gate | x64 and ARM64 AppImage/DEB/tarball workflows exist; final current-revision native-green evidence is pending | `package.json`; `.github/workflows/linux.yml` |
| Keep Windows x64 and ARM64 support | Implemented packaging; open native gate | NSIS/ZIP workflow exists; Windows ARM64 transparently uses an emulated x64 neural worker | `.github/workflows/windows.yml`; `docs/RELEASE.md` |
| Name this stable release v1, not v2 | Implemented | Product metadata is `1.0.0`; docs describe the prior `0.1.0` state only as beta | `package.json`; `package-lock.json`; `scripts/verify-release.mjs` |
| Delete the added BitNet, snnTorch, and NCPS resource folders | Implemented in source state | Research source trees are removed; exact upstream license texts remain in `licenses/` | `THIRD_PARTY_NOTICES.md`; `licenses/BitNet-MIT.txt`; `licenses/snnTorch-MIT.txt`; `licenses/NCPS-Apache-2.0.txt` |
| Preserve third-party licensing after deleting those folders | Implemented | Package resources include the preserved MIT and Apache texts and the notice | `package.json`; `THIRD_PARTY_NOTICES.md` |
| Make every build green | Open | Final current-revision local and six-native-host gates must still run | — |
| Merge all verified work to `main` | Open | Repository operation belongs after the final verification sweep | — |
| Tag and publish `v1.0.0` with checksums | Open | Guarded release workflow exists; no publication is claimed | `.github/workflows/release.yml`; `scripts/verify-release-artifacts.mjs` |
| Leave only the `main` branch | Open | Delete the feature branch locally/remotely only after verified merge and release | — |
| Sign and notarize | Open/credential-dependent | Workflows consume credentials when present; unsigned artifacts must be labeled honestly | — |

## Outcome requests that software tests cannot establish

| User wording | Audit conclusion |
| --- | --- |
| “Best unaligned, self-modifying AGI” | Not claimed. The implementation is a small self-modifying research platform with explicit host permissions, not proof of AGI or a quality superlative. |
| “Replicate the human brain” | Not claimed. LIF, STDP, liquid state, distributed assemblies, working memory, and consolidation are engineering analogies, not a biological replica. |
| “Mega intelligent human” / “super smart even if slow” | Not claimed. Capability depends on architecture scale, training data, compute, and evaluation; the bundled starter is intentionally small. |
| “Understand the entirety of an idea” | Not provable. Whole-input latent integration and distributed assemblies are implemented, but semantic understanding is an empirical capability question. |
| “Remember everything” | Not possible from parameters alone. Exact retained source bytes require Total Recall, and even then retrieval and reasoning can fail. |
| “Think freely” / “build its own ethics” | No hidden persona, RLHF, or preference objective is present, but subjective autonomy or moral agency is not a software acceptance claim. |
