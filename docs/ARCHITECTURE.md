# OmniCortex architecture

## Implementation status

OmniCortex is an experimental, small-scale research architecture. The current implementation exercises real training, plasticity, persistence, and modality paths, but it is not evidence of AGI, consciousness, biological equivalence, or frontier-model quality.

External models are not used to answer chat requests. A build starts from recorded random initialization or an explicitly imported `.omni` checkpoint whose version-1 container, architecture declaration, schema version, license ledger, file lengths, checksums, and tensor contents pass validation.

The word “brain” is a product metaphor. The implementation keeps these states distinct:

- dense slow model parameters;
- effective ternary weights and their higher-precision master weights;
- plastic fast weights, LIF activity, and STDP traces;
- liquid recurrent state;
- distributed substrate neurons and assemblies, VSA hypervectors, and typed
  signed ternary synapses;
- bounded conversation and working-memory state;
- optional retained source material;
- modality-pack parameters;
- tool permissions, which are enforced outside the model.

## Process boundary

```text
sandboxed React renderer
        |
        | grouped, typed preload API
        v
Electron main process
  - validates IPC
  - owns dialogs, storage, jobs, and tool grants
  - supervises the worker
        |
        | JSON-RPC 2.0, one JSON object per line, protocol version 1
        v
local OmniCortex Python/PyTorch worker
  - owns model tensors and neural state
  - writes model state only below the selected brain directory
  - reads user-selected ingestion paths passed by Electron
```

The renderer has no direct Node.js, filesystem, process, or credential access. Electron denies unhandled web permissions and sends validated work to the local worker. Stable v1 does not substitute a second TypeScript “brain” when the authoritative PyTorch worker is unavailable; neural creation, chat, training, and inspection fail visibly until that worker recovers.

## Build initialization and hardware selection

`Automatic` hardware selection is resolved once in Electron before brain creation. The current profiler considers logical CPU count, system memory, and Electron GPU status:

- Workstation at 48 GiB RAM, or at least 32 GiB plus 24 logical CPUs;
- GPU at 12 GiB RAM plus an available GPU process;
- Personal at 16 GiB RAM;
- Micro otherwise.

That selection scales transformer width/layers, sequence length, media dimensions and duration, batch size, gradient accumulation, and related limits. A GPU or Workstation worker prefers CUDA, then an installed usable DirectML path, then CPU. This is build-time selection, not a claim of continuous runtime re-profiling.

Blank creation records random engine state as the immutable origin. A starter
build can load a materialized, compatible `.omni`; it applies only shape-safe
configuration updates and never randomizes the imported tensors. When no
starter URL is supplied, the worker instead materializes the small,
project-authored Omni Starter locally by training the same OmniCortex
architecture on its published seed corpus and action trajectories, then
records that trained state as the immutable origin. This bundled baseline is
auditable and initially trained, but it is not a frontier checkpoint.
Optional user-selected initial training runs only after the immutable origin
exists and mutates the current copy.

## Per-turn data flow

1. Text crosses a UTF-8 byte-token boundary; supported media crosses its trainable pack.
2. Electron derives a bounded structural list of enabled tool IDs, actions, and grants. The worker normalizes it and binds deterministic VSA symbols for each capability; it does not serialize tool descriptions into prompt text.
3. `NeuralSubstrate` creates or recalls a distributed neuron assembly, blends its VSA representation with the tool-capability vector, and remains the sole authority for associative memory.
4. Signed ternary recurrent pathways spread excitation and inhibition through related assemblies. Fan-in normalization and damping settle the state without a fixed hop count.
5. A stateful LIF router integrates activity and emits spikes.
6. Pair-based STDP updates recurrent fast weights when plasticity is enabled.
7. CfC or LTC-like state evolves and emits retention, threshold, noise, and ponder controls.
8. The active assembly vector conditions the ternary decoder or a modality generator.
9. Novelty and sampled uncertainty can allocate competing generation branches until score convergence, workspace pressure, or host-resource pressure settles them; there is no fixed thought-count budget.
10. The selected output, seed, spike metrics, routes, active assembly IDs, parameter delta, prompt-token digest, and tool-channel status are recorded in the neural trace.
11. The worker persists model state atomically and appends checksummed operational events to SQLite.
12. Slow training and consolidation run as candidate updates with a baseline/final-loss gate.
13. A passing candidate is saved; a regressing, non-finite, or failed candidate is restored from the in-memory backup and recorded as rejected.
14. Candidate phases and a complete pre-candidate safe-tensor backup are durable. A
    load after worker termination quarantines unfinished training and restores
    that backup when promotion had started, before any model tensor is read.

Default parameter-only generation does not convert recalled long-term memories or tool schemas into hidden prose. It uses recalled vectors, capability vectors, fast weights, recurrent state, learned parameters, and the explicitly reported bounded recent-dialogue token ring. Trace fields distinguish truthful recent-context expansion from hidden-prompt expansion, which remains false; they also identify the VSA-internal tool channel and hash the actual input and recent token IDs. The trace's generated explanation is a model self-report, not a guaranteed private chain-of-thought transcript.

## Text cortex

The current engine uses a deliberately small decoder-only transformer so CPU fixtures can perform real optimization:

- byte-safe vocabulary with non-text role-boundary tokens;
- learned embeddings;
- RMS normalization;
- causal multi-head attention with rotary positions;
- gated feed-forward blocks;
- `BitLinear` projections with effective `{-1, 0, +1}` weights;
- higher-precision trainable master weights and a straight-through estimator;
- shared assembly-space conditioning;
- next-symbol prediction.

Every eligible `BitLinear`, ternary convolution, and recurrent STDP projection
is ternary during the forward pass. Sparse recurrent propagation uses the exact
effective sign (`-1` for inhibition, `+1` for excitation); the higher-precision
latent master does not scale a live edge. Embeddings, activations, optimizer
state, liquid state, plastic traces, and floating learning masters are not
1.58-bit. Stable v1 exposes no dense-forward toggle, and its packed inference
shards must cover every eligible projection and dynamic synapse.

## Plastic and growable state

The spiking router stores recurrent weights, pre/post traces, stability, use counts, membrane state, and spike counts in `plasticity.safetensors`. A presynaptic spike followed by a postsynaptic spike potentiates the connection; reversed timing depresses it. Metaplastic stability reduces repeated updates, while consolidation decays weakly used structures.

The neural substrate stores distributed neurons, assemblies, and their
effective ternary synapses as authoritative memory. Concept and idea labels are
inspection views derived from that state. Recurrent recall uses signed edge
contributions, including negative signals that suppress competing assemblies,
then applies a fan-in-normalized damped update until convergence or
working-memory/resource pressure. Stable v1 has no configured neuron,
assembly, synapse, idea, concept, expert, recall-hop, or shard-count ceiling.
Host RAM and disk reserve checks gate allocation, record measured pauses, and
allow later growth to resume.

The substrate persists in immutable, content-addressed generations rather than
inside one ever-growing plasticity tensor. Hash-bucketed JSON shards hold
neuron, assembly, and synapse records; safe-tensor shards hold hypervectors and
higher-precision sparse learning state. The atomically promoted engine metadata
selects one generation, so an interrupted save cannot make an orphaned newer
root pointer authoritative. Unchanged shards are reused by saves, snapshots,
origins, forks, evolution rollback points, and exports. See
[SUBSTRATE_PERSISTENCE.md](SUBSTRATE_PERSISTENCE.md).

Sustained novelty can add small residual decoder experts under the same
resource guard. Expert growth is live and sparse; the next atomic save persists
the new structure. Growth does not resize the live transformer's dense base
tensors.

Slow parameters have a second metaplastic protection path. Squared gradients update persistent importance tensors, and an EWC-like penalty resists movement away from persistent anchors. Successful online steps move anchors gradually; promoted training or consolidation candidates commit them. Rejected or interrupted candidates restore weights, anchors, and importance together.

## Memory recipes

| Recipe | Exact source text | Idea/vector memory | Fast weights | Slow consolidation | Current generation path |
| --- | ---: | ---: | ---: | ---: | --- |
| Human Consolidation | No | Yes, lossy | Yes | Yes | Recent-dialogue tokens plus neural/vector conditioning |
| Total Recall | Optional local retention | Yes | Yes | Yes | Recent-dialogue tokens plus neural/vector conditioning |
| Synapses Only | No | Yes, without source text | Yes | Yes | Recent-dialogue tokens plus neural/vector conditioning |

Total Recall's retained text is available for explicit source-selected training and private archival export. The worker never silently retrieves it into chat prompts. Both modes prepend only the bounded, persisted recent human/brain role-boundary token ring. In `parameter-only` mode, all older knowledge enters through learned parameters, semantic/VSA recall, fast weights, and liquid state. In `working-memory` mode it additionally blends a bounded, recency-weighted recurrent activity vector. The trace reports recent-token occupancy, hashes and eviction count separately from hidden/long-term injection, which remains absent.

For newly materialized Blank and bundled Starter builds, hardware profiling
derives token capacity and latent assembly slots: Micro resolves to 256 tokens /
128 recurrent slots, Personal to 1,024 / 256, GPU to 2,048 / 512, and
Workstation to 4,096 / 1,024. The optional Extended working memory checkbox
doubles both.
Compatible imported checkpoints retain their recorded context/model shape;
their effective values remain visible in the architecture manifest and Runtime
Card. Token context and latent assembly slots are separate temporary working
spaces. Whole-input integration latents scale at one quarter of the recurrent
workspace without the former fixed 32-latent ceiling. Response generation has
a separate hardware- and organic-state-derived budget, so a large temporary
context does not force equally long output. These values are not personality or
curiosity controls. Working memory remains temporary and does not guarantee
exact recall.

Decay lowers transient activation, eligibility, and unconsolidated latent
strength but never cardinality-deletes a neuron, assembly, synapse, or vector
record. Metaplastic stability and replay reduce catastrophic interference; the
immutable origin and snapshots provide rollback. Parameter-only memory remains
lossy, so exact source recovery requires Total Recall / Retain exact sources.

## Multimodal baselines

All current packs share the configured assembly/hypervector dimension. A blank build initializes them randomly; a compatible materialized starter may supply trained pack weights:

- a compact convolutional vision encoder;
- a VQ image autoencoder with a ternary, idea/time-conditioned latent transformer trained against an explicit latent noise-prediction objective;
- a two-stage residual-vector-quantized audio codec with a ternary latent-token generator;
- a compressed video autoencoder with factorized spatial/temporal latent denoising, an explicit noise-prediction objective, and a CfC-like liquid temporal gate.

Fixtures prove that every pack can overfit a miniature example, save as safe tensors, reload, and generate byte-identical seeded output. Image input supports Pillow-compatible images. Audio accepts PCM WAV directly, formats supported by libsndfile such as FLAC/OGG, and FFmpeg-decoded MP3/M4A/AAC where available. Video accepts animated GIF/WebP and FFmpeg-backed MP4/WebM/MOV. Generated artifacts are PNG, WAV, and H.264 MP4; animated PNG remains a fallback when the bundled FFmpeg encoder cannot run. These are tiny functional baselines inspired by DiT, EnCodec, and Latte, not quality-equivalent reproductions.

## Persistence and lineage

On Windows the default repository root is:

```text
%LOCALAPPDATA%\OmniAGI\brains\
```

`OMNI_AGI_DATA_DIR` overrides the base directory; the app appends `brains`. If neither Windows location nor an override is available, Electron's `userData` directory is used.

The implemented layout is:

```text
brains/
  .blobs/
    <sha256>                         content-addressed source/tensor blobs
  .trash/
    <brain-id>-<timestamp>/          recoverable app deletions
  <brain-id>/
    brain.json                       current inspectable application state
    origin.json                      immutable inspectable origin
    snapshots/
      <snapshot-id>.json
      <snapshot-id>.meta.json
      <snapshot-id>/engine/
        brain.json
        core.safetensors
        plasticity.safetensors
        substrate/                     exact referenced sharded generation
    artifacts/
      browser/                       guarded browser-task screenshots and evidence
    engine/
      brain.json                     current neural metadata
      core.safetensors               decoder, liquid, adapters, modalities
      plasticity.safetensors         SNN, replay, and liquid activity
      substrate/
        manifest.json                atomic pointer to committed generation
        generations/<sha256>/
          manifest.json              immutable bounded-shard index
        blobs/
          <sha256>.json              neuron/assembly/synapse record shards
          <sha256>.safetensors       hypervector and synapse-learning shards
      packed-ternary/
        manifest.json                exact eligible-forward coverage
        manifest.sha256              canonical-manifest checksum
        ternary-*.bin                four exact 2-bit values per byte
      events.sqlite3                 append-only operational event log
      origin/
        brain.json
        core.safetensors
        plasticity.safetensors
        substrate/                   immutable-origin referenced generation
        packed-ternary/              immutable-origin exact ternary shards
      candidates/<candidate-id>/
      snapshots/<snapshot-id>/
      artifacts/
```

JSON and safe-tensor replacements use temporary files plus atomic rename.
Substrate generation manifests and blobs are immutable and content-addressed;
the metadata commit record moves last. The event database uses WAL mode, full
synchronization, payload hashes, and triggers that reject updates or deletes.

Forks receive independent application state and neural metadata. Immutable tensor bytes are materialized from the repository's content-addressed store with hard links where supported and copy fallback elsewhere; subsequent atomic replacement makes the branches diverge. Merge preview inventories novel neurons, assemblies, ternary synapses, replay examples, evidence records, and branch-local artifacts. The authoritative worker streams a canonical digest over each brain's ID, engine/config identity, neural parameter checksum, substrate records and vectors, and replay vectors; it also reports addition, duplicate, and divergence counts. Electron binds that digest and every reviewed file hash into the review token. The merge RPC recomputes and requires the exact worker digest before mutation, so either the source fork or target base changing after review is rejected. Every regular file is separately hashed and assigned a content-addressed target path; symlinks, non-regular files, corrupt blobs, and out-of-branch paths are skipped and surfaced as review conflicts. Merge copies reviewed overlays and replay examples, emits a hash-backed merge manifest, preserves target-side divergent nodes, and never averages complete dense checkpoints.

## `.omni` boundary

`.omni` is the only supported portable whole-brain checkpoint container. It
carries both the selected current/origin payload and the immutable origin
payload, including each selected substrate generation, JSON state, safe
tensors, packed inference shards, a model card, lineage, and SHA-256 records.
The exact version-1 contract and privacy limitations are documented in
[OMNI_FORMAT.md](OMNI_FORMAT.md). Declarative builder recipes and modality-only
`.omnipack` files have separate non-executable contracts in
[CATALOG_FORMATS.md](CATALOG_FORMATS.md).

Downloaded bundles are treated as data. A streaming ZIP/ZIP64 reader validates
paths, duplicate names, overlapping payloads, encryption/symlink flags,
compression methods, CRCs, declared lengths, executable extensions, checksums,
the `OmniCortex` architecture name and current schema version, materialized
engine format, substrate manifests/blobs, secret-redaction declaration, source
license ledger, and safe-tensor headers. There is no product-defined archive
byte, entry-count, or expanded-size ceiling; extraction pauses before crossing
the disk reserve, and ordinary filesystem/platform limits still apply. Recipes
reject unknown fields and contain no command field. Modality packs are
revalidated by both Electron and the neural worker and may replace only
declared `modalities.<kind>.*` tensors with exact compatible shapes and finite
values. No path loads pickle data or runs repository setup scripts.

Current, origin, and private-archive exports carry materialized current and
immutable-origin safe tensors, the exact referenced substrate generations, and
both exact packed-ternary trees. Export writes entries incrementally and
upgrades to ZIP64 when physical ZIP fields require it; import extracts and
hashes entries incrementally before an atomic brain promotion. A
`referenced-local` export replaces safe tensors with valid placeholders and
packed files with local-reference markers, recording every real object hash.
Import resolves every object before verifying safe-tensor headers and packed
coverage. It succeeds only when the local repository still has all referenced
objects, so it is a storage convenience, not a shareable checkpoint.

## Tools and agents

Tool schemas are structured VSA model inputs; grant enforcement stays in Electron. They describe capability, not persona or behavior, and create no additional language-prompt tokens. A learned action head selects among `talk`, `tool`, `imagine`, `agent`, `ponder`, `learn`, `evolve`, and `stop`. Its dedicated typed worker channel materializes enabled tool IDs, actions, and validated arguments; response prose, slash commands, and tagged text are ignored. An edit-free `evolve` decision routes to a substrate candidate with latent replay. A source candidate exists only when the action carries exact typed path/content/parent-hash edits, preventing objective-only actions from producing guaranteed-empty Git candidates. Tool results are displayed and returned as visible structured experience. Long-running imagination calls stream previews, remain cancellable, and enter experience only after final artifact metadata is available.

`Off` rejects execution. `Ask` issues a five-minute, single-use approval token bound to the exact brain ID, tool ID, action, and SHA-256 digest of the serialized JSON arguments. `Auto` executes its safe subset but still asks for writes and other risky operations; its file reads are confined by real-path checks to the selected brain directory. `Full Authority` executes a valid invocation without an approval token. All levels append permission, invocation, result, cancellation, and failure stages to the same operational trace used by the brain. Arguments are represented by names and digests rather than copied file contents. Active processes, fetches, browser loads, and modality jobs have cancellation paths; a cancelled worker job is interrupted and late results are ignored.

The browser executor runs a sandboxed persistent partition scoped to one brain. Public-network validation is applied to page, redirect, websocket, and subresource destinations; browser permission requests, downloads, popups, private-network targets, and non-web navigation are denied. Typed steps can navigate, click, type, press keys, wait for selectors, extract bounded DOM data, and capture screenshots. A permission-approved visible session can retain sign-in cookies without exposing renderer filesystem or Node access.

Files, PowerShell, code execution, guarded web fetch/search, browser automation, modality generation, brain agents, and source evolution have local executors.

Source evolution requires an explicitly authorized Git clone. `propose` creates a separate branch/worktree and applies only declared typed UTF-8 compare-and-write edits whose existing-file SHA-256 (or new-file absence) still matches. It publishes the candidate only after all paths and temporary files pass traversal, symlink, protected-evaluator, setup-script, binary, size, and stale-input checks; the lineage records every before/after hash and the complete authored diff hash. `diff` validates the worktree boundary and inventories tracked changes plus bounded untracked-file hashes. `test` runs only the allowlisted typecheck, unit-test, and build commands plus `git diff --check`, recording validation against the exact diff digest. Candidate checks temporarily link only the authorized clone's existing `node_modules`, disable npm lifecycle hooks, never install dependencies, and remove and revalidate the link afterward. Empty candidates cannot pass. `promote` requires that digest, a matching passing validation, and a clean target clone before committing and merging the candidate branch. Under Full Authority, the trusted Electron host first snapshots the brain and builds a native unpacked runtime into a separate app-managed slot. It independently verifies the executable, complete artifact tree, current-executable hash, source/evaluator lineage, and unchanged candidate before merge; a packaged host then schedules a delayed side-by-side relaunch, while development/test hosts defer it. Ask and Auto never activate source binaries. Generated response prose is never interpreted as source code, and the running binary is never overwritten mid-execution.

An `agent.fork` action creates one to four copy-on-write brain forks and runs one objective turn in each isolated identity. The parent receives result summaries but no neural mutation. Merge remains a separate, previewed user action that copies reviewed neurons, distributed assemblies, ternary synapses, deduplicated evidence metadata, retained source blobs allowed by the target memory recipe, branch-local artifacts, replay examples, and related overlays. `Synapses Only` targets receive evidence provenance but no raw source text or source blob. Whole-model weights are never averaged. The current executor runs these bounded fork turns sequentially, so this is not a claim of an open-ended parallel autonomous society.

## Cross-platform packaging

GitHub Actions uses native `windows-latest` x64 and `windows-11-arm` ARM64 runners. Each runner installs a native Node runtime, runs the full neural/unit/UI suites, builds the PyInstaller worker, and asks electron-builder for matching NSIS and ZIP artifacts. Each matrix leg expands the ZIP, silently installs the NSIS artifact into a clean temporary directory, and runs the packaged worker from both layouts. Playwright then points at the installed executable and requires the packaged worker while it builds a brain, chats, invokes tools and subagents directly in chat, navigates every primary surface by accessible name, exports a trace, generates and downloads image, audio, and video artifacts, closes the app, relaunches it, and verifies persisted identity and chat. Each successful leg emits a JSON evidence file with artifact hashes and runtime versions. The workflow forwards `WINDOWS_CSC_LINK` and `WINDOWS_CSC_KEY_PASSWORD` into electron-builder's signing variables. Artifacts are unsigned when those repository secrets are absent; “signed-ready” is not a claim that a particular artifact is signed.

The x64 package contains an x64 Electron shell and x64 PyInstaller worker. The ARM64 package contains a native ARM64 Electron shell and an x64 PyTorch/PyInstaller worker that Windows 11 runs through its x64 emulation layer. This split is explicit because PyTorch does not currently publish a stable Windows ARM64 wheel; it is not represented as a native ARM64 neural runtime. The package is still built, installed, launched, restarted, and exercised on a native ARM64 runner. Workflow configuration is not treated as release evidence until both native-host jobs finish green and upload their smoke records.

macOS packages are built natively on Intel and Apple Silicon runners as DMG
and ZIP artifacts. Linux packages are built natively on x64 and ARM64 runners
as AppImage, DEB, and tar.gz artifacts. Their PyInstaller workers live at
`resources/engine-runtime/omni-engine`; Windows uses
`resources/engine-runtime/omni-engine.exe`. The supervisor selects the
platform-specific executable and package smoke rejects a shell or worker whose
machine architecture does not match the matrix leg.

The macOS and Linux workflows run the neural and Node suites, package the
desktop, create and reload a safe-tensor/SQLite brain through the embedded
worker, and drive the packaged application through the Playwright restart
scenario. Artifact hashes and runtime evidence are uploaded per architecture.
Configured workflow coverage becomes release evidence only after the matching
native-host job completes successfully.

Tags do not bypass these gates. The stable-release workflow accepts only the
exact `v<package.version>` tag when its commit is contained in `main`, rebuilds
all six native package legs, validates every expected file and smoke record,
writes `SHA256SUMS.txt` plus `RELEASE-MANIFEST.json`, and only then publishes
the GitHub release.
