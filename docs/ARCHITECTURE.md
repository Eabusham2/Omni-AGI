# OmniCortex architecture

## Implementation status

OmniCortex is an experimental, small-scale research architecture. The current implementation exercises real training, plasticity, persistence, and modality paths, but it is not evidence of AGI, consciousness, biological equivalence, or frontier-model quality.

External models are not used to answer chat requests. A build starts from recorded random initialization or an explicitly imported `.omni` checkpoint whose version-1 container, architecture declaration, schema version, license ledger, file lengths, checksums, and tensor contents pass validation.

The word “brain” is a product metaphor. The implementation keeps these states distinct:

- dense slow model parameters;
- effective ternary weights and their higher-precision master weights;
- plastic fast weights, LIF activity, and STDP traces;
- liquid recurrent state;
- VSA idea vectors, concept nodes, and typed relations;
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

The renderer has no direct Node.js, filesystem, process, or credential access. Electron denies unhandled web permissions and sends validated work to the local worker. If the worker cannot start, the app exposes a deterministic TypeScript research fallback; the fallback is not presented as equivalent to the PyTorch engine.

## Build initialization and hardware selection

`Automatic` hardware selection is resolved once in Electron before brain creation. The current profiler considers logical CPU count, system memory, and Electron GPU status:

- Workstation at 48 GiB RAM, or at least 32 GiB plus 24 logical CPUs;
- GPU at 12 GiB RAM plus an available GPU process;
- Personal at 16 GiB RAM;
- Micro otherwise.

That selection scales transformer width/layers, sequence length, media dimensions and duration, batch size, gradient accumulation, and related limits. A GPU or Workstation worker prefers CUDA, then an installed usable DirectML path, then CPU. This is build-time selection, not a claim of continuous runtime re-profiling.

Blank creation records random engine state as the immutable origin. A starter build must import a materialized, compatible `.omni`; it loads the imported tensors and applies shape-safe configuration updates without invoking random creation over them. Optional initial pretraining runs only after the immutable origin exists. It mutates the current state with user-selected local data, so “starter knowledge” means imported or explicitly trained knowledge rather than a hidden bundled model.

## Per-turn data flow

1. Text crosses a UTF-8 byte-token boundary; supported media crosses its trainable pack.
2. Electron derives a bounded structural list of enabled tool IDs, actions, and grants. The worker normalizes it and binds deterministic VSA symbols for each capability; it does not serialize tool descriptions into prompt text.
3. The concept memory creates or recalls a continuous VSA idea representation and blends it with the tool-capability vector.
4. Sparse concept relations activate associated ideas.
5. A stateful LIF router integrates activity and emits spikes.
6. Pair-based STDP updates recurrent fast weights when plasticity is enabled.
7. CfC or LTC-like state evolves and emits retention, threshold, noise, and ponder controls.
8. The idea vector conditions the ternary decoder or a modality generator.
9. Novelty and sampled uncertainty can allocate a bounded number of generation branches.
10. The selected output, seed, spike metrics, routes, recalled idea IDs, parameter delta, prompt-token digest, and tool-channel status are recorded in the neural trace.
11. The worker persists model state atomically and appends checksummed operational events to SQLite.
12. Slow training and consolidation run as candidate updates with a baseline/final-loss gate.
13. A passing candidate is saved; a regressing, non-finite, or failed candidate is restored from the in-memory backup and recorded as rejected.
14. Candidate phases and a complete pre-candidate safe-tensor backup are durable. A
    load after worker termination quarantines unfinished training and restores
    that backup when promotion had started, before any model tensor is read.

Default parameter-only generation does not convert recalled long-term memories or tool schemas into hidden prose. It uses recalled vectors, capability vectors, fast weights, recurrent state, and learned parameters. Trace fields explicitly report that prompt text was not expanded, identify the VSA-internal tool channel, and hash the actual input token IDs. The trace's generated explanation is a model self-report, not a guaranteed private chain-of-thought transcript.

## Text cortex

The current engine uses a deliberately small decoder-only transformer so CPU fixtures can perform real optimization:

- byte-safe vocabulary with non-text role-boundary tokens;
- learned embeddings;
- RMS normalization;
- causal multi-head attention with rotary positions;
- gated feed-forward blocks;
- `BitLinear` projections with effective `{-1, 0, +1}` weights;
- higher-precision trainable master weights and a straight-through estimator;
- shared idea-space conditioning;
- next-symbol prediction.

Only participating `BitLinear` projections are ternary during the forward pass. Embeddings, activations, convolutional packs, optimizer state, liquid state, and plastic traces are not 1.58-bit. A configuration can disable ternary forwarding and use dense floating-point projections.

## Plastic and growable state

The spiking router stores recurrent weights, pre/post traces, stability, use counts, membrane state, and spike counts in `plasticity.safetensors`. A presynaptic spike followed by a postsynaptic spike potentiates the connection; reversed timing depresses it. Metaplastic stability reduces repeated updates, while consolidation decays weakly used structures.

The VSA memory stores concept and idea metadata in engine JSON and their hypervectors in safe tensors. Fixed and elastic builds use their configured capacities and reuse or evict low-value entries at pressure. An unbounded build doubles pressured concept/idea/relation capacities while disk and RAM reserves permit, records every expansion, and pauses with the measured resource reason before exhaustion.

Novelty can add small residual decoder experts. `fixed` disables expert growth, `elastic` observes `max_experts`, and `unbounded` removes that model-count limit but still pauses before crossing runtime disk or memory reserves. Expert growth is live and sparse; the next atomic save persists the new structure. Growth does not resize the live transformer's dense base tensors.

Slow parameters have a second metaplastic protection path. Squared gradients update persistent importance tensors, and an EWC-like penalty resists movement away from persistent anchors. Successful online steps move anchors gradually; promoted training or consolidation candidates commit them. Rejected or interrupted candidates restore weights, anchors, and importance together.

## Memory recipes

| Recipe | Exact source text | Idea/vector memory | Fast weights | Slow consolidation | Current generation path |
| --- | ---: | ---: | ---: | ---: | --- |
| Human Consolidation | No | Yes, lossy | Yes | Yes | Neural/vector conditioning |
| Total Recall | Optional local retention | Yes | Yes | Yes | Neural/vector conditioning |
| Synapses Only | No | Yes, without source text | Yes | Yes | Neural/vector conditioning |

Total Recall's retained text is available for explicit source-selected training and private archival export. The worker never silently retrieves it into chat prompts. In `parameter-only` mode, generation receives learned parameters, semantic/VSA recall, fast weights, and liquid state. In `working-memory` mode it additionally blends a bounded, recency-weighted recurrent activity vector. Both paths leave the input token sequence unchanged; the trace reports the selected channel, vector count, and token digest.

Working-memory slots, short-term half-life, replay threshold, forgetting rate, and consolidation rate are configurable. They improve control over retention; they do not guarantee exact recall.

## Multimodal baselines

All current packs share the configured idea dimension. A blank build initializes them randomly; a compatible materialized starter may supply trained pack weights:

- a compact convolutional vision encoder;
- a VQ image autoencoder with a ternary, idea/time-conditioned latent transformer denoiser;
- a two-stage residual-vector-quantized audio codec with a ternary latent-token generator;
- a compressed video autoencoder with factorized spatial/temporal evolution and a CfC-like liquid temporal gate.

Fixtures prove that every pack can overfit a miniature example, save as safe tensors, reload, and generate byte-identical seeded output. Image input supports Pillow-compatible images. Audio accepts PCM WAV directly, formats supported by libsndfile such as FLAC/OGG, and FFmpeg-decoded MP3/M4A/AAC where available. Video accepts animated GIF/WebP and FFmpeg-backed MP4/WebM/MOV. Generated artifacts are PNG, WAV, and animated PNG. These are tiny functional baselines inspired by DiT, EnCodec, and Latte, not quality-equivalent reproductions.

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
    artifacts/
      browser/                       guarded browser snapshot PNGs
    engine/
      brain.json                     current neural metadata
      core.safetensors               decoder, liquid, adapters, modalities
      plasticity.safetensors         SNN, VSA vectors, replay, liquid activity
      events.sqlite3                 append-only operational event log
      origin/
        brain.json
        core.safetensors
        plasticity.safetensors
      candidates/<candidate-id>/
      snapshots/<snapshot-id>/
      artifacts/
```

JSON and safe-tensor replacements use temporary files plus atomic rename. The event database uses WAL mode, full synchronization, payload hashes, and triggers that reject updates or deletes.

Forks receive independent application state and neural metadata. Immutable tensor bytes are materialized from the repository's content-addressed store with hard links where supported and copy fallback elsewhere; subsequent atomic replacement makes the branches diverge. Merge preview inventories novel concepts, ideas, relations, evidence records, and branch-local artifacts. Every regular file is hashed, size-bounded, and assigned a content-addressed target path; symlinks, non-regular files, corrupt blobs, and out-of-branch paths are skipped and surfaced as review conflicts. The preview token binds the exact source/target state and file hashes, so a changed branch must be reviewed again. Merge copies those overlays and replay examples, emits a hash-backed merge manifest, preserves target-side divergent nodes, and never averages complete dense checkpoints.

## `.omni` boundary

`.omni` is the only supported portable whole-brain checkpoint container. It carries both the selected current/origin payload and the immutable origin payload, with JSON state, safe tensors, a model card, lineage, and SHA-256 records. The exact version-1 contract and privacy limitations are documented in [OMNI_FORMAT.md](OMNI_FORMAT.md). Declarative builder recipes and modality-only `.omnipack` files have separate non-executable contracts in [CATALOG_FORMATS.md](CATALOG_FORMATS.md).

Downloaded bundles are treated as data. The importer validates ZIP paths, size limits, duplicate names, encryption/symlink flags, executable extensions, checksums, exact manifest byte lengths, the `OmniCortex` architecture name and current schema version, materialized engine format, secret-redaction declaration, source license ledger, and safe-tensor headers. Recipes reject unknown fields and contain no command field. Modality packs are revalidated by both Electron and the neural worker and may replace only declared `modalities.<kind>.*` tensors with exact compatible shapes and finite values. No path loads pickle data or runs repository setup scripts.

Current, origin, and private-archive exports carry materialized tensor bytes. A `referenced-local` export replaces those tensor entries with valid placeholder safe tensors and records the real tensor hashes; import succeeds only when the same local repository still has every referenced object. It is a storage convenience, not a shareable checkpoint.

## Tools and agents

Tool schemas are structured VSA model inputs; grant enforcement stays in Electron. They describe capability, not persona or behavior, and create no additional language prompt tokens. The model may emit one `<omni-tool>` JSON request, while users can call the same path directly with `/tool <tool.id> <action> {JSON}`, `/imagine image|audio|video`, or `/agent <objective>`. Tool results are displayed and returned to the brain as a visible structured experience. Recursive model-produced calls are capped at four actions for a turn.

`Off` rejects execution. `Ask` issues a five-minute, single-use approval token bound to the exact brain ID, tool ID, action, and SHA-256 digest of the serialized JSON arguments. `Auto` executes its safe subset but still asks for writes and other risky operations; its file reads are confined by real-path checks to the selected brain directory. `Full Authority` executes a valid invocation without an approval token. All levels append permission, invocation, result, cancellation, and failure stages to the same operational trace used by the brain. Arguments are represented by names and digests rather than copied file contents. Active processes, fetches, browser loads, and modality jobs have cancellation paths; a cancelled worker job is interrupted and late results are ignored.

The browser executor is a real but deliberately constrained snapshot provider. It performs guarded fetches with private-network and redirect checks, sanitizes bounded remote markup into an inert local document, and loads that document in a hidden sandboxed Electron window with an ephemeral partition. Remote scripts and active content are blocked; only the app's fixed extraction routine executes. Navigation and popups remain blocked. The executor returns bounded title/text/link data and writes a PNG under the brain's artifact directory. It does not operate signed-in sessions or provide general interactive browser automation.

Files, PowerShell, code execution, guarded web fetch/search, modality generation, browser snapshots, brain agents, and source evolution have local executors.

Source evolution requires an explicitly authorized Git clone. `propose` creates a separate branch/worktree and an out-of-tree task record. Edits are made inside that candidate through separately permissioned file/code tools. `diff` validates the worktree boundary and inventories tracked changes plus bounded untracked-file hashes. `test` runs only the allowlisted typecheck, unit-test, and build commands plus `git diff --check`, recording validation against the exact diff digest. `promote` requires that digest, a matching passing validation, and a clean target clone before committing and merging the candidate branch. The running binary is never replaced or restarted mid-execution.

An `agent.fork` action creates one to four copy-on-write brain forks and runs one objective turn in each isolated identity. The parent receives result summaries but no neural mutation. Merge remains a separate, previewed user action that copies novel ideas, relations, deduplicated evidence metadata, retained source blobs allowed by the target memory recipe, branch-local artifacts, replay examples, and related overlays. `Synapses Only` targets receive evidence provenance but no raw source text or source blob. Whole-model weights are never averaged. The current executor runs these bounded fork turns sequentially, so this is not a claim of an open-ended parallel autonomous society.

## Windows packaging

GitHub Actions is configured to test on `windows-latest`, build and exercise the PyInstaller worker over JSON-RPC, and ask electron-builder for NSIS and ZIP artifacts for x64 and ARM64. Each matrix leg expands the ZIP, silently installs the NSIS artifact into a clean temporary directory, and runs the packaged worker from both layouts. The x64 leg points Playwright at the installed executable and requires the packaged worker while it builds a brain, chats, navigates every primary surface by accessible name, exports a trace, generates and downloads an image, closes the app, relaunches it, and verifies persisted identity and chat. The ARM64 Electron executable cannot run on the hosted x64 runner. Each successful leg emits a JSON evidence file with artifact hashes and runtime versions. The workflow forwards `WINDOWS_CSC_LINK` and `WINDOWS_CSC_KEY_PASSWORD` into electron-builder's signing variables. Artifacts are unsigned when those repository secrets are absent; “signed-ready” is not a claim that a particular artifact is signed.

The x64 package contains an x64 Electron shell and x64 PyInstaller worker. The current ARM64 package targets the ARM64 Electron shell but includes the same x64 worker built on the hosted x64 runner, relying on Windows 11 ARM64 x64 emulation. These are configured checks, not evidence of a green run by themselves. A native ARM64 PyTorch/PyInstaller worker and an actual Windows 11 ARM64 desktop launch remain open release gates.
