# Ground-up OmniCortex contract

This is the contract for every **new** Studio Build. It answers the origin
question directly: a new Build is one OmniCortex, not one AI wrapped around a
second AI. Studio does not take Falcon, Llama, a Hugging Face checkpoint, or an
API model and quantize it into OmniCortex.

## What Build creates

1. The worker resolves a project-authored OmniCortex architecture for the
   selected hardware tier. Public Build requests cannot supply private
   snake-case tensor shapes; persisted/research configurations use a separate
   loader and cannot enter the new-Build path.
2. PyTorch initializes fresh packed ternary synapses from the recorded local
   seed. No model checkpoint is opened or downloaded.
3. Every eligible projection stores and uses exact `{-1, 0, +1}` weights.
   Bounded online updates change those packed codes directly; the native path
   has no full floating master copy of each projection.
4. Before the first durable origin is published, the core learns the bundled,
   project-authored tool/action curriculum. The v3 receipt proves complete
   record coverage, changed native-core and substrate state, unchanged random
   modality/selector parameters, and the checksum bindings later verified by
   exact ternary packing.
5. Explicit Build attachments are then learned through the normal streamed
   dataset/media/web paths. Chat stays closed until creation, requested initial
   learning, and readiness verification all finish.

If any required training or verification step fails, the desktop retains an
`initializing` or recoverable `failed` state. It does not make the incomplete
brain chat-ready.

The built-in ground-up curriculum is intentionally narrow. Version 3 contains
exactly 35 project-authored structured action examples, 27 tool trajectories,
and six negative no-action examples. Each group has its own immutable SHA-256
ledger entry. Derived whole-tool and action-route views have separate hashes
and explicitly add no source records. Its manifest reports zero general
language-corpus and zero dialogue-answer records; the old bundled Starter
module is not part of the current Build.

Version 3 also hash-binds two empty fixture sets: synthetic modality training
and synthetic imagination-selector training. Image, vision, audio, video, and
selector parameters therefore remain finite random initialization until
attributable user-selected or visibly captured media trains them. Structural
readiness means those random parameters are finite and included in the exact
ternary pack; it does not relabel them as trained or claim useful media quality.

The current curriculum and receipt schema are exact import boundaries, not
migration suggestions. Previous curriculum identities, Starter/Blank origins,
foundation adapters, and other legacy receipts are not selectable or accepted
by the current whole-brain importer. Existing files remain untouched on disk
when rejected. Every new Build uses v3 and training-receipt format 2.

Its host capabilities are platform-neutral. Neural experience uses
`system.files` and `system.shell`, symbolic explicit-path examples, and no
Windows command/path fixture. At execution, the visible desktop adapter uses
noninteractive PowerShell on Windows and `/bin/sh` on macOS/Linux. Whole-brain
import does not expose legacy Windows tool IDs as a compatibility mode.
Host-native commands and paths may be materialized for a no-gradient readiness
probe, but those values are never optimizer inputs. The symbolic probe contract
is independently hashed and marked training-ineligible.

## Permitted learning sources

After random initialization, learned state may change only from an attributable
source and a visible learning operation:

- the hash-described local capability curriculum above;
- files or folders the user selects, including streamed Parquet, Arrow,
  JSON/JSONL, tabular data, documents, archives, and source files;
- image, audio, or video the user selects or visibly captures;
- a URL/crawl the user explicitly starts;
- visible conversation and action outcomes under the configured continual
  learning policy;
- an API-teacher job the user explicitly configures and starts with explicit
  questions.

Merely having a model cache, foundation environment variable, network access,
or API credential does not authorize learning. API credentials remain outside
brain state, and an API teacher is training data provenance—not a runtime
fallback or a hidden second brain.

## Provenance and no-fallback rules

A completed ground-up runtime/export reports all of the following:

- `origin_kind: ground-up`;
- `baseFrozen: false`;
- `pretrained: false`;
- no foundation-model identifier or adapter field;
- a seeded `randomInitialization` with an exact parameter count;
- a content-hashed `ground_up_training_manifest` and content-hashed training
  receipt bound to the exact v3 dataset and derived-view ledgers;
- equal before/after modality and imagination-selector checksums during the
  built-in curriculum phase;
- no `pretrained_text_cortex` object and no foundation adapter;
- packed-forward metadata bound to the same origin, curriculum, training
  manifest, post-training parameter checksum, and complete eligible-tensor
  coverage.

The public worker rejects attempts to create a new brain with a legacy
`starter`, `blank`, Falcon, or automatic foundation origin. Environment
variables cannot opt a new Build back into one. There is no “try OmniCortex,
then fall back to a pretrained model” branch.

Whole-brain Import uses the same native boundary. Both the current payload and
immutable origin must prove `origin_kind: ground-up`,
`baseFrozen: false`, and `pretrained: false`, with no foundation identifier,
Starter manifest, foundation cortex, adapter, or incompatible schema.
Falcon-backed, bundled Starter, random-only Blank, foundation-adapter, and
other legacy `.omni` files fail closed rather than being migrated, relabeled,
or opened in a compatibility mode. Rejection does not delete or rewrite the
selected file or other saved user data.

## Truthful scale and resources

[`architecture/omnicortex-ground-up-v1.json`](../architecture/omnicortex-ground-up-v1.json)
is the versioned preflight contract. The counts below are logical learned
elements for Auto working memory—not FP32 master allocations or marketing
parameter estimates:

| Tier | Auto memory items | Workspace latents | Logical parameters | Exact-ternary projection parameters |
| --- | ---: | ---: | ---: | ---: |
| Micro | 8,192 | 2,048 | 404,025 | 320,151 |
| Personal | 32,768 | 8,192 | 1,301,985 | 736,991 |
| GPU | 65,536 | 16,384 | 4,097,377 | 2,457,311 |
| Workstation | 131,072 | 32,768 | 9,908,193 | 5,615,583 |

The count changes when the selected working-memory population changes because
the native global-workspace latents are learned parameters. The resource
planner recalculates rather than displaying a hard-coded Falcon parameter or
file size. It separately reports:

- 2-bit packed bytes for ternary projection weights;
- higher-precision non-projection and transient training bytes;
- bounded packed-update scratch and checkpoint headroom;
- resident context, runtime overhead, cold-memory paging, scratch, reserves,
  and estimated storage slowdown.

Optimizer moments, gradients, buffers, and dynamic sparse synapses are not
misreported as dense model parameters. Dynamic synapses are counted separately
after they grow. Storage offload can hold cold replay, optimizer scratch, and
inactive working patterns; it cannot replace the live layer/context minimum.

## Capability boundary

“Human-brain-like” describes design inspiration: persistent identity,
distributed associative memory, spiking/STDP plasticity, temporal liquid state,
working activity, spreading assemblies, rehearsal into slow learned weights,
salience, interference, stability, decay, and continuing local learning.
It is not evidence that this small architecture reproduces a biological brain,
is conscious, or has frontier language competence. A ground-up build starts
without pretrained world/language knowledge, and the narrow local curriculum
does not make it broadly capable. Useful capability must be trained and
measured; readiness proves construction and learning integrity, not AGI. The
old trained Nova hybrid is not evidence for this path because its origin,
architecture, and training history do not satisfy the native contract.
