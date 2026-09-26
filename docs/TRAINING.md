# Training and continual learning

This document describes the stable `v1.1.0` training and recovery contract.

## Ground-up builds and native imports

Every new Studio Build is initialized from a recorded random seed into the
project-owned OmniCortex architecture. It has no inherited language or world
knowledge. Before Chat opens, it trains on the hash-described local
tool/action curriculum and any initial resources the user explicitly selected,
then must pass its learning and exact-ternary readiness checks. Early output
can still be repetitive or incoherent; readiness is an integrity result, not a
capability benchmark.

The current v3 curriculum has 68 unique source records: 35 structured actions,
27 typed tool trajectories, and six negative no-action examples. Its split
dataset ledger, derived training views, symbolic readiness probes, and receipt
contract are independently SHA-256 bound. It admits zero synthetic modality
and zero imagination-selector fixtures. Those parameter groups remain random
and finite until selected or captured user media trains them; a native
host-path or shell probe is evaluation-only and cannot enter optimizer loss.
Older Starter, Blank, and external-foundation checkpoint formats are rejected,
not migrated or offered as Build/Import choices. Native `.omni` imports retain
their own verified ground-up tensors and immutable origin.

The authoritative new-Build rules and exact hardware-profile counts are in
[GROUND_UP_OMNICORTEX.md](GROUND_UP_OMNICORTEX.md).

## Hardware profiles

Profiles change scale, not the meaning of a module:

| Profile | Typical machine | Text scale | Media baseline | Logical update target |
| --- | --- | --- | --- | --- |
| Micro | 4–8 GB RAM, CPU | Architecture/smoke scale | Tiny images, short low-rate audio, tiny clips | 8 rows/windows |
| Personal | About 16 GB RAM | Small research model | Moderate latent sizes | 8 rows/windows |
| GPU | NVIDIA/DirectML-capable PC | Larger local research model | Larger packs | 16 rows/windows |
| Workstation | High-memory local host | Largest bundled baseline | Larger packs | 16 rows/windows |

A native tier that cannot keep its minimum active state resident is not made
feasible merely by waiting. The builder blocks an unsafe selection or asks for
a smaller native tier/context; it never substitutes an external model. Replay
and stability tensors remain CPU-resident between operations and are durably
offloaded to safe-tensor checkpoints; only the active micro-batch moves to the
selected CPU, CUDA, DirectML, or MPS device. The standalone ground-up trainer
additionally supports `torchrun` DDP and large-CUDA FSDP as described below.
DirectML and MPS remain single-device paths.

The target above is a logical optimizer-group target, not a promise to place
that many rows in accelerator memory at once. In **Auto** mode the live resource
planner considers at most four physical rows and chooses the largest safe
physical size that exactly divides the target. Gradient accumulation is then
the exact quotient, so a target of 16 uses `4 × 4`, `2 × 8`, or `1 × 16`
without changing the logical cardinality. All profiles retain activation
checkpointing. Manual mode remains bounded by its explicit safe-resource
selection rather than this Auto divisor policy.

## Experience lifecycle

Every valid input is treated as a whole experience. The mechanisms below can
overlap, recur, or fade according to the live state; they are not a fixed
pipeline and there is no manual consolidation step:

- bounded working activity keeps the current whole experience available;
- fast temporal and episodic synapses record immediate relationships;
- signed recurrent activation spreads through related distributed assemblies;
- replay can carry selected activity into slower learned-weight updates; and
- salience, timing, novelty, reuse, prediction error, interference, stability,
  rehearsal, and decay continuously determine what strengthens or fades.

The stored policy names `encode`, `consolidate`, `pretrain`, and `archive` select
scheduling emphasis and source-retention behavior; they are compatibility/API
identifiers, not four memory stages. Human Consolidation schedules automatic
replay into slow weights. Synapses Only removes raw training material after
feature conversion. Total Recall may retain exact source bytes with provenance,
separately from neural learning.

## Continual-learning protections

- A complete safe-tensor checkpoint remains readable while a candidate trains.
- Candidate promotion records a durable phase and pre-candidate backup. If the
  worker stops between the three atomic file replacements, the next load
  conservatively restores the complete prior checkpoint before reading tensors.
- Reservoir or latent replay reduces catastrophic forgetting.
- Stable, frequently used synapses have lower plasticity.
- Persistent squared-gradient importance and slow anchors add an EWC-like
  stability penalty to language, dialogue, replay/slow-learning, and modality
  updates.
- Background corpus training forms padded physical micro-batches. Auto chooses
  a physical divisor no larger than four and adjusts accumulation so their
  product remains the configured logical target before one clipped optimizer
  step. A short final group still trains every remaining row/window once.
- Micro, Personal, GPU, and Workstation profiles checkpoint decoder-block
  activations during training. Replay remains on CPU until used, and all
  durable checkpoints are non-executable safe tensors.
- Optimizer moments and step counters are persisted as typed safe tensors in
  the same atomic mutable-state generation as parameters, slow anchors,
  importance, replay state, and the ingestion cursor; pickle is never loaded.
  Corpus ingestion normally publishes that generation every 512 committed
  source records and at the final partial group, rather than writing the whole
  mutable state after every optimizer step. This bounds crash replay while
  avoiding per-step SSD writes.
- Before the first successful mutation of a new ingestion transaction, an
  allocator OOM restores RNG state and retries the complete logical group with
  the next exact physical divisor (`4 → 2 → 1`). If necessary after physical
  size 1, it can reduce the token window and emit more windows without skipping
  source bytes. The first successful shape is hash-bound into the learning
  schedule and frozen. A later OOM or any failure after slow-state mutation
  fails closed and lets the worker restore the last atomic generation; it does
  not silently rewrite a committed schedule.
- Interrupted training candidates are marked `interrupted` and quarantined on
  the next load; a killed-worker integration fixture verifies that stable model
  parameters and counters survive.
- The user can pause slow learning without erasing fast neural state.

These mechanisms reduce forgetting; they cannot guarantee perfect retention.

## Distributed ground-up training

`engine/distributed_train.py` is the production, non-interactive path for the
next full-folder or rented-GPU run. It constructs only a randomly initialized
`ground-up` OmniCortex whose foundation ID is `none`. The optional
`--initial-ground-up` speed path may locate only a verified, pristine immutable
`engine/origin` from the current curriculum; it never copies that brain's
mutable checkpoint, prior user sources, conversation, replay, or installed
packs. The command has no pretrained-model, reward-model, preference-label,
external-teacher, or RLHF option.

Rank zero first writes a canonical, source-free `DatasetManifest`. Its records
are a batched SQLite ordinal index with an incremental content chain; neither
manifest construction nor loading materializes a corpus-sized Python list or
JSON document. Every valid record has one global ordinal. In each requested
epoch ordinal `n` belongs only
to rank `n % WORLD_SIZE`; the union of rank shards is exactly the manifest and
has no padding duplicates. A fixed global record wave keeps optimizer
boundaries unchanged across world sizes. Uneven tail gradients are weighted by
the true global token-window count before DDP reduction. Deterministic
per-record noise, rank-synchronized AMP overflow handling, `no_sync` gradient
accumulation, and committed-wave cursors make restart and one-/multi-process
results equivalent within normal floating reduction precision.

DDP is the default when `WORLD_SIZE > 1`. `--strategy auto` selects FSDP only
for CUDA model parameters at or above `--fsdp-min-parameter-bytes` (2 GiB by
default); `--strategy fsdp` rejects CPU/MPS instead of pretending to shard.
Linux CUDA uses NCCL. Windows CUDA uses Gloo because official Windows PyTorch
wheels do not provide the same NCCL path. A launch without `torchrun` uses one
CUDA device, one MPS device, or CPU.

At each checkpoint, all ranks first finish the same optimizer wave. Rank zero
then merges source-free dynamic-neural operations in strict `(epoch, ordinal)`
order, so substrate dictionaries cannot acquire duplicate neuron, assembly, or
synapse rows and STDP order does not depend on which rank finished first. The
native safe-tensor/optimizer/substrate generation commits before the external
cursor pointer. If a process dies in that interval, resume restores the
previous pointer's exact `brain.json`; native recovery materializes its named
generation and truncates an uncommitted replay suffix. Packed ternary export
and output-directory promotion occur only on rank zero after the final gate.

Tool/action capability knowledge is not rehearsed only during creation. The
reusable capability scheduler runs at start, at fixed committed-global-wave
intervals, and immediately before final promotion. Only rank zero applies a
scheduled event, then its neural action heads/metaplastic state are synchronized
to every dense replica. Rehearsal optimizer inputs are the same hash-declared,
platform-neutral action, tool, and negative records. Eight concise model-side
probes cover files, native shell, web, the imagination action, agent fork,
learning, evolution, and settings; the full receipt separately covers all 27
bundled tool/action trajectories and six negative no-action examples. Native
probe arguments are no-gradient evaluation data, never hidden rehearsal input.
They use structural IDs/actions/JSON-schema fields—never tool-description or
system prose—and record scores, confusion, selected IDs/actions, and argument
validity. The scheduler does not train the imagination selector; selected media
does. Final promotion fails on a route or confidence regression.

The run directory exposes `distributed-status.json`, append-only aggregated
resource/disk telemetry, per-rank cursors, recent rank failures, and the cancel
sentinel path. A cancel is observed collectively before the next wave; the last
published checkpoint remains resumable. Resume currently requires the same
`WORLD_SIZE`, because silently remapping per-rank cursors would weaken the
exact-once claim.

Final promotion writes a canonically sealed v2 receipt. It binds the dataset
manifest and record-chain hashes, exhaustive final cursors, record/epoch high
water, the final capability regression gate, media coverage, before/after
parameter evidence, exact parameter accounting, sanitized per-rank resource
measurements, the append-only telemetry-ledger hash, current ground-up
curriculum and initialization receipts, and the verified packed-ternary content
hash/count. The bounded receipt is also stored inside the aggregate distributed
training-source entry in `brain.json`, so portable `.omni` export/import keeps
the evidence even though the local `distributed-training.json` convenience
file is not part of the portable container.

### Local Linux CUDA

```bash
PYTHONPATH=engine python3 -m torch.distributed.run \
  --standalone --nproc-per-node=gpu \
  engine/distributed_train.py train \
  --dataset /data/full-folder \
  --output /brains/omni-rented-run \
  --profile gpu --strategy auto --amp bf16 \
  --epochs 3 --global-batch-records 64 --micro-batch-records 2
```

The equivalent convenience command is
`scripts/train-distributed.sh DATASET OUTPUT [options]`.

### Multi-node rented Linux hosts

Use the same shared dataset/run path and rendezvous values on every node,
changing only `--node-rank`:

```bash
PYTHONPATH=engine python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --rdzv-backend=c10d --rdzv-endpoint=10.0.0.10:29400 \
  --rdzv-id=omni-ground-up-2026-09 \
  --max-restarts=3 \
  engine/distributed_train.py train \
  --dataset /shared/full-folder \
  --output /shared/brains/omni-rented-run \
  --run-dir /shared/runs/omni-rented-run \
  --profile workstation --strategy auto --amp bf16 --resume auto
```

### Windows CUDA

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train-distributed.ps1 `
  -Dataset 'D:\data\full-folder' `
  -Output 'D:\brains\omni-rented-run' `
  -Processes 2 `
  --profile gpu --strategy ddp --amp fp16 --resume auto
```

### Status, disk telemetry, cancellation, and resume

```bash
python3 engine/distributed_train.py status --run-dir /shared/runs/omni-rented-run
python3 engine/distributed_train.py cancel --run-dir /shared/runs/omni-rented-run
python3 engine/distributed_train.py clear-cancel --run-dir /shared/runs/omni-rented-run
```

Re-run the identical `torchrun ... train ... --resume required` command after a
rank failure. `--max-restarts` can let TorchElastic do this automatically.
Never point two independent jobs at one run directory.

Current verification deliberately uses tiny Parquet data and two CPU ranks; it
does not perform the deferred full-folder native OmniCortex training run. The
old Falcon-backed Nova run is excluded from native acceptance evidence.
CUDA/FSDP is gated and implemented but still needs a rented Linux CUDA
acceptance run. Text and
typed structured rows drive the distributed dense objective. A monotonic
rank-zero replay cursor reopens each newly committed image/audio/video record
once and runs the existing real modality trainer before high-water publication;
promotion fails unless decoder coverage is complete and the modality parameter
checksum changes. This is transactional rank-zero modality training, not a
claim that modality gradients themselves are sharded across GPUs.

The source/runtime CLI is covered by the tests below. A fresh packaged-desktop
rebuild is intentionally not triggered solely for this trainer change; packaged
worker/app discovery remains an explicit proof item for the next normal release
artifact run.

## Conversation learning

The current human turn and the explicitly reported, capacity-bounded ring of recent human/brain dialogue tokens are ordinary model input. Long-term sources are never silently pasted beside them. Parameter-only mode conditions activations with learned semantic state; working-memory mode additionally blends bounded recency-weighted recurrent vectors. The trace separately reports recent-dialogue expansion and confirms that hidden/long-term prompt expansion is absent. After the turn:

- human text is eligible for self-supervised learning;
- generated text receives a lower default replay weight to limit self-amplifying errors;
- explicit corrections receive higher surprise, rehearsal, and slow-learning priority;
- style, vocabulary, and slang can change through ordinary continuing prediction;
- no hidden rule or persona string is inserted.

## Web and dataset training

Catalog and crawl jobs record URLs, hashes, timestamps, quarantine state, and
declared licensing. Dataset traversal has no product-defined file, byte, row,
or chunk ceiling: it writes a deterministic manifest, streams every regular
file into an incremental SHA-256 snapshot, and records processed/rejected
coverage. File size and modification identity are rechecked before learning,
and the worker verifies the committed content hash while reading. Symbolic
links, unreadable paths, changed downloads, malformed records, and unsupported
binary data are explicit rejections rather than silent omissions. Rejection
counts remain exhaustive; equivalent diagnostics are aggregated/sampled so a
multi-million-row invalid shard cannot become a second in-memory dataset.
Supported readers include
PDF/text/source, CSV/TSV, JSON/JSONL, Parquet, Arrow IPC, SQLite, ZIP/TAR and
WebDataset shards, EPUB/Office archives, and local Hugging Face-style
`data_files` manifests. PyArrow handles columnar batches without loading an
entire table. Rows with a typed `messages` field train human/brain dialogue
targets while system/developer persona text is excluded. Metadata-only blob
indexes are rejected until their referenced content payload is available.

The web crawler uses a brain-local SQLite frontier. It follows the same site by
default, automatically sizes concurrent fetching from the host's available
processors, obeys `robots.txt` by default, and continues
until stopped, an optional page/depth condition is reached, or its frontier is
empty. Resume requeues in-flight pages and retains visited URLs, failures, and
provenance. Per-URL learning receipts and aggregate modality/error coverage stay
in SQLite across resumes; only a bounded recent diagnostic window is returned
to the renderer, so an indefinite crawl does not accumulate an indefinite
in-memory result list. Explicit dataset epochs revisit every valid record in the
committed deterministic manifest/source snapshot even when its content was
learned before that manifest, whereas a repeated one-off upload remains
deduplicated. Fatal neural, resource, or worker failures preserve incomplete
coverage and a resumable cursor rather than advancing it or claiming
completion. Neither datasets nor crawls are concatenated into a prompt or
in-memory corpus. The training engine does not execute code found in a dataset
or repository unless the separate code tool is explicitly granted authority.

Declarative build recipes always create a ground-up OmniCortex and may select
the hardware tier, memory policy, modalities, and initial tool grant. Import
accepts only exact-schema native ground-up OmniCortex packages; Blank, Starter,
Falcon/foundation, hybrid, and other legacy checkpoints are rejected rather
than offered through a separate compatibility flow. A recipe cannot contain
commands.
Modality packs carry only a model card, manifest, checksum ledger, and safe
tensors; installation is namespace-, shape-, architecture-, and
license-validated twice. See [CATALOG_FORMATS.md](CATALOG_FORMATS.md).

## Media decoder coverage

Image discovery includes PNG/JPEG/WebP/GIF/BMP/TIFF/AVIF/HEIF, JPEG 2000,
JPEG XL, and common camera-raw extensions. Audio discovery includes WAV,
MP3/FLAC/AAC/Opus/AIFF/WMA plus CAF, ALAC, AMR, AU, and Matroska audio.
Video discovery includes MP4/WebM/MOV/Matroska/AVI/MPEG/WMV/FLV plus 3GP,
Ogg video, MPEG transport-stream containers, and VOB. These are
decoder-dependent inputs, not a promise that every codec variant is present on
every operating system. Pillow or the installed image plugin decodes stills;
FFmpeg/soundfile decode audio and video in bounded streams. A missing decoder,
malformed stream, or ambiguous `.ts` file is reported as an attributable
rejected record instead of being counted as learned. TypeScript `.ts` source
keeps its source-code interpretation; MPEG transport streams should use `.mts`
or `.m2ts` locally, while crawled `video/mp2t` responses are normalized to
`.m2ts`.
