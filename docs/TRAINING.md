# Training and continual learning

## Blank versus starter brains

A blank brain is initialized from a recorded random seed. It has architecture but no language or world knowledge. Its early output may be repetitive or incoherent.

A starter brain is an OmniCortex checkpoint trained by the same engine. Installing one is not an API call to another AI; its tensors become the brain's starting parameters. Its model card must disclose all upstream data and post-training.

The bundled default is documented in
[OMNI_STARTER.md](OMNI_STARTER.md). Its materialized manifest records the exact
project-authored dataset hashes, objectives, parameter checksums, and local
loss/pack metrics before the immutable origin snapshot is created.

## Hardware profiles

Profiles change scale, not the meaning of a module:

| Profile | Typical machine | Text scale | Media baseline | Training strategy |
| --- | --- | --- | --- | --- |
| Micro | 4–8 GB RAM, CPU | Architecture/smoke scale | Tiny images, short low-rate audio, tiny clips | Batch 1, eight-step accumulation, activation checkpointing |
| Personal | About 16 GB RAM | Small research model | Moderate latent sizes | Batch 2, four-step accumulation, activation checkpointing |
| GPU | NVIDIA/DirectML-capable PC | Larger local research model | Larger packs | Batch 8, two-step accumulation, activation checkpointing |
| Workstation | High-memory local host | Largest bundled baseline | Larger packs | Batch 16, direct block execution |

A large model that cannot fit active state is not made feasible merely by waiting. The builder selects a smaller compatible model. Replay and stability tensors remain CPU-resident between operations and are durably offloaded to safe-tensor checkpoints; only the active micro-batch moves to the selected CPU, CUDA, or DirectML device. The current release reports distributed capability but does not claim a multi-node trainer.

## Experience lifecycle

Every input can take one or more learning paths:

- **Encode:** immediate VSA, graph, spike, and fast-weight updates.
- **Consolidate:** short background training against the new experience plus replay.
- **Pretrain:** longer sequential prediction training over a selected corpus.
- **Archive:** retain exact source bytes with provenance; this is separate from learning.

Human Consolidation uses Encode and scheduled Consolidate. Synapses Only erases raw training material after feature conversion. Total Recall can retain Archive data.

## Continual-learning protections

- A complete safe-tensor checkpoint remains readable while a candidate trains.
- Candidate promotion records a durable phase and pre-candidate backup. If the
  worker stops between the three atomic file replacements, the next load
  conservatively restores the complete prior checkpoint before reading tensors.
- Reservoir or latent replay reduces catastrophic forgetting.
- Stable, frequently used synapses have lower plasticity.
- Persistent squared-gradient importance and slow anchors add an EWC-like
  stability penalty to language, dialogue, consolidation, and modality
  updates.
- Background corpus training forms padded physical micro-batches and performs
  the profile's configured number of gradient-accumulation passes before one
  clipped optimizer step.
- Micro, Personal, and GPU profiles checkpoint decoder-block activations during
  training. Replay remains on CPU until used, and all durable checkpoints are
  non-executable safe tensors.
- Optimizer internals are not persisted because the checkpoint format avoids
  pickle. Dataset manifests, per-file cursors, coverage, and crawler frontiers
  are persisted separately, so an interrupted corpus job resumes at its next
  atomic file without skipping a committed file.
- Interrupted training candidates are marked `interrupted` and quarantined on
  the next load; a killed-worker integration fixture verifies that stable model
  parameters and counters survive.
- The user can pause slow learning without erasing fast neural state.

These mechanisms reduce forgetting; they cannot guarantee perfect retention.

## Conversation learning

The current human turn and the explicitly reported, capacity-bounded ring of recent human/brain dialogue tokens are ordinary model input. Long-term sources are never silently pasted beside them. Parameter-only mode conditions activations with learned semantic state; working-memory mode additionally blends bounded recency-weighted recurrent vectors. The trace separately reports recent-dialogue expansion and confirms that hidden/long-term prompt expansion is absent. After the turn:

- human text is eligible for self-supervised learning;
- generated text receives a lower default replay weight to limit self-amplifying errors;
- explicit corrections receive higher surprise and consolidation priority;
- style, vocabulary, and slang can change through ordinary continuing prediction;
- no hidden rule or persona string is inserted.

## Web and dataset training

Catalog and crawl jobs record URLs, hashes, timestamps, quarantine state, and
declared licensing. Dataset traversal has no product-defined file, byte, row,
or chunk ceiling: it writes a deterministic manifest, streams every regular
file, and records processed/rejected coverage. Supported readers include
PDF/text/source, CSV/TSV, JSON/JSONL, Parquet, Arrow IPC, SQLite, ZIP/TAR and
WebDataset shards, EPUB/Office archives, and local Hugging Face-style
`data_files` manifests. PyArrow handles columnar batches without loading an
entire table.

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

Declarative build recipes may select a blank or compatible starter origin, architecture recipe, hardware tier, memory policy, modalities, and initial tool grant. A recipe cannot contain commands. Modality packs carry only a model card, manifest, checksum ledger, and safe tensors; installation is namespace-, shape-, architecture-, and license-validated twice. See [CATALOG_FORMATS.md](CATALOG_FORMATS.md).
