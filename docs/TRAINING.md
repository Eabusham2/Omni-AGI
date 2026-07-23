# Training and continual learning

## Blank versus starter brains

A blank brain is initialized from a recorded random seed. It has architecture but no language or world knowledge. Its early output may be repetitive or incoherent.

A starter brain is an OmniCortex checkpoint trained by the same engine. Installing one is not an API call to another AI; its tensors become the brain's starting parameters. Its model card must disclose all upstream data and post-training.

## Hardware profiles

Profiles change scale, not the meaning of a module:

| Profile | Typical machine | Text scale | Media baseline | Training strategy |
| --- | --- | --- | --- | --- |
| Micro | 4–8 GB RAM, CPU | Architecture/smoke scale | Tiny images, short low-rate audio, tiny clips | Small batches, aggressive checkpointing |
| Personal | 16 GB RAM | Small research model | Moderate latent sizes | Gradient accumulation and disk-backed data |
| GPU | NVIDIA/DirectML-capable PC | User-selected | Larger packs | Mixed precision |
| Workstation | Multi-GPU or remote worker | Scalable | Scalable | Distributed data/model parallelism |

A large model that cannot fit active state is not made feasible merely by waiting. The builder selects a smaller compatible model or a stronger worker.

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
- Optimizer internals and data cursors are not persisted. The checkpoint format
  deliberately avoids pickle, so an interrupted slow job restarts rather than
  resuming at an exact optimizer step.
- Interrupted training candidates are marked `interrupted` and quarantined on
  the next load; a killed-worker integration fixture verifies that stable model
  parameters and counters survive.
- The user can pause slow learning without erasing fast neural state.

These mechanisms reduce forgetting; they cannot guarantee perfect retention.

## Conversation learning

The current conversation is ordinary model input and bounded working memory. After the turn:

- human text is eligible for self-supervised learning;
- generated text receives a lower default replay weight to limit self-amplifying errors;
- explicit corrections receive higher surprise and consolidation priority;
- style, vocabulary, and slang can change through ordinary continuing prediction;
- no hidden rule or persona string is inserted.

## Web and dataset training

Catalog and crawl jobs always record URLs, hashes, timestamps, and declared licensing. They stream into shards rather than loading an entire corpus into RAM. The training engine does not execute code found in a dataset or repository unless the separate code tool is explicitly granted authority.
