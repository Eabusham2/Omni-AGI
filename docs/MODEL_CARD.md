# OmniCortex model card template

Every distributed `.omni` checkpoint must include a completed copy of this card in its manifest.

## Identity

- Model/checkpoint name:
- Omni architecture schema:
- Parameter count:
- Effective weight format:
- Modalities:
- Native ground-up origin receipt:
- Training run checksum:

## Training

- Random initialization seed and native parent lineage:
- Datasets and versions:
- Dataset licenses:
- Tokens/samples/hours:
- Hardware and duration:
- Self-supervised objectives:
- Continual-learning settings:
- Raw archive included:

## Post-training disclosure

- RLHF: **none for official Omni training**
- Preference/reward model: **none for official Omni training**
- Supervised behavioral tuning:
- Synthetic data producers:
- Imported upstream/foundation model: **none (required for whole-brain import)**

Whole-brain packages with an upstream/foundation model, retired Starter/Blank
origin, legacy hybrid, or unknown origin provenance are rejected rather than
described into compatibility. Generated datasets and modality-only packs must
still disclose unknown or preference-trained producers. “No app-level RLHF”
must never be used to hide training-data provenance.

## Evaluation

- Held-out prediction loss:
- Continual-learning retention:
- Idea recall:
- STDP/plasticity checks:
- Text samples:
- Vision/image metrics:
- Audio metrics:
- Video metrics:
- Tool-protocol accuracy:

## Known limits

A newly initialized or lightly trained native OmniCortex is not expected to
converse fluently. Ternary effective weights do not make the entire runtime
1.58-bit. Parametric and semantic memories are lossy. Observable personality or
self-report is not evidence of consciousness. Tool execution can affect the
host according to its configured grant.

The new native path keeps learned linear, convolutional, embedding, spiking,
normalization-gain, residual-gain, and grown expert-route weights as packed
ternary codes and mutates those codes directly during learning. It does not
retain a full FP32 master copy or Adam moments for those synapses. It decodes
bounded rows/positions for arithmetic; this is not yet a fused BitNet kernel.
Working activations, fixed normalization and gain bases, liquid activity,
eligibility, and bounded update scratch still use higher precision; they are
not additional learned weight copies. In the new v3 substrate source, adaptive
neuron/assembly vectors also use one shared packed ternary row, with transient
normalized floating reads for VSA math. The v3 persistence and inspector
readers are source-wired, but native learning/recall quality and disk-paged
scaling remain unverified. A floating trainable `nn.Parameter` in the native
module roots fails the packed-runtime audit; v3 VSA rows have their own
canonical packed validation.
Sparse idea links are exact ternary weights and their persisted v3 shards are
packed; live graph records also contain non-weight metadata. The paged v3
writer bounds row buffers and verifies immutable shards, but it still scans
all neuron/assembly metadata on each checkpoint. This is not yet evidence of
low-wear or fast whole-corpus training.

Thus "1.58-bit" describes learned ternary weights, **not** the entire
process's memory footprint. The new control schema is incompatible with older
floating-control checkpoints; strict loading rejects them. Packed projections
have source-level checkpointed uint8 output-row resistance, not a resident
FP32/Fisher/anchor model or proof of non-forgetting. The native packed
distributed path currently supports a single rank; synchronized multi-rank
mutation remains an implementation gap.

The scoped v1.1.0 code/UI/package/worker-health and all-target release
verification completed, as recorded in `docs/IMPLEMENTATION_STATUS.md`.
It did not train or qualify a native brain. The new uncommitted audit
corrections and their focused source checks are separate from those published
artifacts and do not establish current trained-native retention/quality.
Full native training quality, cold-load speed, GPU/MPS performance and useful
conversation/recall remain unproved and deferred to a later user-initiated
goal; no model test is claimed by this documentation correction.
