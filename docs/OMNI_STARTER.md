# Bundled Omni Starter model card

## Scope

`omni-starter-bundled-1` is the initially trained local baseline included with
Omni AGI Studio v1. It is trained from random OmniCortex weights before the
immutable origin snapshot is created. It is intended to prove the complete
starter, continual-learning, action, modality, export, and restoration paths on
ordinary hardware. It is not a frontier foundation model or a claim of AGI.

## Architecture and forward weights

The starter uses the same custom OmniCortex architecture as a Blank Brain:
ternary cortical projections, a growable neural substrate, LIF/STDP dynamics,
CfC temporal state, a whole-input global workspace, a learned structured action
head, and the four baseline modality packs. Every eligible inference weight is
verified as exactly `-1`, `0`, or `+1` and exported in packed two-bit shards.
Higher-precision master weights, activations, normalization, liquid state, and
learning traces remain necessary for training.

## Training data and provenance

The bundled source is entirely project-authored:

- 12 short text passages about connected ideas, memory, uncertainty, tools,
  imagination, agents, continual learning, and experimental improvement.
- 16 typed action examples covering `talk`, `tool`, `imagine`, `agent`,
  `ponder`, `learn`, `evolve`, and `stop`.
- deterministic synthetic fixtures used only to initialize and verify the
  vision, image, audio, and video learning paths.

No third-party model or model-generated dataset is used. The materialized
starter manifest records the exact record counts, canonical SHA-256 hashes,
license, training parameter checksums, the per-passage corpus loss curve,
action-training metrics, and modality-training metrics. Because the baseline is
materialized locally for the selected hardware profile, these measurements are
stored with that brain rather than claimed as one universal loss curve.

## Objectives

The baseline performs next-byte corpus prediction, whole-experience idea
reconstruction, temporal prediction, spike homeostasis, structured-action
trajectory imitation, and synthetic cross-modal training. Later user
experiences follow the same neural learning path through immediate plasticity,
assembly consolidation, replay, and slow-weight updates.

## Excluded post-training

The starter uses no RLHF, DPO, preference labels, reward model, refusal or
persona tuning, imported preference-tuned weights, or hidden behavioral system
prompt. Tool capability schemas are typed neural inputs; they are not prose
instructions about how the identity must behave.

## Limitations

This small baseline cannot supply broad factual knowledge, reliable coding
ability, frontier media quality, human-level reasoning, consciousness, or
perfect recall. A compatible, separately published `.omni` starter may provide
more training, but its own manifest, hashes, data licenses, objectives, and
model card must be verified before installation.
