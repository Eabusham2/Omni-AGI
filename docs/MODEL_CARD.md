# OmniCortex model card template

Every distributed `.omni` checkpoint must include a completed copy of this card in its manifest.

## Identity

- Model/checkpoint name:
- Omni architecture schema:
- Parameter count:
- Effective weight format:
- Modalities:
- Parent checkpoint:
- Training run checksum:

## Training

- Random initialization or parent:
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
- Imported upstream model:

If any imported checkpoint or generated dataset has unknown or preference-trained provenance, it must be stated here. “No app-level RLHF” must never be used to hide upstream training.

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

A blank or lightly trained OmniCortex is not expected to converse fluently. Ternary effective weights do not make the entire runtime 1.58-bit. Parametric and semantic memories are lossy. Observable personality or self-report is not evidence of consciousness. Tool execution can affect the host according to its configured grant.
