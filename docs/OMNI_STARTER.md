# Retired Starter, foundation, and Nova research record

This page is historical documentation, not a model catalog. Current Studio has
one whole-brain architecture and origin boundary:

- every new Build initializes and trains native OmniCortex locally under the
  [ground-up contract](GROUND_UP_OMNICORTEX.md);
- whole-brain Import accepts only the exact supported `.omni` schema when both
  current state and immutable origin prove native ground-up OmniCortex;
- the retired bundled Starter, random-only Blank origins, Falcon/FoundationCortex
  adapters, Nova, and other legacy hybrids are not selectable, loadable, or
  migration sources;
- rejecting an old file does not delete or rewrite it.

No historical result on this page establishes the quality of a current native
Build.

## Retired bundled Starter

`omni-starter-bundled-1` was an earlier project-authored bootstrap. It began
from random OmniCortex weights, trained on a small corpus, supervised dialogue
pairs, typed action examples, and synthetic modality fixtures, then recorded an
immutable origin. The mechanism exercised optimization, action, modality,
export, and restoration paths, but its small curriculum never proved coherent
general conversation, cleared-context recall, broad knowledge, coding ability,
or useful media quality.

The current Build does not use that corpus or Starter identity. It uses the
hash-bound native capability curriculum and readiness receipt described in
`GROUND_UP_OMNICORTEX.md`. A Starter manifest in either current or origin state
is therefore rejection evidence, not an import option.

The retired bootstrap itself used no project-authored RLHF, DPO, preference
labels, reward model, refusal/persona tuning, or hidden behavioral prompt. That
historical statement cannot establish anything about an upstream foundation
whose training history was incomplete.

## Retired Falcon/FoundationCortex experiment

Earlier research implemented a frozen packed foundation cortex with a mutable
Omni adapter and reviewed two Falcon-E Base revisions:

| Historical artifact | Pinned revision | Recorded `model.safetensors` SHA-256 |
| --- | --- | --- |
| `tiiuae/Falcon-E-1B-Base` | `f4001b8b1c26d28a717d79a8ece14901816d92e8` | `f62c270b5640c9fce1dedf1ffe84aa8f138443ad1b065e90c7508a0eebc7c79d` |
| `tiiuae/Falcon-E-3B-Base` | `ad18b0713c10a8696144b1112accf4ccd28af6d3` | `25dbd87d68cf853de3bea1ae4615bf2fa027f252b77ec2137a07e94c1a1a6ba2` |

Those records are retained for audit and attribution only. Current Studio does
not search `OMNI_FOUNDATION_ROOT`, download or load Falcon weights, construct a
`FoundationCortex`, expose a Falcon choice, convert Falcon into OmniCortex, or
accept a Falcon-backed `.omni`. Their model cards also did not establish
complete corpus, dataset-license, training-stage, or preference-training
provenance. The applicable Falcon license links remain in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Reviewed but never compatible: Ultron 0.3B Base

`1bitLabs/ultron-0.3b-base-experimental` was reviewed at revision
`cc4f22f1b997dbf6e41a6c9bf2b2939e44ca81e4`. Its recorded base artifact,
`ultron_r008_305m.pt`, was a 1,222,496,651-byte PyTorch ZIP checkpoint with
SHA-256 `c67ebb6ed4910aea1935c878fda84caf669104f171ca2cac4689be69fcd027ae`.
Loading it would have required pickle-compatible deserialization and upstream
custom architecture code. It was never accepted by Omni's data-only,
safetensors-only, no-remote-code boundary and remains neither a Build nor an
Import option. Dataset names in its model card did not provide the exact
revision, row-level provenance, and license ledger required by Omni.

## Nova is not native-build evidence

The saved trained Nova brain was a Falcon-backed legacy hybrid. Its frozen base,
mutable adapter, substrate, replay, and optimizer state belonged to the retired
foundation experiment. Consequently:

- successful persistence or parameter mutation in Nova did not validate fresh
  native OmniCortex initialization;
- Nova's chat output did not validate the current ground-up curriculum or
  readiness gate;
- Nova recall or routing failures do not measure a newly trained native Build;
- Nova cannot be selected or imported to stand in for missing native evidence.

A current capability claim requires a newly constructed native OmniCortex,
complete attributable training coverage, an exact readiness receipt, held-out
evaluation after restart, exact ternary coverage, and the relevant packaged
platform evidence. At the time of this record, a complete post-cleanup native
Build and competence run still has to supply that evidence; architecture and
unit tests alone are not proof of conversational quality, AGI, consciousness,
human biological equivalence, perfect recall, or factual reliability.

## Why this record remains

Keeping hashes, upstream identities, negative findings, and license links makes
old logs and user-held files interpretable without preserving a compatibility
path. Research references—including BitNet, snnTorch, NCPS, and an ignored
`.runtime/bitnet-src` checkout—remain useful inputs to independent OmniCortex
engineering. They are not pretrained brains, fallback runtimes, or selectable
product models.
