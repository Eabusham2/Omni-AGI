# Native retention research and comparison gate

## Decision and authority

The approved [plan](PLAN.md) treats the names in the [original request](REQUEST_VERBATIM.md)
as research suggestions, not a mandate to stack every algorithm. Native packed
signs, individual integer strengths, plasticity, ordinary experience, replay,
and one connected multimodal brain remain authoritative. No foreign foundation,
floating learned weight master, preference/reward model, RLVR, or RLCD loop is
introduced here. No runtime default changed.

The new [comparison harness](../../engine/omni_core/retention_comparison.py)
measures the existing packed projection's resistance/rehearsal tradeoff. One
candidate passed the tiny component screen; it is eligible for a larger native
comparison, **not adopted**. The two failed candidates remain failed even though
one improved average retention. Neither paper results nor this constructed
classification fixture establish whole-brain knowledge retention or intelligence.

## Primary-source method survey and native mapping

The compatibility judgments below are engineering inferences from the cited
papers and the named repository mechanisms. “Related” does not mean that Omni
implements the paper or inherits its reported guarantees. Sources were checked
October 7, 2026. Paper-page lookup was unavailable for some older papers; original
papers, author material, and proceedings were used instead.

| Suggestion / primary source | Actual mechanism and compatibility decision |
| --- | --- |
| **PNN** — [Rusu et al., Progressive Neural Networks](https://arxiv.org/abs/1606.04671) | New columns read frozen older columns through lateral connections. Freezing/routing is part of the paper's preservation argument; capacity grows with tasks. Omni's `OmniDecoder.grow_expert` / `grow_depth` and connected `RegionExchange` already provide native growth and cross-region connections, but shared routing/readouts remain plastic. **Retain the useful growth/connectivity idea; do not claim literal PNN or a no-forgetting guarantee.** A frozen-column variant needs its own retention, routing, growth-cost comparison before adoption. The paper's RL experiments do not authorize a reward loop here. |
| **LwF** — [Li and Hoiem, Learning without Forgetting](https://arxiv.org/abs/1606.09282) | Preserve old outputs on new-task inputs using distillation while learning the new task. It avoids requiring old training examples, not the cost of old-function targets. A frozen **own native packed checkpoint**, streamed targets, and output-consistency objective could be compatible; no external foundation is needed. Existing replay is related but is not LwF. **Defer** until a new-only-data comparison measures acquisition, retention outside the new distribution, teacher-pass cost, and checkpoint residency. |
| **SI** — [Zenke et al., Continual Learning Through Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html) | SI accumulates parameter-trajectory contributions to loss improvement and uses importance to protect past learning. Native update-count resistance in `_apply_packed_gradient_rows` is a bounded integer proxy for persistence, **not SI's path integral or task-importance estimate**. **Adaptation only:** compare existing discrete resistance first; a quantized utility estimate would require additional measured state and migration. No SI implementation or paper-equivalent result is claimed. |
| **EWC** — [Kirkpatrick et al., Overcoming catastrophic forgetting](https://arxiv.org/abs/1612.00796) | EWC penalizes movement from previous parameters, weighted by a diagonal Fisher approximation. The packed model's active mechanism is discrete resistance; `AdaptiveBrain._stability_penalty` has no floating weight parameters to penalize on an all-packed core. A packed anchor plus bounded importance codes is conceivable, but **literal differentiable FP-weight EWC is not active and must not be reported as implemented**. Existing resistance was measured below; exact Fisher-weighted protection remains deferred. |
| **CLS** — [McClelland, McNaughton and O'Reilly, original paper](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf), [authors' later account](https://doi.org/10.1111/j.1551-6709.2011.01214.x) | Separate rapid episodic acquisition from slower integration across interleaved experiences. `OrganicMemoryLifecycle`, `DurableReplayBuffer`, pending chat-learning cursors, capability rehearsal, and `_latent_rehearsal_step` already provide related fast/slow paths. **Retain, avoid a duplicate subsystem.** The experiment isolates a small rehearsal schedule, not hippocampal equivalence or the production lifecycle's full behavior. |
| **Predictive coding** — [Rao and Ballard](https://doi.org/10.1038/4580) | Top-down predictions and bottom-up residual errors are a specific recurrent learning architecture. Existing next-byte prediction, modality reconstruction/denoising, and prediction-error retention signals are useful related objectives; they are **not** that full architecture. **Retain current self-supervised objectives; defer a new error-unit hierarchy** until packed recurrent stability, convergence, quality, and cost are compared. No wholesale learning-rule replacement. |
| **MAML** — [Finn, Abbeel and Levine](https://proceedings.mlr.press/v70/finn17a.html) | Learn an initialization through within-task adaptation and across-task optimization. Native discrete in-place updates are not an ordinary differentiable inner optimizer; exact tiled attention explicitly does not support higher-order gradients. **Defer literal MAML.** Any first-order/discrete meta-adaptation would be a new method with task-distribution, branching-state, gradient, and resource tests—not a rename of existing growth or Best Work. |
| **Gradient projection / OGD** — [Farajtabar et al.](https://proceedings.mlr.press/v108/farajtabar20a.html) | Project new gradients away from stored old-output gradient directions. A bounded regional sketch is a possible adaptation, but stored bases cost memory, and projecting a continuous gradient does not guarantee that subsequent stochastic sign/strength transitions preserve old outputs. **Deferred candidate**, requiring post-discrete-update retention checks and explicit basis storage accounting; no full floating learned master is necessary or introduced. |
| **FIM** — [Amari, Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746), [Amari et al., Fisher Information and Natural Gradient Learning](https://proceedings.mlr.press/v89/amari19a.html) | Fisher information is a statistical metric/importance tool, not a stand-alone continual-learning feature. Full matrices have quadratic storage; diagonal/block/sketched estimates change the approximation. Integer native resistance is **not measured Fisher information**. **Defer as a separately budgeted estimator**, or as part of an EWC/OGD candidate, rather than layering redundant penalties by name. |
| **LoRA-style adaptation** — [Hu et al.](https://arxiv.org/abs/2106.09685) | Freeze a base and learn low-rank updates. A small packed regional adapter on Omni's own state might reduce update cost, but native connection strengths, repeated contacts, and new bridges are not automatically low-rank updates. **Defer literal LoRA**; do not import a foundation or silently freeze the shared brain. A packed adapter needs acquisition/retention, rank/capacity, composition, export, and actual resident-byte measurements. |
| **Attention improvements** — [Vaswani et al.](https://arxiv.org/abs/1706.03762), [Dao et al., FlashAttention](https://arxiv.org/abs/2205.14135) | Attention supplies contextual communication; FlashAttention improves exact attention's memory traffic using tiling. `exact_causal_attention` already has bounded query/key tiles, online softmax, recomputed derivatives, and paging; connected decoding uses a shared token frontier. **Retain compatible exact tiling**, not a new retention regularizer. Backend speed claims still require matching outputs/gradients and measured end-to-end device/I/O behavior. This experiment does not benchmark attention. |
| **CLMR** — [Spijkervet and Burgoyne, Contrastive Learning of Musical Representations](https://arxiv.org/abs/2103.09410) | The request gives only an acronym. This is a documented primary-source interpretation, **not confirmation of the intended expansion**. The paper learns music representations from augmented raw audio without labels. A same-brain audio contrastive objective could complement existing reconstruction, but augmentation must preserve relevant sounds and must not silently replace shared modality learning. **Deferred pending acronym confirmation and audio ablations**; no pretrained CLMR weights imported. |

Rehearsal also has a direct continual-learning precedent in
[Rolnick et al., Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682).
The implementation here uses supervised constructed fixtures, not that paper's
reinforcement-learning setup. Replaying observed training records does not create
a reward model or imply that generated self-reports are true.

## Recorded dispositions and current source reconciliation — October 13

These are **research decisions**, not claims that every named paper was
implemented, compared experimentally, or shown to improve a trained brain. The
approved plan allows an adapted, redundant, rejected, or deferred decision.
Only the existing resistance/rehearsal policies were compared in the retained
24-trial experiment below; that is not an ablation of the other algorithms.
No learning objective, native representation, runtime default, or user setting
changed in this reconciliation.

| Method | Recorded disposition | Existing source and deliberately unclaimed boundary |
| --- | --- | --- |
| PNN | **Adapted growth idea; literal frozen-column variant deferred.** | `OmniDecoder.grow_expert`, `grow_depth`, and connected `RegionExchange` provide packed expansion and lateral exchange. Shared routing and readouts remain plastic. Preserving old columns alone would not preserve outputs if their routing changes; no PNN preservation theorem or no-forgetting result is claimed. A frozen-column candidate requires its own routing/retention/resource comparison. |
| LwF | **Deferred own-checkpoint candidate.** | Current observed-record and latent rehearsal are not old-function distillation. Matching old outputs only on new inputs does not establish retention elsewhere. A separate own-native teacher pass, target coverage, residency and acquisition/retention comparison would be required; no external teacher or extra objective is added. |
| SI | **Adapted stability idea; literal SI deferred.** | `_apply_packed_gradient_rows` increases bounded integer contact/row resistance after successful changes. A change count does not measure signed loss improvement along a parameter trajectory. The current proxy is retained, not relabeled SI; a quantized utility estimator would need measured state and migration. |
| EWC / Fisher information | **Existing resistance retained; exact Fisher/anchor candidate deferred.** | The active packed transition policy has neither a measured Fisher estimator nor EWC's previous-weight anchor penalty. `_stability_penalty` applies only to actual floating `named_parameters`; it is empty for the inspected all-packed owners. No floating anchor/master is introduced to make the acronym appear implemented. |
| CLS | **Related fast/slow mechanism retained; another subsystem is redundant.** | `OrganicMemoryLifecycle`, ordinary fast changes, pending observed replay, `DurableReplayBuffer`, capability rehearsal and `_latent_rehearsal_step` already provide complementary acquisition/integration paths. They do not require a rigid one-way memory-stage pipeline, a second answer store, or every observation becoming permanent immediately. Biological equivalence and production retention quality remain unproven. |
| Predictive coding | **Related prediction objectives retained; full error-unit hierarchy deferred.** | Corpus prediction, modality reconstruction/denoising and prediction-error-based retention are present. They are not Rao and Ballard's top-down/error-unit architecture. Replacing the current rule without recurrent convergence, packed-update, acquisition/retention and resource evidence is not adopted. |
| MAML | **Deferred; current learning is not literal MAML.** | Packed backward changes native contacts in place; packed embedding and tiled attention backward are explicitly first-order (`once_differentiable`). No differentiable inner/outer meta-optimizer or task distribution is implemented. A discrete/first-order variant would be a new measured method, not a rename of growth, ordinary replay or Best Work. |
| Gradient projection / OGD; request's “OGP” | **OGD-inspired candidate deferred; “OGP” expansion unconfirmed.** | Orthogonality of a continuous gradient need not survive stochastic individual sign/strength transitions. A bounded direction sketch, resource accounting and actual post-transition retention checks would be necessary. The request's acronym is not silently converted into an implemented algorithm. |
| LoRA-style adaptation | **Floating learned-delta variant rejected by the storage contract; packed candidate deferred.** | Existing native strengths, repeated contacts and zero-initialized packed regional bridges are not a low-rank optimizer. A compatible packed adapter needs composition, rank/capacity, acquisition/retention, persistence and resident-byte evidence. No imported base, silent whole-brain freeze or learned floating adapter is added. |
| Attention | **Existing exact tiled mechanism retained; another retention feature is redundant.** | `exact_causal_attention` has online softmax, bounded tiles, recomputed derivatives and paging. This is contextual computation, not SI/EWC or an independent memory database. No attention speed or whole-Load claim follows from the retention experiment. |
| CLMR | **Acronym unconfirmed; music-contrastive interpretation deferred.** | The cited musical-representation paper supplies one possible meaning, not evidence of the user's intended expansion. Current shared audio reconstruction is not that augmentation/contrastive objective. No CLMR weights, labels, augmentation rule or new objective are introduced without resolving the meaning and evaluating native audio effects. |

### FIM has two materially different primary-source interpretations

The earlier survey used **Fisher information matrix**, a possible importance or
natural-gradient estimator. The original `Fim? ogp?` request does not expand
either acronym. **Fill-in-the-Middle** is also a language/code-training method:
the original [Bavarian et al. paper](https://arxiv.org/abs/2207.14255) trains
infilling by rearranging document spans. This is a data/objective choice, not a
Fisher estimator or a synonym for gradient projection.

Both interpretations are recorded rather than choosing one by assumption.
Fisher estimation remains the separately budgeted candidate above. Infilling
would require an explicitly approved native training contract, preserved
document/span provenance, objective weights and paired acquisition/retention
evidence before default use. The paper's reported results do not establish
benefit or absence of harm for Omni. **Neither FIM interpretation is newly
implemented or enabled.** The incomplete Hugging Face Markdown response was not
used as evidence; the original arXiv paper page supports this distinction.

### Learned state versus permitted activity/control — source review only

The inspected constructor path in `brain.py` selects `OmniDecoder` (or preserved
connected native regions), packed memory/idea bridges, spiking router, liquid
controller and modality hub. `_trainable_modules` and `packed_runtime_audit`
cover all six roots rather than counting only eligible language projections.
The connected-region constructor uses the same native owners. In the inspected
sources:

- `PackedAdaptiveBitLinear` stores packed signs and integer strength/resistance;
  repeated contacts are separately stored. Its floating scalar forward scale,
  online learning-rate control and nonpersistent autograd trigger are not
  gradient-learned projection weights or a full-precision weight master.
- Embedding rows, RMSNorm's learned scale delta, residual scales, liquid maps,
  modality codebooks/positions and regional exchange maps use native packed
  owners. Fixed unit/base gains and rotary caches are distinguished from learned
  deltas, not counted as learned floating projections.
- Tool-route schema identity features are fixed deterministic descriptors;
  packed query/candidate maps are the learned route. Fixed features are not a
  learned embedding shadow or stored cue-to-answer keys.
- Floating activation, liquid/membrane state, timing traces, softmax and transient
  derivatives remain computation/control. They must not be confused with
  authoritative learned projections. The generic floating-parameter stability
  fallback is not evidence that an inspected packed owner actually has such a
  parameter.

This review found **no confirmed active learned floating projection in the
inspected source-selected paths**. No model was instantiated or loaded for this
pass, so this is **not a fresh runtime inventory**, proof for every saved
checkpoint, or full reader/backend coverage. Conditional imported/merged owners
must still pass the actual loaded-root audit; frozen learned floating Parameters
and unexpected packed-owner fields remain blockers. An eligible-layer count
alone is insufficient.

### Retained measurement is pinned to its measured sources

The old verified report remains valid evidence of its own source-stable capture,
not of every later edit. The October 13 read-only reconciliation found these
source hashes differ:

| Source | Verified capture SHA-256 | Reconciliation snapshot SHA-256 |
| --- | --- | --- |
| `model.py` | `d7c78726abe4ed03af8daef6ccbcba1c38799a47997afcd3655af46f5363db29` | `6ff7469fa4c6b920018f8ea889066dcc07d189e684451a224ec32a5f5f18c725` |
| `packed_native_kernel.py` | `1158b897518926fbefc82b17dfa1812392f80ca46b8f426b0231609aca605ac4` | `28a4fe45124c354b953ecf9308f910d0b5fdc0b6e9e8e75413bafdfa48efd36f` |

The harness and `packed_connections.py` hashes still matched the retained
report. This does not erase the differing owner/kernel hashes, retroseal the
report, or establish current-tree acquisition, retention or speed. No old trial
was repeated. The proposed larger own-native sequential quality comparison
remains the adoption gate below; the full user-corpus run is still separate.

## Existing native mechanism under test

The harness imports the production `PackedAdaptiveBitLinear` and calls its real
packed forward kernel and `learn_from_gradient`; accuracy comes from that model's
outputs, not a scripted success score. It checks that there are no learned
floating parameters, no latent floating master, and no non-scalar floating model
buffers. Floating input activity and bounded transient gradient blocks remain
computation, not another learned weight model.

`configure_packed_stability(enabled=True, strength=0.025)` maps the existing
coefficient to `0.10`. Output-row uint8 resistance and the connection's uint4
resistance suppress discrete transition probabilities; actual successful changes
raise integer resistance. Strength/sign movement remains possible. The test's
`row-resistance-off` tag means the **configuration flag is off**, not that every
resistance disappears: the current per-contact floor remains, with coefficient
`1/16` rather than `0.10`. Consequently this is a coupled existing-policy ablation,
not clean causal isolation of row versus per-contact resistance.

The four fixed arms are default resistance, flag-off resistance, default plus
25% rehearsal, and flag-off plus 25% rehearsal. “Default” describes the resistance
setting only: the standalone fixture uses its own declared learning rate and
scale and is not the whole application's default training recipe.

## Fixed protocol and integrity

- Two conditions, three tasks each, four classes, 12 inputs, and seeds **17, 29,
  43**. Each task has 64 training and 96 independent held-out noise realizations.
  Shortcut-shift has a stable class cue plus a stronger nuisance cue whose class
  permutation changes per task; all tasks have a compatible shared solution.
  The control uses disjoint feature blocks and requires no old-task forgetting.
- 64 updates per task, batches of 16, learning rate 0.30, fixed scale 0.25.
  Initial packed state, data, order, and per-step random seeds are paired across
  arms. There are **24 trials and 4,608 actual packed update calls**. No tasks,
  steps, seeds, or parameters were tuned after seeing the results.
- Rehearsal keeps the first 16 balanced **training** records per completed task.
  Four old examples replace four current examples per batch after task one.
  Thus each later task sees **768 current + 256 replayed examples**, versus
  **1,024 current** in non-rehearsal arms. This is an equal-step/equal-batch
  comparison, not equal exposure to new data; the tradeoff is part of the report.
- Acquisition is held-out accuracy immediately after each task. Retention is
  final accuracy on tasks one/two. Forgetting is each old task's best measured
  post-training accuracy minus its final accuracy. The full evaluation matrix,
  initial accuracy, cross-entropy, changed-contact counts, and data/state hashes
  are retained. Held-out examples never enter updates or the replay reservoir.
- One subprocess and one native thread per trial, sequential execution, at most
  60 seconds per worker. `perf_counter_ns` measures update loops after initial
  kernel warmup, excluding evaluation and subprocess startup. Exact tensor bytes
  are separate from whole-worker high-water RSS, which includes Python, Torch,
  data, libraries, allocator and temporary overhead. Batch-byte accounting is
  **not** a claim to include every temporary allocation.
- The verified run checks source hashes before/after each worker and across all
  paired workers. Native CPU kernel loading separately authenticates its binary
  and source manifest. A changed source aborts the comparison rather than mixing
  implementations. The report pins the measured source hashes; later edits need
  a new run before claiming the same result for changed code.

Protocol SHA-256:
`c4c32be559173769ce466cd47c57f0ae7457bb0387f7ffc7992cbc984c7bd198`.

## Measured result: fixed native component only

Decision evidence: [source-stable raw report](research-retention-comparison-verified.json).
The capture completed in **27.863 seconds** on macOS 26.3 arm64, Python 3.10.0,
Torch 2.13.0. Percentages below are means across the three declared seeds; timing
is mean update-loop milliseconds for all three tasks within a trial.

| Condition | Existing resistance policy | Replay | Acquisition | Old-task retention | Update time |
| --- | --- | ---: | ---: | ---: | ---: |
| Shortcut shift | Default | 0% | 100.00% | 37.15% | 86.87 ms |
| Shortcut shift | Flag off; contact floor retained | 0% | 100.00% | 25.87% | 117.37 ms |
| Shortcut shift | Default | 25% | 97.69% | 74.83% | 79.87 ms |
| Shortcut shift | Flag off; contact floor retained | 25% | 100.00% | 77.78% | 83.05 ms |
| Disjoint control | Default | 0% | 100.00% | 100.00% | 77.48 ms |
| Disjoint control | Flag off; contact floor retained | 0% | 100.00% | 100.00% | 109.14 ms |
| Disjoint control | Default | 25% | 98.84% | 100.00% | 142.54 ms |
| Disjoint control | Flag off; contact floor retained | 25% | 100.00% | 100.00% | 82.41 ms |

All trials had **80 bytes** of checkpoint tensors, **84 bytes** of resident model
tensor storage, **zero floating learned-parameter bytes**, and 1,408 accounted
batch-tensor bytes. Rehearsal added a measured maximum **2,688 bytes** of retained
example tensors; no-rehearsal arms retained zero. Maximum whole-worker RSS was
**204.28 MiB** across the run. These tiny model bytes do not describe a complete
brain's RAM, graph, history, attention, or replay footprint. No parallel contacts
were allocated in this small fixture, so their growth cost is not qualified here.

The screen rejects **any** paired seed/task accuracy decrease (zero tolerance),
less than 10 percentage points of task acquisition over initial accuracy, state
plus replay above 64 KiB, worker RSS above 768 MiB or unavailable, training time
above 2× its paired baseline, a missing pair, changed inputs/source/initial state,
evaluation leakage, or floating-master violation. All branches always return
`defaultChangeAuthorized: false`.

- **Flag off alone: rejected.** Retention fell in all three shortcut seeds;
  worst paired decrease was 22.92 percentage points. Two timing pairs also
  crossed the declared bound in the verified capture.
- **Default plus rehearsal: rejected despite its higher mean retention.** The
  shortcut seed-43 third task fell to 79.17% acquisition, and disjoint seed-17
  third task to 89.58%. One retained task fell by 6.25 points. One timing pair
  also crossed the bound. Averaging these regressions away would be misleading.
- **Flag off plus rehearsal: passed this component screen only.** No paired
  acquisition/retention decrease was observed on these six condition/seed pairs;
  mean old-task retention improved by 20.31 points when both conditions are
  pooled. Maximum paired training-time ratio was 1.180. It remains an
  experimental candidate, not a new default or a guarantee of non-regression.

Timing loops are subsecond and affected by scheduling, warm state, and concurrent
machine activity. The timings are measurements, **not speedup evidence**. Three
seeds, easy finite tasks, known task boundaries, and a single projection provide
neither statistical assurance nor open-world retention. Rehearsal's reduced new
data exposure confounds any claim that its benefit is free.

An [earlier exploratory capture](research-retention-comparison.json) is retained
for audit, not used for the decision. It predates the per-worker source-freeze
guard. Its `model.py` hash differs from the verified capture during concurrent
repository work, although all 24 data/final-state hashes match. This is why the
source-stable capture, not agreement of aggregate scores, is the evidence above.

## Reproduction and promotion boundary

### Canonical replay safety repair (October 7)

The single-parent `rehearse_capabilities` maintenance path now applies one
bounded joint native replay transaction to the language action, internal action,
typed-route, and typed-argument owners that are eligible for generic maintenance.
Confirmed-grounded argument owners and independent-experience route owners are
read-only for this maintenance pass, with exact protected-owner reasons recorded.
Their ordinary confirmed-experience and newly installed tool/schema learning
paths remain active; no permanent freeze or new limit on ordinary learning is added.
It retains the existing start/middle/final
committed-cursor cadence. Features come from the existing cortex but are
detached: this operation is **not cortical language training**. Argument coverage
explicitly records eligible, visited, and deferred records under the byte/budget
limit. Initial curriculum teaching and ordinary experience learning remain their
own existing paths.

The obsolete fallback that accessed floating `.weight` / `.norm.scale`, reset
learned heads from a seed, doubled the budget, and excused a probability
regression was removed. Maintenance no longer adds fabricated generic episodes
or rewrites the global episode clock. It does not execute a public readiness quiz
or require perfect tool scores. Candidate measurement covers loss, target
probability, and each previously correct declared record; averages cannot hide
an exchanged or lost route. A declined candidate restores exact packed codes,
strengths, resistance, repeated-contact topology, pending update counters, and
training modes. A historical probability shortfall is reported honestly rather
than erased or used to block chat. Tool permission and grounded-execution
authority are unchanged.

Ten focused tests passed in 5.381 seconds, including actual production packed
heads on tiny detached features, native update/rollback, a forced historical
shortfall, reserve refusal, schema checks, and committed distributed-window
cadence. A separate opt-in saved-native test passed in **221.689 seconds**:
it privately copied the genuinely initialized width-8 fixture, authenticated and
loaded it, ran two replay steps, saved, and freshly reloaded the same parameter
checksum. The source/origin, tested cursors, substrate counts, and grounded
authority stayed unchanged; the initial curriculum was not rerun.

That native candidate reduced aggregate action loss **3.274259 → 3.244991**,
route loss **0.141841 → 0.141459**, and eligible argument loss
**4.010012 → 3.969437**. Counts stayed at **20/35** declared actions and **33/33**
declared routes, with 14 argument records eligible for the fixture's context.
Nevertheless a per-record retention condition declined it, and the exact prior
owners were restored. This is proof of bounded real learning and safe rejection,
not a claim that the candidate was promoted, every tool worked, or corpus-wide
retention improved. No user brain/corpus was trained.

An additional actual packed-owner test checks nonmutation of confirmed private
owners during the maintenance candidate, preserves their pending event counters,
then registers and learns a new schema through the same argument head afterward.
This verifies the maintenance-only boundary, not full retention over arbitrary
private experiences.

### Component comparison commands

From the repository root, choose a **new** output filename (existing reports are
not overwritten):

```sh
PYTHONPATH=engine .venv/bin/python -m omni_core.retention_comparison --output work/retention-comparison.json
PYTHONPATH=engine .venv/bin/python -m unittest engine.tests.test_retention_comparison -v
```

The four harness tests passed in 0.048 seconds, covering deterministic disjoint
partitions, real repeatable packed updates, exact exposure/storage accounting,
rejection of incomplete/drifting pairs, per-task regressions, resource violations,
and the invariant that a screen never authorizes changing defaults. Their
constructed gate inputs are unit tests, not experimental performance evidence.

Before proposing any default change: preregister larger **own native** sequential
text/action/audio/visual holdouts; preserve acquisition as well as older behavior;
test unfriendly orders, contradictions and longer sequences; compare equal new
exposure as well as equal compute; measure paired repeated timings, paging,
storage and transfer cost; verify restart/fork/merge/export and replay cursors;
include regional/repeated-connection growth and stop rollback. Only then review
whether no regression was detected in that broader declared scope. The future
user-corpus training/learned-capability task remains separate.
