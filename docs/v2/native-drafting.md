# Native auxiliary drafting evaluation

## Decision

**Evaluated, not enabled.** No app switch or runtime default was added. The
correct serial prefix verifier was slower than ordinary cached native decoding
in the fixed short comparison: median 5.932 versus 3.171 ms/token, or 0.535x
ordinary throughput. It still needed one target-model call per output token.
The decision preserves OmniCortex's existing behavior rather than calling an
incorrect parallel verification path a speedup.

This satisfies the plan's investigation/comparison gate, not a claim that a
production DFlash2 accelerator is implemented. No foreign model, floating
learned master, hidden prompt, reward loop, user corpus or long initialization
was used. The full v2 release goal is separate.

## Primary sources and deliberate adaptation

[DFlash2's official Speculators documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash2/)
describes local causal mixing and a predecessor-conditioned low-rank selector.
It explicitly distinguishes its experimental training objective from what
the public inference/checkpoint source establishes. The
[authors' technical description](https://inco.ai/blog/dflash2/) and
[pinned Z Lab inference source](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py)
were inspected; their published speed results are not Omni measurements.

[Leviathan et al.](https://arxiv.org/abs/2211.17192), Algorithm 1 and Appendix
A.1, supply the lossless speculative-sampling rule. Given the actual proposal
law `q` and target law `p` at the same prefix, acceptance is `min(1, p/q)`;
rejection samples the normalized positive part of `p-q`. Substituting an
unmodified `p` after rejection biases the result. Top-K/temperature changes
must already be included in each actual law. Matching distributions does not
mean matching ordinary generation's individual seeded output: the algorithms
consume random draws differently.

Omni's independently written `native_draft_evaluation.py` is a **small research
adapter**, not a reproduction of DFlash2's diffusion/attention backbone. It
reads the same brain's packed token embedding and language readout, without
registering or copying another decoder. Its own context projection, position
rows, local two-tap mixing, feed-forward path and selector projections use
native signs, integer strengths/plasticity and ordinary packed learning.
Current target activity conditions its parallel candidate features; a small
selector walks candidate choices. Training uses own-target unary/path
imitation, with training-only teacher-candidate inclusion. Actual proposals
never insert the verifier's answer into their candidate set.

The upstream repository is MIT-licensed. Its notice is preserved in
`licenses/DFlash-MIT.txt`; no upstream code module or weights are imported,
bundled or executed. Research mapping is also recorded in `RESEARCH.md`.

## Actual target-verification boundary

The ordinary workspace/block path is prefix-causal, but grown experts select
their routing from the **complete current prefix**. One long
`target.forward(prefix + guesses)` therefore gives its earlier positions the
route for the longer prefix. Those are not generally the target distributions
that ordinary generation would use at each shorter prefix.

The actual tiny two-expert fixtures measured maximum logit differences from
0.000538 to 0.006925. With temperature 0.7 and Top-K 20, maximum probability
total-variation difference reached 0.045652. Small is not zero; accepting such
logits as the target would not preserve the requested sampling semantics.
The no-expert control matched within stated float tolerances, not a universal
proof for arbitrary evolved or connected brains.

The evaluated correct path uses the existing incremental `_generation_step`
for each real prefix. It retains no live production cache. Its own cache is
closed at the atomic research-round boundary. This includes re-prefill,
proposal, selection, rejection and cleanup overhead, and prevents the current
test from pretending a transactional block-cache/crop API exists.

A future faster path could compute each verified position's actual running
prefix routing rather than changing the learned brain to force fixed routing.
It would also need exact paged cache frontiers, discard/crop behavior, resource
admission, observations, action receipts and a versioned resumable draft/
verifier cursor. The existing token-boundary cursor must not silently discard
or resample its random draws. None of that runtime integration is claimed here.

## Isolation and bounded scope

Each research round uses separate draft/verifier generators, read-only learned
target/auxiliary state, private cache ownership and restored generation
metadata. It publishes no token callback, action, observation or Ponder text.
An exception, admission failure or interrupted private round restores both
random frontiers and closes the private cache. Shared target embedding/readout
training modes are rejected before auxiliary imitation can mutate them.

The current evaluation supports only static, unmasked, no-logit-noise sampling.
Production UTF-8 publication, live actions/perception, voluntary Ponder,
Best Work and cold resume continue through the existing ordinary path. This
research module is not called by that path. There is no decorative Draft
indicator or unsupported drafting checkbox.

## Fixed native comparison

`engine/tests/benchmark_native_drafting.py` measured:

- Three fixed random native target seeds: 4129, 4131 and 4133.
- Width 8, one decoder block, two real growable experts, 261-token vocabulary;
  two fixed role/token prefixes; three alternating-order repetitions each.
- Ordinary cached `OmniDecoder.generate` versus the serial-verified auxiliary
  path, at temperature 0.7, Top-K 20, noise zero and no printable mask.
- Eighteen paired comparisons, 12 actual output tokens per run. The two
  algorithms' individual sampled strings are not expected to be equal.
- Twenty-four actual native auxiliary updates per target, using two trajectories
  generated by that same native target: 72 total updates. The private auxiliary
  loss improved in all three fixtures (3.124→2.849, 3.216→2.951,
  3.139→2.846); this is tiny imitation, not broad language competence.
- Target checkpoint tensor storage: 278,418 bytes. Auxiliary: 1,005 bytes.
  Floating learned `Parameter` bytes: zero. These totals include real strength/
  control state, not a claim that the whole brain occupies 1.58 bits/parameter.
  Proposed scratch admission: 148,976 bytes, not whole-process RSS.
- Native target bytes remained unchanged during auxiliary learning and
  inference; learned auxiliary bytes remained unchanged during inference.
  Relevant source hashes matched before/after the comparison.

Raw timings, output/call counts, training receipts, source/state hashes and
verification differences are preserved in
[`native-drafting-comparison.json`](native-drafting-comparison.json).
Timings use `perf_counter_ns` and include the actual decoder generation path,
not just one selector arithmetic operation. This short CPU-only fixture is not
representative of a trained large model, whole-app latency or Metal/CUDA/NPU
throughput. Poor draft acceptance and repeat prefill are visible costs rather
than omitted from timing.

## Focused tests and remaining enablement gate

Thirteen distinct cases in `test_native_draft_evaluation.py` passed across the
focused runs: the initial eleven in 0.514 seconds, mixed shared-head protection
in 0.077 seconds, and the added positive-residual case plus strengthened
post-sampling interruption in 0.069 seconds. These cover actual packed learning,
safe reload, proposal support, acceptance/residual algebra and runtime choice,
neural/global-RNG/metadata isolation, real paged cache cleanup, admission,
private rollback and the expert-routing counterexample.

No default change is authorized. Before any optional production integration,
require a valid general block verifier/cache/cursor contract, sampling and
Stop/Steer/observation equivalence for every supported route, and matching
representative end-to-end speed/retention/resource evidence. The present
negative result does not authorize weakening those requirements or replacing
the native brain with a foreign fast model. No full training, CI, commit,
package or release was performed by this research lane.
