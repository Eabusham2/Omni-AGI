# Native Ponder and Best Work implementation evidence

Native turn-time Ponder and optional Best Work now use the same weighted
ternary brain. Ponder has learned continuation and no fixed pass or elapsed-time
cutoff. Best Work maintains private inference branches and exposes only the
selected result to the existing action and observation pipeline. The evidence
below covers source behavior and small deterministic fixtures, not broad
intelligence, truthfulness, or training-folder qualification.

Authority is the settled decisions in [PLAN.md](PLAN.md), especially voluntary
Ponder, optional resource-admitted search, the 0.95 temperature ceiling, and
the exclusion of reward-model/RLCD/RLVR training. Original request lines
198–200 remain in [REQUEST_VERBATIM.md](REQUEST_VERBATIM.md).

## Voluntary Ponder

`AdaptiveBrain._native_pre_speech_ponder` in `engine/omni_core/brain.py` is entered
after an actual learned action proposal. It does not insert a reasoning
instruction or require generated prose. It refines this turn's internal memory
through the existing liquid state, LIF router, and native idea adapter, then
evaluates the learned internal action head after each completed pass.

The loop ends on a learned transition away from Ponder, numerical convergence,
Stop, Steer, or actual memory/disk pressure. Resource pressure is recorded as
incomplete work, not successful thinking. There is no fixed pass count or
10-second completion cap. A scheduler yield releases the interpreter between
safe operations; it is not a model-facing instruction or a reasoning deadline.
Stop and Steer are checked between native operations, so their latency includes
the currently executing operation rather than claiming instantaneous kernel
interruption.

Ponder uses a private random generator and private liquid scratch. Router
membrane and spike counters are restored on normal exit and exceptions. The
returned internal vector conditions generation; Ponder inference itself does
not update learned weights. Existing experience admission, working memory,
replay, and discrete ternary learning remain separate and unchanged.

The trace records actual passes, router steps, elapsed time, state delta,
continuation action, convergence, and stop reason. No hidden prose transcript
or guaranteed chain of thought is fabricated. `handledInTurn` prevents the
host from starting the same Ponder action again after it was already handled,
including an incomplete resource-pressure result.

Selecting Ponder does not implicitly enable multi-answer search. With Best
Work off, an ordinary answer retains its single live decode and streaming
token counters.

## Optional Best Work

`engine/omni_core/best_work.py` provides an inference scheduler. The settings
are `bestWorkEnabled` (default false) and `bestWorkExplorationBranches` (integer
1–16, default 16). The latter counts additional candidates, not the ordinary
branch. Enabling search does not replace model geometry or load another model.

The initial branch request uses measured normalized routing-logit entropy:

| Normalized entropy | Requested additional branches |
| --- | --- |
| Below 0.25 | None; retain direct single-pass output |
| 0.25 to below 0.50 | Up to 4 |
| 0.50 to below 0.70 | Up to 8 |
| 0.70 or greater | Up to 16 |

The user's budget and real resource admission can reduce those counts. An
admitted search always contains the ordinary branch with its usual sampling
temperature. Additional temperatures follow a 0.20–0.95 ladder. When Best Work
is enabled, a conflicting ordinary temperature above 0.95 is rejected rather
than silently altered. These thresholds are search heuristics, not calibrated
measures of task difficulty or intelligence.

Disagreement can expand the pool within the configured budget. Strong support
for a completed ordinary candidate, with agreement from another completed
candidate, can stop unfinished exploration. Resource refusal for an additional
candidate preserves already admitted ordinary work. A failure of one branch
does not discard other completed candidates.

Selection uses whitespace-normalized exact output agreement multiplied by
sampled token likelihood. It does not claim semantic equivalence or calibrated
confidence. Current local verification checks Python fenced-code syntax and
structured action schemas. It neither executes arbitrary candidate code nor
proves a program correct. When all completed candidates fail static checks,
the ordinary fallback is explicitly labeled, not reported as verified truth.
No reward training, preference model, PRM, RLCD, or RLVR loop is introduced.

## Private state and selected effects

`engine/omni_core/cooperative_generation.py` advances candidate sessions over
the native decoder loop in `engine/omni_core/model.py`. Each session owns its
seeded sampler, prefix, attention cache, activity ledger, and decoder generation
metadata. Learned model storage is shared read-only; there is no copied
floating-point learned model or averaging of branch weights.

The scheduler is **cooperative and interleaved**, not simultaneous GPU-batched
decoding. Each branch advances at a token boundary. A private Ponder operation
within an advance can take multiple recurrent passes; it remains Stop/Steer
responsive, but other branch tokens wait for that operation. Pager activation,
cancellation scopes, and ContextVars are exited before control returns to the
scheduler. Cache pages and resource leases are released on completion,
interruption, failure, and discarded branches.

Private candidates cannot publish tokens or supply a world-observation
callback. Their action proposals are collected without host dispatch and
without private self-observation learning. Only the selected seed and
temperature are replayed through the existing live action/observation path.
Existing action receipts prevent duplicate host effects. New external evidence
may change the selected live continuation at an observation boundary; it is
not represented as an unchanged private prediction.

Resource accounting admits branch token/ledger state and uses the existing
working-attention pager for cache allocations. It is not a promise that every
backend allocation or whole-process resident-memory limit is OS-enforced.

## Settings and progress interface

The Python configuration validates typed enable/budget fields, external input,
saved round trips, and enabled-temperature compatibility. `brain.update_config`
accepts those runtime settings without changing the learned architecture.
The shared `BrainConfig` and desktop defaults now preserve these settings.
Repository normalization rejects invalid enabled values and budgets, while
older native configurations receive the disabled default. Device & runtime
has an optional Best Work checkbox and a collapsed search-budget control;
saves use the existing preflighted worker/config transaction. Two focused
desktop cases passed for save/reload, invalid-input rejection, worker delivery
and unchanged architecture. Both TypeScript projects passed. Screenshot and
full interaction acceptance remain open.

The worker forwards `cognition-progress` as `chat-cognition`, in the normal
notification order. It is not a reply-completion event. Measured progress
contains a Ponder or Best Work phase, `active: true`, and applicable real
counts/rates. A terminal cognition event uses `active: false` and a stop reason,
including zero-pass exit, Steer, resource pressure, and failure. The ordinary
reply may continue after the cognition indicator ends.

The host normalizer, forwarding, and compact expandable presentation are
implemented separately in `src/shared/cognition.ts`, `brainService.ts`,
`chatActionController.ts`, and `App.tsx`. Their focused checks do not replace
the requested all-theme screenshot audit.

## Focused verification record

Completed runs recorded for this implementation slice:

| Evidence | Verified contract |
| --- | --- |
| 20 scheduler, private-generation, and Ponder cases passed in 0.499 s | Eight scheduler cases, six private-generation cases, six adaptive Ponder cases; resource refusal, disagreement expansion, ordinary fallback, interruption, scoped state, real paged-cache cleanup, and failure cleanup. |
| Three configuration cases passed in 0.001 s | Default off; saved/external round trips; strict budget typing; enabled ceiling without silent ordinary-temperature changes. |
| Three tiny native chat cases passed within the earlier 18-case run in 17.065 s | Ordinary streaming occurs during the one decode; Steer preserves already streamed text; private search publishes one selected replay and truthful progress. This overlaps the earlier scheduler/Ponder coverage and is not an additional 18 distinct cases. |
| Selected-Ponder chat and exception-cache cases passed in 6.393 s | Selecting latent Ponder still produces one ordinary answer decode; an exception releases pages and restores generation metadata. |

Fixture files are `test_best_work.py`, `test_best_work_config.py`,
`test_cooperative_generation.py`, `test_native_pre_speech_ponder.py`, and
`test_chat_single_pass.py` under `engine/tests/`. The Ponder continuation fixture
uses a real packed sign head and exceeds the superseded pass/time limits with
a controlled clock. The chat fixtures run a tiny native decoder; selected
entry actions are controlled so the test isolates execution rather than
pretending to prove learned conversational competence.

A seventh private-generation fixture subsequently passed in 0.025 seconds.
It verifies that registered child-region generation metadata is restored after
each private advance and retained inside its branch. This narrow helper fixture
does not establish a trained connected brain's search quality.

## Remaining acceptance boundaries

- Finish the affected visible UI interactions and screenshot acceptance,
  including real progress termination while ordinary output continues.
- Complete connected-decoder metadata-isolation evidence and broader
  selected-action/observation integration without duplicate effects. Current
  tiny tests do not establish every possible tool interaction.
- Measure latency, memory, and answer-quality tradeoffs on a trained native
  model before claiming Best Work is smarter or faster. It deliberately adds
  inference work and may delay first visible text while candidates run.
- Full-folder training, broad retention/learned-capability qualification, and
  hardware-specific performance remain the separately deferred work. No user
  brain, corpus training, CI job, commit, or release was performed by this
  cognition evidence pass.
