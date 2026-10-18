# Mixed sequential regions inside complete decoder operations

## Actual supported boundary

The existing operation Auto controller may compare `native-mixed-sequential`
plans for six **already supported** operation kinds: ordinary and connected
retained cached steps, complete prefixes, and uncached full forwards. These
are alternative placements of the same captured decoder, not another model
or a new work-kind declaration. The registry still has thirteen work kinds.
Perception, media generation, and training retain their existing homogeneous
complete-operation consumers; they do **not** advertise mixed-region support.

The implementation is sequential. It does not claim concurrent device
execution, cross-rank scheduling, collective custody, or distributed speedup.
It also does not claim a hardware or neural-quality result from the new inert
contract tests. No model was created, loaded, trained, or hardware-qualified
to verify this source change.

## Issued routes and execution

`mixed_region_native.NativeMixedRegionPool` walks the real captured closure's
exact `DecoderBlock` objects. A registry digest binds their names, current
object identities and actual packed geometry. A route carries the reviewed
outer/home candidate plus ordered, non-overlapping spans of that registry.
Every span refers to an exact base device descriptor from the current review.
The declared completion participants are the union of the real used devices;
CUDA retains its explicit packed-CPU-tile fallback acknowledgement.

Candidate generation is linear in region count and alternate device count:
single-target prefixes/suffixes and, when useful, one multi-target contiguous
partition. It does not enumerate a Cartesian product of assignments or impose
a fixed neuron/model/region ceiling. Metadata, the registry, and the exact
descriptor index must fit current RAM reservations. If that extra metadata
cannot fit, the existing homogeneous candidate list remains available.

The production `_Target` uses the existing `regional_block_call` seam, not a
metadata-only wrapper. For an assigned block it admits the exact reviewed
device again, moves actual input/control activity to that target, selects that
target for packed projections and attention, then returns output activity to
the outer/home device. A fresh compute reservation includes the actual block
geometry. Buffers and compute hints are restored even when the block raises
or Stop arrives. The containing complete operation retains its existing
packed-residency restoration, native-owner, activity, RNG and source fences.

Cached mixed regions require the existing private paged K/V frontier. They
do not materialize the entire retained cache on another accelerator or silently
change context size. Connected recall, active memory, STDP, learning effects,
working-memory ownership and home state remain governed by their existing
owner protocol. No memory update is replayed to time an alternative route.

## Whole-path evidence and bounded results

The existing complete cached/prefix/forward operation encloses all block
placements, transfers, native work, home outputs and restoration. Eligibility
comes from those **complete** authenticated observations and the existing
exact output/cache/activity/sampling gates, not isolated projection rates or
made-up device throughput. Physical resource observations include every
participating target and home device, with the existing shared-memory-group
deduplication; CPU and Metal on one unified memory group do not become two RAM
budgets.

An expanded mixed pool retains only the reference result and one current
private trial, rather than one large output/cache copy per hypothesis. After
selection it re-executes the selected private computation once and rechecks
the exact complete result before the existing real publication boundary.
Message delivery, sampling commit and actual learning are not repeated.
Existing homogeneous-only behavior is unchanged.

The fixed-memory whole-path selector accepts a derived route only through
`admit_mixed_primitive`: the current runtime operation must still own its
issued descriptor, exact registry, review, session and complete identity.
Fresh per-device admissions remain mandatory. A matching public hash, saved
plan, imported descriptor or copied JSON is not execution authority. The
runtime-issued pool is retired with the captured operation's `ExitStack`.
The retained-step controller can reuse a qualified layout only under its
existing explicitly labelled fresh-frontier inference contract and a newly
issued matching live registry, never by reusing a previous output or cache.

## Verification added for this change

- `engine/tests/test_mixed_region_schedule.py`: new standard-library-only
  schedule, exact custody, pressure fallback, decorated dispatch, buffer/hint
  restoration, paged-cache, retirement, publication and no-real-evidence-from-
  inert-receipts contracts. Fake tensors and clocks exercise control flow,
  not neural outputs, provider responses, real devices or measured speed.
- `tests/nativeMixedRegionCandidate.test.tsx`: strict main-process descriptor
  parsing, nested copy isolation, hash/home/participants/range checks, six-kind
  discrimination and honest sequential participant labels.
- Python syntax compilation and Node/Web TypeScript checks cover the touched
  cross-language seams. Existing closed neural/provider/UI suites were not
  rerun as part of this focused change.
