# Bounded transient memory-cue composition

`NeuralSubstrate.vector_for_labels` previously retained every distributed row
and then allocated a second label-count × VSA-width stack. This temporary
activity—not packed learned weights—could exhaust RAM on large current-task
inputs. The same pattern also existed in continuous assembly composition.

Both now use `streamed_vsa_activity.py`. It visits every label/position, reads
the existing weighted/repeated packed neuron row or the same deterministic
unlearned symbol, and processes at most32rows or64KiB of row payload per block
(at least one full-width row). Continuous activity uses compensated float32
block sums; short inputs use the original Torch stack/mean. There is no label,
neuron, context or learned-memory ceiling. Wide VSA dimensions reduce block size
rather than dropping values. Current attention cancellation is checked before
requesting the next row.

The fallback symbol is no longer generated eagerly when the learned row already
exists. No symbol seed, positional binding/permutation, learned row, resistance,
strength, repeat, graph or persistent activity metadata is rewritten.

## Contract and evidence

- Eight focused cases cover short bitwise parity, long numerical parity,
  weighted/repeated learned rows, unknown symbols, zero-superposition fallback,
  continuous assembly means, paged-state immutability, actual full label
  traversal, bounded stack sizes and cancellation before another row allocation.
- Eighteen affected adaptive-vector, weighted-memory and substrate-inspection
  cases also passed in0.951s; these are memory components, not model training.
- Long floating reductions may differ in final rounding; the tested tolerance
  is `rtol=2e-6, atol=2e-7`. This is not a promise of arbitrary-input bitwise
  identical generation. Short32-row/64-dimensional fixtures are bit-identical.
- `activity_reduction_scratch_bytes` estimates the reduction's bounded
  simultaneous tensor/list working set. Callers must additionally admit
  text/token/label storage, row decoding, repeated-contact metadata and other
  model work. It is not a total-process-RAM guarantee.
- The paired CPU measurements are in `streamed-vsa-comparison.json`: three
  pairs each for32/1,025/4,097labels. Small profiles were1.47x/1.17x as fast;
  the largest was0.970x (about3% slower), not a universal speed improvement.
  Its calculated retained-row-plus-stack payload fell from8,390,656bytes to an
  81,920byte conservative reduction estimate. These are accounting values, not
  measured RSS; row decoding and labels are outside that comparison.
- All three measured cues happened to be bit-identical and packed state stayed
  unchanged. General learned conversational quality, giant-context performance,
  accelerator throughput and full-corpus training were not tested here.

The first local test command used system Python, which lacks `jsonschema` and
failed during import; no test ran. The existing `.venv/bin/python` was used
without installing dependencies. The paged test initially called a nonexistent
cleanup method; its assertion had passed, and the test-only cleanup was removed
because this store scopes its own connections. Only that affected case was
rerun. Earlier successful tests were not needlessly repeated.
