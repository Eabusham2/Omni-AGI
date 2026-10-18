# Direct-packed floating-activity SNN recurrence

October 8, 2026. This closes the specific `STDPSynapses.recurrent` forward
weight-expansion gap, not the whole model/device qualification gate.

## Implemented contract

`STDPSynapses.recurrent` now passes compact uint8 sign bytes and uint8
connection-state bytes to the native kernel. Each multiply decodes its sign
and low-nibble integer strength in a register. The resistance high nibble
does not multiply activity. The forward path no longer calls
`iter_effective_tiles`, `unpack_tile`, `weighted_ternary_levels`, or `torch.mv`,
and creates no signed or floating learned-weight tile. The inspection and
STDP-learning readers remain separate; this change is not a claim that all
other bounded learning operations have been eliminated.

Sparse recurrence visits only allocated blocks, including every independently
stored repeated-contact layer, in the existing sorted-key and row/column tile
order. Absent blocks contribute zero without scanning potential connectivity.
Dense native compatibility owners use packed signs with unit strength. RAM
admission precedes tile copies; `try/finally` and generator closure release
mapped tiles on success, cancellation, admission refusal, or kernel failure.
Read-only recurrence does not change signs, gains, timing controls, topology,
or learned-state checksums.

Activity is still floating point, not a floating learned master and not
quantized to int8. CPU accepts float16, bfloat16, float32, and float64;
half/bfloat16 activity promotes exactly to float32 before arithmetic and
returns in its original dtype. Float64 stays float64. The CPU kernel uses a
double accumulator; the Metal kernel uses float32. Both accumulate columns
in ascending order. Normal output-dtype rounding/overflow still applies.
General results are not promised bit-identical to the former BLAS reduction,
between CPU and Metal, or across different tile budgets. Exactly representable
small dyadic cases were checked exactly; general Metal values were checked
with explicit floating tolerances. Nonfinite or nonfloating activity is
rejected, and reserved sign codes are rejected, including unused padding.

The existing CPU-backed router placement is preserved: even when its small
live input/output vectors are on MPS, recurrence reads admitted CPU tiles and
returns the output to MPS. The standalone Metal floating kernel was executed
and checked on actual hardware, but this change does not select it as the
production router backend. No float64 Metal, CUDA, NPU, multi-GPU, or
whole-brain speed/quality claim is made. The int8-forward and input-gradient
prototypes and ABI version remain unchanged; two floating-forward exports
were added. A source-matching CPU library rebuild is required.

## Validation actually run

- Rebuilt the small CPU library with `scripts/build-native-kernels.py` on
  macOS arm64. The exact source/binary hashes are in the comparison JSON.
- The existing 71-case native-kernel, weighted-router, sparse-router, and
  paging suites passed in 7.988 seconds after the implementation/rebuild.
  They were not repeated during the later targeted continuation.
- The new 16-case `engine.tests.test_packed_float_recurrence` suite passed in
  1.688 seconds. It includes actual CPU and Metal arithmetic, odd widths,
  strength/resistance separation, double values `1 + 2^-40` and `2^-200`,
  half/bfloat16 dtype retention, reserved padding, empty input dimensions,
  fail-on-Python-decoding guards, repeated and border blocks, reversed
  insertion order, absent-connectivity zero, dense compatibility,
  noncontiguous input, hot RAM/mapped storage, failure/cancellation/admission
  release, unchanged controls/checksums, and MPS-vector/CPU-router placement.
- The focused LIF leak/fire/reset and causal/anti-causal STDP tests passed.
  The metaplasticity test initially used the obsolete dense-tensor `.abs()`
  return API; only its two receipt reads changed to
  `STDPUpdateSummary.delta_absolute_sum`, then that one test passed in
  0.037 seconds. No learning-rule or production receipt change was made.

These were tiny constructor-free or one-cell dynamics fixtures, with no user
brain loading, user corpus, training curriculum, commit, CI, or release run.

## Bounded CPU comparison

Run the opt-in comparison with:

```sh
PYTHONPATH=engine .venv/bin/python -m engine.tests.benchmark_packed_float_recurrence
```

The benchmark alone retains the former decoded-tile/`torch.mv` path as its
baseline; it is not a production fallback. Both paths use the same fixed
67-by-69 activity shape, tile budget, pager, and allocated contacts. Sparse
cases have five allocated blocks including repeated layers and 12,623 contact
slots; dense compatibility has 4,623 slots. Both outputs were exactly equal
for the fixed dyadic activity, and learned-state checksums were unchanged.
Measurements use one CPU thread, two paired warmups, seven alternating-order
rounds, and five calls per sample. Raw samples, environment, source hashes,
and backing/payload byte counts are saved in
[packed-recurrence-comparison.json](packed-recurrence-comparison.json).

| Case | Tiles/call | Decoded median ms | Packed median ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Sparse/repeated, RAM, 64 KiB tile budget | 42 | 3.650 | 2.021 | 1.81x |
| Sparse/repeated, mapped, 64 KiB tile budget | 42 | 7.436 | 5.511 | 1.35x |
| Sparse/repeated, RAM, 1 MiB tile budget | 5 | 0.897 | 0.754 | 1.19x |
| Dense compatibility, RAM, 64 KiB tile budget | 28 | 1.794 | 1.014 | 1.77x |

For the 64 KiB sparse cases, the largest raw sign/gain payload was 400 bytes,
versus 1,280 bytes for the former floating-weight tile alone; at 1 MiB these
were 5,120 versus 16,384 bytes. Those are actual tensor payload sizes, not
measured process RSS or whole-call peak memory. Persistent sparse state was
142,059 bytes in both paths. The mapped case used real file-backed state and
the pager release path, but OS page cache residency was not controlled; it is
not a physical cold-disk latency benchmark. These modest fixed-shape results
do not establish large-brain, device-placement, acquisition, or retention
performance, and no default sizing/tile policy was changed from them.
