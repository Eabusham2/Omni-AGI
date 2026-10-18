# Weighted native modality table evidence

Image, audio, video, and latent-position table reads now use their authoritative
packed ternary signs, integer strengths, resistance state, and repeated
contacts. The correction removes full decoded-codebook reads from production
modality execution. It does not add another model or alter the existing
checkpoint orientation.

## Confirmed gap and correction

`PackedTernaryTable` in `engine/omni_core/modalities.py` previously called
`projection.effective_weight()` and transposed the full sign matrix. That
inspection helper returned signs only, omitting learned strength and repeated
contacts. VQ nearest-code selection and audio token scoring then used that
decoded table. Those were actual forward-read omissions, not evidence that
all prior modality learning was inactive.

The existing full-gradient learning method already ran inside
`packed_row_owner`, through which `_apply_packed_gradient_rows` resolves the
owner's weighted state. The new selected-code path preserves that owner and
the collective derivative contract, supplies connection state explicitly,
and tests actual gain, resistance, and repeated-contact mutations.

## Runtime computation

- Table checkpoint storage remains `[dimensions, codes]` inside the original
  `BitLinear` projection. No learned bytes are silently transposed or reset.
- `table(indices)` gathers only requested code columns. Each contribution is
  the base ternary sign times its integer strength, plus actual repeated
  contacts at that coordinate, followed by the existing projection scale.
- VQ streams nearest-code scoring across sample blocks and reads only the
  selected codes for commitment learning and reconstruction. It does not build
  a full token-by-code distance matrix or a full decoded codebook.
- Audio token logits use the direct native packed transpose operation. The
  code norm is accumulated over bounded integer contribution tiles. This
  computes the squared-distance logits without quantizing the predicted
  activity to int8 and without decoding a full floating weight table.
- Audio token-loss backward updates predicted activity through bounded code
  tiles. It does not change target codebooks to make a prediction appear
  correct. VQ commitment learning updates selected codes separately.
- Position tables read only active positions. Audio generation samples real
  code IDs, then looks up those selected IDs through the same native table.

All production audio scoring/loss callers pass `PackedTernaryTable`, not a
decoded tensor fallback. `modalities.py` contains no `effective_weight()`
call. Float tensors here are temporary activities, loss gradients, logits,
and fixed scales—not a learned master-weight copy or optimizer moments.

## Memory and learning boundaries

Norm computation and activity-backward tiles span at most 64 code columns by
64 dimensions. Selected lookup processes at most 256 requested IDs by 64
dimensions per decode tile; its returned activation necessarily contains the
requested values. Direct packed scoring creates bounded sample blocks of
logits across the code count. Audio's returned logits still occupy
`sample_count × code_count` floats because softmax consumes those logits.

Selected-code learning now sums repeated occurrences into derivative tiles of
at most 64 dimensions **by 64 codes**, with at most 256 occurrences examined
per gather. The tile shape shrinks from the actual working-attention/native
transfer budget before admission. Ascending scans identify touched code tiles
without allocating an index covering every possible code. The input is fully
validated before the first mutation. Unselected code signs, gains, and
per-contact resistance do not mutate; existing repeated contacts retain their
owner identities. Shared output-row resistance advances once after all code
tiles, using the original resistance throughout that logical update.

Distributed selected derivatives use canonical one-dimension/64-code SQLite
partitions. Different local sample or computation tile boundaries therefore
sum raw derivatives before one shared native update; no sample, code-count,
tile-count, or rank divisor is added. Repeated contacts stream in bounded
batches. Exact base-byte, per-contact, row-resistance, topology, and counter
rollback covers a later tile failure. Changing stochastic draw grouping does
not promise bit-identical random trajectories to the former full-width path;
the discrete transition probabilities are unchanged.

Each eligible contact takes a discrete sign/strength step. Saturation can
allocate an independent repeated contact rather than overflowing a strength
or creating a floating shadow. The existing transaction/collective owner and
paging mechanism remain authoritative. Admission is resource accounting, not
a claim that the OS imposes a hard whole-process resident-memory limit.

## Focused results

Eighteen distinct focused cases passed across the affected runs:

| Run | Result and scope |
| --- | --- |
| Eight new weighted table cases plus six affected existing cases | 14 passed in 0.411 s: exact weighted/repeated reads and activity gradients, resistance-dependent learning, saturation growth, untouched codes, collective duplicate sums/orientation, bounded tiles, selected VQ reads, safe reload, unit-strength migration, previous packed modality tests, and audio token protocol. |
| Cold paging and tiny audio integration | 2 passed in 0.122 s: mapped native table values/updates survive cooling; actual audio forward/backward and progressive generation use the packed tables. |
| Actual Metal table parity | Passed on available Metal hardware, not skipped: selected reads, native packed logits, and activity backward match the CPU reference within stated float tolerances. |
| Tiny complete modality hub | Passed in 0.336 s after correcting an invalid test configuration that mismatched `idea_dim` and `d_model`: actual image/audio/video forward and backward, generation, safetensor reload, and identical seeded generated tensors. |

The complete-hub fixture uses width 4, 2 modality channels, 8×8 images,
16 audio samples, and 2 video frames. It installs non-unit strengths and
repeated contacts into every code/position table, forbids the removed full
reader throughout actual production calls, and confirms no `Parameter`
objects are retained in the hub. It is a small mechanism fixture, not a
trained media-quality demonstration.

Evidence lives in `engine/tests/test_weighted_modality_tables.py`,
`test_packed_modality_tables.py`, `test_audio_codec_token_protocol.py`, and the
specific table-derivative case in `test_collective_derivative_normalization.py`.
No user brain, full corpus, CI job, package, or release was used for this repair.

### Two-axis learning continuation, October 10

All ten cases in `engine/tests/test_selected_modality_tiling.py` passed in the
24-case focused rollback/journal/tiling run (6.134 seconds). The Metal case ran
on actual available hardware, and the two-rank case used a real Gloo process
group. These cover duplicate raw sums across code tiles, exact deterministic
native state against the full-width reference, existing repeated owner IDs,
once-per-row resistance, untouched packed neighbors/padding, a mapped
4,097-code/65-dimension table under a 32 KiB compute budget, early admission
and invalid-input refusal, mixed full/tiled collective cancellation, late
collective failure rollback, safe reload/unit migration, and rank agreement.

`engine/tests/benchmark_selected_modality_tiling.py` compares five paired CPU
updates of an 8,193-code/65-dimension table with five selected occurrences,
including a repeated contact. Construction, source copies, and checksums are
excluded from timing. Every result matched the actual native state of the
full-width reference; the source remained unchanged. Results are preserved in
`selected-modality-comparison.json`.

| Update path | Median CPU time | Derivative tile bytes | Admitted learning peak |
| --- | ---: | ---: | ---: |
| Former full-width reference | 27.804 ms | 2,097,408 | Not bounded by the new budget |
| Two-axis, normal 1 MiB budget | 8.037 ms | 16,384 | 400,032 bytes |
| Two-axis, 32 KiB pressure budget | 31.765 ms | 1,024 | 27,072 bytes |

The normal case was 3.46x faster in this small sparse-selection fixture. The
pressure case was 14% slower than the reference while reducing its derivative
tile by roughly 2,048x. This is a useful memory/speed trade-off, not an
unconditional speedup or a whole-process RSS/whole-model throughput claim.
Returned activities and audio logits still have their real output sizes.

## Remaining scope

This closes the identified modality-table reader gap and its affected native
callers. It does not close all native readers elsewhere in the product,
visual media acceptance, long-run retention, media quality, or large-codebook
throughput/memory qualification. CUDA uses the existing admitted packed CPU
fallback when no direct kernel exists; no CUDA hardware parity claim is made
by these CPU/Metal fixtures.
