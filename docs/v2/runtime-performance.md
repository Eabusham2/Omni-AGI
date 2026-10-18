# Native runtime sequencing and resource sampling

## Observed bottleneck

On the development Mac, profiling two tiny native chat fixtures took 177.8
seconds. `ResourcePolicy.status` was called 6,091 times and consumed 161.6
seconds cumulatively. System-memory sampling launched 18,273 commands and
consumed 106.9 seconds; fresh managed-process sampling launched 2,377 process
inventories and consumed 47.5 seconds. These cumulative categories overlap;
they must not be summed as independent elapsed times.

The fixtures used a tiny random native decoder, not the user's training corpus
or a saved trained instance. They test transport and state transitions, not
answer quality.

## Changes and exact boundaries

- Normal talk now streams its first decode. It does not privately finish a
  reply, rescore it, and regenerate it before publishing text. Selected Ponder
  search remains separate.
- Token events carry actual native output-token counts and monotonic measured
  speed, including nonprinting boundary tokens. The renderer does not turn
  output characters into a purported native-token count. It batches metric
  updates with text frames, with a clock update for nonprinting events.
- The direct Darwin sampler calls `host_statistics64` and `sysctlbyname`.
  Available memory is read each time; only API handles, host port and page size
  are cached. The existing available-memory calculation and RAM ceiling remain
  unchanged. Unsupported interfaces retain the old measured fallback.
- An empty repeated-contact range returns without allocating or reserving page
  cursors. Nonempty ranges still use the admitted streaming cursor path.
- Auto selects a backend with available native packed support. Generic Torch
  CUDA/DirectML discovery alone no longer opts every projection into CPU round
  trips. Explicit backend choices retain admitted raw-packed CPU-tile fallback.

Darwin ABI references: [XNU VM statistics declarations](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h),
[host statistics revisions](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/host_info.h),
and [read-only memory-status interface](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus.c).
The FFI adapter is an original implementation against those interfaces.

## Verification

Four sampler cases passed, including direct real-Mac sampling while subprocess
launches were patched to fail. Two real tiny-model reply fixtures passed:
first-pass streaming/counts and steering after a published prefix without a
second generation. Their combined run with the sampler cases took 57.7 seconds.
This is not a controlled speedup ratio: the earlier measurement included
profiling overhead and the steering fixture was corrected to leave a generation
step after its first complete UTF-8 character.

Eleven packed-runtime/accounting checks passed, including retained floating
weight-mirror detection; two host token-measurement validation cases and the
renderer typecheck passed. A separate empty-contact no-probe case passed.

Fresh managed-family enumeration now uses Darwin `libproc` rather than launching
`ps`. Each sample observes the owned descendant tree twice, verifies parent and
process-start identities, and takes the larger observed RSS per process. Changed
membership, an unreadable process, truncated output, or an observation beyond
one second invalidates the sample; admission pauses. Only the API handle is
cached. Allocation checks still force a fresh sample. Numeric ABI offsets were
checked against the installed Apple SDK `sys/proc_info.h`; no process names,
command lines, or credentials are decoded.

Fourteen native sampler checks and twelve existing managed-RAM-ceiling checks
passed. The real-Mac fixture samples a child and grandchild with subprocess
launching patched to fail, confirms their membership, and excludes the caller.
A current-process-only comparison measured a 0.026 ms median for 30 fresh native
samples and a 16.698 ms median for five old `ps` samples. This is a sampler
comparison, not a claimed full-model speedup or atomic OS snapshot.

Full-size load, generation TPS, peak RAM, long-run latency and packaged
cross-platform performance remain unqualified.

## Verification-cache I/O window, October 7

An actual connected-load stack showed repeated proof-cache transaction closes
inside `_verified_forward_from_staging`. Each last SQLite close checkpoints
the WAL. A bounded `proof_io_window` now keeps one idle handle alive for that
verification operation, while individual proof writes still commit. The handle
holds no read snapshot and closes before staging publication/retargeting, on
failure as well as success. Missing headroom falls back to the original path.
Source hashes, file identities, HMAC authentication and cold-process rehashing
are unchanged. It does not certify unverified source content.

Four new cases pass for committed rows, changed identities, tampered proofs,
exception cleanup/publication and optional headroom refusal. Thirteen affected
cache-rebuild/tamper/new-session cases pass in 24.092 seconds. The first cleanup
fixture caught a read-only last handle leaving WAL data; the keeper now opens
the existing writable cache (`mode=rw`, which cannot create a missing DB),
issues read queries only, and performs SQLite's normal final-close checkpoint.

`engine/tests/benchmark_proof_io_window.py` measured three paired runs of 250
authenticated writes/lookups with final close included. Baseline durations
were 555.540, 365.701 and 403.287 ms; the window durations were 178.269,
170.197 and 174.650 ms. Medians were 403.287 and 174.650 ms. The measured
source SHA-256 was `675605fafcf9dc52a7afeca8f3660cf0fc64790eadbcafebf3aaeebafd45aee1`.
These are bounded database-operation timings under concurrent local activity,
not a full-load/TPS benchmark. No model or user dataset was built for them.
