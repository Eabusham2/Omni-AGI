# Local GGUF teachers

R158 adds an external local teacher-data producer, not a foreign Omni runtime,
checkpoint conversion, Q/A lookup, or an alternative learned identity. The
native brain remains its existing ground-up state. Existing API-teacher code
and secure bring-your-own-key settings are separate and unchanged by this lane.
No provider request, model download, training run or real GGUF inference was
performed during this implementation.

## User and tool path

`omni.localTeacher` supplies selection/progress/cancel, current measurements and
recommendations, jobs, paged lesson receipts, Pause, Resume and Stop. It is
mounted in the existing Learn workspace; see [UI evidence](local-teacher-ui.md).
`data.distill-local` is an Ask-default tool. Starting/resuming external runtime
work remains a risky one-action review, including when ordinary tools use Auto.
The tool cannot choose arbitrary runtime/model paths, flags, setup commands or
downloads. Main-owned file selection is required before execution.

Selection hashes every byte with fixed 64 KiB scratch and cancellable byte
progress. A header scan traverses all supported metadata without loading
tensors or materializing tokenizer arrays. Immediate header reading checks the
same exact inode/change identity instead of rehashing the whole GGUF a second
time merely to inspect its header. Declared geometry is not payload validation.
Supported parsing is little-endian GGUF v2/v3 with scalar and flat arrays;
big-endian, nested arrays and unknown encodings fail explicitly. License fields
are user/GGUF declarations, always `verified:false`, never invented legal
training-use permission. [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

The exact selected executable is hash-bound and probed only with `--help`.
Required raw/offline flags must appear in its actual help; unsupported installed
versions stay unavailable instead of silently falling back to a chat template.
Dependencies remain external referenced files. Execution authorization is
process-private, current-owner/current-policy bound and not exported.

`plan` measures the current app family, already-loaded neural/inspection owners,
selected RAM ceiling, physical free RAM and CPU count. Declared GGUF KV/context
geometry produces an explicit estimate, not an OS resident-memory guarantee.
Unknown geometry or insufficient overlap blocks Start rather than inventing
values. Actual family RSS, metadata headroom and storage writes are guarded
during execution. The initial implementation uses explicit CPU/no-offload
flags; GPU acceleration has not been implemented in this local-teacher runner.
Timeout 0 means no wall-clock deadline, not an ETA or exemption from Stop.

## Structured route and exact output

`llama-server` is preferred. It uses the raw `/completion` endpoint, not an
OpenAI/chat-completion endpoint, chat template, system message or behavioral
persona. A POSIX session binds an owned private Unix socket and private
authentication file. Windows requires a private loopback bind, the selected
live child's readiness and an OS-verified listener plus exact established
peer before sending any authentication/prompt bytes. Unknown ownership refuses
the request. The implementation uses public `node:http` socketPath support,
not internal undici imports or an ambient proxy dispatcher.
[Official server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

One warm server session handles the selected prompts sequentially. Generated
UTF8 content is streamed separately from the full SSE transport and process
logs. Fragmented Unicode, terminal frames, explicit output-byte envelopes and
prompt/context truncation reports are checked. Unsupported aggregate terminal
content is refused, not guessed or duplicated. Transport bounds include the
actual full prompt size. Operational server logs stream to disk under current
write admission rather than a hidden per-session lesson/log-count limit.
HTTP completion has `exitCode:null` and `completionProof:http-terminal`;
intentional later server shutdown is not fabricated as a successful process
exit for each lesson. Partial/failed transport is retained but never marked a
complete learning source. Auth/socket files live outside exported brain
custody and are removed only after the exact owned child closes.

Explicit completion/CLI modes remain optional teacher-console data. Their
complete stdout, including any framing, is attributed as console output;
nothing is heuristically stripped or mislabelled pure generated tokens.
Token-output logging is not disabled because completion itself emits tokens
through that path. These modes do not substitute for the preferred structured
route. [Completion documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md),
[CLI documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md).

## Custody, learning and interruptions

The existing native supervisor admits one host-compute slot without starting,
loading, RPCing or killing a neural owner. Earlier native/inspection startup
finishes first; later native startup waits. Cancellation/shutdown releases the
exact slot. The brain writer is acquired before this slot to avoid a writer
cycle. Both are released **before** ordinary native source ingestion.

Runtime and GGUF verified FDs remain open for the teacher session. Post-session
pathname/FD identity and current authority checks precede a durable custody
seal. Complete HTTP/process receipts are candidates until that seal commits.
Pause can retain a verified completed prefix; Resume generates only unfinished
positions. A revoked/uncertain dependency closure does not become eligible
merely because output files exist. Atomic immutable receipt publication never
overwrites a conflicting prior receipt. POSIX directory fsync and SQLite FULL
are used; no portable Windows directory-fsync/power-loss guarantee is claimed.

The full exact prompt/response becomes a typed JSONL `messages` dialogue with
declared provenance. Existing DatasetManifestStore and LearningCoordinator
own exhaustive native source traversal, parameter-learning mechanics,
transaction deduplication and cursors. This is not a new fact extractor, answer
key store or memory lookup path. All selected lessons and requested epoch
visits must finish without rejected records/files before the job says complete.
Generated and native committed-source counts are separate; neither proves
truth, comprehension or retention.

Request identities, result files, custody seals, source-manifest IDs, native
task IDs and source progress persist. A lost start ACK returns the same job.
A fsynced post-shutdown seal can reconcile a lost SQL ACK without regenerating
or learning. Native-task request identity reconciles a lost learning-start ACK.
Stop preserves committed data; unfinished/cancelled output does not pretend to
be learned. When native learning needs an unloaded brain it remains visibly
waiting for explicit Load. Restarts pause jobs without executing them.

## Portable state

The complete local-teachers SQLite ledger, raw job history, lesson prompts,
responses, transport/logs, source data and custody seals are part of the
existing learning-workspace snapshot/fork/export/import/recovery/full-merge
closure. Active generation/selection must first be paused; snapshot does not
silently cancel or claim a coherent copy of an active writer. Source archives
and immutable learned/output identities remain exact. Only current operational
owner projections change. Pending copied/merged jobs are dormant, never copied
as live grants or blindly replayed. Later history retained during a rollback
is dormant, even when its original task had completed; it is not proof that
the restored older neural checkpoint contains those later changes.

External runtime/weight files are not copied into the brain. Importing their
references does not authorize execution. A current host must explicitly select
and approve its dependencies. The existing UI keeps dormant jobs historical;
it offers a new reviewed selection/job rather than a false Resume grant.

## Focused evidence and open qualification

38 distinct backend cases passed across bounded runs; closed cases were not
repeated merely to inflate totals:

- 8 real-supervisor scheduling cases with constructor-free synthetic transport.
- 11 protocol/metadata/actual tiny owned-child cases; no real teacher model.
- 8 actual small SQLite/filesystem source, lost-ACK and portability cases.
- 10 orchestrator cases using explicitly synthetic teacher/native receipts around
  actual SQLite/files/manifests, including a tiny owned help-only executable,
  selected-file hash cancellation, immutable publish and multi-epoch accounting.
- 1 separate pre-spawn regression: current policy was revoked during async
  setup; the boundary recheck refused spawn and both prepared owned FDs closed
  (9ms test time). The earlier 37 cases were not rerun for this followup.

These tests do **not** establish real supported-runtime generation, native
parameter-change/retention quality, Windows/macOS/Linux packaged acceptance,
external-model license rights, GPU performance or the full API/media teacher
scope of R063/R239. The four presentation cases and synthetic-host computer-use
checks are recorded separately in [local-teacher-ui.md](local-teacher-ui.md).
Final shared compiler evidence will be recorded at the coordinated source
freeze; no CI, package, release, model run or commit occurred in this lane.
