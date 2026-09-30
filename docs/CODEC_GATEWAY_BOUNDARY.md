# First-use codec setup boundary

Ordinary health, chat, image work and text training do not initiate downloads.
Explicit media RPC preparation remains supported. Mixed ingestion and organic
inline video/audio can also request the runtime at the actual subprocess codec
boundary while the one neural RPC is already active.

The worker creates a nonce bound to the exact raw request, brain, job, turn and
optional inline action. Main authorizes it against a live RPC or an inline action
registered before its parent text RPC ended. Main prepares only the compiled,
pinned runtime catalog and replies through a private out-of-band control lane.
No private configuration is queued behind the RPC waiting for it. Renderer APIs
cannot forward a path setter or fabricate a codec receipt.

The reader deposits the receipt without loading, training, saving or executing a
brain. The owning thread reverifies the binary and takes a codec lease before
execution. Changed selectors wait for actual old codec leases, not unrelated
image decoders. Pinned binaries are reverified under each lease before use;
existing externally selected or PATH FFmpeg stays a separate external runtime.

Setup progress exposes actual checking/download/verification state and bytes in
existing job/action activity lanes. It never becomes neural-generation progress
or reopens completed text generation.

An inline artifact owns its setup independently of its saved reply. Its local
cancel withdraws only that challenge; a host job that later claims the inline
decode cancels the same original action scope. Direct artifact cancellation has
an exact request/brain/job/action control too, without a process-wide signal.
Cancellation is acknowledged only after the owning RPC unwinds, including any
ingestion cleanup. If acknowledgement stalls, the UI remains pending and main
reports it; these scoped controls never kill/restart the warm worker or claim
cleanup happened from a flags-only receipt. Shared setup remains available to
other live owners when one artifact stops.

Validation here is protocol/state/file/stub evidence. No neural constructor,
training, app or codec-in-neural run was used. Runtime acquisition under actual
hardware I/O and neural workload, all-platform app behavior, and speech/video
quality remain separate acceptance gates. Runtime catalog builds/publication
are owned separately from this gateway implementation.
