# DMA: OS-permitted process memory

## Contract and source

This implements the settled R043/A05 meaning in [PLAN.md](PLAN.md): an optional,
advanced host-process inspection/modification tool, not direct physical RAM or
an escape from OS protection. The original request remains verbatim in
[REQUEST_VERBATIM.md](REQUEST_VERBATIM.md). Existing neural state, learning,
packed parameters, and working-memory mechanisms are not replaced by this tool.

- `src/shared/processMemory.ts`: versioned settings, identity, request, and receipt.
- `src/main/processMemoryService.ts`, `processMemoryStore.ts`: current authority,
  process-private leases, installation-local CAS settings and write journal.
- `src/main/processMemoryBridge.ts`: bounded two-phase helper protocol.
- `engine/process_memory_helper.py`: independent stdlib-only OS adapter.
- `ToolExecutor`, main composition, IPC/preload, catalog and capability-pack hooks
  are additive. The worker has an early helper-only entry point before Torch.

Both the DMA user switch **and** the current `system.memory` tool policy must
allow an operation. Both default Off. A tool or plugin cannot enable the DMA
switch itself. Auto reviews attach, region inspection, read, and write; normal
nested permission checks still apply. Current instance ownership must be loaded
for an operation; preference inspection does not Load a model. Root still owns
the actual small settings-panel mount and its UI qualification.

## Exact target and effects

`attach` accepts only an explicitly selected PID and `read` or `read-write`
access. It returns a process-creation/executable identity and a current-host
lease. This service does not enumerate processes. Kernel/Omni authority,
neural/inspection workers, and its own helper processes are protected targets.
An unsupported sandbox/VM environment is rejected, never redirected to the host.

`regions` pages one selected process, at most 200 entries per call. `read` and
`write` use exact hexadecimal addresses and at most 1 MiB per call, with resource
admission and explicit page continuation rather than a whole-process dump.
Writes require a read-write lease and the exact observed preimage SHA-256.
That check is **not atomic** against a concurrently running target. Region
protections are checked before an operation; no API changes those protections.
In particular, Linux's debugger-authorized `/proc/PID/mem` interface and those
userspace checks do not provide an atomic page-protection guarantee.

Each operation opens a pinned OS process object/port/FD and prepares an exact
request hash and one-use nonce. Main rechecks current setting revision, tool
permission revision, lifecycle, lease, target, Stop signal, and actual managed
resource headroom before sending the commit nonce. It checks authority again
after awaited work. A revoked read delivers no raw bytes. `detach`, Unload,
recovery, permission changes, and shutdown revoke the session leases; none
terminates the selected target or undoes a write.

A committed write is allowed to return its bounded real result even if Stop
arrives. A timeout/transport loss after dispatch is `write-uncertain`, never
"cancelled, nothing happened." Partial counts and Mach's unknown failure count
remain honest. Actual result/journal-save failures retain the typed receipt.
The durable local write journal prevents automatic repetition of an exact
request, including a dispatched request whose final ACK was lost. It stores
hashes/results, not a memory dump. Normal private tool-outcome/history retention
still preserves whatever the authorized tool actually returned.

Disabling revokes live access immediately, even if disk admission subsequently
prevents saving the preference. That failure explicitly says the revocation is
current-session-only and must be rechecked after restart; it is not presented
as a durable saved Off setting.

## Portability and OS limits

The settings/effect journal live under installation `host-process-access`, not
inside a brain archive. Copies, full merges, imports and recovery do not carry
live process leases, OS handles or one-use approvals. Historical tool receipts
are evidence, not a restored grant. A fresh target attachment must satisfy
current local settings/permissions and the actual OS identity/access checks.

Adapters use public OS interfaces only:

- macOS: permitted `task_for_pid` task port, pinned port identity, Mach VM region
  queries/read/write. No signing, entitlement, TCC or SIP change is attempted.
  [Apple Mach VM APIs](https://developer.apple.com/documentation/kernel/mach/mach_vm),
  [public region ABI](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_region.h),
  [public process identity ABI](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h).
- Windows: minimal query/VM rights, pinned `OpenProcess` handle, exact process
  creation FILETIME, `VirtualQueryEx`, `ReadProcessMemory`, `WriteProcessMemory`.
  No debug privilege or elevation is enabled.
  [Read](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory),
  [write](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-writeprocessmemory),
  [region query](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualqueryex),
  [creation time](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes).
- Linux: pinned selected `/proc/PID` directory and `mem` FD, boot/start-tick and
  executable identity, bounded region/byte reads. PID reuse cannot redirect the
  already-open memory FD to a new process. No ptrace attach, sysctl, capability
  or process signal is used.
  [proc memory access](https://man7.org/linux/man-pages/man5/proc_pid_mem.5.html),
  [kernel Yama access policy](https://docs.kernel.org/admin-guide/LSM/Yama.html).

These are independently written wrappers around documented/public ABI
declarations, not copied OS implementation. The small helper is source-wired to
the packaged executable; an actual rebuilt/package qualification was not run.
Packaged mode refuses a missing helper binary rather than installing or falling
back to an arbitrary external interpreter.

## Exact evidence and remaining gates

Focused checks, run in distinct affected slices (closed cases were not rerun):

- 12 constructor-free Python cases: initial 11 passed in 0.001 s, one added
  identity-after-read disclosure case passed separately in <0.001 s.
- 15 actual small SQLite host settings/lease/write-journal cases: initial 11
  passed in 107 ms, two no-file/low-disk cases in 29 ms, two lifecycle/failed
  terminal-journal cases in 24 ms. Native OS callbacks here are explicit fixtures.
- Six real task-owned subprocess **protocol** cases passed in 371 ms. Their
  OS receipts are synthetic; they do not prove foreign-process read/write.
- Four actual ToolExecutor/catalog/curriculum checks passed in 75 ms, with
  fixture adapters, including Auto/Off and retained uncertain write receipts.
- Shared Node/Web type checks passed around these hooks. Final compiler and
  whitespace boundaries are reported in the agent handoff, not as platform proof.

One actual native attempt created a harmless child with a selected **16-byte**
buffer, inspected no other process, and attempted the ordinary adapter once.
macOS refused `task_for_pid` with OS code 5 after **0.030476 s**. It reported
`actual-owned-access-not-proven`; no protection/signing/privilege was changed and
no retry was made. Therefore successful actual OS read/write is **not proved**.

Root mounted `ProcessMemorySettings` inside the existing Tools & permissions
authority surface; no workspace or app was redesigned. It defaults disabled,
uses the current installation-only revision, does not optimistically grant
access, refreshes after a rejected write without retrying a grant, and fences
late responses to the mounted instance. The separate `system.memory` policy
still applies. Two new renderer cases passed in 10 ms; both TypeScript targets
passed after this mount.

Bounded computer use with an explicitly synthetic preference API confirmed one
revision-bound update, then an unchecked independent owner, readable expanded
OS details and wrapping at 360 px in light/dark states. Captures are in
`test-results/process-memory-settings-ui-manual/`. No real setting or process
grant was changed by those UI checks.

Open: successful OS-permitted target read/write under an actually granted
configuration; Windows/Linux execution;
packaged helper/code-signing/runtime qualification; supported sandbox/VM memory
adapter if later requested. No model/corpus/provider call, CI, package build,
commit or release was performed for this slice.
