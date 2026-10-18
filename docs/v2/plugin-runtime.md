# Optional lifecycle feature modules — source contract

This adds optional executable lifecycle modules to the existing Electron
application. It does not replace Studio with a website or substitute static
panels for programmable features. Old packages without `runtimeModules` keep
their previous canonical bytes and hashes. Import/install alone runs no module.

## Activation and execution boundary

`runtimeModules` declares immutable JavaScript, subscription topics, allowed
Studio contribution kinds/slots, and whether typed dynamic tools may be
registered. The human activates an exact package content hash, catalog revision,
instance-access revision, module, and request identity. Current enabled access
and resource admission are checked again while starting and before each effect
or callback. Unload, Stop, revision changes, and recovery retire the session.

The process factory creates a separate hidden sandboxed Electron renderer with
an ephemeral partition, no Node integration, context isolation, restricted
devices/navigation/resources, and a separately bundled minimal preload. Only
two fixed IPC channels exist for that renderer; actual sender/frame identity,
session and fresh private nonce bind every request. No main process objects,
Studio preload, credentials, or reusable approval tokens are exported to it.

This follows Electron's [sandbox](https://www.electronjs.org/docs/latest/tutorial/sandbox)
and [context-isolation](https://www.electronjs.org/docs/latest/tutorial/context-isolation)
boundaries. Node explicitly does not treat [node:vm](https://nodejs.org/api/vm.html)
as an untrusted-code security mechanism; it is not this implementation's
production boundary. The owned `new Function` test is only a test harness for
the known fixture body, not evidence of a security sandbox.

Page load is **not** module readiness. The generated bootstrap sends a checked
`ready` RPC with `{ok:true}` only after its top-level declarations finish. A
caught initialization error sends `{ok:false,error}`. Missing acknowledgment
(including parse failures or a stuck body) reaches the bounded startup deadline;
the module is not presented as active. The startup deadline defaults to 15
seconds and is separate from module lifetime. Whole-family resource checks
already run before the factory returns a child PID. Startup abort retires the
exact owned process, including late factory completion.

`onActivate` runs after the positive acknowledgment and current authority check.
Host effects before active startup are refused, not silently run. Callback
exceptions become durable diagnostic events. Stop closes the public SDK and
refuses new effects immediately, starts cancellation of existing host requests,
then gives local `onDeactivate` callbacks a bounded 100 ms grace. A correlated
private bootstrap acknowledgment reports callback outcomes without granting
capabilities; it is self-report, not hostile-code attestation. Missing ACK,
observer failure or refused diagnostic persistence cannot prevent process
termination and resource release. Unresponsive code need not cooperate.

## Generic SDK, typed tools, and effects

`globalThis.omniRuntime` provides lifecycle/event subscriptions, arbitrary local
JavaScript computation, declared host `call`, CAS state get/read/put, result byte
pages, dynamic typed tool registration, and dynamic UI put/remove. Runtime UI
payloads are inert presentation; their event callbacks return to the exact
current module. Studio-wide CSS supports ordinary layout/color rules and safe
responsive `@media`, `@supports`, `@container`, `@layer`, `@keyframes`, and
`@scope` rules. Escapes/comments that obscure resource syntax and resource-bearing
rules/functions are refused. This is a documented supported subset, not a claim
that a regex parses arbitrary CSS; Studio CSP remains an independent boundary.

Every host effect passes both the explicitly enabled package capability and
ordinary current ToolExecutor permissions, including nested Off/Ask/elevation
rules. A module cannot grant itself permission. Dynamic tool identities use
`extension.<full registration SHA-256>` (74 characters); readable local IDs and
titles remain in descriptors. The registration hash binds the exact instance,
package, module, and definition. Per-action schemas use the same locally resolved
JSON Schema dialects as MCP. Validation neither coerces arguments, adds defaults,
strips fields, fetches remote references, nor accepts asynchronous schema bypasses.
Actual typed operands are copied before asynchronous dispatch.

Public schema text has its own exact text hash and enters the loaded brain's
existing structured capability-learning path. A queued job or fake-host receipt
is not proof that an untrained brain understands the tool. Package code, CSS,
and raw results are not hidden behavioral prompts.

## Durable lifetime, bounded live transfers

`plugin-runtime/workspace.sqlite3` holds hashed canonical operational sessions,
actual RPC operands/replies, state revisions, complete event history, and schema
learning receipts. It is operational custody, not neural memory or a Q/A store.
The database uses a checked fixed schema, disabled extensions/trusted schema,
FULL-sync writes, and bounded SQL cache. It does not store activation nonces,
process handles, or approval tokens. Actual tool approval authority remains
device-private and is invalidated when its runtime is retired.

RPC preparation commits before an effect. Exact repeated IDs coalesce or return
the retained result; changed operands are refused. A dispatched request with a
lost acknowledgment is uncertain and is **not** rerun. Ask keeps the private
token in memory while exposing only the historical proposal/receipt. Large
results are retained whole and returned as checked byte-page descriptors. A
large retry need not parse the whole old reply merely to return its descriptor;
the original result must be read before interpreting its terminal outcome.

There is no lifetime count quota for this new runtime. Live requests use
backpressure; history and state have explicit bounded pages/cursors, not silent
omission. Session pages are newest-first so old history cannot precede the newest
Stop control. Full-value imported reads require transient shared-family RAM
admission before parsing. Large values and events stay complete and can be read
in verified byte pages. Integrity checking currently streams/hash-checks the
selected stored blob (and immutable source database) again; this bounds transfer
RAM but is not a claim of constant-time I/O or a large-history speed benchmark.

## Copy, import, recovery, and full merge

The existing complete learning-workspace envelope includes the runtime store
and SHA-named immutable source archives. SQLite snapshots use isolated online
copies, never a bare live WAL-file copy. Fork/import rekeys current owner metadata,
retires sessions and pending schema jobs to dormant, and preserves exact original
history and state. No copied process, nonce, grant, pending effect, or learning
job auto-runs. Recovery retains later real effects as inert historical sources
without overwriting the recovered variable state.

Full merge retains every complete parent workspace as a checked source archive.
Only equal or unambiguous small transfer-page values enter the default namespace.
Conflicting values have **no first-parent winner**: they stay namespaced/inert,
and activation of the affected module is blocked until the human selects exact
source values with current revision checks. Oversized values similarly remain
whole in their sources pending admitted explicit selection; a low-RAM destination
may postpone selection, never erase them. There is no product-data limit implied
by the 64 KiB merge transfer page. Archive lookup verifies source SHA, owner,
size and file identity, then checks again before publishing read data. This does
not claim atomic OS-level attestation against unrestricted hostile host code.

## Evidence and remaining gates

New source fixtures only; no model, neural load, corpus, provider, personal file,
real host command, or real device operation was used in this lane.

- Initial runtime fake-host run: 11 passed in 638 ms; one fixture format constant
  failed and was corrected. Its affected retry plus the distinct startup-family
  pressure case passed (2 checks, 16 ms).
- Original seven small custody checks completed in 493 ms; the original live
  test handle was recovered after interruption, not restarted. They cover the
  actual export envelope, dormant copy, full-state conflicts, exact archive retry,
  recovery history, oversized descriptor-first reads, cancellation/tamper refusal,
  and newest-first session paging.
- The distinct ready/error/typed-input/maximal-ID affected run passed 3 checks in
  170 ms. The final focused async-schema/concurrent-ACK/large-retry/source-race
  run passed 3 checks in 187 ms (21 unrelated cases skipped).
- Owned file syntax transpilation and scoped tracked-file whitespace check passed.
  Root subsequently passed coordinated main/renderer semantic compilation and
  one affected actual Electron owned-JavaScript retry, including local cleanup
  acknowledgment and process release, after retaining the original failed Stop
  attempt. See [actual boundary evidence](plugin-runtime-host-evidence.md).
  Three distinct new shutdown fixtures cover ACK/cancellation, timeout/refused
  diagnostic persistence, and throwing observers. No brain or provider ran.

At the lifecycle-runtime slice freeze, the older **interactive scripted-panel**
transport still had its separate `MAX_REQUESTS_PER_VIEW=256` lifetime guard.
That was subsequently addressed in a distinct source slice, documented in
[scripted-panel durable replay](plugin-panel-durable-replay.md); it was not
implicitly removed by this new runtime. Platform packaging,
arbitrary third-party module quality, all host capabilities, and trained native
tool comprehension remain separate evidence boundaries.
