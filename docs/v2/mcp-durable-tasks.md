# MCP server-owned durable tasks

This is the source-lane contract and focused evidence. The compact task panel
and chat receipt view are mounted in the existing Electron interface. Native
interactive acceptance and final coordinated compiler verification are separately
recorded by the parent lane; this file alone does not establish complete
MCP/provider/platform support.

## Dated protocol, not a blended draft

The adapter is explicitly `mcp-2025-11-25`. It follows that negotiated version's
task capability and tool execution declarations, deferred task state/results,
and related-task message association. A task-creation acknowledgement is not
the original tool's completed result. The upstream feature is designated
experimental in that dated specification.
[MCP 2025-11-25 Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks),
[official dated schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-11-25/schema.ts).

The newer Tasks extension draft changes creation, result and cancellation
semantics. It is **not** silently negotiated, accepted, or described as supported
by this adapter. No `tasks/update`/inline-result behavior is mixed into the
older `tasks/result`/`tasks/list` protocol.
[Current draft extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks).

## Existing tool permissions remain authoritative

`McpClientService` retains and validates the receiver's task capabilities and
each tool's `execution.taskSupport`. A required-task tool cannot fall back to
an ordinary synchronous call. Optional tools retain ordinary `call`; forbidden
or absent declarations do not receive task-start actions. Task-aware actions
are generated locally on the **original remote tool ID**, with separate typed
input schemas:

- `call-task` wraps the exact original tool arguments and optional requested TTL.
- `task-status`, `task-result`, and negotiated `task-cancel` take an opaque local
  `taskRef`, never an arbitrary remote ID or replacement server.
- Reserved `tasks/list` has its own normal Ask-level client-operation policy.
  It returns one account-scoped provider page as untrusted data. Listed foreign
  task IDs do not create local task records or confer management authority.

An unsolicited dated or draft task acknowledgement from an ordinary synchronous
call is not presented as a completed result and does not create task authority.
It is not retried or automatically cancelled; external effects remain uncertain.

Schemas enter the existing declared/neural capability path, including each
action-specific schema. They are not hidden persona instructions or claims of
executed tools. Original JSON-Schema reference scopes are preserved when tool
arguments are wrapped. There is no external model replacement or task-augmented
client sampling/elicitation receiver advertisement.

Every management operation requires a fresh host action/permission lease, the
original tool grant, the exact current connection/account boundary and schema,
and the existing instance/controller signal. Endpoint equality alone is not
proof of the same account. There is no automatic reconnect, account rebind,
watch, polling after restart, or action resend. Retained history without a
proved current lease is visibly dormant.

## Real creation, unknown outcomes, and results

Before transport dispatch, `McpTaskStore` commits the original host request/action
correlation, argument hash, source/schema identity and attempted operation. A
stable logical intent reserves one deterministic local task reference. Repeated
or parallel presentations of that same intent do not send another `tools/call`.
The server still generates its own distinct remote task ID.

The pre-send phase is durably recorded before the actual transport call. A lost
acknowledgement, Stop or transport failure after that boundary is an unknown
creation outcome, not an invented failed/cancelled remote operation. A received
ACK is saved even if the current permission dies before delivery; that receipt
does not revive its old grant. Raw decoded creation metadata remains attributed
untrusted data rather than being used as a system prompt.

`task-result` validates the exact related-task binding and actual tool content,
then commits a checksummed payload and stable `taskResultReceiptId` before
publication. Repeated reads return the committed payload with the same receipt,
not a second remote effect. Invalid/expired lookup errors are not final failures
of the original tool. A real underlying RPC error is retained as a final result
only after actual failed task state is confirmed.

The raw decoded result remains immutable. A separate observed result passes
through the existing host-owned MCP App capture/provenance path using the
original remote tool arguments. Remote `omni/mcpApp` metadata cannot fabricate
a live host view. Receipt identity prevents presenting a saved result read as
another remote execution. Distinct explicit reads still retain their own
action/execution/evidence identities and may reinforce ordinary learning, linked
to the same source result receipt. Source verification is not proof that a
trained brain has learned anything here.

## Input, Stop and retirement

Task-related input is matched by the real related-task ID to an exact currently
owned task operation, never assigned to whichever unrelated RPC is active.
Pre-ACK input waits for the actual acknowledgement to prove the receiver ID
before any host review. Sampling and elicitation still use their existing
independent input/output review and same-brain/native rules. Successful replies
echo the task association. Foreign, stale or unowned task messages cannot grant
new work. Optional status notifications are not a substitute for final results
or explicit status polling.

`stopTaskWait(brainId, taskRef)` stops only that owned local management wait and
its review. It does not cancel the remote task, kill/restart a process, delete
learned data or retire a still-valid session lease. A fresh permissioned status
or result operation may subsequently use that same current session. Task waits
also suppress generic RPC-cancel notifications: closing a result wait must not
be misrepresented as remote task cancellation.

Remote `task-cancel` is a separate ordinary permissioned operation. Its
acknowledged remote state is not a claim that previous external effects were
undone. Unload/recovery/account retirement/exit abort local operations, retain
uncertainty, and drain pending custody writes through `retireTaskAuthority` and
`settleTaskWrites`. No automatic remote cancel is performed under Off.

## Compact native presentation

The existing Tools screen contains a collapsed remote-task history panel. It
reads local metadata only when explicitly opened, refreshed or paged; rendering
does not start a connection or poll a provider. Current-session and dormant
source-original history are distinct. Status, result and cancellation buttons
use the original tool ID and local task reference through ordinary
`tool.execute`, including exact Ask approval/decline and the independent reviewer.
Stop waiting uses only the scoped local stop API, never a brain-wide cancel.

Late local metadata, approval and action responses are fenced to their original
brain/API/source owner. Closed or replaced UI owners cannot publish responses
into another instance. The task metadata page itself is not an authorization
grant; main rechecks the current policy/account boundary on every operation.

Chat's compact action label describes the remote task, so a successful creation
ACK is not shown as Done. Result failures remain failures. A native
`McpTaskCustodyError` carries the actual committed receipt on a failed or unknown
wait; ToolExecutor preserves failure state plus a separate typed receipt and
ordinary structured output. No UI code parses error prose to manufacture state.
Source/result details are expanded explicitly; remote text is not rendered as
trusted HTML and cannot fabricate the host-owned MCP App view.

## Portable history, never portable grants

Operational records, immutable previous versions and outcome payloads live in
the instance's `engine/operational-mcp-tasks` subtree. Source-original IDs and
raw decoded observations are retained. No session IDs, live callbacks, OAuth
credentials, approval tokens or management/watch grants are serialized.
Received raw task IDs remain sensitive unsanitized historical data, but do not
act as an Omni grant after copying or importing.

The source reader pins canonical ownership and FD identity, admits exact-size
JSON bytes, reads only that admitted length, and checks for growth/replacement.
Closure descriptors and accumulated metadata pages also receive physical/app
budget admission; there is no arbitrary total task-history ceiling.

`captureMcpTaskCustody` holds the canonical per-brain lease only while making an
exact verified clone/copy into caller-owned staging. Later multi-gigabyte
neural export does not hold that lease or read mutable task source paths.
Source/staging roots must be disjoint. Merge seals each actual current/ancestor
group independently and flattens identical proven closures rather than nesting
or repeatedly copying shared history. It does not rewrite original IDs.

Recognized interrupted `.json.<uuid>.next` publications are preserved byte-for-
byte as inert partial evidence, never parsed, promoted into current task state,
finished or executed. Unknown paths, links and corrupt committed payloads still
refuse. Import/copy/merge records remain dormant without current verified
authority; no unproved account-rebind API is provided.

## Focused evidence boundary

New synthetic HTTP/SSE transport cases cover negotiated required/optional/
forbidden execution, original arguments/action schemas, accepted versus final,
lost ACK, stale permission/account isolation, exact related-task input, pre-ACK
input, real versus missing-task failure, stable outcome/App presentation,
logical-intent deduplication, account-list data, cancellation and scoped Stop.
The separate owned Node stdio fixture verifies the same process remains alive
after a local task wait stops, with zero cancellation notifications.

Tiny filesystem cases cover unknown custody, stable payloads, canonical aliases,
symlink/corruption refusal, exact snapshots, shared-ancestor merge deduplication,
dormant source ownership, descriptor admission and interrupted publications.
New renderer SSR/presentation/flight checks cover no render-time calls, dormant
controls, ACK/unknown/final labels, source bindings and late-owner responses.
Host fixtures verify typed failed custody, Stop plus exact Ask, and separate
explicit read experiences with a stable source result receipt. The isolated
Electron fixture uses production React components with synthetic callbacks;
opening it alone is not evidence that a live provider or model works.
These are new scoped source/transport/filesystem checks, not public-provider,
OAuth-account, model learning, native modality, all-platform, full UI, package,
CI or release qualification. No model was created, Loaded or trained.
