# MCP roots, diagnostics, and App change notifications

Internal source/evidence boundary, October 15, 2026. This extends the existing
raw MCP client and sandbox host; it is not universal provider/platform
acceptance. No brain was created, Loaded, trained, sampled or queried, and no
provider/account or real tool command was invoked by these checks.

## Implemented surfaces

| Surface | Actual boundary |
| --- | --- |
| `roots/list` | Installed handler returns only currently reviewed selected file URIs, or an empty list. A server request never opens a picker. |
| `notifications/roots/list_changed` | Real selection/sharing/clear/revocation updates notify only the exact already-ready session; no connect/retry. |
| `logging/setLevel` | Reserved client-owned operation exists only with negotiated server logging, through normal `ToolExecutor` schema/current-permission/reviewer admission. |
| Server `notifications/message` | Validated bounded/redacted operational diagnostics, not chat, hidden prompts or learning. |
| App `notifications/message` | Notification-only logging from the exact ready/token-bound App when a real sink is installed. No tool invocation. |
| App list changes | Negotiated tool/resource changes reach only the current ready view told that support exists; no automatic discovery, refresh or tool replay. |

Primary contracts:
[MCP roots](https://modelcontextprotocol.io/specification/2025-11-25/client/roots),
[MCP logging](https://modelcontextprotocol.io/specification/2025-11-25/server/utilities/logging),
and [MCP Apps](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx).
Roots use file URIs; logging uses the eight protocol severities; App host
notification/logging capabilities are separate from server log declarations.

## Selected roots are disclosure, not a filesystem sandbox

`src/main/mcpRoots.ts` stores chosen paths in private device-owned settings,
outside every brain and portable closure. `McpRootsApi` in
`src/shared/mcpHostServices.ts` exposes status/select/review/clear, with no paths
in renderer commands. Main obtains paths from its trusted picker. No home,
brain, application, credential, storage-pool or working directory is automatic.

The existing scrollable private review presents the endpoint, exact selected
names/URIs, revision and fingerprint. Approval shares URI boundary metadata
with that connection; it does not change `system.files` policy, fetch remote
files or confine an already-launched local process. Execution remains separately
permissioned. The review explicitly says disclosure-only / no OS sandbox.

Live sharing binds brain/server, endpoint/launch configuration, fresh main
connection ID, transport epoch, exact selection and file-policy revision.
Chosen settings may survive restart; their live sharing does not. Reconnect,
Unload, recovery, credential/account replacement and connection retirement need
fresh review. New identities inherit no root choice or grant. Changed launch
configuration/endpoint requires a new explicit selection, not a silent rebase.

Every list rechecks connection access, `system.files` policy, canonical root
identity and accessibility around awaits. Off returns no roots. Symlinks,
symbolic ancestors, unsupported/unavailable files and replaced inode identities
refuse sharing. No user file content or directory tree is read to list roots.
Changed/inaccessible selections become review-required, not a silently selected
subset. Oversized protocol envelopes refuse explicitly rather than omit a tail.

`status()` rereads current connection identity after metadata/accessibility/
policy awaits; it cannot return captured old shared state. The `connectionRetired`
callback aborts that server's pending review even with no prior live grant or
resource subscription. Other servers are unaffected. Notification `sent` means
transport handoff, not proof the peer processed it; not-ready/failed outcomes
remain explicit and never reconnect a replacement session.

## Diagnostics are not native memory

`src/main/mcpDiagnostics.ts` keeps current-process-only logs: at most 64 KiB per
log, 256 records / 4 MiB retained globally, and 60 attempts per ten seconds per
brain/server. These operational bounds are not model cardinality limits.
Malformed, cyclic/accessor/nonfinite or oversized data refuses. Rate suppression
and oldest-record eviction have counters. Main guards current connection access
before exposing read-only paged history.

The consumer redacts sensitive field names, bearer syntax, configured MCP bearer/
environment/OAuth secret snapshots and an App's private view token. Snapshot
reads never refresh tokens or contact a provider; exact runtime object and
connection ID are rechecked after vault awaits. Receive rechecks cancellation/
access after resource/secret awaits. Unknown PII/credential formats are not
claimed universally detectable; sanitized logs remain untrusted operational
content. Only sanitized data and a source fingerprint are retained, never a raw
log archive, prompt, native-learning receipt, App artifact export or tool effect.
Restart clears history; this is not complete durable brain history.

`setStandardHostServices` installs these handlers without replacing existing
sampling/elicitation. Roots/logging are advertised only with real consumers.
Negotiated `logging/setLevel` still uses normal tool permissions. Actual log data
does not enter the capability-learning document or completed tool outcome.

## Exact session/account and App relay

The new main host identity has a fresh connection ID as well as integer epoch,
preventing remove/re-add from reviving an old disclosure. Bearer/environment
replacement and disconnect already retired transports. Interactive OAuth
`authorizing → authorized` publication now also retires old sessions/views/root
authority without reconnecting; ordinary same-account refresh is unchanged.

`McpClientService.serverMessage` calls the real App notification hook for
negotiated tool/resource list changes. `McpAppService` remembers what it actually
advertised at initialization. Events bind brain/server/view, main view token,
epoch and sequence, and contain only an empty protocol notification. Before-ready,
closed, wrong-owner, stale-endpoint/session and revoked views get nothing.
Observer failures never trigger a remote request or replay.

Root owns current trusted IPC/preload, existing Tools controls and renderer
iframe relay. These host cases do not independently establish visible Electron
delivery, a remote peer's reaction or every account/platform.

## Focused evidence

Sixteen distinct **new** cases passed, without repeating older MCP suites:

- `mcpHostServices.test.ts`: eight passed in 72 ms for empty/explicit selection,
  dormant restart/session replacement, clear during review, awaited Off, root
  source/metadata refusal, pending owner cancellation, redaction/rejection,
  rate/retention and paging.
- One added affected case passed in 13 ms with those eight skipped: same-server
  pending-review retirement, unrelated-server preservation and status replacement
  during policy await.
- `mcpStandardHostSurfaces.test.ts`: three new production-client cases passed in
  the initial scoped command. The fourth logging-wrapper case initially failed
  because its fake repository directory had the wrong brain-owner basename;
  correcting that fixture made the affected case pass in 59 ms, the other three
  skipped. It checks real `ToolExecutor` Off/Full, reserved schema/input handling.
  The other cases cover new method routing, absent/invalid roots, no automatic
  invocation, credential/account retirement and stale secret-await refusal.
- `mcpAppHostNotifications.test.ts`: three new cases passed in 8 ms for negotiated
  current-ready-view forwarding, absent/revoked consumers and notification-only
  attributed/deduplicated/redacted App logs.

These use owned tiny filesystem data and synthetic in-memory HTTP/SSE through
production host code, not a live provider. The OAuth case injects a status
publication, not an account login. Final TypeScript/renderer/UI evidence is
root-owned and recorded separately. No native/model/package proof is inferred.

## Still unsupported or unqualified

The later [dated durable-task slice](mcp-durable-tasks.md) adds a genuine
2025-11-25 outbound server-owned consumer with original tool permissions,
explicit local history, unknown/no-resend custody and source-bound final results.
It does not implement the 2026 Tasks extension draft or client-receiver tasks.
Its separate scoped source tests and parent-owned UI acceptance do not broaden
this roots/logging report into universal interoperability evidence.
Fullscreen/PIP, raw devices, view-exposed tools, sampling tools/context/video/
output media and provider-specific authentication extensions remain explicit
unsupported surfaces. Provider/account behavior, packaged OS sandbox behavior,
complete accessibility and native competence remain separate gates. Earlier
bounded App/media proofs are neither reopened nor broadened by this slice.
