# Saved HTML artifacts

Updated 2026-10-13. Omni AGI Studio can save a visualization, canvas, local
mini-app, simulation, or other bounded HTML presentation as the completed
result of `studio.artifact.create`. The result stays in the conversation at its
original action position and can be reopened after restart.

This is a local presentation and declared-tool boundary. It is not a general
browser, a Node/Electron extension, a native-code loader, or authority to run
arbitrary setup scripts.

## Creating an artifact

The `create` action accepts:

| Field | Meaning |
| --- | --- |
| `title` | Visible title, at most 256 characters |
| `html` | Local body markup for controls, SVG, canvas, tables, and text |
| `css` | Optional local styles with no imported or addressed resources |
| `script` | Optional isolated browser script |
| `tools` | Exact `{toolId, actions}` declarations the script may request |
| `height` | Saved inline height from 160 through 1,200 pixels |

The action is a normal typed Studio tool. It writes one content-addressed
artifact and returns a renderer-safe identity; it does not return the raw HTML
or script to the Studio renderer. Reusing the same source-action identity with
the same canonical content is idempotent. Reusing it with different content is
rejected rather than silently replacing the saved artifact.

Markup may contain local controls, SVG, and canvas elements. It may not contain
script/style document elements, event-handler attributes, forms, frames,
navigation, or addressed resources. CSS may not import or address resources.
The optional script is saved separately and inserted only by the main-owned
artifact host under its exact SHA-256 Content Security Policy.

## Interaction and tool review

Every open gets a fresh random view handle and nonce bound to the current
brain, artifact content hash, and saved revision. The artifact is served from
the dedicated `omni-html-artifact:` scheme in an opaque sandboxed iframe. It
has no preload, Node, filesystem object, credentials, Studio DOM access,
same-origin privilege, device grant, network access, dynamic evaluation,
child frames, workers, or navigation.

A script can call only a tool/action pair listed in the saved declaration.
Main validates the message source, nonce, schema, size, unique request ID,
current artifact revision, and allowlist. It then sends the operation through
the ordinary `ToolExecutor`, so current Off, Ask, Auto, Full, risky-operation,
and cancellation rules still apply. A child never receives an approval token.
When Ask is required, trusted Studio shows the resolved tool, action, and exact
arguments outside the frame. Concurrent requests retain separate review cards.

Closing a view, changing its revision, disposing the host, or replacing the
component aborts pending work and invalidates pending approvals. Completed
external effects keep their truthful receipt; closing a view does not pretend
an already-started external operation never happened.

## Copies, import, recovery, and merge

Artifact registries and canonical content blobs participate in the ordinary
learning-workspace snapshot used by supported copy and `.omni` paths:

- same-owner recovery retains the saved operational state, but no live browser
  handle, nonce, pending request, or approval token is serialized;
- a new owner receives exact presentation bytes and provenance with the
  operational bridge dormant;
- full merge keeps each distinct parent artifact and lineage, also dormant;
- imported or merged interaction can be enabled only through an explicit
  current-owner review of its exact saved revision;
- request idempotency receipts that could carry old authority are not adopted
  as grants for a new owner.

The registry rejects unsupported files, filesystem links, noncanonical
content, missing or unreferenced blobs, changed hashes, wrong owners, and
colliding provenance.

## Renderer API

`window.omni.artifacts` exposes five narrow operations:

| Operation | Purpose |
| --- | --- |
| `open(request)` | Validate a saved identity and return a fresh opaque launch |
| `activate(request)` | Explicitly enable one dormant exact revision |
| `request(request)` | Relay one declared structured tool request |
| `review(request)` | Trusted Studio approves or denies one exact Ask receipt |
| `close(brainId, viewId, nonce)` | Revoke the view and cancel pending work |

The renderer component is `HtmlArtifactView`. It accepts an actual completed
`studio.artifact.create` action event, never arbitrary output that merely looks
similar. Raw saved markup does not cross this API.

## Evidence and limits

Focused store, archive, permission, protocol, relay, preload, ToolExecutor, and
component tests cover content-addressed creation, lost acknowledgements,
tamper/extra-file/executable-schema rejection, dormant copy/merge/adoption, exact CSP and request
isolation, main-owned Ask review, concurrent review preservation, keyed
identity reset, late-activation fencing, and renderer source checks. Node and renderer TypeScript checks passed for this source
slice.

These checks do not establish packaged Electron behavior on every platform,
hostile-package qualification, accessibility or resize quality in every theme,
trained natural artifact selection, useful generated designs, or support for
arbitrary executable plugin/native hooks. The existing app layout remains
unchanged; this feature supplies a small inline result component for the
existing chat action card.
