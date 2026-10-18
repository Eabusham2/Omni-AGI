# Reviewed integrations and private map adapters

This is a scoped source/fixture report, not proof of every provider account,
trained tool choice, operating system or packaged UI. No provider login, paid
API, mail/repository write, public map request or model training was performed.

## Connection templates

`shared/mcpPresets.ts` supplies inert reviewed templates. Selection fills the
existing MCP form only; it cannot install a package, authenticate, connect,
discover tools, grant access or call a tool. Clear previous token/environment
drafts when changing templates. Local templates intentionally leave the
absolute installed executable blank instead of invoking a package downloader.
Only a subsequent explicit connection uses the existing MCP/authentication,
permissions and actual-schema learning paths.

- GitHub offers its documented full remote endpoint and an explicitly separate
  read-only endpoint. Account permissions and actual discovery still decide the
  available tool surface. [Official configuration](https://github.com/github/github-mcp-server/blob/main/docs/server-configuration.md).
- Microsoft's Playwright template requires an explicitly installed executable
  and browser runtime. It is browser automation, not a website security sandbox.
  [Official server](https://github.com/microsoft/playwright-mcp).
- Adobe Express Developer exposes add-on documentation/types. This is not
  Photoshop, Illustrator or Premiere control; those need a separately reviewed
  API/bridge or permitted desktop action. No Adobe endpoint was invented.
  [Adobe documentation](https://developer.adobe.com/express/add-ons/docs/guides/getting-started/local-development/mcp-server).
- Gmail is explicitly a developer-preview setup with administrator/security
  requirements. Its documented tool list does not justify assuming a send-mail
  operation. [Reference](https://developers.google.com/workspace/gmail/api/reference/mcp),
  [setup and security considerations](https://developers.google.com/workspace/gmail/api/guides/configure-mcp-server).
- Resend has an official hosted MCP endpoint and OAuth/API-key setup. Sending
  mail is an external effect, not an automatic connection test.
  [Official server](https://github.com/resend/resend-mcp).
- Microsoft365 Work IQ needs the organization's tenant/client/licensing policy.
  Do not adopt another application's client ID. Its separate Copilot tools are
  explicit external services, never a native-brain fallback.
  [MCP overview](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/mcp/overview),
  [official endpoint configuration](https://github.com/microsoft/work-iq/blob/main/plugins/workiq/skills/workiq/SKILL.md).

These descriptions are dated October 11, 2026. They are not prepopulated fake
negotiated schemas or a guarantee that generic OAuth can provision a tenant.
Provider versions, eligibility and scopes can change; actual discovery remains
authoritative.

## Compatible map path

`MapsService` implements an explicitly configured **private Photon-compatible**
search endpoint and offline supplied-GeoJSON/coordinate presentation. It does
not choose a public geocoder or paid account. Photon permits a local service
and documents its search response and parameters; the adapter also accepts the
documented features-only response envelope, records that projection and hashes
the exact original response bytes. [Photon API](https://github.com/komoot/photon/blob/master/docs/api-v1.md).

Configuration is installation-private, with explicit retained-results review.
An optional bearer lives only in `SecureSecretStore`, encrypted when supported
by the OS or session-only otherwise. It never enters tool arguments, model
schemas, status, neural state or exported connection configuration. Redirects
are not followed, credentials are not forwarded to a changed endpoint without
explicit replacement/clear, and a reflected credential causes response
rejection rather than learning it. HTTPS or loopback HTTP is required.

Every save has an expected revision. Removed IDs retain revision tombstones so
re-adding an ID cannot reuse an old approval or cache scope. Config changes
cancel its in-flight reads; existing per-instance tool Off/Ask/Auto/Full policy
still applies. Tool calls use the original `ToolExecutor`, durable intent and
actual-outcome observation path:

- `maps.search.connections` reports configured source metadata, not credentials.
- `maps.search.search` requires the exact connection ID/revision and query;
  optional result count/language/bounds are passed to the service without
  claiming exhaustive geographical coverage.
- `maps.display.show` accepts one owner-bound saved result or exact supplied
  GeoJSON with attribution. A different instance cannot read another instance's
  lookup ID. Completed display output contains its geometry, so retained
  displays need no live credential/session on copy/export/import.

Requests have configurable pacing, timeout and query-cache duration. Returned
cache directives and Retry-After are honored; failed requests never retry
automatically. Cache hits retain original source time/hash and say they are
cached. A corrupt cache requires explicit refresh. Disk reserve failure returns
the actual already-read geometry with a truthful unavailable-cache warning,
not a fabricated lookup ID. Physical transfer RAM is checked while reading.
The cache is an external-service cache, not the brain's recall mechanism.

The append-only `private-photon-offline-coordinates-1` supplementary capability
pack teaches declared schemas/examples through existing initial/supplementary
learning. It contains no claimed executions, hidden persona or preference
labels. No learning was run for this implementation check, and useful native
map-tool choice remains unproven.

## Display and data rights

`MapResultView` is accurately labelled an **offline coordinate canvas**, not a
street basemap. It draws supplied point/line/polygon/collection geometry over a
mathematical WGS84 grid, progressively rather than creating one DOM node per
coordinate. Fit/zoom/pan/world-grid controls perform no network requests;
attribution stays visible and URL protocols are checked. It invents no coast,
road, tile or other geographical feature. GeoJSON foreign members remain inert
data. [GeoJSON/WGS84 reference](https://www.rfc-editor.org/rfc/rfc7946).

No service's CODE licence automatically grants rights to its DATA. Photon code
is Apache2, while OSM-derived data needs its own attribution/ODbL compliance.
The adapter is independently written and vendors neither Photon nor a map SDK.
[Photon licence](https://github.com/komoot/photon#license),
[OpenStreetMap data licence/attribution](https://www.openstreetmap.org/copyright).

Public Nominatim is not a default integration: its policy has application-wide
traffic/privacy restrictions and a specific restriction on generic geocoder
offerings by no-code/low-code/vibe-coding platforms. This is not described as a
blanket ban on every AI app. Public Photon is a donated demo without an SLA and
permits only reasonable usage. Neither public service is silently used here.
[Nominatim policy](https://operations.osmfoundation.org/policies/nominatim/),
[Photon demo policy](https://github.com/komoot/photon#demo-server).

The default private/offline slice is unchanged. A later additive implementation
provides optional user-reviewed keyless/keyed raster XYZ sources and a separate
contract-gated Mapbox Static Images adapter; see [Optional retained basemaps](map-basemaps.md).
No provider is automatically selected, and a declaration does not establish a
license grant. Mapbox temporary/permanent geocoding remains unimplemented:
temporary results cannot be stored, while the standard permanent offering has
own-use/distribution restrictions. Omni does not silently disable ordinary
learning or sanitize complete exports to fit those terms. No real Mapbox request
was executed for this source implementation.
[Storage documentation](https://docs.mapbox.com/api/search/geocoding/#storing-geocoding-results),
[Permanent Geocoding pricing/usage description](https://www.mapbox.com/pricing).

## Focused evidence and remaining boundaries

- Seven `mapsService.test.ts` cases passed in 1.68s, including actual local HTTP
  search, source hash/attribution/cache, main-only bearer, owner isolation,
  redirect/error/no-retry/cancel, disk fallback and inert supplied geometry.
  The subsequent removed-ID revision fixture passed separately.
- Two `mapsToolBridge.test.ts` cases use the actual `ToolExecutor`/intent path
  and a clearly synthetic provider. They verify Ask/Off/stale source revision,
  one real dispatch and delivery to the existing completion observer, not native
  learned competence.
- Two preset/declared-pack cases and three static presentation cases passed.
  Static markup proves escaping/structure/no automatic calls, not canvas pixels,
  colors, focus, all layouts or provider-account compatibility.
- The final four-file affected run passed14 cases in11.37s, overlapping those
  results. One additional storage-permission/non-regular-target case passed
  separately; a focused display check also covers numerical zoom safety.
  Node/renderer compilation and owned-path whitespace checks passed before
  Root's App mount; Root owns its final integration compilation.
- Root mounted `McpPresetPicker`, `MapConnectionPanel`, `MapResultView` and
  `mapDisplayFromAction` in App, clearing prior token/environment drafts when
  selecting a template. This is source integration, not actual visual/provider
  acceptance. Full street-map/provider qualification remains open. No
  simultaneous CUA or provider tests were performed by this lane.

Remaining evidence includes current provider-account/OAuth compatibility,
broader Adobe-document automation, real native learned use, actual licensed
provider basemap output, all-platform packaged/UI/large-geometry acceptance,
and public-service deployment policy. The optional source paths above do not
establish those outcomes through local transport/static fixtures.
