# Optional retained basemaps

Updated October 17, 2026. This adds to the existing private Photon search and
offline GeoJSON view; neither is replaced. No external map provider, account,
key, public endpoint, or paid plan becomes a default.

## Supported optional sources

- **Raster XYZ:** an exact HTTPS or explicit loopback HTTP template with
  `{z}/{x}/{y}` path substitutions. It may be keyless, bearer-authenticated,
  or use a named query credential. Configure a service you operate or are
  authorized to use for the retained learning/export workflow. The reviewed
  origin cannot change through tile substitutions.
- **Mapbox Static Images:** an explicitly selected style owner/id, a device-owned
  `access_token`, and the user's recorded compatible license/contract
  authorization. Logo and attribution remain requested. This does not enable
  Mapbox Geocoding, an external language model, or Mapbox's separate AI products.
  Provider-supported dimensions are checked rather than silently resized.
  [Official Static Images API](https://docs.mapbox.com/api/maps/static-images/).

An enabled profile records an authorization reference and license link for
retention, learning, and redistribution through complete instance exports.
This is the user's declaration, **not** a license grant, legal determination,
or assertion that a normal subscription permits those uses. The code cannot
verify a private commercial contract. Keep a source disabled if its terms do
not support this workflow.

Public `tile.openstreetmap.org` is not an offline/export preset and cannot be
configured through this adapter. Its service policy prohibits bulk/offline
archives; free OSM data does not mean the donated public tile service is
unrestricted. Self-hosted or separately authorized OSM-derived services remain
usable with their required attribution.
[OSMF tile policy](https://operations.osmfoundation.org/policies/tiles/),
[OSM data copyright/attribution](https://www.openstreetmap.org/copyright).

Mapbox temporary geocodes cannot be stored. Its standard permanent offering
allows storage but limits results to the customer's own use rather than
distribution. Therefore this implementation does not add either geocoding
path or silently remove learning/history/export to fit it.
[Geocoding storage](https://docs.mapbox.com/api/search/geocoding/#storing-geocoding-results),
[Permanent Geocoding usage description](https://www.mapbox.com/pricing).

## Device configuration and permission boundaries

`MapBasemapService` stores profiles, revision tombstones, pacing, and optional
raster caches in a device-owned SQLite database outside brain archives.
Credentials use `SecureSecretStore`: OS-protected encryption where available,
otherwise session-only. A key is never part of a tool schema, returned profile,
renderer URL, neural checkpoint, origin, journal, or `.omni` connection grant.
Source/authentication changes require explicitly replacing or clearing the
prior key. A deleted/re-added profile cannot reuse an old approval revision.

Settings are inert: listing/reviewing/saving a profile does not contact its
provider. Opening or redisplaying a retained map does not contact it either.
The normal `maps.display` ToolExecutor policy controls each actual request:

- `basemaps` lists reviewed metadata, including current profile revisions.
- `basemap` takes `profileId`, `expectedRevision`, requested WGS84 bounds,
  viewport `width`/`height`, and optional XYZ `zoom`. No token or arbitrary
  per-tile URL is a model operand.

The host supplies its current action-owner/permission fence. The service
rechecks it and the profile/credential through paced requests, response
headers, streaming reads, cache updates, and final publication. Stop,
instance authority retirement, settings replacement/removal, and shutdown
abort the owned requests. Retiring one instance does not cancel another's
independent request.

Requests omit cookies, do not follow redirects, and do not retry failures.
`Retry-After` postpones the next separately authorized request. Configured
pacing and provider cache headers remain active. Cache corruption is a visible
failure, not an automatic fresh/billable lookup. Resource admission happens
before tile-plan/pixel allocation and while receiving bytes, using the host's
selected app-family RAM budget when supplied. This is not an OS hard-memory
limit or a measured provider throughput claim.

## Actual raster output and display

The result retains the exact received PNG/JPEG/WebP bytes, source times,
SHA-256 hashes, profile revision, requested bounds, attribution links, and
authorization provenance. HTML/SVG/scripts, incorrect declared MIME types,
incorrect dimensions, and raw reflected credentials are refused. The renderer
verifies each retained checksum and decoded dimensions before drawing it; it
does not install a remote SDK or load remote images.

XYZ tiles have exact tile bounds. A Mapbox image instead says
`provider-fit-to-request`: fitting a geographic box to an aspect ratio may
include more area than requested. Its image is displayed without a fabricated
coordinate overlay, and the caption explicitly distinguishes requested bounds
from actual image edges. Mapbox/OSM attribution links accompany the required
branding request. Other source attribution stays visible and is inert text.

Only visible requested XYZ tiles are planned; there is no automatic world,
city, multi-zoom, or background prefetch. Geometry/zoom/physical-resource
constraints remain real rather than being described as unlimited network
capacity. The retained result follows ordinary tool history, experience
processing, copy, and complete export; installing an archive does not install
the device's profile, keys, or permission grant.

The additive `reviewed-retained-raster-maps-1` capability pack contains typed
schemas and two declared examples for future initial/supplementary learning.
It is not an execution transcript, preference tuning, hidden persona, or proof
that an untrained brain understands these tools.

## Evidence boundary

Eight new host-service cases passed in 144 ms using tiny synthetic HTTP
responses and temporary databases. They cover absent defaults, source/rights
review, exact bytes/cache/provenance, main-only keys, Mapbox branding and fitted
extent labeling, source replacement/tombstones, authority revocation,
per-instance cancellation, Stop/backoff, physical pre-admission, corrupt cache,
and non-raster refusal. Four static presentation/schema cases passed in 25 ms.
These are not public-provider/account, trained-use, cross-platform package,
large-state, visual interaction, or real-world license qualification. Root's
separate native Electron UI audit and host bridge results have their own
evidence records; no new model or training was started for this addition.
