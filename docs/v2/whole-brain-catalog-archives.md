# Complete saved whole-brain catalog archives

The approved format-2 export retains actual current and immutable-origin
catalog closure, including all pointer-reachable retained lineage. It does not
invent missing registries, rewrite an immutable origin, implicitly Load a
model, adopt a foreign backend or preserve an older-app export mode. See
[`OMNI_FORMAT.md`](../OMNI_FORMAT.md) for the public container contract.

## Exact copied-state custody

Trusted duplicate/fork and collision import retain source IDs and exact
catalog bytes as lineage rather than relabeling source neurons as a new live
registry. Re-export selects a real lineage registry only when every non-owner
binding matches current saved metadata: core and substrate generations,
attention-overlay digest, connected-region descriptor and graph-state digest.
One unique exact pointer is required; stale or distinct ambiguous matches fail.

The archive-only `projection.json` identifies preserved current state, binds the
actual current owner/state digest to the original lineage entry/pointer and
sets `destinationRegistry:false`. It has no grant or executable authority.
Import recomputes its expected bytes before identity reservation. The record
is not installed into the live engine as a registry. Future exports derive it
again from real metadata/lineage; no Load/Save is needed to invent a map.

The complete exact catalog namespace is checksummed alongside other `.omni`
payloads. The validator rejects undeclared/orphan entries, mismatched ZIP entry
length/path, invalid chains/index locations and changed projection bytes.
Native identity/catalog rekey edits retain unrelated raw JSON lexemes, so
sealed native numeric descriptors such as `1.0` do not become `1`.

## Admission and storage

`WholeBrainCatalogReadOptions.admit(bytes,purpose)` and the repository option
`catalogAdmission` admit metadata, shard and index scratch against the current
host budget. The helper also checks actual free RAM/V8 heap before allocation
and after an awaited host check. Reads pin exact descriptor size/identity and
canonical source path, with cancellation and checksum fences. SQLite walks
distinct shard locations with a small page cache and file-backed temporary
work. Catalog, lineage and recovery/import accesses use the same admission
hook, including clone/rekey and final publication checks. No total model or
lineage-count cap substitutes for physical resource limits.

## Evidence boundary

The new `tests/wholeBrainCatalogArchive.test.ts` is constructor-free file,
JSON and tiny SQLite/ZIP validation. It does not instantiate or train a neural
model, prove learned recall, measure large-state throughput, or validate a
packaged/platform runtime. Existing repository export fixtures now make
explicit test-only catalog saves backed by actual inert committed mutable and
substrate manifests, exact attention/connected descriptor bindings and tiny
SQLite indexes. Production export/import never repairs stale state for them.

Ten new focused cases passed in 184 ms: feature gates, direct current/origin
ZIP closure, missing/empty registry refusal, catalog-free-only format-1 import,
two-generation copied re-export, stale/ambiguous lineage refusal, projection
tampering/orphans, native numeric-lexeme preservation, pre-decode allocation/
cancellation, and repository admission propagation. One additional declared
record-kind count case passed alone in 13 ms with the earlier ten skipped.
The first run exposed a real macOS `/var` versus `/private/var` canonical-path
false rejection; the corrected path/FD/root fences passed the affected cases.
An initial test-only wrong ZIP helper import was corrected before the recorded
successful run. These are scoped source results, not complete repository,
large-state, live-model, CI or package qualification.

## Repository fixture modernization

The selected host fixture file contains no neural construction, Python worker,
Load, provider request or training. Its largest tensor-file payload is about
256 KiB and its long-label fixture is about 293 KiB. Forty affected test titles
expand to 51 cases; 25 unrelated cases were skipped. The initial one-worker run
passed 46 and found 5 failures in 44.4 s of tests. Four affected cases subsequently
passed after updating stale fixture assumptions or the corresponding scoped
source defect; the successful initial 46 were not rerun.

New actual BrainRepository coverage includes portable format-2 export, same-ID
import, collision import, lineage-only re-export with an exact archive-only
projection, and another import without any model Load. This passed in 1,085 ms.
An actually stale saved-catalog export refuses without changing its source or
publishing an archive; this passed in 66 ms. These are host filesystem/custody
proofs, not native capability qualification.

Recovered optional workspaces must remain genuinely absent rather than being
invented as empty directories. Historical incomplete captures without an
original continuation declaration do not gain an invented partial declaration
solely because catalogs are rekeyed; exact catalog custody is still validated,
preserved and outer-checksummed. Modern declared required payload failures stay
fail-closed. The existing attachment recovery proof was extended with optional
root and exact selected/later API-teacher custody assertions. Those assertions
passed, then exposed a distinct teacher copy-projection identity collision.
The projection identity now includes its exact source job ID as well as the
source database digest and original provenance. Distinct selected/later rows
can no longer collide merely because they share an original job ID. The
conflicting-history validator remains strict and copied provider/learning
authority remains dormant. The extended recovery→export→import case passed
alone in 1.75 s with the other 75 file cases skipped. All 51 selected distinct
affected host cases have now passed across scoped runs; this is not a claim
that all 76 file cases or the complete repository suite were rerun green.
