# Selected web images — R023

## Implemented source contract

Web image metadata is not reported as seen pixels. `web.search` can return
attributable references from actual provider image/thumbnail fields; its
optional `images` input selects the SearxNG image category. The public RSS
fallback still returns ordinary page hits when that provider has no image
metadata. The brain can explicitly fetch a hit's page, receive its literal
`img`, Open Graph, Twitter-image or image-link references, and then choose an
image. No page script/DOM executes during this extraction and discovery does
not download thumbnails.

`web.fetch` also distinguishes a supported binary image response from text.
It returns a reference instead of UTF-8-decoding pixels. In this particular
case the already-fetched image SHA must match a later selected download;
changed pixels require a newly observed reference.

`web.fetch` action `inspect-image` accepts a `referenceId` actually registered
for the current brain and chat turn, plus an optional user-requested `maxBytes`
stop condition. It cannot manufacture search attribution from a model-supplied
URL or copy another turn's reference. References are SHA-bound rows in the
**existing task-attachment registry**, not a neural memory, answer index or
second authoritative brain database.

## Download and exact ownership

- Existing `safeFetch` handles production HTTPS, public-address and redirect
  validation. Every selected redirect also rechecks the current learning scope.
  This slice does not replace or certify the entire network stack.
- Current tool permission revision, loaded owner, source scope and physical
  resource admission are checked during streaming, after the download, and
  before native observation/queued learning. A changed grant or learning policy
  does not silently keep the former authority.
- Bytes stream to a managed temporary file in 64 KiB write windows, with
  incremental SHA-256 and disk/RAM admission. There is no default whole-image
  RAM buffer or new fixed image-file ceiling. Actual reserves and an explicit
  `maxBytes` still stop work honestly.
- Supported MIME and container signatures are PNG, JPEG, WebP and GIF. HTML,
  SVG, executable/data URLs, truncated declared bodies, changed sources and
  unsupported MIME fail without a pixel-observation or learning claim.
- The existing attachment CAS stages the exact bytes and private download
  provenance. Each selected action has a separately hashed `webimg-…` task scope
  bound to its original chat turn, action event and reference. It **does not**
  alter the original turn's sealed attachment set or grant general historical
  file access.
- Stop/failure removes only this operation's temporary staging directory;
  already committed source custody and effect receipts are not undone.

The source, source-response SHA, current reference, requested/final URL,
redirects, permission revision, optional learning-policy SHA, fetch time,
MIME, byte count, exact content SHA, task manifest and original action identity
remain attributable. Labels/metadata are untrusted source data, not behavioral
instructions or captions invented by another model.

## Same brain: perception is separate from learning

The ordinary live tool-observation callback recognizes only a completed,
source-bound selected-image result. Native checks the reference/attachment
registry, action scope, source JSON, public manifest and selection checksum.
It hashes the **opened exact file** before pixel decoding and checks file
identity/change indicators while and after reading. The decoder cannot close
the owner's verified descriptor.

Pillow/NumPy decoding uses the existing admitted image-window path, including
decoded-pixel expansion admission. Every frame reaches this instance's own
native perception region. Shared-space accumulation is bounded; no base64,
source passage, URL-only substitute, foreign vision model or caption is
inserted into a language prompt. The callback's typed
`omni-web-image-perception` receipt reports actual window/source coverage and
explicitly says `parametersChanged: false`: that phase is read-only perception.

Lasting updates follow the **existing completed-action learning lifecycle**,
not forced immediate permanent memory. When that learning queue owns a safe
boundary, the selected raw file enters ordinary atomic image/video ingestion,
including its existing fast/sensory/slow-weight learning and replay. Animation
is recognized from the actual container even when CAS filenames end `.data`;
temporal learning requires this same brain's enabled video region.

The returned `omni-web-image-learning` receipt binds selection/reference/source
SHA and bytes to the genuine committed media source and ingestion transaction.
URL JSON alone cannot satisfy it. The existing transaction key and exact
content identity prevent a lost acknowledgement from repeating already
committed ingestion. Missing modality training, incomplete coverage or absent
completion receipts remain failures, not learned results.

## Portability and limits

Snapshots/export/import preserve complete reference and selected-source
provenance through task-file custody. Forks/recovery projections retire image
reference authority; merge retains reference rows inert. Original source rows
and bytes are not rewritten as new observations. Copied pending selected-image
work cannot replay an inherited tool grant: explicit reselection under the new
current identity is required. Source/manifest histories remain unsanitized.

On Windows, native `O_NOFOLLOW` may be unavailable. Existing owned-path,
opened-file, SHA and identity checks remain, but this is not a claim of an
atomic Windows no-follow/reparse-point API or hostile-local-owner filesystem
certification. Codec/platform packaging and genuinely trained vision ability
are separate gates. A primitive/untrained native region is not represented as
understanding the image.

## Scoped evidence

- Six new host cases passed in **571 ms** using fake HTTPS transport and tiny
  private SQLite/files only: unfetched discovery; redirects/CAS/exact original
  turn isolation; retry without redownload; MIME/signature/length/size/private
  destination refusal; current permission/scope/RAM/cancellation checks;
  action-evidence binding; and inert copied provenance.
- Two further distinct host cases passed in **49 ms** for hostile reference
  schema refusal and previously fetched image SHA pinning. One additional
  trusted-fallback action/turn evidence case passed in **109 ms**. The earlier
  six/eight cases were deliberately skipped during those affected checks.
- Five constructor-free native cases passed in **101 ms**. They use fake
  decoder/perception/ingestion functions with real small custody files/SQLite.
  They cover read-only pixel routing/typed receipts, current action binding,
  corrupt/dormant/pressure refusal, exact ordinary image transaction/lost-ACK
  identity, and refusal to claim incomplete learning.
- One further constructor-free animation/CAS-container/temporal-window case
  passed in **33 ms**, without rerunning the previous five native cases.
- The new helper-scoped strict/noUnchecked TypeScript check passed. Seven
  touched integration/test files passed syntax-only transpilation, not a
  whole-project semantic compile. Changed Python sources compiled. The shared
  project compiler is owned by the parent integration lane after source freeze.
- The parent's one coordinated Node compiler identified one fixture-only
  widened boolean; it was corrected to literal `true as const` without a
  production change or rerunning closed tests/compiler. Public user and
  integration manuals now describe this route in ordinary documentation
  curriculum **v13**. One new declaration/manual-byte/hash case passed in
  **2 ms** with no host action or neural construction.
- These are source/control-contract tests, **not** actual trained visual
  comprehension, a model load, remote/provider call, media/codec run, UI
  demonstration, packaged-platform test, CI or release evidence. Previously
  closed web, attachment, model and lifecycle suites were not rerun.

Implementation: `shared/webImages.ts`, `webImageReferences.ts`,
`webImageService.ts`, `webImageCapabilityPack.ts`, the narrow ToolExecutor,
BrainService/composition and task-custody hooks, and native
`web_image_observation.py` with the ordinary callback/ingestion entry points.
The declared capability pack contains project-authored schema examples, not
invented successful executions or a hidden prompt.
