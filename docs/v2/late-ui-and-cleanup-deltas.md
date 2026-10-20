# Late affected checks, not a second full matrix

October20 follow-up to published `d948371`. Its immutable384-cell cohort
continues unchanged in the managed `omni-v2-ui-final` checkout. Earlier failed
cohorts and captures remain preserved. Source fixes below are PRIMARY-only.

## Capture cleanup

The new hosted fixed stdlib lane ran136 cases;135 passed and one failed an old
text-adjacency assertion at `test_reply_execution_planner.py:421`. The same
failure was read directly in macOS ARM, Linux x64 and Windows x64 logs. The six
desktop jobs ultimately failed that gate; Android/iOS passed. No job was
cancelled or rerun merely to hide the failure.

Review also found a real cleanup guarantee: if capture rejection delivery
raises, a straight subsequent reply-control retirement call could be skipped.
`Worker._chat` now uses nested `try/finally`. The old assertion is replaced by
an exact AST once-only/enclosing-finally contract, not removed. The affected
old case passed separately (0.298s); a new extracted actual-cleanup/BrokenPipe
case passed (0.059s), proving inbox removal, byte-lease release and once-only
reply-control retirement. No neural imports or model work were required.

## Compact preview disclosure

Direct inspection of the new cohort's cell000390x844 Home capture showed no
preview disclosure. Existing final `max-width:650px` CSS hid the entire sole
global badge. The changed Titlebar keeps one compact **Preview** indicator,
with full disconnected accessible wording/hover explanation. At wider widths
the full visible wording remains. Native runtime behavior, brand, Appearance
handlers and OS-control spacing are unchanged. No extra Home panel is added.

Three focused source assertions passed. A short isolated no-preload Electron
run then checked Home and Conversation at390,650 and651pixels: six header
views, one badge each, correct compact/full-detail breakpoint, visible brand
and Appearance, measured no overlap/clipping, no renderer/external errors,
clean child exit0. Its six saved screenshots were directly inspected. Evidence
is PRIMARY `.runtime/compact-preview-delta-0NoUkF/result.json` and sibling PNGs.
The record binds exact dirty App/CSS hashes; those bytes are committed with
this follow-up, not substituted into the older immutable cohort.

The first650/651 Home captures caught early body-heading color transitions;
their body text is **not** accepted as stable visual evidence. One separate
650 Home diagnostic waited for the selected light/standard attributes and
actual finite CSS transitions, without changing production. Its heading/fill
resolved to `rgb(23,21,28)` and the directly inspected screenshot is readable.
That result is `.runtime/compact-preview-delta-DmJi31/result.json`. Do not turn
the early capture into a confirmed persistent app defect or silently replace
its evidence. Header geometry/disclosure evidence is distinct from body QA.

The three breakpoint samples qualify this narrow late header delta, not every
theme/state/platform or full accessibility compliance. The full384-key audit
and final exact-source CI/package/release gates remain open. No brain was
created, loaded, trained or quizzed; synthetic history is not real model output.
