# Voice settings and plain task attachment UI

October 11, 2026. Bounded Electron presentation/interaction check using the
real React controls, `useInstanceVoice`, `useTaskAttachmentDraft`, and the
metadata-only `LiveVoiceController` route preparation. Host storage and audio
capabilities are **explicitly synthetic**. No real brain, microphone, file,
provider, tool effect, ingestion, or neural learning runs in this fixture.

The native window was operated through computer use. Each accepted screenshot
was saved and reopened from disk. This is not full app, platform, speech,
accessibility, neural capability, or corpus qualification.

## Observed steps

1. **Wide dark controls — healthy.** Independent input/output selectors,
   delivery and pace are legible; unchanged Save is disabled. Settings are
   available before starting capture, not hidden behind a running call.

   ![Wide dark voice controls](../../test-results/voice-attachment-ui-manual/01-wide-dark.png)

2. **Manual unavailable native output — healthy.** Selected and saved native
   output remains selected. Route preparation reports that it is unavailable
   and explicitly does not substitute platform TTS. The highlighted button
   retains readable text. Capture is not started.

   ![Unavailable native output](../../test-results/voice-attachment-ui-manual/02-native-unavailable-no-fallback.png)

3. **Instance isolation — healthy within the synthetic host.** Switched from
   A to B (B showed Auto), then back to A (A retained native output). The B
   observation is in the current accessibility result; this screenshot shows
   the returned A. This does not prove real SQLite restart durability.

   ![Returned instance A settings](../../test-results/voice-attachment-ui-manual/03-instance-choice-retained.png)

4. **320px light controls — healthy.** The pairs stack and remain readable.
   No horizontal overlap was observed in the controls. The fixture toolbar is
   outside the 320px product-control surface and is not a compact app-toolbar
   acceptance claim.

   ![Compact light controls](../../test-results/voice-attachment-ui-manual/04-compact-light.png)

5. **Active call/utterance locks — healthy.** A synthetic active call disables
   input/output route changes but not idle delivery/pace. A synthetic active
   utterance disables all four. Disabled selections and the reason remain
   legible in light mode. This tests control presentation, not live barge-in.

   ![Active control locks](../../test-results/voice-attachment-ui-manual/05-active-control-locks.png)

6. **Attach/expand and bottom content — fixed and healthy.** Staging an inert
   document shows task attachment, not training. Expansion now reveals its
   complete name, exact bytes and SHA with wrapping at 320px. Scrolled to the
   bottom and inspected the full expanded area. The initial discovery only
   exposed the truncated summary name; the full-name expansion was added.

   ![Expanded task attachment](../../test-results/voice-attachment-ui-manual/06-compact-attachment-expanded.png)

7. **Remove an unsent attachment — fixed and healthy.** Removal clears the
   card and reports “Attachment removed; no learning was started.” The first
   interaction exposed a stale “Attached” label; the affected path was fixed
   and checked once more. Bound historical files are not removed by this UI.

   ![Correct removal feedback](../../test-results/voice-attachment-ui-manual/07-removed-status.png)

8. **Missing count evidence — healthy.** Auto stays Auto; preparation reports
   that verified native parameter accounting is unavailable rather than
   silently guessing platform output. The expanded details and bottom content
   were inspected without any microphone or model request.

   ![Missing evidence does not invent a route](../../test-results/voice-attachment-ui-manual/08-auto-missing-evidence.png)

## Source integration and focused checks

`App.tsx` now mounts these controls in Call, refreshes saved instance settings
and current capabilities before Start, fences stale instance/Unload responses,
and does not write new choices to the old renderer-global record. Only an
actually present valid old record is passed to a host-eligible migration.
Route changes are blocked during a call; delivery/pace changes are blocked
during an active utterance. Unknown count/unavailable native routes never
silently become another route.

The composer separates plain Attach from explicit Learn. Dropping files stages
task bytes rather than calling dataset ingestion. Exact attachment IDs follow
pending/queued/steered messages and delivery receipts. Unsent files can be
removed; failed sends restore only still-staged IDs, not files already bound to
an earlier turn. Historical cards query exact IDs/turns lazily, not all history.
Source initialization reads are fenced against later draft changes.
The initial draft query now requests only staged records. Turn/ID projections,
failure restoration and the read-result card use indexed exact queries rather
than constructing all historical handles. One new real SQLite query fixture
passed in165ms: unrelated corrupt history was not loaded, the turn index was
used, and a guarded full-history `list()` was never called for those operations.
This does not measure whole-application or large-file hashing throughput.

Four new voice controls/strict-legacy cases passed in21ms test execution;
three new attachment presentation/dedup cases passed in47ms. The one updated
static voice/steer contract passed in10ms (12 unrelated cases skipped). The
owner's 21 voice backend cases and task-attachment host cases are distinct
evidence, not re-run or relabeled as this UI run. Renderer TypeScript passed
after the integration. No full test suite, CI, packaging, or release ran here.

Vite source edits reloaded the synthetic page between some affected checks.
Its in-memory fixtures are not durable brains; these reloads are not a native
restart/retention proof. Actual task-tool reading, attached-media understanding,
queued/steered native execution, real input/output audio and packaged permissions
remain separate acceptance gates. Screenshots cannot certify assistive
technology behavior, every theme/size, device grants or actual learned speech.
