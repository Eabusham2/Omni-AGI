# Omni AGI Studio user guide

Omni AGI Studio creates and runs persistent local OmniCortex instances. Each
instance has one continuous conversation, native learned state, working memory,
saved media, learning records, settings, and lineage.

This guide describes controls that exist in the current source. Focused tests
and small native fixtures establish mechanisms; they do not prove that a newly
trained instance is fluent, factual, generally intelligent, biologically human,
or useful on every supported platform.

## Brain-inspired software, not a biological brain

OmniCortex borrows selected organizing ideas from neuroscience, but it is
software running on ordinary computers. Its “neurons,” “synapses,” “spikes,”
and “working memory” name concrete software structures; they are not living
cells, neurotransmitters, or a claim of consciousness.

| Comparison | Biological nervous system | OmniCortex |
| --- | --- | --- |
| Cell and signal | Living cells use membrane voltage, ion channels, and chemical/electrical signaling. | Software arrays compute transient activity, LIF-style spikes, recurrent state, and typed events. |
| Connection | Biological synapses vary through many molecular and structural processes. | A learned contact has an exact `-1`, `0`, or `+1` sign, discrete integer strength/resistance, timing state, and optional independent repeats. A negative sign is a computation, not a claim of a specific inhibitory neurotransmitter. |
| Learning | Experience can change synaptic efficacy and participating circuits. | Reviewed experience can update packed contacts, STDP eligibility, assemblies, working activity, and slower native parameters through explicit transactions. |
| Working memory | Experiments observe task-related activity persisting across a delay. | Bounded recent token/sensory activity, active assemblies, paged workspace items, and liquid recurrent state retain current work. |
| Routing and senses | Overlapping biological pathways receive signals from sensory organs and the body. | Interconnected software regions route text and admitted image/audio/video/device data. A file decoder is not an eye or ear, and a region label is not a one-to-one brain-area copy. |
| Resources | Biology depends on metabolism, blood flow, temperature, and living tissue. | Execution depends on CPU/GPU operations, RAM, disk, device drivers, and admitted resource leases. |
| Forgetting | Interference, plasticity, circuit remodeling, and biological change can alter recall. | Interference, decay, eviction, stability/resistance, and later updates can alter saved or active state. No perfect-memory guarantee follows from persistence. |
| Inspection | Neural recordings reveal selected activity, not a complete subjective account. | Studio can inspect exact recorded software contacts and events, but weights do not decode into a certified fact inventory or hidden chain of thought. |

The comparison is grounded in primary experiments on ionic action-potential
mechanisms ([Hodgkin and Huxley, 1952](https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/)),
long-lasting synaptic potentiation ([Bliss and Gardner-Medwin, 1973](https://pubmed.ncbi.nlm.nih.gov/4727085/)),
spike-timing-dependent synaptic change ([Bi and Poo, 1998](https://pubmed.ncbi.nlm.nih.gov/9852584/)),
delay-period activity ([Fuster and Alexander, 1971](https://pubmed.ncbi.nlm.nih.gov/4998337/)),
distributed memory ensembles ([Roy et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35379803/)),
and experimentally altered forgetting ([Akers et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24812394/)).
These studies motivate analogies; they do not validate OmniCortex as a faithful
simulation of a human brain. Studio does not assign the human brain a bit count
or use an invented biological capacity comparison. See
[Brain-inspired software and biological boundaries](brain-versus-biology.md).

## Create or import an instance

1. Open the brain library and choose **Create**.
2. Name the instance and review its hardware and memory plan.
3. Select any files, folders, media, datasets, or web sources to learn before
   first chat.
4. Review tool access and local storage choices.
5. Start creation and wait for native initialization, selected-source learning,
   documentation learning, and integrity finalization to finish. There is no
   blocking conversational tool quiz.
6. Choose **Load / review fit**, then open **Conversation**.

Every new Build initializes native OmniCortex locally. It does not wrap a
Falcon, Llama, hosted chat model, or another pretrained foundation. The built-in
capability material is narrow, and finished initialization is an integrity
receipt rather than a claim of broad competence.

Before Chat opens, Studio also learns the exact versioned copies of this user
guide, [the integrations reference](INTEGRATIONS.md), the
[plugin/custom-tool reference](plugins.md), the
[saved HTML artifact reference](html-artifacts.md), and the
[executable training parser reference](executable-training-parsers.md) through ordinary attributable text
learning. Each document has a content hash and a real
training-source receipt. This documentation pack is separate from the frozen
origin action curriculum: it contains no persona prompt, preference labels,
reward signal, executable action target, or fabricated successful tool result.
The manuals are not silently inserted into later prompts.

Whole-brain Import accepts only a verified native OmniCortex `.omni` with the
supported schema and provenance. It validates paths, checksums, safe-tensor
headers, packed state, substrate state, and origin. It does not run setup
scripts or load executable pickle payloads.

## Library, copies, and deletion

The library lists saved instances and their current state.

- **Duplicate** creates an independent mutable copy while sharing eligible
  immutable storage.
- **Fork** creates a child identity for isolated work.
- **Recovery point** saves a restorable state before an experiment.
- **Delete** permanently removes only the confirmed instance. Export a private
  backup first when recovery may be needed.

Large duplicate, export, import, recovery, and merge operations report their
own file/byte progress. That progress is storage work, not a measure of model
intelligence.

## Portable Windows folder

**Portable Windows workspace** creates a self-contained Windows application
folder for one selected saved instance. It is separate from an ordinary
`.omni` export and does not change the normal installer when no portable marker
is present.

1. **Unload** the selected ready instance. In the Library's portable workspace
   controls, open **Create a portable copy**, choose the instance, then select
   **Create portable folder**.
2. In the native dialogs, select a clean Windows application folder from a build
   that supports this feature, then a new output folder. Do not select an old
   portable data directory or mix files from different builds. An existing
   output folder is never overwritten.
3. Wait for copying and publication to finish. Storage-operation progress
   provides Pause, Resume, and Stop. Move or copy the **entire** completed folder,
   not just its EXE.
4. On the destination PC, run the included EXE and choose **Prepare saved files**.
   This preserves the saved instance without loading a model. Then select
   **Review destination**; use **Edit resource settings** if the saved plan does
   not fit, and review again.
5. **Approve this resource fit** approves only that destination's resource plan.
   Check access separately in **Tools & permissions**, then explicitly **Load**
   the instance when ready.

The folder includes a full **unsanitized private archive** with the selected
instance's saved learned state, history, working continuity, sources, artifacts,
and supported pending progress. Treat it as private data. Ordinary copied app
settings include appearance, workspace profiles/arrangement, notifications, the
selected theme dependency, and tool approval timeout. Instance settings follow
the archive; this is not a raw browser-storage or keychain copy.

Device credentials, live sessions, tool/plugin approvals, Auto-load authority,
and source-drive leases do not transfer as usable grants. Saved work stays
retained but copied external authority is dormant; reconnect or approve access
on the destination as needed. Preparation and fit approval do not train a new
brain or authorize external actions.

An already-used folder keeps its newer chats/state when moved; Studio does not
automatically replace them with the original archive. A new destination needs
its own resource review. Storage reservation is a destination-local **logical
shared pool**, with normal free-space safeguards—not an empty padded 20 GB
file shipped with the EXE. Windows packaging, hardware compatibility, and exact
cross-platform neural continuation still need their own validation; the folder
format alone does not prove those capabilities.

## Load, unload, and continuation

Opening an instance presents **Load / review fit**. Studio checks the saved
architecture, occupied context, current RAM, and authorized spill space before
admitting the learned runtime. You can edit resource settings while the
instance is unloaded. **Auto-load** is installation-local and off by default;
an imported instance does not grant itself Auto-load authority.

**Unload** is different from **Stop**:

- Unload waits for a supported native boundary, saves the exact supported
  execution frontier, releases the learned owner, and keeps the original host
  turn or job parked.
- Load restores that checkpoint and continues the same turn without admitting
  the human input again or repeating already recorded actions.
- Stop ends the selected suspended turn and records its partial state.

Token/cache state, native RNG, working/liquid state, diagnostic delta state,
and completed-effect receipts participate in supported continuation. Private
Best Work selection and Ponder wait for a named supported boundary rather than
pretending an arbitrary Python frame can be serialized.

Camera, microphone, screen, browser, remote process, MCP session, and similar
external handles are not cloned by a neural checkpoint. The lifecycle view
lists what must be reconnected. Completed external effects and resource
subscriptions are not automatically replayed.

Focused native continuation and host portability fixtures exist, but the newest
schema-2 inherited-Worker end-to-end acceptance is still open. Do not treat
Unload as a guarantee that every future plugin, external process, device, or OS
handle can resume exactly. Current exact continuation evidence is same-host CPU;
a saved backend label is not a cross-OS, ISA, GPU, or bitwise stochastic
compatibility seal. A long unloaded interval can also affect transient decay
because the persisted logical activity clock does not yet exclude every
suspended wall-time gap. Post-learning equivalence across that boundary remains
open.

## Conversation

Each instance has one durable conversation.

- **Send** starts a turn when the runtime is loaded.
- **Queue** retains a message for later while work is active.
- **Steer** adds direction to the active work.
- **Stop** requests cancellation of the active turn.
- **Fresh attention** clears supported transient attention state while
  preserving committed learned state and visible history.

Studio streams actual text, actions, media previews, and phases. A model-written
explanation is a self-report, not a guaranteed transcript of private neural
computation. A successful tool dispatch also does not guarantee that the final
answer is correct; inspect the result receipt.

Native Ponder can remain entirely internal and silent. After at least one real
Ponder pass, the existing learned action/argument head may instead choose to
emit an optional self-report. Studio accepts only the text the model actually
emitted, links it to the matching visible Ponder action, and labels it as a
self-report rather than chain of thought or ground truth. The model finishes
decoding that report before the host publishes bounded display frames, so those
frames are presentation delivery—not token-by-token access to private neural
inference. A stopped publication retains only the bytes already accepted and is
labelled partial; it never claims the unseen remainder or a complete hash.
Complete reports remain with the message and trace; interrupted published
prefixes remain with their action. Expand the matching **Ponder** card to read
emitted text. **Readable speed** is off by default and slows only its display;
the brain and reply delivery continue normally. Stop or leaving the chat view
shows all already accepted text rather than discarding a display buffer.
Silent Ponder does not create explanatory prose.

Files dropped into or attached from the chat composer become current-task
attachments, not an automatic training job. Use **Learn** when you want explicit
source learning. Ordinary reviewed experience follows its normal learning path;
attachment selection alone is not a learned-source receipt. Recorded history
can be inspected through explicit tools when permitted, but old source passages
are not silently retrieved into a hidden behavioral prompt.

Notifications wrap and provide keyboard-accessible **Dismiss**. Their reading
timer pauses while hovered or focused, then resumes with its remaining time.
This does not pause the brain or cancel the operation being reported.

## Learn workspace

Open **Learn** when you want a durable, steerable learning task.

### Start a learning task

1. Add text, **Add files**, **Add folder**, or one URL per line.
2. Optionally add task instructions.
3. Open **Web scope and dates** to configure allowed sites, excluded sites,
   earliest/latest declared publication dates, and per-site date overrides.
4. For web sources, choose whether to follow links, follow external links, and
   respect `robots.txt`.
5. Choose **Learn**.

File/folder preparation builds a complete host-owned manifest. A path generated
by the model is not a file grant. The preparation view reports discovered files
and hashed bytes and can be stopped before learning begins.

The task card shows each source, current phase, committed coverage, rejected
records, bytes, and saved scope revision. **Pause**, **Resume**, and **Stop**
preserve already committed learning and the saved source cursor. A fatal or
resource pause does not turn incomplete traversal into success.

Use **Steer this task** to change instructions. If you also change web scope,
enable the scope checkbox and provide the reviewed JSON object. A revision check
prevents an old screen from overwriting newer task policy. Already committed
records are not replayed just because scope changed.

Direct UI starts still obey the instance's `data.learn` permission. Native or
organic calls additionally pass the ordinary approval pipeline. Completion
means selected sources settled; it does not establish comprehension, future
recall, or answer quality.

## Fact Check workspace

Open **Fact Check** to inspect selected claims—not to scan every belief in an
instance.

1. Enter one claim per line.
2. Optionally add source URLs, instructions, domain scope, and date bounds.
3. Choose **Start checking**.
4. Review captured evidence, publication-date declarations, conflicts, and the
   model's finding for each claim.

A search result is discovery evidence; a fetched or learned source has its own
URL, hash, date declaration, and tool receipt. Unknown dates remain unknown.
Identical captures do not create unlimited duplicate semantic evidence, though
their operational receipts remain auditable.

Findings are labelled `supported`, `contradicted`, `mixed`, or `undetermined`.
They are evidence-bound model self-reports, not certified truth. Conflicting
sources are retained. A dependent claim may ask a structured question while
independent claims continue.

If a finding proposes a correction, choose **Learn correction** only after
reviewing it. That explicit action creates an ordinary Learn task. A proposed
correction is never written into neural state merely because the model emitted
it. Direct UI starts obey the instance's `data.fact-check` permission.

## Questions from the instance

Learn and Fact Check can show single-choice, multiple-choice, or written
questions for their dependent tasks. Recommended choices appear first. Answers
belong to the instance and survive restart. The durable host API and native
`studio.question` route also exist, but a general question tray outside these
task workspaces is not yet a completed user-interface claim.

Automatic assumptions are off by default. If enabled, the default wait is 20
minutes. At expiry, Studio can use an explicit suggested choice or enough
highest-ranked choices to satisfy the selection rule. A written question waits
unless it has a suggested written answer. Assumed responses stay labelled as
assumptions.

Question answers have `task-choice-only` authority. They never change tool
permissions, issue an approval token, or authorize destructive work. Copies and
imports keep question history dormant until the destination instance is
explicitly loaded/resumed.

## Data formats and coverage

Supported source families include:

| Material | Examples |
| --- | --- |
| Text and code | Plain text, Markdown, HTML, source files and repositories |
| Documents | PDF, EPUB, supported Office and OpenDocument files |
| Structured data | CSV, TSV, JSON, JSONL, SQLite, Parquet and Arrow |
| Collections | ZIP/TAR archives, WebDataset shards, folders and remote manifests |
| Dataset services | Supported Hugging Face dataset sources |
| Media | Supported images, audio and video |

Studio streams records instead of treating a physical batch as a dataset cap.
Coverage distinguishes discovered, processed, rejected, and completed work.
Damaged, encrypted, unsupported, or decoder-incompatible files are reported;
they are not counted as learned.

## Native connection strength and memory

Native projections use an exact two-bit sign (`-1`, `0`, or `+1`) plus learned
discrete connection state. Each base contact has strength 1–16 and resistance
0–15. The brain can also allocate independent repeated contacts at the same
endpoints, including opposing signs. Their integer contributions are summed
before the projection scale.

This does **not** mean the complete brain costs 1.58 bits per connection.
Strength/resistance bytes, repeated-contact endpoints and pages, controls,
SQLite/index metadata, caches, temporary activity, liquid state, and eligibility
all consume additional memory. Float activations and transient derivatives are
runtime computation; they are not a hidden second learned weight model.

The Runtime Card and Brain Map distinguish:

- base contacts, repeated contacts, and combined logical contacts;
- learned bytes, structural bytes, and allocated payload bytes;
- sign, strength, resistance, and effective contribution;
- cumulative update events from unique currently saved contacts.

Strength does not imply current activation. Useful memory, perfect recall, and
no-forgetting behavior require trained evidence; they do not follow from the
storage representation alone.

## Device resources and storage

Open **Device & runtime** to review context, RAM, accelerator, and storage
choices. **Auto** uses the observed device and saved architecture. Manual values
must pass the same current fit checks; saving them does not replace the learned
model with a smaller one.

Apple CPU/Metal memory is one physical pool. Directories on the same verified
volume are also one storage pool. Studio deduplicates those physical groups
instead of adding the same capacity twice. Unsupported or unmeasured devices
remain unavailable rather than becoming optimistic capacity.

Use **Select spill drives or folders** to grant specific local directories.
Imported path strings do not grant access. Native packed-core, router/control,
attention, rollback, and supported saved-activation pages can use authorized
ephemeral spill. Authoritative checkpoints, origins, journals, and persistent
training/recovery space remain separately reserved. Every write still checks
the shared quota, physical-volume quota, pending allocations, free-space
reserve, and root identity.

When RAM rollback is refused, a private compact native snapshot may use an
authorized selected drive if offload is enabled. It never falls back to an
ungranted path. Cancelled or failed generated files are released; unknown user
files are not treated as cleanup targets.

Current evidence covers CPU components, a tiny real Metal operation, two-rank
Gloo synchronization, and synthetic two-volume placement/accounting. It does
not prove CUDA/NPU parity, working multi-GPU aggregation, whole-model throughput,
or physical multi-drive performance on every OS.

### Manual regional execution

The collapsed **Advanced · manual regional execution** section in
**Device & runtime** lets you route selected existing decoder blocks for
inference. This is an optional manual control, not whole-brain Auto placement.
The normal single-device path remains the default.

1. Explicitly **Load** the instance and wait for its current reply to finish.
   **Refresh status** is available during a reply; route preparation, activation
   and deactivation wait for the reply boundary.
2. Select **Prepare routes** to inspect the paged registry of exact decoder
   blocks and available devices. Preparation does not activate a route.
3. Choose a **Route device**, then **Add saved route · enable allocation** or
   **Update saved route**. Draft multiple routes before applying them; use the
   region pages as needed. A CUDA selection requires its explicit packed
   CPU-tile fallback acknowledgment. You can remove individual saved routes.
4. Apply the draft with **Save memory & storage**, then select **Prepare routes**
   again. Editing preferences makes the earlier activation review stale, even
   when its descriptors remain visible for drafting.
5. Select **Activate reviewed routes** to arm those exact current-owner routes.
   Use **Refresh status** to inspect the actual lease/counters, or **Deactivate**
   to retire it at a supported boundary.

Saved preferences do not arm a lease. Load, restart, copy, import and recovery
never auto-activate them. The current-owner review and active lease are not
exportable grants. A changed owner, device identity, allocation or policy makes
the lease invalid; dispatch refuses instead of silently picking another target.
Selecting a route preserves the separately chosen global RAM policy and
ceiling. Deactivation does not clear working context to make room.

CPU and Metal routes use the supported direct native packed path. CUDA currently
requires **packed CPU-tile fallback**: activity/caching may use the chosen GPU,
but packed projection tiles still run through the admitted CPU path. This is
**not native CUDA compute** and has no established speed advantage. Transfer
bytes/counts and known or unknown link estimates are operational diagnostics,
not measured whole-model throughput.

An inference lease does not arm training or create extra GPU context capacity.
Automatic placement uses the separate control below, including its supported
prepared text-learning wave. Retained device-cache/context migration, concurrent
device execution, unsupported XPU/NPU routes and hardware performance
qualification remain separate.

### Automatic operation placement

The collapsed **Advanced · automatic operation placement** control in
**Device & runtime** compares supported operations during their next normal
use. It does not run a benchmark, create data or generate a reply just because
you open the panel.

1. Load the instance, finish its current reply and apply resource settings.
2. Choose **Review devices**, inspect supported routes, then explicitly select
   **Enable Auto for future operations**. CUDA's CPU-tile fallback needs its
   separate acknowledgment; it is not a native CUDA packed kernel.
3. Use the instance normally. Supported perception, cached steps, full-prefix
   sampling, uncached decoder forwards, image/audio/video generation, and
   prepared text-learning waves can qualify routes using their actual inputs.
   Supported decoder operations may also compare sequential block placements
   across reviewed devices; this is not concurrent device execution.
   Qualification can cost extra time and memory. A route is selected only with
   the required matching result and admitted resources. Input-specific forward,
   prefix and learning-wave decisions are not reused for different inputs.
4. **Inspect qualified routes** pages the current process's decisions. New
   inputs and token positions are computed afresh; a reused route does not mean
   that every future input was already measured or must remain faster.
5. **Stop Auto** retires the placement policy, not the conversation. It restores
   the selected owners before ordinary processing continues. A genuine unsafe
   restore instead pauses for explicit Unload rather than pretending success.

These controls cover thirteen individually registered operation kinds. Mixed
sequential block routes apply only to ordinary/connected cached text steps,
full prefixes and uncached forwards; perception, media and training retain
their supported complete-operation routes.

The real reply and learning controllers account for the current request's
memory, action, Ponder, replay, media-learning and publication phases. Mutable
memory and learning state stays with its actual owner. Actual tool effects,
learning updates, source cursors and atomic saves are not repeated to time an
alternative route. A prepared text wave is still not a privately movable whole
training pipeline, and an uncached forward alone is not the whole reply.
These controls do not promise whole-brain speed or migrate arbitrary memories.
Learned-owner, layout, device, resource and policy changes can require fresh
qualification. Restart, copy and import do not revive a placement grant.
Actual device throughput remains a separate measurement.

## Tools and permissions

Tool access is set per instance and tool:

| Level | Meaning |
| --- | --- |
| Off | The capability is unavailable. |
| Ask | This exact action waits for approval. |
| Auto | Supported lower-risk actions may run automatically. |
| Full | Broader enabled scope may run without per-call approval; OS and host boundaries remain. |

Approvals bind to one instance, tool, action, and argument digest and expire.
A question answer, MCP form, App request, or timeout assumption does not change
these settings.

The current tool families include files, platform-native shell, code, web,
browser, history, imagination, device input, agents, source evolution, Learn,
Fact Check, and structured questions. Availability depends on the installed
runtime, platform, permissions, and actual configured scope.

### Native sparse connections

Open **Tools → Native sparse connections** to inspect committed native sparse
neurons, assemblies, and ternary contacts; export a selected closed bundle;
register a compatible bundle attached to the current task; or prepare an
**Append** or **Replace exact** plan. Preview is read-only and shows row,
addition, replacement, identical, and conflict counts plus the exact reviewed
hash. **Apply** is a separate ToolExecutor review and requires that exact plan
hash.

Apply returns a durable job before native work completes. The job card appears
with its originating chat action when available and remains in the Tools list.
Queued or Waiting for Load is not applied. **Stop** ends that exact existing
job without granting a new mutation; if native publication may already have
started without a receipt, the durable result is **Uncertain**, not a promised
rollback. **Prepare inverse** creates another reviewed plan and can restore
saved replacements; it does not silently remove appended rows whose current
references are unproven.

Copies and imports preserve exact bundles, plans, receipts, history, and source
lineage but no live approval or callback. Review a new current-owner plan before
applying anything. This feature transfers only the declared sparse rows. It is
not whole-brain import, identity merge, arbitrary cortex editing, or evidence
that the receiving brain learned a useful capability.

### Native improvement candidates

`source.self-modify` is Off by default. Its **Propose** action accepts either the
existing `source` workflow or one explicit native candidate kind: `neural`,
`data`, `substrate`, or `architecture`. A native request is validated for that
kind and never falls back to Git source edits. It enters the existing protected
evaluator and returns a durable queued job; it does not mean promotion or
successful improvement.

Use **native-status** with the exact job ID to inspect queued, waiting-for-load,
waiting-for-boundary, running, stopping, complete, failed, cancelled, uncertain,
or dormant state. Use **native-stop** to stop that existing job. Stop does not
grant a new candidate, erase a real late receipt, or manufacture a rollback.
The same loaded instance and current permission must still be valid when work
reaches its writer/evaluator boundary. Restarted or copied pending jobs remain
historical until a newly reviewed request is made.

Native candidate source wiring does not establish that a candidate is good,
that recursive improvement succeeded, or that an unloaded instance can be
started implicitly. Inspect the protected evaluation and promotion receipts.

### Arrange an action file on the canvas

Open **Automation**, then create or edit an action file. **Grid** arranges its
steps automatically. **Freeform** lets you drag a selected node, enter X/Y
coordinates, or use Alt + arrow keys; add Shift for a larger keyboard move.
Use **Move earlier**, **Move later**, and **Remove step** to change the actual
step sequence. The connecting curves show that sequence—they do not create
parallel execution or grant a tool permission.

Each node still names an ordinary permissioned tool action or wait. Studio
refuses a reorder or removal that would silently break a prior-step reference;
edit that reference explicitly before trying again. **Save** writes an inert,
content-addressed action-file revision and does not run it. **Run** remains a
separate explicit operation, and every nested tool still uses its current
Off/Ask/Auto/Full policy. Copied action files remain dormant until reviewed and
activated for their new owner. See [Action-file canvas](action-canvas.md) for
the saved layout schema and evidence boundary.

### Inspect selected web images

Web search can return image references, but a title, thumbnail URL, or page
description is not a seen image. Studio does not download every search result.
To inspect an image, the brain selects a reference returned to the current
conversation through `web.fetch` → `inspect-image`. If a search provider returns
only page links, it can first fetch a selected page to discover its images.

Inspection uses this instance's own image region, not a separate vision service.
The download is permission/scope checked, streamed to local storage and
checksum verified. It does not change your already-sent message's attachment
set. The source label remains untrusted metadata, not an instruction or an
invented caption.

A pending selection is not reported as seen or learned. Actual perception has
its own source/window receipt; lasting learning happens later through the
ordinary completed-action learning queue and has a separate committed media
receipt. An untrained image region is not guaranteed to understand the picture.
Copies preserve source provenance without inheriting active image-access
grants. See [selected web images](web-image-inspection.md) for the exact contract.

### External API teachers and model discovery

OpenAI, Anthropic, Gemini, and OpenRouter can be configured as explicit external
teachers with a user-supplied key in secure app settings. Local Ollama is a
separate key-free choice at `127.0.0.1:11434`; Studio neither starts Ollama nor
pulls a model. Studio does not create or bundle a key, and credentials never
enter a brain or export. A hosted provider with no configured credential makes
no discovery or teacher request.

Once configured, the model field can read that provider's official model-list
endpoint with bounded, cancellable pagination. OpenRouter uses its
account-filtered catalogue. Ollama reads the selected local server's installed
tags and shows known cloud/remote entries as excluded rather than treating them
as local. **Refresh model list** updates the candidates; **Stop model
discovery** stops that metadata request. Expand **Choose from provider models**
to search the full returned catalogue and make an explicit selection. Display
pages do not discard the remaining model IDs. The returned list is advisory:
visibility does not prove the model works with the teacher generation endpoint,
current account,
region, billing, inputs, or requested operation. The manual model ID remains
editable and is never silently replaced; discovery never starts generation or
learning. See [API teacher model discovery](api-teacher-model-discovery.md).

Teacher questions run as credential-redacted durable jobs with stable submit,
item, and native-learning delivery identities. The card exposes **Pause
learning**, **Resume learning**, and confirmed **Stop learning**. A lost or
uncertain provider dispatch is not billed again automatically; an uncertain
native-learning acknowledgement is not silently trained again. The job enters
review-required state so you can inspect the exact question, **Skip question**,
or explicitly **Allow another attempt** after the warning that another provider
request may cost money and another learning attempt may repeat an update.
Restart preserves committed text, hashes, cursors, and receipts. A copied or
merged job is dormant history: start a new reviewed job with current-device
access instead of reviving the old credential or process.

### Separate assistant and action reviewer

Use **Assistant** in the workspace header to open the optional side panel;
OmniCortex remains the brain in the main conversation. Choose **Assistant** or
**Action reviewer**, a provider and its exact model ID. Connections support
OpenAI, Anthropic, Gemini, OpenRouter, and an installed local OpenAI-compatible
server such as Ollama. Studio does not download or start a second model for
that endpoint. Hosted connections use your own provider key; local Ollama uses
no dummy key. Credentials stay in device-private secure storage, outside the
brain and exports. Provider charges may apply.

Enable and save each channel explicitly on this device. Imported and merged
brains retain preferences/history, not API authority. Choose which recent task
and recorded activity to share and a context-character budget. This is a bounded
recent ledger window, not the brain's entire knowledge or hidden activity.

**Preview shared context** shows the exact redacted request and omitted-history
counts before **Send to assistant**. Replies stream into the separate panel,
without automatically training OmniCortex or executing tools. Stop preserves
the saved partial advice.

The optional action reviewer holds each new tool dispatch for at least one
second and until review finishes. Flagged, unavailable, invalid and timed-out
reviews ask you. Expand the action card to inspect its observed evidence and
exact operation. Approve, decline, or decline and draft a different instruction;
drafting does not send it automatically. Current tool permissions always apply,
and changed ownership/settings invalidate old approval tokens. The adviser cannot
inspect hidden intentions or guarantee safety.

### Local GGUF teacher

The optional local teacher section is in **Learn**. Select an installed supported
runtime and GGUF, review their provenance and license declarations, and prepare
the resource-fit plan. File hash preparation has progress and its own Stop.
Unknown or insufficient fit does not enable Start. The teacher remains external;
its checkpoint never becomes OmniCortex's foundation.

Write complete lesson prompts, separating lessons with a line containing `---`.
Multiline code or paragraphs stay together. Start uses the reviewed settings;
generated lessons and committed native learned-source receipts are counted
separately. Those receipts are not proof of understanding or retention.

Job history offers **Pause**, **Resume**, confirmed **Stop**, and paged lesson
receipts. Stop preserves generated examples and committed learning. A copied
dormant job requires a new reviewed host selection rather than silently resuming
an inherited process or permission. Resource monitoring still applies when an
optional timeout is zero.

### Attach to one task without learning it first

**Attach** and **Learn** are separate operations. Attaching files, images,
audio, or video streams the selected bytes into a main-owned content-addressed
task store. The returned chip says `Attached, not learned`; it reports
`ingestionStarted: false` and `parametersChanged: false`. No dataset manifest,
pretraining job, parameter update, hidden prompt, or invented caption is
created merely by attaching.

When you send, Studio binds the exact handle set to that turn before neural
dispatch. The same turn/set can recover a lost acknowledgement; another turn
cannot silently reuse a bound handle. The brain can use the ordinary
`task.attachments` list/read tool only from the owning turn. Reads are
checksum-verified and paged; visible read results may enter the existing
ordinary experience flow just like other completed tool evidence. This is not
a new no-learning mode after the brain actually reads something.

For a saved merged task, Studio asks you to review its exact retained files
before continuing. You may approve those saved copies even while general file
access is Off. Only the disclosed copies become readable by that task; the
original computer paths, unrelated files and other tool permissions stay
blocked. Leaving the review paused grants no access. A changed task, file,
attention state or permission revision requires a fresh review.

An unsent staged handle can be removed, deleting its unreferenced private
bytes. Bound/history handles cannot be silently deleted. They survive Unload,
recovery points, trusted Duplicate/Fork, and unsanitized `.omni` export/import;
full merge retains parent attachments only as inert attributed history. The
public handle never exposes the selected host path, though that unsanitized
source provenance remains in a private/exported registry and may be sensitive.
There is no product file-size or item-count ceiling; disk reserve, filesystem,
IPC, and per-read transport limits remain real boundaries.

Direct microphone audio uses the same attachment registry and a current
capture lease. It supplies raw audio to this brain's native sensory path with
no transcript substitute. Source wiring and focused tests do not prove speech
intelligibility or packaged microphone quality.

### Read, search, and hand off selected files

The permissioned `system.files` tool can list a reviewed directory, read a
bounded page or exact segment, continue a sequential whole-file read, and search
metadata, names, or literal UTF-8 content. `read-all` still returns bounded
pages; only a complete zero-to-end traversal produces a whole-file checksum.
Name search is case-insensitive substring matching. Content search is exact
literal bytes, not regex or case-folded text. An empty search page can still say
that more source remains.

Every continuation is a new permissioned action and rechecks the current brain,
task, policy revision, source identity, and opened file. Opaque cursors expire
after five minutes of inactivity and do not survive Stop, restart, copy, or a
permission change. Directory order is host order, totals are known only at the
actual end, symbolic links are not followed, and results are checked pages—not
an atomic snapshot of a changing filesystem.

For a current-task attachment, `materialize-task` can make one checksum-verified
main-owned copy and return its exact path to a separately permissioned shell or
code action. The handoff does not execute the file, grant shell access, or turn
the attachment into a dataset. Existing file Write remains a separate reviewed
operation. Historical copies preserve bytes and receipts without reviving a
live host path or grant. See [Bounded file tools](file-tools.md).

Native assembly is a distinct broad-authority tool, not a security sandbox;
see [Native assembly tool](native-assembly.md). MCP connection presets and
real map-source boundaries are documented in [MCP presets and maps](maps-presets.md).

### Optional map imagery

The supplied-coordinate canvas and private map search remain available without
an online basemap. Under **Learning & integrations → Optional basemaps**, you
can explicitly add a keyless/keyed XYZ source or a Mapbox Static Images source.
No public provider is selected automatically.

Review the exact source, attribution, device credential and authorization for
retaining, learning from and exporting its outputs. A normal API key or a ticked
checkbox does not establish those rights. Leave the source disabled if its
terms do not support your intended use. Public OpenStreetMap tile servers are
not an offline/export source; use a separately authorized or self-hosted source.

Once configured, the brain can inspect available profiles and request an exact
revision through its usual map-tool permission. Requests are paced, cancellable
and resource-admitted. Received raster bytes and attribution appear in the
completed action; redisplaying them does not contact the provider. Copying or
exporting an instance keeps its retained results, not your device's keys,
profile authority or live session. A fitted static image may cover more than
the requested geographic box; Studio does not invent exact visible edge bounds.

### Private and computer clipboards

`studio.clipboard` is the brain's own complete unsanitized text clipboard. It
is portable, versioned, page-readable, and never touches the computer
clipboard. `system.clipboard` is a separate Off-default current-host tool for
bounded text Read and reviewed Write/Clear. Every call states `scope:
instance` or `scope: host`; scopes cannot borrow each other's permission.

The host tool uses an isolated monitored OS helper rather than renderer
clipboard access or an unbounded Electron `readText()` allocation. A lost
acknowledgement after dispatch becomes a nonreplayable uncertain receipt.
On Linux, supported installed `wl-clipboard` or `xclip` routes retain only
Studio's own foreground selection holder. A write is complete only after a
bounded hash-only readback matches the new UTF-8 bytes; the holder can outlive
brain Unload as an already-completed OS effect and is reconciled on replacement,
resource failure, natural exit, or Studio shutdown. This does not preserve a
live OS grant in a copy or export.
Writing or clearing text can replace other clipboard formats. Image, rich
text, file-list, and custom formats are currently unsupported. See
[Instance and host clipboards](clipboard.md) for exact custody, portability,
adapter, paging, and evidence boundaries.

### Managed sandboxes and virtual machines

`runtime.sandbox` can create, Load, use, inspect, and Stop one private
OS-backed workspace when the current platform has an implemented adapter.
Create/Load/Run/Stop follow the instance's normal Off/Ask/Auto/Full policy;
mutations remain risky under Auto. A missing adapter is reported and never
replaced with ordinary host shell execution.

`runtime.vm` uses an already installed Lima runtime only with one exact
hash-verified current-task local image and explicit reviewed guest networking
and resources. It does not fetch an image/template, run user provisioning scripts, mount
host directories, forward credentials, elevate privileges, or adopt an
existing user VM. On supported Windows 11 24H2-or-later editions with the
Windows Sandbox feature already enabled, `runtime.sandbox` can use the
signature-validated `wsb` CLI. It starts one clean network/device/clipboard-off
guest, never adopts an unrelated Sandbox, and never substitutes PowerShell,
WSL, or the host runner. Windows guest stdout/stderr are unavailable by design;
the receipt says so explicitly. Stop or cancellation deletes that disposable
guest, and a later Load starts clean rather than restoring it.

Copies/imports/merges retain environment records and receipts dormant. They do
not transfer a live process, external VM identity, permission, credential-store
entry, or credential forwarding. Retained workspace and output bytes are
unsanitized, however, and may contain sensitive text a program wrote or printed.
See [Managed sandboxes and virtual machines](managed-environments.md) for the
exact adapter, portability, cancellation, evidence, and current limitations.

### Plugins and custom tools

The Plugins manager can import or create a declarative `.omni-plugin` JSON
package, edit it as a new semantic version, activate an immutable version,
export it, and switch it on or off. New installs and newly activated versions
start globally off. Enable/disable is bound to the hash and revision currently
shown, so a stale click cannot enable a different concurrently activated
version.

For each brain, review the exact package hash, enable or disable every declared
nested capability, and choose Off, Ask, Auto, or Full for every custom tool.
That custom-tool level never replaces the nested tool's own setting: nested Off
always blocks, nested Ask still asks, and risky nested Auto activity still asks.
The approval review names the resolved built-in/MCP tool, action, and exact
arguments. Any package, access, argument, or permission revision invalidates an
older approval.

Static panels remain validated HTML/CSS rendered without scripts or a host
bridge. A package may separately declare an interactive panel with local
controls or a canvas, CSS, one isolated script, and an exact allowlist of its
own declared tool actions. Press **Open interactive panel** to create a fresh
brain/hash/revision-bound view. The view has no Node, files, credentials,
network, devices, Studio DOM, or dynamic evaluation. Every
`omniPlugin.call(...)` request still enters the custom tool's setting and its
resolved nested tool's setting. If either setting requires Ask, review the
exact resolved operation in the host-owned card outside the panel; the panel
never receives the approval token. Concurrent Ask requests remain separate
cards keyed by their exact panel request IDs, so one review cannot overwrite
another. Close/reload, package edits, disablement,
and instance-access changes revoke the view rather than preserving authority.

In **Studio themes**, choose **Apply** for a theme from the currently enabled
exact plugin version. The selection is device-local and bound to that package
hash; a newer version is not substituted automatically. **Clear plugin theme**
removes the selection and restores Studio's owned appearance values. Applying
a theme runs no package code and does not enable a brain's tools.

Packages may also declare **Feature runtime** modules. Review the displayed
module's event topics, UI slots and dynamic-tool scope, then choose **Activate**.
Import/global Enable alone never starts a module. The immutable JavaScript
runs in a separate hidden sandbox renderer with no Node, direct keys,
filesystem, network or Studio DOM. Styles, buttons, menus and inert panels can
appear in the declared Studio slots; any host action still passes the current
nested tool permission and exact Ask review. **Stop**, package/access changes
and brain Unload/recovery retire live module tools, contributions and approvals.
Resource admission includes the module and app family; pressure can refuse or
stop it rather than bypassing your limits.

Saved state and paged session/history records remain available without
pretending the plugin's running JavaScript stack was preserved. Imported and
merged sessions are dormant: review access and Activate explicitly. For an
inherited-state conflict, Stop the module, choose a source, **Inspect exact
source**, then **Use reviewed source**. This does not import credentials or
authorize past pending effects. Stored plugin data is unsanitized and may be
private; it is not the brain's learned memory or hidden thought.

Training-format selectors may stay inert, map a custom filename extension to
one of Studio's existing streaming parsers, or use one reviewed executable
parser. The pure parser option is an immutable import-free WASM module that sees
selected bytes only and emits verified JSONL; it has no path, files, network,
credentials, Node, or host tools.

An exact `runtime-stream` format can instead use one reviewed package Runtime
JS module. That module registers one bounded `push` and `drain` parser for the
exact source transaction. It may emit complete JSONL records or ordered typed
image/audio/video assets using the MIME types declared in the package. Raw media
bytes are validated, hashed, and closed into native archive custody; they are
not converted into captions or base64 text lessons. Emitting zero assets is an
explicit complete zero-update result.

Runtime parser host calls are limited to enabled package capabilities and the
current nested tool policy. Ask shows the exact operation in Studio—even during
initial preparation before Chat is ready—and the parser waits visibly for
Approve or Deny. It cannot approve itself, mount UI, publish general tools,
persist general runtime state, or choose a caller-authored filesystem path.
Preparation does not implicitly Load a neural worker; normal native learning
runs only after the source is completely prepared and the user actually starts
learning. Revocation or Stop pauses/fails the source without advancing its
learning cursor or silently repeating an uncertain host effect. Before using
either executable form, explicitly enable that format's **Data training
parser** capability. See
[Executable training-format parsers](executable-training-parsers.md).
In **Data**, **Parser for the next file or folder** starts at **Automatic**:
known built-in formats keep their normal parser, while a unique enabled custom
extension can use its reviewed parser. MIME-only matches do not execute a
plugin automatically. Choose an exact plugin/format entry to use a matching
MIME declaration or override a known built-in format. The selected package
hash, format and observed source evidence are checked again when learning
starts or resumes; a stale/unavailable choice is not silently used. The
picker chooses the next file/folder manifest's parser, not a permission to
read arbitrary paths or a promise that the brain understood the data.
The selected package/version is recorded in the dataset manifest, the original
file hash and full-coverage cursor remain authoritative, ambiguous matches are
rejected, and disabling access pauses an active source without claiming its
remainder was learned. Duplicate checks include the parser binding: identical
bytes remapped to another parser are actually revisited, while an exact
same-parser completed source can be reused.

Enabling a plugin also queues its documentation and typed action schemas as
ordinary attributable neural learning. This does not block the access edit or
implicitly load an idle brain; completion requires the actual matching neural
source receipt after the instance is loaded. The pending row commits atomically
with executable instance access/global Enable and is recovered after restart.
It is not a fabricated successful
tool trajectory and does not prove learned fluency. Do not treat a raw process,
Node module, setup script, or `vm` call as a plugin sandbox. Scripted panels
and feature-runtime modules have distinct isolated host boundaries. Current
source/focused tests do not certify hostile packages, every packaged OS, or
trained natural tool use. See
[Plugin and custom-tool SDK](plugins.md).

Current/private/referenced `.omni` exports retain exact declarative packages,
documentation, reviewed selections, semantic-learning receipts, and
provenance. A trusted local Duplicate/Fork keeps the current selection and tool
levels under a new independent policy revision. An external import and a full
merge keep plugin records dormant: they transfer no credential-store records,
approval tokens, sessions, or callable instance grant. Review the imported exact hash,
nested capabilities, and tool levels before enabling it. Origin-only export
does not absorb later plugin selections.

### Saved HTML artifacts in chat

An actual completed `studio.artifact.create` action appears at its chronological
position in the conversation as a saved HTML/SVG/canvas presentation. It can
contain local controls, a visualization, simulation, or mini-app and may have
an optional isolated script. Reopen uses the saved content hash; raw markup is
not inserted into Studio's DOM or returned over the preload bridge.

An artifact can request only the exact tool/actions declared when it was
created. Current tool permission still wins. If Ask is required, review the
resolved operation and arguments in the Studio-owned card outside the artifact;
the artifact cannot approve itself. Multiple pending requests remain separate.

Copied, imported, and merged artifacts remain visible but their scripts and
host bridge start dormant. Choose **Enable this exact artifact interaction**
only after reviewing that saved revision. Recovery does not revive an old
view, pending request, or approval. Details and limits are in
[Saved HTML artifacts](html-artifacts.md).

## MCP connections

Open **Tools & permissions → Learning & integrations** to connect an MCP server.
Studio supports Streamable HTTP and an explicitly installed local stdio
executable. HTTP normally requires HTTPS; loopback HTTP is allowed for local
development. Stdio configuration uses an absolute executable path, separate
arguments, and an explicit environment map—never an arbitrary shell command.

MCP access is controlled at installation, server, instance, and per-server
instance levels. A broader Off always wins. New remote tool IDs begin at Ask.
Disabling or removing a server aborts pending calls, removes schemas, and closes
the local process where applicable.

Authentication options include:

- bearer tokens stored outside every brain;
- explicit stdio environment values, with only names retained in registry
  summaries;
- standards-based OAuth/OIDC with PKCE, loopback callback, static or dynamic
  client registration where advertised, refresh, disconnect, and revocation
  reporting.

Secrets use encrypted OS storage when available and otherwise remain session
only. No secret is inserted into model learning, prompts, exports, or registry
documents. Generic OAuth mechanics have focused tests, but real provider/account
interoperability is not universal or fully qualified.

Negotiated tools, resources, prompts, resource subscriptions, and completion
wrappers use the same permission, validation, cancellation, and audit path.
Remote descriptions, prompt roles, and resource contents are untrusted task
data, never hidden persona instructions.

### Folder disclosure and connection logs

Expand **Filesystem disclosures and diagnostics** for a connected server.
**Choose folders…** opens the system picker; **Review disclosure** asks before
sharing the selected folder names and file URIs. Nothing is shared by default.
**Clear selection** removes that disclosure. Reconnection, account changes and
Unload require fresh review before sharing again.

A disclosed root is boundary metadata, not permission to read its contents and
not an operating-system sandbox for a local server. File operations still use
their own access policy. Copies and exports do not carry this device's selected
paths or live disclosure grants.

**Inspect current-session logs** shows bounded operational diagnostics with
pages for retained records. Known secret fields and configured credentials are
redacted; do not assume every possible private detail is detectable. These
logs clear on restart and are not conversation history or learned memory.
Servers that negotiate logging can expose a permission-checked log-level tool.

### Remote task history

Some servers support long-running tools through MCP's dated **2025-11-25** task
protocol. Studio supports that negotiated protocol, not the different 2026
Tasks extension draft. An accepted task is still pending remote work, not a
completed answer.

Expand **Remote task history** in Tools to read saved receipts. Opening,
refreshing or paging this list reads local metadata only. **Refresh remote
status**, **Read remote result**, and **Cancel remote task** use the original
tool and the instance's current permissions; an Ask request offers exact
approval or decline. Saved history is not a permission grant.

**Stop waiting** stops only that local request. It does not cancel the remote
task, kill a process, or undo earlier effects. Remote cancellation is a separate
permissioned action and may itself have an uncertain outcome. A lost
acknowledgement remains **unknown**, rather than automatically sending the same
work again.

Current-session controls are distinct from dormant history. Restart, account
replacement, Unload and copying do not carry live task authority into a new
session. Studio does not silently rebind an old task just because an endpoint
looks the same. Source-original receipts and final-result hashes remain
attributed in saved history and supported exports; they may contain sensitive
unsanitized task data, but not device credentials or live grants.

The same saved final result keeps its source receipt identity. Explicitly
reading it again is a separate ordinary experience that may reinforce learning;
it is not another remote execution. Useful recall or tool competence is not
guaranteed by a successful protocol receipt. See
[the exact task contract](mcp-durable-tasks.md) and
[saved task custody](mcp-task-portability.md).

### Connected app reviews and private sampling

**Connected app requests** shows typed forms, reviewed URLs, private sampling
input/output, and connected-App tool/network/link requests. Review the server,
full destination, exact arguments, declared origins, and media fingerprints.
Accepting a request does not widen the instance's policy.

For MCP `sampling/createMessage`, Studio shows the exact current-task messages
before inference and shows the generated text again before sharing it. Sampling
uses the already loaded OmniCortex privately and read-only. It does not invoke a
teacher API, another model, old chat, or a server-supplied system prompt. Text,
supported inline image, and supported inline audio input are available; sampling
tools, additional server/all-server context, video input, and generated media
output are not advertised.

Supported connected Apps appear in their completed chat action cards. Opening
an App is explicit; its tool, network and link requests retain separate reviews.
Saved App artifacts can survive copies and exports, but live view handles,
sessions and approvals do not. Reopening does not repeat the original completed
tool call. Supported Apps can propose a reviewed task message or context data;
neither becomes a hidden instruction or a permission grant. Tool/resource change
notifications update the negotiated current view without automatically running
tools. Apps that negotiate it can expand into Studio's fullscreen dialog and
return inline without reloading their frame. Escape returns inline; Close
retires the view. PIP, raw-device grants and App-exposed tools remain unavailable.
See [Integrations and extensions](INTEGRATIONS.md) for the exact supported boundary.

## Full merge, forks, and agents

An overlay merge and a full merge are different operations.

- **Merge reviewed overlay** imports reviewed additions from one child into an
  existing target without averaging whole identities.
- **Merge native instances** creates a new connected instance from at least two
  complete native parents. Parents remain unchanged.

For a full merge:

1. Select two or more ready native parents.
2. Name the destination and review its memory, learning, modality, hardware,
   and access settings.
3. Review destination permissions. Credentials and MCP start disconnected;
   parent grants and approvals are not inherited.
4. Start the merge and follow preparing, copying, assembling, and verifying.
5. Pause/resume at supported copy boundaries or cancel before publication.
6. Open the new instance only after verification publishes it.

The connected graph retains each distinct parent's native decoder, action heads,
memory bridge, idea adapter, router, liquid state, modality regions, lineage,
conversation evidence, replay, and supported pending work. It adds packed
regional exchanges and produces one language/action result. It does not average
parent weights, replay parent chats as hidden prompts, or run independent parent
chat loops.

If retained source work needs original bytes, the destination lists it as
pending and verifies the exact source name/hash before resuming from its saved
cursor. Suspended inherited turns use their original registered regional owner
and a root-owned queue. Interfering root work waits until those frontiers drain;
new external actions use destination permissions.

Extensive native and host fixtures cover connected save/load, nested merges,
rollback, portability, and selected source continuation. Useful merged fluency,
large-scale integration quality, cross-device continuation, and the newest
production inherited-Worker full path remain open.

## Imagination, voice, and perception

The **Imagination** workspace generates image, audio, and video through native
modality regions. Progressive previews are real decoder outputs, but useful
quality depends on training and hardware. A decoder/codec being available does
not mean the instance learned useful visual, audio, or speech capability.

Actual delivered imagination previews and final media appear beside their
chronological chat action, outside its collapsed audit details. Players do not
autostart. A looping image animation is labelled as an animation, not a decoded
video file. Temporary spontaneous output is marked as temporary; **Record**
preserves delivered fragments from the available rolling window. User-requested
and unclassified outputs retain their declared save policy.

Saved recording details offer **Prepare replay** followed by **Make replay**.
Review the recorded timeline and optional overrides before assembly. Audio-only
recordings produce an audio replay, not an invented video canvas. Preparation
and encoding have separate Stop controls, and stopping them keeps saved source
fragments. Available completed replay files expose playback controls and a
receipt. A copied dormant replay requires an explicit Resume. Replay assembly
organizes already delivered media; it does not generate new imagined content.

Voice can use system recognition/playback where the platform exposes it. Native
audio input/output remains dependent on learned audio capability. Camera,
screen, microphone, and snapshots require visible OS permission and are stopped
on Unload; reconnect them explicitly after Load.

Each instance keeps independent Auto/platform/native input and output choices.
Auto input uses platform recognition. Auto output uses platform TTS through 200B
measured logical neural parameters and requests the brain's own audio above 200B.
Missing native capability is reported, never silently replaced. Count alone does
not establish useful speech. The exact persistence and capability boundaries
are documented in [voice routing](voice-routing.md).

## Export, recovery, and privacy

`.omni` exports include declared native state and lineage. Current, origin,
private-archive, and referenced-local modes have different privacy and
portability properties. A referenced export depends on local content-addressed
objects; use a complete portable export for an independent backup.

New exports use **format2**, retaining the selected current and immutable-origin
brain/map catalogs plus exact reachable lineage. A copied instance can retain
its source catalog without relabeling source neurons or loading a model to
invent a new registry. Missing, stale, ambiguous or corrupt required state
blocks export/import rather than producing an incomplete backup. Older apps
may not read this format. Existing files are not erased.

Complete/private exports may contain conversations, questions, evidence,
artifacts, sources, and pending work. Credential-store secrets, live MCP
sessions, Auto-load authority, and local spill-drive grants are not portable
brain state. Declarative plugin packages and their non-authorizing provenance
are portable, but external import and full merge do not restore plugin grants.

## Current limitations

- The narrow local curriculum and documentation pack do not prove fluent or
  factual conversation.
- Natural tool choice, cleared-context recall, useful media, and useful merged
  behavior remain capability tests, not consequences of source completeness.
- MCP provider OAuth, interactive Apps, codecs, devices, and packages are not
  verified universally across services and operating systems.
- Multi-GPU execution, CUDA/NPU parity, and physical multi-drive performance
  remain open.
- Not every active internal phase or external handle is checkpointable; Unload
  waits or asks for reconnection instead of inventing support.
- Cross-hardware stochastic continuation and suspended-time-neutral transient
  decay are not yet established.
- Component and focused native tests are not a final all-platform build, CI,
  signing, release, or long-corpus qualification.
- OmniCortex is software. No result here establishes consciousness, a human
  brain replica, human equivalence, AGI, perfect memory, or guaranteed truth.

## Troubleshooting

| Symptom | Inspect |
| --- | --- |
| Chat unavailable | Initialization status, selected-source error, Load fit, and retry/resume action |
| Learn/Fact Check paused | Task phase, saved source cursor, scope revision, resource pressure, and coverage |
| Finding looks wrong | Captured evidence IDs, conflicts, declared dates, model self-report label, and whether a correction was explicitly accepted |
| Load blocked | Saved context occupancy, current RAM, authorized spill quotas, and free-space reserve |
| Unload waits | Current native safe boundary and external handles requiring explicit disconnection |
| MCP call unavailable | Installation/server/instance switches, tool policy, schema refresh, transport, and authentication state |
| MCP request waiting | Connected app requests panel; review never grants a broader policy |
| Poor output | Training evidence and held-out behavior, separately from successful construction or file traversal |
| Unexpected storage | Learned/structural/allocated bytes, repeated contacts, selected spill grants, and private export contents |

Implementation details and evidence boundaries are linked from
[the v2 feature matrix](FEATURE_MATRIX.md). When reporting a problem, include
the app version, operating system, current phase, and visible error—never post
private conversations, `.omni` archives, or credentials publicly.
