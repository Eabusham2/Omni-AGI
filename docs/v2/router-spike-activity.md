# Real LIF/router spike activity

October15 latest: the primary Brain Map is now one unified whole-brain view;
the older three-domain-tab evidence below is a source-point history, not the
current primary navigation. Router observation is a collapsed activity
drill-down that feeds the same canvas without inventing cross-axis mappings.
Contact paging remains reachable when its separate map is hidden. See
whole-brain-inspection.md and whole-brain-map-ui-2026-10-15.md for current
source/UI boundaries and the controls-only affected check.

## Source and meaning

The optional observer reads the exact binary `spikes` computed inside
`LIFPopulation.step`, after its real threshold/reset/count update and before
the differentiable surrogate is added. It does not threshold a VSA activation,
confidence, similarity, weight sign or semantic UUID. With no observer the
step adds only an attribute lookup/conditional; it allocates no capture data.

The current source is specifically `root.router.population`. Channel `i` has
identity `brainId:root.router.population:i`. It is an actual channel of the
loaded LIF population, not evidence that token `i`, an idea UUID or any
particular dynamic/cortical synapse fired. Other connected regional populations
are not silently included. Contact firing and semantic mapping remain explicitly
unobserved until an actual identity-bearing source exists.

## API and ownership

`brain.routerActivity(brainId, RouterSpikeQuery)` uses the existing explicitly
loaded neural worker only. It never starts a worker, loads a brain, enters
neural generation, preempts current work or timeout-restarts the worker.

1. A fresh `captureId`, `enabled: true`, `start` and `count` configure observation
   on the serial writer. This can wait behind current work rather than modifying
   learned state concurrently. Shared RAM admission precedes ring allocation.
2. The reply supplies the actual total channel count, admitted viewport, epoch,
   current sequence and ring capacity. Subsequent polls include the same
   `captureId`, exact `epoch` and `afterSequence`.
3. Polls and matching Stop use the existing metadata-control stdin lane. These
   operations read only the admitted byte ring and checked ownership; they do
   not allocate another observer or run Torch operations on that lane.
4. A matching `enabled: false` control retires only that observer. An old scope
   cannot release a replacement's lease. Loaded brain/storage path, capture
   identity, epoch and sequence are checked before output.

`RouterSpikeActivity` identifies its source as `actual-lif-hard-spikes` and
always reports `semanticNeuronMapping: false` and `synapseFiringObserved: false`.
Each frame has its exact sequence, channel start, 0/1 values, observed wall time
and a monotonic nanosecond string that preserves precision beyond JavaScript's
integer range. Main-process validation rejects different owners, nonbinary
values, wrong epochs/ranges and duplicate/out-of-order continuation frames.

## Bounded storage and lifecycle

The transient ring holds one bit per admitted channel plus a fixed frame
header. The default raw ring budget is 64 KiB; at most 256 channels and 32
frames are transferred per viewport/page. Those are IO/memory envelopes, not a
cap on the LIF population: a requested start can address every valid channel.
The response reports the requested/admitted range, capacity, first/last/next
sequence, dropped frames and frames missed by an old cursor.

Only the admitted range is sliced before a CPU transfer. A retained shared RAM
lease also escrows bounded conversion/JSON scratch and metadata. The allowance
is conservative admission accounting, not a measured whole-process RSS claim.
It remains retained escrow rather than falsely marking future scratch as already
present in an RSS sample. Refusal leaves the previous capture intact.

LIF reset, successful population growth, Unload and brain close retire capture,
discard its transient ring and release the lease. A retired epoch cannot emit
its old frames as current activity; a new capture needs a fresh ID/epoch.
Observer failure retires inspection with an explicit reason without changing
learned bytes or turning a completed neural step into a generation failure.
No capture is serialized into neural checkpoints or treated as learned memory.

## New source/component evidence only

- Eight new non-model event-ring/source-hook cases passed in **36 ms**.
  They cover exact binary/silent observations, bounded bit packing, real drop
  and continuation accounting, epoch rejection, a spy proving transfer of
  only 5 of 1,024 input channels, source-hook placement, resource refusal,
  lifecycle retirement and bytes-only live-control behavior.
- A distinct late-old-owner retirement case passed separately in **<1 ms**;
  the initial eight were not rerun.
- A constructor-free call through the actual Worker control/storage-check
  methods passed separately in **16 ms**. It used a fake already-existing owner
  and rejected a different storage path; it did not initialize the Worker or
  execute a LIF step.
- Four new TypeScript receipt/controller cases passed in **8 ms** (598 ms
  runner): exact channel/owner/epoch/frame validation, explicit unavailable
  state, existing-child control and writer configuration that refuses a
  replaced process without starting another.
- The root lane mounted `RouterSpikeView` as a third domain within the existing
  Brain Map. It never projects these events onto semantic-neuron UUIDs or
  unknown contact links. Observation is off until an explicit click; changing
  owner or leaving the view retires only that capture. Exact ranges page across
  the real population rather than imposing a population cap.
- Three new renderer cases passed in **8 ms**, covering no automatic observer,
  binary silence, exact channel identity/monotonic strings, and cross-owner,
  wrong-epoch, nonbinary or repeated-frame rejection. Computer use with the
  explicitly synthetic host verified no automatic calls, arbitrary channel
  offset 700, waiting without fabricated cells, then an actual-shaped 16-channel
  receipt with only channels 702 and 705 lit. Narrow dark selection preserved
  exact identity. The initial end-exclusive range label was corrected to
  inclusive 700–715 before the accepted screenshot.
  The below-fold epoch, exact monotonic timestamp, allocation and page controls
  were inspected after fixing a fixture-only scroll container. Stop cleared the
  frame; Next moved the offset, and changing owner stayed off without a new
  configure call. Exact-owner cleanup may repeat an idempotent Stop; it does
  not create another observer or replay a neural step. Captures are under
  `test-results/router-spike-ui-manual/`; `03-details-scroll-final.jpg` is the
  accepted lower-page inspection, replacing the incomplete diagnostic `02`.
- Scoped TypeScript compilation of the affected service/IPC/preload dependencies
  and Python compilation of the capture/spiking/brain/worker/transport source
  passed. The CLI source check explicitly included the repository's existing
  Markdown `?raw` declaration after the first invocation omitted it. This was
  check configuration, not a product source defect. Whitespace checks passed.

These tests use small synthetic tensors and fake existing owners. They do not
construct a LIF/brain/model, execute a trained forward, start a process, capture
a device, train data or establish real-model event throughput. Learned behavior
and all-platform qualification remain separate acceptance evidence.
