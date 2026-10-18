# Native v2 memory inventory

The planner now includes the byte of connection strength/plasticity stored beside each row-packed ternary sign. A strength byte is part of one connection's learned state; it is not an additional parameter or a floating-point master weight.

## Accounting contract

Python `native_core_inventory` and TypeScript `nativeCoreInventory` mirror the same constructor layout.

| Field | Meaning |
| --- | --- |
| `logicalParameters` | Logical connections, including biases. Sparse fresh routers contribute only their actually allocated recurrent contacts: initially zero. |
| `packedWeightBytes` | Ternary sign bytes with the actual per-row padding. |
| `connectionStateBytes` | One unsigned byte per base connection: four strength bits and four resistance bits. Includes every projection, table, and bias. Archived dense-router inventories also reserve the recurrent contact states required during migration. |
| `packedLearnedStateBytes` | Sign bytes plus connection-state bytes. |
| `staticNonweightTensorBytes` | Existing row resistance, gains/rates, router timing/activity, route controls, and liquid state. Learned strength is not hidden in this category. |
| `parallelTopologyControlBytes` | Eight persistent host bytes per projection/table owner, recording its next repeated-contact page identity, including empty pools. Included in static nonweight tensor bytes. |
| `maximumProjectionBytes` | The largest complete packed owner, including its signs, connection state, row controls, scalar gains/rates, and four-byte autograd trigger. |
| `maximumUpdateScratchBytes` | The largest actual matrix-row or bias update, using the shared weighted scratch admission rule. |

The scratch rule admits at most 64 matrix rows at once and charges 256 bytes per padded logical element. This is a conservative transient allowance for gradient/probability/draw arrays, discrete-state and packing intermediates, and saturated-contact coordinate tensors/Python lists. It is not a claim of exact OS RSS. A skinny projection's full bias update is checked separately; tool-schema projections wider than the decoder are included.

Inference reservations use the full packed learned state plus controls. A separate 16-KiB-per-owner Python module/empty-registry allowance is explicitly estimated in sizing and `resources.modelMetadataReserveBytes`; it is charged to host RAM and excluded from checkpoint tensor bytes. Saved contact pages add their page-registry allowance and derived two-byte-per-slot ordering cache. Training adds weighted update scratch. Checkpoint, origin, rollback, growth, and storage-pool plans inherit the applicable costs. Workspace tables include both packed signs and one state byte per connection when separating the workspace from the rest of the cortex.

The live Python training planner uses the same scratch helper and includes bias rows. Distributed training already consumes that live plan. CPU/GPU admission retains separate activation/device checks and actual pager measurements; the scratch estimate does not introduce a second full-sized learned model.

## Descriptor versions and existing native instances

- New main-selected architecture descriptors use `formatVersion: 2` with the additional fields.
- Version 1 descriptors retain their original arithmetic and SHA-256. Loading does not rewrite their immutable origin, shape, or lineage.
- Geometry lineage reconstruction explicitly uses each descriptor's recorded inventory version.
- Runtime planning recomputes the current weighted cost for a saved descriptor where no materialized byte inventory is available.
- For saved safetensors, the planner reads headers only. Existing state bytes are counted once. Missing old-native state receives a conservative unit-strength migration allowance of four state bytes per packed sign byte; this may include up to three padding bytes per row. Missing repeated-contact pool counters reserve eight bytes per owner. Neither weight payloads nor neural models are loaded to make this estimate.
- Invalid, redirected, or incomplete tensor headers produce a planning error instead of silently understating the instance. Header parsing is checked against available host/process memory.

Zero-initialized state bytes encode unit strength and preserve the prior signed values. This migration concerns earlier OmniCortex checkpoints only and does not admit external foundation models.

The mirrored legacy fixture hash remains `7caef23bd630bb834fe0c03bad59472466ebe4814e423060b872f1cb9e9dc7a2`. The corresponding v2 fixture hash is `461e65e7b9289a1a5a3803c44bd6cc29b7ebe170f9f64c02e56caf25ae76128d`.

## Growth and repeated contacts

Depth/expert growth includes every new sign, strength byte, row control, and update scratch allocation. Router growth reserves both complete replacement projections while the original owners still exist. Recurrent copies account for actual allocated blocks, expanded border blocks, and repeat layers; they do not invent a dense population-squared allocation.

Each allocated recurrent block uses row-packed signs plus 11 bytes per contact: one strength/plasticity byte, two eligibility bytes, and two four-byte timing/use controls. Block registry metadata is reserved separately. Growth retains a conservative three-copy allowance for new projection state plus its nonweight transfer allowance; resource checks still run before real allocations and writes.

Fresh inventories contain no invented repeated contacts. Repeated-projection contact pooling is maintained by its live owner/pager: it must report actual allocated page tensors, registry allowance, and physical backing. This planner does not freeze the earlier 8-KiB-per-individual-contact implementation into a new architecture descriptor.

## Verification and remaining allocator work

Focused tests cover mirrored Python/TypeScript v1/v2 counts and hashes, tiny native module constructor counts (under 1 MiB of learned state), exact largest-owner bytes, biased update scratch, header-only migration, fresh resource scaling, saved settings, and growth with sparse repeat layers. They do not train user data or claim neural quality, actual full-model throughput, or a universal hard RSS guarantee.

Recorded checks: both TypeScript projects passed; seven focused Node suites passed 55 checks, then the added pooled-page header case passed with its four-case suite (56 distinct Node cases). The two Python inventory suites passed 13 checks, and the separately run old-descriptor lineage case passed (14 distinct Python cases). Python compilation and the diff whitespace check passed. No full dataset or user-saved brain is constructed by these checks.

Per-device topology, budget selection, current-device admission and Build/runtime selectors now have scoped implementation and tests in [device-resources.md](device-resources.md). Actual multi-GPU placement remains distinct from projected capacity; NPU execution and hardware-specific performance validation also remain open. Live pool and scratch measurements must continue to account for growth that occurs after the initial descriptor was selected.
