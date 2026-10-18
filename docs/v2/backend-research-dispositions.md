# Native backend compatibility decisions

October18 source and primary-documentation review for R124, R197, R201 and
R218. These are conditional research decisions, not new defaults, installed
integrations or measurements of a trained brain. No framework was downloaded,
no brain was constructed, and no hardware benchmark or old test was rerun.

## Findings and decisions

### MLX and Metal

MLX exposes custom Metal kernels and custom derivatives; its documented
contiguous-input option can introduce copies. It is a possible implementation
interface, not automatic support for Omni's packed contacts.
[MLX custom kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

Omni already dispatches its own `weighted_ternary.metal` through
`packed_native_kernel._metal_library`, including packed signs and integer
connection state. **Retain the existing direct Metal backend.** A second MLX
execution adapter is deferred, not declared impossible or faster: it would
need equivalent packed readers/updates, explicit array and stream ownership,
bounded copies, persistence and a paired whole-operation comparison. Switching
frameworks solely because it also targets Metal does not establish a benefit.

### vLLM

vLLM supports registered out-of-tree models; multimodal models must implement
its corresponding interface. Registration alone is not an execution adapter.
[vLLM model registration](https://docs.vllm.ai/en/stable/contributing/model/registration/)

**Do not substitute a supported foreign model or advertise a vLLM switch.**
Omni's mutable packed owners, recurrent workspace, same-brain action/media
outputs and execution-cursor custody currently use its own Worker and decoder.
A compatible adapter remains a candidate, requiring those contracts and exact
sampler/cache/learning semantics before a measured serving comparison. The
source review does not prove that such an adapter cannot be built.

### DeepSpeed

The documented stock ZeRO path partitions optimizer, derivative and parameter
state, with floating-precision training configurations and CPU/NVMe offload.
[DeepSpeed configuration](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)

Omni's authoritative contact buffers change through discrete native updates,
not an Adam weight master. `distributed_training._resolve_strategy` selects
its existing `PackedCollectiveController`, whose canonical update/broadcast,
rollback and checkpoint protocol must remain intact. **Reject drop-in stock
ZeRO/optimizer conversion for this representation.** A custom transport/offload
adapter is deferred pending packed-state and exact-coverage comparison; this
is not a claim that every DeepSpeed extension requires floating weights.

### NPU and other direct accelerator kernels

PyTorch's backend integration requires operator and, where necessary,
autograd registrations. Renaming or discovering a backend does not implement
those kernels.
[PyTorch backend integration](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)

Omni's packed kernel currently accepts CPU/Metal. Topology discovery explicitly
reports other direct packed backends unsupported or unverified. **Do not enable
an NPU/XPU by its device label.** A candidate needs the actual packed operators,
derivative/update path, transfer accounting and end-to-end evidence on that
device. Available accelerator storage is a separate capability from native
packed arithmetic; retained-context work must not conflate them.

## Comparison gate retained

These engineering dispositions follow the approved investigate/if-beneficial
condition. They neither cancel requested usable RAM/VRAM context capacity nor
require every named framework to replace a working native backend. A positive
adoption still requires a fixed representative baseline, output/update and
retention agreement, current memory/transfer costs, full dataset coverage,
Stop/recovery checks and measured whole-path improvement. No speedup or
all-hardware support follows from this documentation review. The deferred
training-folder/learned-quality task remains separate.
