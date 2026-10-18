# Direct packed native kernels

Status: native projection/convolution dispatch and pooled-contact integration
are implemented and component-tested. This is not whole-model performance or
release acceptance.

## Contract

- `packed_native_kernel.py` accepts contiguous integer activations, packed native
  ternary signs, and optional independent uint8 strength/plasticity bytes.
- CPU C++ reads each packed sign into registers and accumulates weighted products
  exactly in int64. It does not expand a learned weight matrix or use a floating
  master. Input gradients use floating activity with the same discrete weights.
- Metal reads the same packed representation. A 128-thread group reduces each
  dot product as two uint32 limbs, retaining exact signed int64 accumulation on
  GPUs without native int64 arithmetic. Input-gradient activity is float32.
- An unavailable native backend is not labeled a working fused backend.
  Unsupported accelerator calls use admitted raw-packed CPU tiles; they do not
  restore decoded floating learned weight matrices.
- Runtime verification rejects floating learned parameters even if their
  gradients are disabled. A frozen custom projection cannot bypass the native
  representation check by avoiding standard Linear/Conv/Embedding classes.
- The package builder compiles the small CPU library ahead of packaging and
  includes its integrity manifest and shader sources. Loading a brain does not
  invoke a C++ compiler or download binaries.

## Evidence, 2026-10-02

- CPU native library built on macOS ARM64 using the installed C++ compiler.
- Five CPU primitive tests passed: exact odd-width/strength arithmetic,
  unit-strength equivalence, input gradient, invalid buffer/code rejection,
  empty batch.
- Two real Metal primitive tests passed: exact positive/negative results plus
  gradients; a 1,100,000-input contraction exceeding signed int32 range.
- The first sequential Metal implementation was subsequently replaced with a
  parallel integer reduction. An isolated 64×4096 weighted contraction matched
  CPU exactly. Its median local call time fell from 1.609ms to 0.317ms across
  ten groups of ten calls. CPU median was 0.221ms. These are one primitive's
  local timings, excluding neural decoding, data transfers, and model load.
- The changed parallel Metal unittest rerun initially encountered a temporary
  import error while the pooled-contact rewrite was changing `model.py`. After
  that import stabilized, all seven CPU/Metal primitive tests passed again in
  0.806 seconds. Integrated projection tests remain separate.
- Neural worker health now requires the bundled CPU packed library and reports
  actual CPU/Metal backend availability; it does not claim a CUDA kernel exists.
- Native linear and convolution dispatch now consumes these kernels, including
  weighted repeated contacts. The weighted-substrate lane's 96 affected tests
  include forward/backward with expanded-weight helpers patched to fail.
- A small real native chat fixture passed: normal talk streams during its first
  decode, calls generation once, and does not privately rescore/redecode. This
  establishes sequencing, not fluency or large-model speed.

## Remaining

An October 7 audit found two auxiliary readers bypassing the weighted native
path. Sensory assembly admission now uses the direct tied transpose operation,
including strengths and repeated contacts, with a measured regression. Media
codebook table/scoring correction now has eighteen affected cases, including
actual Metal and a tiny image/audio/video forward/backward/generate/reload
integration; see [weighted modalities](weighted-modalities.md). Three CPU
sensory cases passed in 0.037 seconds; one actual native Metal case passed in
0.396 seconds. Stored weights stayed unchanged during perception. Core
projection tests alone did not cover these manual readers.

- Check sustained throughput and memory across representative batch/row sizes;
  primitive correctness and the isolated timings do not establish this.
- Extend native accelerator coverage where supported, retain accurate backend
  reporting and measured CPU/accelerator placement decisions.
- Verify packaged binaries on Windows x64/x64-worker ARM64, both macOS
  architectures, and Linux architectures. Windows compiler environment setup
  still requires CI verification.
- Prove all snapshot/export/pager/rollback/collective paths preserve native
  weighted state; the primitive tests alone do not prove that contract.
