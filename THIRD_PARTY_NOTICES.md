# Third-party notices

Omni AGI Studio is an original research implementation informed by the projects and papers below. The application does not import, link, or execute these upstream source trees. An ignored local checkout such as `.runtime/bitnet-src` may be preserved for developer research, but it is not a runtime dependency, Build/Import choice, catalog asset, or packaged release input. Preserved license texts ship in `licenses/`; the project-level PolyForm license does not replace any third-party terms.

## Research source licenses

| Component | Upstream | Preserved license | Use in Omni |
| --- | --- | --- | --- |
| Microsoft BitNet | <https://github.com/microsoft/BitNet> | `licenses/BitNet-MIT.txt` | Research reference for ternary inference and quantization; Omni uses its own implementation. |
| snnTorch | <https://github.com/jeshraghian/snntorch> | `licenses/snnTorch-MIT.txt` | Research reference for spiking neuron dynamics and spike encoding; Omni uses its own tested STDP implementation. |
| Neural Circuit Policies (NCPS) | <https://github.com/mlech26l/ncps> | `licenses/NCPS-Apache-2.0.txt` | Research reference for CfC/LTC cells and sparse circuit wiring; Omni uses its own implementation. |

The full license texts ship with packaged applications under `resources/licenses/`.

## Runtime dependencies

Electron, React, Vite, TypeScript, Ajv (MIT; local JSON Schema validation),
Python, PyTorch, NumPy, safetensors, JSON Schema, and PDF parsing dependencies retain their
respective upstream licenses. Exact dependency versions are recorded in
`package-lock.json`, `engine/requirements.txt`, and the target-specific,
hash-locked worker closures in `engine/locks/`.

### Native JSON Schema validation

The Python worker uses these MIT-licensed packages to validate structured
action/tool schemas. Frozen-worker builds explicitly collect their modules,
the bundled JSON Schema specification documents, and the target-specific
`rpds` extension. These are validation libraries, not model weights or an
external AI backend. No optional `jsonschema` format extras are installed by
the reviewed engine locks.

| Package | Reviewed release and source | Preserved license |
| --- | --- | --- |
| attrs | [26.1.0](https://pypi.org/project/attrs/26.1.0/) | `licenses/attrs-MIT.txt` |
| jsonschema | [4.26.0](https://pypi.org/project/jsonschema/4.26.0/) | `licenses/jsonschema-MIT.txt` |
| jsonschema-specifications | [2025.9.1](https://pypi.org/project/jsonschema-specifications/2025.9.1/) | `licenses/jsonschema-specifications-MIT.txt` |
| referencing | [0.37.0](https://pypi.org/project/referencing/0.37.0/) | `licenses/referencing-MIT.txt` |
| rpds-py | [0.30.0](https://pypi.org/project/rpds-py/0.30.0/) | `licenses/rpds-py-MIT.txt` |

The four pure-Python packages use exact universal-wheel SHA-256 hashes.
`rpds-py` uses a reviewed CPython 3.11 wheel hash for each supported worker
architecture; Windows ARM64 packaging continues to use its explicitly labeled
emulated x64 worker. The installed upstream license texts above are preserved
verbatim and flow into the existing desktop/mobile legal payload verifiers.

The Android companion includes Kotlin and AndroidX runtime components under
the Apache License 2.0. The iOS companion uses platform Swift and Apple SDK
components under their applicable platform terms. Mobile applications embed
this notice, the Omni license and Required Notice, and every preserved license
file from `licenses/` inside the signed application container.

### Historical Falcon-E evaluation record

Earlier research evaluated pinned Falcon-E-1B-Base and Falcon-E-3B-Base
artifacts. Current Studio does not download, load, convert, package, catalog, or
offer those weights in Build or Import. A historical Falcon-backed brain,
including the saved Nova experiment, is a rejected legacy artifact and is not
evidence for current native OmniCortex construction or capability. Existing
user files are not deleted by this compatibility removal.

The links are retained for attribution and for interpreting old research
records. Any separate possession or redistribution of those weights remains
subject to the Falcon LLM License. Their published model cards did not establish
complete training-stage, corpus, dataset-license, or preference-training
provenance, so historical notes must keep those facts unknown rather than
claiming they were absent.

- Models: <https://huggingface.co/tiiuae/Falcon-E-1B-Base> and <https://huggingface.co/tiiuae/Falcon-E-3B-Base>
- License terms: <https://falconllm.tii.ae/falcon-terms-and-conditions.html>

### Apache Arrow / PyArrow

Streaming Parquet and Arrow IPC ingestion uses PyArrow, the Python bindings for
Apache Arrow, under the Apache License 2.0. Omni does not bundle Arrow datasets
or assign licenses to user-provided data.

- Project and source: <https://arrow.apache.org/>
- License: <https://github.com/apache/arrow/blob/main/LICENSE.txt>

### ijson

Large JSON arrays and maps are traversed incrementally with ijson under its
BSD-3-Clause license.

- Project: <https://github.com/ICRAR/ijson>
- License: <https://github.com/ICRAR/ijson/blob/master/LICENSE.txt>

### imageio-ffmpeg and FFmpeg

Omni's local MP4 paths use `imageio-ffmpeg` 0.6.0, whose Python wrapper is
licensed under the BSD 2-Clause License. Upstream platform wheels carry a
separate FFmpeg executable, and the executable reported by the pinned package
is FFmpeg 7.1 built with `--enable-gpl` and `libx264`; such builds are
GPL-2.0-or-later programs rather than BSD-licensed wrapper code.

- Wrapper and binary provenance: <https://github.com/imageio/imageio-ffmpeg/tree/v0.6.0>
- Wrapper license: <https://github.com/imageio/imageio-ffmpeg/blob/v0.6.0/LICENSE>
- FFmpeg 7.1 source: <https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz>
- FFmpeg license terms: <https://github.com/FFmpeg/FFmpeg/blob/n7.1/LICENSE.md>
- FFmpeg build and external-library licensing notes: <https://ffmpeg.org/general.html>
- GNU GPL version 2 text: `licenses/GPL-2.0.txt`
- Machine-readable distribution policy: `licenses/ffmpeg-runtime-policy.json`

Official Omni release packaging deliberately removes the wheel-provided FFmpeg
executable before the neural worker is embedded. The BSD-licensed Python
wrapper remains and may use an FFmpeg executable explicitly selected through
`IMAGEIO_FFMPEG_EXE` or already installed on the host. When none is available,
the same brain retains its built-in APNG video or PCM WAV fallback. The user
has chosen automatic video-runtime setup; source now provides a cancellable,
hash-pinned on-demand installer. The checked-in catalog now binds separately
published FFmpeg 9.0.2 and x264 builds for all six desktop targets, with exact
executable/corresponding-source/build/license hashes. It does not install an unreviewed wheel binary
or execute downloaded setup scripts. Therefore the current official artifact
still conveys no FFmpeg executable and has no bundled FFmpeg corresponding-
source payload to match.

The package verifier fails if an FFmpeg executable is found or if the embedded
machine-readable policy and notices are absent. A downstream distributor that
changes this design to bundle FFmpeg must also change that policy, bind the
exact binary hash, and provide the complete corresponding source and build and
installation material for FFmpeg and every linked non-system library. Merely
linking to upstream source is not treated as satisfying that requirement.

For automatic provisioning the same exact-binary and complete-source review
is required. Source/build material and license notices must be downloaded and
verified before the standalone executable is selected, and remain alongside
it in the app-owned cache. The declarative catalog ships as
`licenses/ffmpeg-runtime-manifest.json`. The independent runtime and matching
sources are available at <https://github.com/Eabusham2/Omni-AGI/releases/tag/omni-video-9.0.2-r1>.
These GPL components communicate with Omni through ordinary standalone CLI
media formats and retain their own license rights. Technical source/dependency
review and six native codec round-trip checks are not legal advice, patent
clearance or proof of AI-generated media quality. See
`docs/VIDEO_RUNTIME_PROVISIONING.md` for provenance and remaining live integration evidence.

## Research inspirations

The architecture is informed by LLaMA-style decoder components, BitNet b1.58,
liquid time-constant networks, closed-form continuous-time networks,
vector-symbolic architectures/hyperdimensional computing, Perceiver-style
global latent integration, spike-timing-dependent plasticity, adaptive
computation time, elastic/synaptic consolidation, dynamically expandable
networks, sparse growable experts, Darwin-Gödel-style evolutionary lineage,
diffusion transformers, latent video diffusion, and neural audio codecs. See
`RESEARCH.md` for the source-to-feature and license-boundary ledger.

No third-party pretrained model weights are bundled or accepted as an
OmniCortex brain. Imported datasets and modality-only packs require their own
provenance and license manifests; whole-brain import requires verified native
ground-up OmniCortex provenance and the exact supported schema.
