# Third-party notices

Omni AGI Studio is an original research implementation informed by the projects and papers below. Third-party source directories already present in this repository retain their original licenses; the project-level PolyForm license does not replace those terms.

## Bundled source snapshots

| Component | Location | License | Use in Omni |
| --- | --- | --- | --- |
| Microsoft BitNet | `BitNet-main/` | MIT | Research reference for ternary inference and quantization. The uploaded snapshot is missing its `llama.cpp` submodule and is not executed by the custom engine. |
| snnTorch | `snntorch-master/` | MIT | Research reference for spiking neuron dynamics and spike encoding. |
| Neural Circuit Policies (NCPS) | `ncps-master/` | Apache-2.0 | Research reference for CfC/LTC cells and sparse circuit wiring. |

The full license texts ship with packaged applications under `resources/licenses/`.

## Runtime dependencies

Electron, React, Vite, TypeScript, Python, PyTorch, NumPy, safetensors, and PDF parsing dependencies retain their respective upstream licenses. Exact dependency versions are recorded in `package-lock.json` and `engine/requirements.txt`.

### imageio-ffmpeg and FFmpeg

Omni's local MP4 paths use `imageio-ffmpeg` 0.6.0, whose Python wrapper is
licensed under the BSD 2-Clause License. Its platform wheels carry a separate
FFmpeg executable. The executable reported by the pinned package is FFmpeg 7.1
built with `--enable-gpl` and `libx264`, so it is treated as a separate
GPL-2.0-or-later program rather than relicensed as part of Omni.

- Wrapper and binary provenance: <https://github.com/imageio/imageio-ffmpeg/tree/v0.6.0>
- Wrapper license: <https://github.com/imageio/imageio-ffmpeg/blob/v0.6.0/LICENSE>
- FFmpeg 7.1 source: <https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz>
- FFmpeg license terms: <https://github.com/FFmpeg/FFmpeg/blob/n7.1/LICENSE.md>
- FFmpeg build and external-library licensing notes: <https://ffmpeg.org/general.html>
- GNU GPL version 2 text: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt>

The packaged application includes this notice under `resources/licenses/`.
Anyone redistributing a release must also satisfy the corresponding-source and
license-copy obligations for the exact FFmpeg build and its linked GPL
libraries. These links document the known boundary; they are not a substitute
for a downstream distributor's compliance review.

## Research inspirations

The architecture is informed by BitNet b1.58, liquid time-constant networks, closed-form continuous-time networks, vector-symbolic architectures/hyperdimensional computing, spike-timing-dependent plasticity, adaptive computation time, elastic/synaptic consolidation, experience replay, sparse growable experts, diffusion transformers, latent video diffusion, and neural audio codecs. See `RESEARCH.md` for the source-to-feature and license-boundary ledger.

No third-party pretrained model weights are bundled. Imported datasets and model packs require their own provenance and license manifests.
