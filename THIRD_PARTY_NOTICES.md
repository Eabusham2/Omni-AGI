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

## Research inspirations

The architecture is informed by BitNet b1.58, liquid time-constant networks, closed-form continuous-time networks, vector-symbolic architectures/hyperdimensional computing, spike-timing-dependent plasticity, adaptive computation time, elastic/synaptic consolidation, experience replay, sparse growable experts, diffusion transformers, latent video diffusion, and neural audio codecs. See `RESEARCH.md` for the source-to-feature and license-boundary ledger.

No third-party pretrained model weights are bundled. Imported datasets and model packs require their own provenance and license manifests.
