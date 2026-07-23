# OmniCortex research ledger

This ledger maps research ideas to concrete, independently testable parts of OmniCortex. Biological terms describe engineering analogies, not evidence of biological equivalence or consciousness.

| Source | Adopted idea | Omni implementation boundary | Code/license boundary |
| --- | --- | --- | --- |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764) and [Microsoft BitNet](https://github.com/microsoft/BitNet) | Effective linear weights constrained to `{-1, 0, +1}` with higher-precision training state | `BitLinear` uses a straight-through estimator. Embeddings, activations, optimizer state, SNN traces, and liquid state are not falsely described as 1.58-bit. | The vendored Microsoft snapshot remains MIT. Omni's trainer is an independent implementation. |
| [snnTorch](https://github.com/jeshraghian/snntorch) | Leaky integrate-and-fire state, recurrent spikes, surrogate-friendly dynamics | A small local spiking router controls concept salience. STDP is independently implemented and tested because the vendored learner has reset/API defects. | The vendored source remains MIT. The runtime does not import or redistribute its Python package. |
| [Liquid Time-Constant Networks](https://arxiv.org/abs/2006.04439) and [NCPS](https://github.com/mlech26l/ncps) | Input-dependent continuous-time state and sparse wiring | CfC is the default temporal controller; an LTC-like cell is available for research profiles. | The vendored NCPS source remains Apache-2.0. Omni's compact cells are independently written from the published equations. |
| [VSA/HDC survey](https://arxiv.org/abs/2111.06077) and [TorchHD](https://github.com/hyperdimensional-computing/torchhd) | High-dimensional binding, bundling, permutation, and approximate recall | Idea and concept fingerprints connect semantic units without treating stored text as the model's parameters. | Paper-level ideas only; TorchHD code and weights are not copied or bundled. |
| [Differentiable plasticity](https://arxiv.org/abs/1804.02464) | Fast Hebbian weights layered beside slow learned weights | Per-turn synaptic changes are immediate; slow-model consolidation is separately checkpointed. | Paper-level method, independently implemented. |
| [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) | Spend more recurrent computation on uncertain inputs | Liquid control, novelty, and output entropy determine a bounded ponder count; the trace records the decision. | Paper-level method, independently implemented without copied code. |
| [Elastic Weight Consolidation](https://doi.org/10.1073/pnas.1611835114) and [SYNERgy](https://proceedings.mlr.press/v199/sarfraz22a.html) | Stabilize important parameters while mixing new examples with rehearsal | Candidate training combines replay examples with a metaplastic importance penalty before atomic promotion. | Paper-level methods, independently implemented; no third-party checkpoints. |
| [GROWN](https://arxiv.org/abs/2110.00908) and [Switch Transformers](https://arxiv.org/abs/2101.03961) | Add capacity only after sustained error and sparsely route inputs through experts | Omni grows small ternary residual experts at checkpoint boundaries and logs prototype/routing evidence. | Paper-level architectural ideas, independently implemented. |
| [Diffusion Transformers](https://arxiv.org/abs/2212.09748) | Transformer denoising in a compressed visual latent space | The tiny image imagination pack is a trainable baseline, not a claim of Stable Diffusion quality. | Paper-level architecture; no DiT code or weights are bundled. |
| [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) | Discrete residual-quantized audio latents | The audio pack provides a small trainable codec/token path without bundling AudioCraft weights. | Paper-level architecture; no EnCodec code or weights are bundled. |
| [Latte](https://arxiv.org/abs/2401.03048) | Factorized spatial-temporal latent video processing | The video pack operates on tiny clips and shares liquid temporal state. | Paper-level architecture; no Latte code or weights are bundled. |

## Deliberate departures

- OmniCortex is trained from random initialization or an Omni-format checkpoint. It does not silently call an external LLM.
- Long-term parameter-only memory enters the model as neural state and fast weights, never as hidden retrieved prose.
- Exact recall is only possible when the user explicitly enables an episodic archive.
- Growth adds sparse concepts, synapses, and experts. It does not resize a live dense transformer tensor without a checkpoint boundary.
- Operational traces show seeds, activations, routing, tools, and weight deltas. They are not marketed as faithful private chain-of-thought.
- No RLHF, preference model, behavioral policy prompt, or deception objective is part of the training pipeline.
