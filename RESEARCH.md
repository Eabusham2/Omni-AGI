# OmniCortex research ledger

This ledger maps research ideas to concrete, independently testable parts of OmniCortex. Biological terms describe engineering analogies, not evidence of biological equivalence or consciousness.

Current implementation descriptions were reconciled through 2026-09-30 with
`docs/IMPLEMENTATION_STATUS.md` and the packed update/retention source.
Research inspiration is not a claim of identical training machinery or useful
native quality. The v1.1.0 code/package release is complete; newer uncommitted
source corrections for candidate v1.1.1 and deferred native training/recall
evidence are separate. The independent six-target video runtime is published;
the new application packages are not yet published.

| Source | Adopted idea | Omni implementation boundary | Code/license boundary |
| --- | --- | --- | --- |
| [LLaMA](https://arxiv.org/abs/2302.13971) | Decoder-only causal language modeling with RMSNorm, rotary positions, and a gated feed-forward path | `OmniDecoder` uses these published architectural ideas at the language boundary while adding its own global workspace, neural substrate, action head, liquid/spiking paths, and continual learning. | Paper-level architecture only; Meta source and weights are not copied, imported, or bundled. |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764) and [Microsoft BitNet](https://github.com/microsoft/BitNet) | Published ternary-projection ideas, with an explicit Omni training/residency departure | `PackedAdaptiveBitLinear` and the native embedding/convolution/control paths store authoritative packed `{-1, 0, +1}` codes and mutate them directly, not through a resident full-precision master or Adam weight mirror. Bounded temporary derivatives/unpacking, activations, SNN traces and liquid activity are not extra learned weight copies. Physical packing is four trits per byte; no whole-process 1.58-bit or fused BitNet-kernel claim is made. | Omni's trainer is an independent implementation. An ignored developer checkout may be preserved at `.runtime/bitnet-src` for research comparison, but Studio never imports, links, executes, packages, or offers it as a model. The upstream MIT text is preserved in `licenses/BitNet-MIT.txt`. |
| [snnTorch](https://github.com/jeshraghian/snntorch) | Leaky integrate-and-fire state, recurrent spikes, surrogate-friendly dynamics | A local spiking router controls associative salience. STDP is independently implemented and tested rather than importing snnTorch's learner. | snnTorch source and package are not bundled; its MIT text is preserved in `licenses/snnTorch-MIT.txt`. |
| [Liquid Time-Constant Networks](https://arxiv.org/abs/2006.04439) and [NCPS](https://github.com/mlech26l/ncps) | Input-dependent continuous-time state and sparse wiring | CfC is the default temporal controller; an LTC-like cell is available for experimental evolution. | Omni's compact cells are independently written from published equations. NCPS source is not vendored; its Apache-2.0 text is preserved in `licenses/NCPS-Apache-2.0.txt`. |
| [VSA/HDC survey](https://arxiv.org/abs/2111.06077) and [TorchHD](https://github.com/hyperdimensional-computing/torchhd) | High-dimensional binding, bundling, permutation, and approximate recall | V3 neuron/assembly views share an authoritative adaptive packed vector row and typed ternary pathways. Labels/fingerprints and rebuildable indexes are inspection metadata, not a separate answer-memory authority. The cue-to-answer decoder/storage path is removed and matching legacy state is rejected. Useful native recall/generalization is not established by those structures alone. | Paper-level ideas only; TorchHD code and weights are not copied or bundled. |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | A limited latent workspace can integrate large, heterogeneous inputs before task-specific output | Omni's bidirectional whole-input workspace compresses the available text/sensory experience into recurrent neural latents before causal language-boundary decoding. It is not a verbatim Perceiver implementation. | Paper-level method, independently implemented; no Perceiver source or weights are bundled. |
| [Differentiable plasticity](https://arxiv.org/abs/1804.02464) | Fast plastic activity plus slower shared-weight refinement | An ordinary experience first conditions temporary activity; an accepted turn commits fast packed memory changes and queues separately checkpointed cortical refinement. A queued job or checksum mutation is not proof that the slow stage completed or that normal answers now use the new knowledge. | Paper-level method, independently implemented; native retained-answer quality remains deferred proof. |
| [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) | Spend more recurrent computation on uncertain inputs | Liquid control, novelty, uncertainty, neural-energy settling, convergence, workspace pressure, and host reserves determine computation; the trace records measured outcomes. | Paper-level method, independently implemented without copied code. |
| [Elastic Weight Consolidation](https://doi.org/10.1073/pnas.1611835114) and [SYNERgy](https://proceedings.mlr.press/v199/sarfraz22a.html) | Retention/rehearsal inspiration, not a claim of unchanged floating EWC machinery | Native packed projections use checkpointed uint8 output-row resistance (`_PackedRowMetaplasticity`) in direct ternary updates, alongside selective replay and rollback. Legacy floating-anchor/importance helpers have no native packed weight parameters to penalize; they are not the current cortical retention path. This row-level approximation is not a full per-synapse Fisher/anchor model or proof of non-forgetting. | Method-level inspiration and deliberate packed-state departure; independently implemented with no third-party checkpoints. |
| [Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547), [GROWN](https://arxiv.org/abs/2110.00908), and [Switch Transformers](https://arxiv.org/abs/2101.03961) | Add capacity after sustained error while stabilizing old knowledge and sparsely routing inputs through experts | Omni grows distributed substrate structures and small ternary residual experts after sustained novelty, resets the optimizer for new parameters, routes by learned prototype, and persists them at the next atomic save. Arbitrary incompatible live tensor reshaping is deliberately rejected. | Paper-level architectural ideas, independently implemented. |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) and its [reference repository](https://github.com/jennyzzt/dgm) | Preserve an open-ended lineage of self-improvement experiments instead of greedily retaining only one candidate | `EvolutionController` forms isolated neural, data, architecture, or source candidates; binds them to immutable evaluation evidence; archives parentage and promising failures; promotes supported gains; and can recursively reassess the improved improvement process. Host permission, evaluator, origin, and rollback boundaries remain outside candidate-writable state. | Method-level inspiration only; DGM source is not copied, imported, or executed. Omni's controller and Git/safe-tensor isolation are independently implemented. |
| [Model Context Protocol 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18) | Discover and invoke external typed tools over negotiated Streamable HTTP or stdio JSON-RPC | Omni independently implements initialize, initialized notification, paged tool discovery, and calls. Only tool names and structural input schemas enter neural learning; descriptions, credentials, and transport metadata cannot become behavioral prompts. Host permission and audit enforcement remain outside candidate-writable state. | Protocol interoperability only; no MCP SDK or server code is copied or bundled. |
| [OpenAI Responses](https://developers.openai.com/api/reference/resources/responses/methods/create), [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create), and [Gemini Interactions](https://ai.google.dev/api/interactions-api) | Optional teacher-model response imitation from explicit user questions | Provider responses are converted into attributable corpus-prediction trajectories and learned through OmniCortex's existing synapse/assembly/slow-weight path. No provider model is used during normal chat inference; there is no hidden system prompt, preference optimization, or exported credential. | Wire-protocol integration only; provider SDKs, source, and weights are not copied or bundled. API use remains subject to each provider's terms and the user's account. |
| [Diffusion Transformers](https://arxiv.org/abs/2212.09748) | Transformer denoising in a compressed visual latent space | The VQ image pack uses a small ternary, idea/time-conditioned latent transformer with a latent noise-prediction loss and iterative denoising sampler. It is a trainable baseline, not a claim of Stable Diffusion quality. | Paper-level architecture; no DiT code or weights are bundled. |
| [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) | Discrete residual-quantized audio latents | The audio pack provides a two-level RVQ codec and ternary latent-token generator without bundling AudioCraft weights. | Paper-level architecture; no EnCodec code or weights are bundled. |
| [Latte](https://arxiv.org/abs/2401.03048) | Factorized spatial-temporal latent video processing | The video pack operates on tiny compressed clips with separate spatial/temporal paths, latent noise-prediction training, iterative denoising, liquid recurrent temporal gating, and local MP4 encoding. | Paper-level architecture; no Latte code or weights are bundled. |

## Reference-source boundary

BitNet, snnTorch, NCPS, TorchHD, DGM, LLaMA, and the other entries above are
research references, not selectable model families. Preserving a paper,
license, source URL, or ignored local checkout does not make that source part of
the product runtime. In particular, `.runtime/bitnet-src` may remain on a
developer machine for inspection and kernel/file-format comparison; no release
script, worker import, Build, catalog entry, or `.omni` import may depend on it.

## Deliberate departures

- Every new Build trains native OmniCortex from locally initialized packed
  ternary weights, without a resident full-precision learned master copy.
  Import accepts only an exact-schema `.omni` whose current and
  immutable-origin records prove native ground-up OmniCortex state. It does not
  convert or compatibility-load Falcon, the retired bundled Starter, a Blank
  origin, a frozen foundation adapter, or another model family.
- Historical Falcon experiments and the saved Nova hybrid are research records,
  not selectable product paths and not evidence that a current native Build is
  coherent or capable.
- OmniCortex does not silently call an external LLM. An explicitly configured
  API teacher contributes attributable training examples; it is not chat
  inference, fallback, or an imported foundation.
- Long-term parameter-only memory enters the model as neural state and fast weights, never as hidden retrieved prose.
- An unsanitized backup preserves source bytes already retained in saved state,
  but that is separate from neural memory and does not guarantee exact learned
  recall or correct reasoning. The basic memory-recipe/source-retention chooser
  was superseded; do not restore it. Hidden source/history prompt reinjection
  and saved-answer sequence replay are not accepted learning paths.
- Growth adds sparse concepts, synapses, and experts. It does not resize the live dense base transformer tensor.
- Operational traces show seeds, activations, routing, tools, and weight deltas. They are not marketed as faithful private chain-of-thought.
- No RLHF, preference model, behavioral policy prompt, or deception objective is part of the training pipeline.
