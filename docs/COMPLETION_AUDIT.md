# Omni AGI Studio completion audit

## Audit boundary

This audit maps the requested product to inspectable implementation and test
evidence. It does not claim that OmniCortex is conscious, biologically
equivalent to a brain, a frontier model, or proven AGI. A blank brain is a
small randomly initialized research model and is primitive until it is
trained. The image, audio, and video packs are functional trainable baselines,
not production-quality foundation models.

The detailed feature checklist is maintained in
[REQUIREMENTS.md](REQUIREMENTS.md). Research sources and independent
implementation boundaries are recorded in [../RESEARCH.md](../RESEARCH.md).

## Requested capability map

| Requested capability | Implemented boundary | Primary evidence |
| --- | --- | --- |
| Custom model rather than an API wrapper | Local decoder-only PyTorch OmniCortex with RMSNorm, RoPE, gated feed-forward blocks, causal attention, and trainable ternary projections | `engine/omni_core/model.py`; `engine/tests/test_model.py` |
| Initial training and continuing adaptation | Blank random origin, validated pretrained Omni starter, optional local pretraining before first conversation, immediate fast updates, and candidate-gated slow training | `src/renderer/src/App.tsx`; `src/main/brainService.ts`; `engine/omni_core/brain.py`; `tests/brainService.test.ts` |
| 1.58-bit-inspired weights | Effective `{-1, 0, +1}` projection weights with floating-point master weights and straight-through gradients | `engine/omni_core/model.py`; `engine/tests/test_model.py` |
| Neuromorphic plasticity and STDP | Stateful LIF router, causal potentiation, anti-causal depression, decay, metaplastic stability, and persistent fast synapses | `engine/omni_core/dynamics.py`; `engine/tests/test_dynamics.py` |
| Liquid neural dynamics | Trainable CfC and LTC-like recurrent cells that control retention, thresholds, noise, and bounded pondering | `engine/omni_core/dynamics.py`; `engine/tests/test_dynamics.py` |
| Ideas rather than only token strings | VSA binding, bundling, permutation, approximate recall, sparse concept nodes, typed relations, and graph growth | `engine/omni_core/vsa.py`; `engine/tests/test_memory_modalities.py` |
| Working, short, semantic, and long memory | Bounded recurrent working vectors, configurable timescales, replay, semantic graph, fast weights, slow parameters, and three source-retention recipes | `engine/omni_core/brain.py`; `docs/ARCHITECTURE.md`; `engine/tests/test_brain.py` |
| Memory learned into neural state | Parameter-only mode leaves the input token sequence unchanged and conditions generation through weights, internal vectors, spikes, and recurrent state | `engine/tests/test_brain.py`; `engine/tests/test_tool_schemas.py` |
| Structural growth | Resource-guarded dynamic VSA capacity, sparse synapse growth, and novelty-triggered residual experts | `engine/omni_core/brain.py`; `engine/tests/test_release_gates.py` |
| Fuzzy behavior and traceability | Seeded exploration noise, uncertainty-driven branches, voluntary bounded pondering, activation/mutation traces, and model self-reports labeled as non-authoritative | `engine/omni_core/brain.py`; `engine/tests/test_release_gates.py` |
| No RLHF or hidden behavioral persona | No reward/preference model or behavioral system prompt in the training/generation path; role boundaries are non-text tokens | `engine/tests/test_model.py`; `tests/projectIntegrity.test.ts`; `docs/MODEL_CARD.md` |
| One identity and continuous chat | One durable chat per brain, immutable origin, snapshots, origin restore, fork lineage, and restart persistence | `src/main/brainRepository.ts`; `tests/brainRepository.test.ts`; `tests/e2e/electron.spec.ts` |
| Direct tools from chat | `/tool`, `/imagine`, and `/agent` plus bounded model-produced `<omni-tool>` calls use the same permission-checked, cancellable, traced executor; imagination waits for final artifact metadata before returning experience | `src/renderer/src/App.tsx`; `src/main/toolExecutor.ts`; `tests/toolExecutor.test.ts`; `tests/e2e/electron.spec.ts` |
| Windows files, PowerShell, code, web, browser, and media tools | Structured protocols with Off, Ask, Auto, and Full Authority; exact one-use approvals; visible execution records | `tools/catalog.json`; `src/main/toolExecutor.ts`; `tests/toolExecutor.test.ts` |
| Tool knowledge inside the brain without prompt prose | Enabled schemas are encoded as bounded deterministic VSA capability vectors and add no prompt tokens | `engine/omni_core/tool_schemas.py`; `engine/tests/test_tool_schemas.py` |
| PDF/code/document learning | PDF, text, Markdown, JSON, source, folders, images, audio, and video flow through provenance-aware ingestion and selected Encode/Consolidate/Pretrain/Archive policies | `src/main/brainService.ts`; `engine/omni_core/brain.py`; `engine/tests/test_release_gates.py` |
| Vision and imagination | Trainable compact vision encoder, VQ image model with ternary latent noise prediction, RVQ audio model, factorized liquid-gated video denoising, and local MP4 output | `engine/omni_core/modalities.py`; `engine/omni_core/brain.py`; `engine/tests/test_memory_modalities.py`; `engine/tests/test_brain.py` |
| Forked subagents and reviewed merging | Copy-on-write brain forks execute isolated turns; merge previews bind exact overlay/file hashes and never average full dense models | `src/main/brainService.ts`; `src/main/toolExecutor.ts`; `tests/toolWorkflows.test.ts` |
| Self-modification | Authorized Git worktree proposal, diff, allowlisted test/build, exact-diff validation, tamper rejection, and optional merge promotion | `src/main/toolExecutor.ts`; `tests/toolWorkflows.test.ts` |
| Download, copy, share, and restore | Strict non-executable `.omni` current/origin/private/referenced exports, content-addressed tensors, safe import, and separate declarative recipe/pack formats | `src/main/brainRepository.ts`; `src/main/catalogInstaller.ts`; `docs/OMNI_FORMAT.md`; `docs/CATALOG_FORMATS.md` |
| Windows 11 native-packaged interface | Electron main/preload isolation, Windows Mica/Fluent styling, Build and Run workspaces, keyboard-accessible navigation, NSIS and ZIP packaging | `src/main/index.ts`; `src/preload/index.ts`; `src/renderer/`; `tests/e2e/electron.spec.ts` |

## Verification snapshot

The following local gates passed on the development host after a clean
`npm ci`:

| Gate | Result |
| --- | --- |
| Python neural/integration suite | 43 of 43 passed |
| Node/Electron unit and workflow suite | 44 of 44 passed |
| TypeScript checks and production Electron build | Passed |
| Real built Electron UI test | Passed, including build, chat, neural worker health, all primary surfaces, trace export, image generation/download, close, relaunch, identity persistence, and chat persistence |
| Python bytecode compilation | Passed |
| `npm audit` | 0 known vulnerabilities |
| Whitespace/error-marker audit | Passed |
| Native Windows x64 package | Passed again in run 30066558290; ZIP and NSIS uploaded with a comprehensive installed-app smoke record |
| Native Windows ARM64 package | All build/worker/package paths passed in run 30066558290; installed desktop remains unproven after NSIS omitted the top-level app executable, with a direct-ZIP extraction fix intentionally not rerun |

The Windows release workflow uses two matching GitHub-hosted runners:

- x64 on `windows-latest`;
- ARM64 on `windows-11-arm`.

Run [30063378852](https://github.com/Eabusham2/Omni-AGI/actions/runs/30063378852)
produced the current successful x64 release record. Its installed NSIS app
passed accessibility navigation, direct chat/tool/subagent paths, image,
audio, and video generation, slow/fast parameter mutation, real PDF
ingestion, safe-tensor and SQLite persistence, and full close/relaunch state
recovery. The uploaded ZIP SHA-256 is
`c9935bd703bb8fda9aa7835e5eec13dfe1a2cf15e9f4fc7b6c5de94b6e6fc761`;
the NSIS SHA-256 is
`a8218ac9dc2e15be636a61bb68a3008f93a6e8013a199204b2e7e5d80e5f541c`.

Run [30064377772](https://github.com/Eabusham2/Omni-AGI/actions/runs/30064377772)
completed both architecture source suites, builds, built-UI tests,
self-contained workers, and Windows package builds. The x64 installed app
made its CDP endpoint and WebSocket available but exceeded Playwright's
30-second connection handshake. The ARM64 package passed the ZIP worker and
the comprehensive silently-installed worker exercise, then the verifier
searched only the installation root for the desktop executable. The source
then gave the CDP handshake 120 seconds and resolved the installed executable
from its worker application root with a recursive exact-name fallback. Run
30066558290 exercised both changes: x64 passed, while ARM64 exposed a separate
NSIS extraction defect rather than another lookup failure.

Run [30066558290](https://github.com/Eabusham2/Omni-AGI/actions/runs/30066558290)
fully passed and uploaded the x64 release. On native ARM64 it passed both
source suites, the production and built-UI gates, the self-contained worker,
package creation, ZIP worker, silent NSIS installation, and the comprehensive
installed-worker neural exercise. The installed resources tree was complete,
but the NSIS 7z staging/copy path omitted only the top-level native Electron
executable. Packaging now sets `nsis.useZip` so NSIS extracts directly into
the install directory, and the smoke verifier checks the executable
immediately after installation and again after worker exercise. Per user
direction this final packaging fix was not rerun, so ARM64 remains the sole
unproven release item.

The x64 package uses an x64 shell and worker. The ARM64 package uses a native
ARM64 shell with an x64 PyTorch worker under Windows 11 emulation because a
stable native PyTorch Windows ARM64 wheel is not available. A configured
workflow or an unexecuted fix is never counted here as successful evidence.

## Deliberate limits

- Exact source recall requires Total Recall and a retained private archive.
  Parameter-only and Synapses Only memory are lossy by design.
- The browser tool is an isolated, inert snapshot tool. It does not borrow a
  signed-in browser session or expose remote active content.
- Subagent turns are bounded and currently scheduled sequentially; the
  isolation and overlay merge are real, but this is not an open-ended parallel
  agent society.
- Sparse experts and associative structures grow; the live dense base
  transformer's tensor shapes do not resize in place.
- Distributed capability is described by the hardware profile but a
  multi-node trainer is not claimed.
- Full Authority can materially affect the host. The grant is explicit and
  every invocation remains auditable, but auditing does not make arbitrary
  commands harmless.
- Windows artifacts are signing-ready. They are unsigned unless the repository
  provides signing credentials to the workflow.
