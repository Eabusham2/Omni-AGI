# Completion evidence matrix

This file preserves the requested scope. A checked item has current inspectable implementation evidence and, where practical, an automated fixture. A checked research item is not a claim of model quality. Windows items are checked only where a native Windows Actions run produced direct logs or an uploaded smoke record; an implemented but unexecuted CI fix is not counted as release evidence.

## Custom adaptive brain

- [x] Custom model answers without a hosted or local third-party LLM.
- [x] Blank and compatible starter-checkpoint build paths exist.
- [x] Effective ternary weights are exactly `-1`, `0`, or `+1`.
- [x] New input produces persistent parameter/synapse mutations.
- [x] Slow self-supervised weights can consolidate without RLHF.
- [x] Hardware-scaled physical batches, gradient accumulation, and decoder activation checkpointing change the real slow-training path.
- [x] Fast STDP updates have tested potentiation and depression.
- [x] CfC and LTC-like liquid state are implemented and persistent.
- [x] VSA ideas bind, bundle, connect, and recall approximately.
- [x] Synapses and sparse experts can grow after build creation.
- [x] Noise/fuzziness is configurable and replayable by seed.
- [x] Operational traces expose activations, routes, tools, and deltas. Permission, invocation, result, cancellation, and failure stages join the persistent neural trace without copying file contents into it.
- [x] No hidden behavioral prompt or reward/preference model exists.
- [x] Enabled tool capabilities enter the brain through a deterministic, bounded VSA channel with no tool-schema prose or extra prompt tokens.

Evidence: `engine/tests/test_model.py`, `test_dynamics.py`, `test_memory_modalities.py`, `test_brain.py`, `test_release_gates.py`, `test_tool_schemas.py`, `tests/toolExecutor.test.ts`, and `tests/projectIntegrity.test.ts`.

## Memory and identity

- [x] Each build owns exactly one persistent chat/identity.
- [x] Human Consolidation, Total Recall, and Synapses Only behave differently.
- [x] Parameter-only memory never injects retrieved prose.
- [x] Working, short-term, semantic, and long-term controls are configurable.
- [x] Repeated/useful ideas gain stronger and more stable connections.
- [x] The immutable origin and current evolved copy are exportable.
- [x] Forks evolve independently and merge through stale-safe reviewed deltas, including ideas, relations, deduplicated evidence, allowed retained files, content-addressed artifacts, and neural replay examples without whole-weight averaging.
- [x] Continuing dialogue training can strengthen a learned slang response; the fixture measures the target next-token probability and does not claim a fixed personality.

Evidence: `engine/tests/test_model.py`, `engine/tests/test_brain.py`, `engine/tests/test_release_gates.py`, `tests/adaptiveCore.test.ts`, `tests/brainRepository.test.ts`, `tests/brainService.test.ts`, and the hash-bound merge preview/apply path in `src/main/brainService.ts`.

## Inputs, imagination, and learning

- [x] Text, Markdown, JSON, source, folder, and PDF ingestion paths exist; real PDF extraction is tested.
- [x] Image understanding and generation execute real model paths, including latent noise-prediction training and iterative ternary-transformer denoising.
- [x] Audio input and generation execute real model paths for PCM WAV, tested FLAC, and common soundfile/FFmpeg-backed formats.
- [x] Video input and generation execute real model paths for animated GIF/WebP and tested FFmpeg-backed MP4, with factorized latent noise prediction, liquid temporal gating, iterative denoising, and generated H.264 MP4 output.
- [x] Text, image, audio, and video miniature fixtures decrease loss, reload safe tensors, and reproduce identical seeded output.
- [x] Uploading data can change model weights rather than only appending context.
- [x] Dataset catalog, quarantined web-corpus mode, and source provenance are visible.
- [x] Low-end hardware profiles retain every modality at reduced scale.
- [x] Automatic build selection profiles CPU, RAM, and GPU status, then resolves to a scaled Micro, Personal, GPU, or Workstation configuration.
- [x] A compatible starter preserves its imported tensors, and optional local pretraining mutates only the current copy after immutable-origin creation.

Evidence: `engine/tests/test_memory_modalities.py`, `engine/tests/test_brain.py`, `engine/tests/test_release_gates.py`, `tests/brainService.test.ts`, `catalog/catalog.json`, and the creation/ingestion paths in `src/main/brainService.ts`.

## Windows application

- [x] Windows 11 Build and Run modes are complete and accessible. The installed x64 Playwright pass exercised keyboard focus, roles, and named navigation.
- [x] Brain Library, Build wizard, one-chat workspace, Data Studio, Brain Map, Trace, Journal, Tools, Agents, Catalog, and Imagination are usable in the packaged Windows app.
- [x] x64 Windows package builds in clean CI. The native-host job uploaded both NSIS and ZIP artifacts plus a checksum-bearing smoke record.
- [ ] ARM64 Windows package builds in clean native CI. The workflow builds a native ARM64 Electron shell with an x64 PyTorch/PyInstaller worker for Windows 11 emulation. Run 30066558290 passed every source, built-UI, worker, and package stage, and its installed worker passed the comprehensive neural exercise. The NSIS install then lacked only the top-level desktop executable while retaining the full resources tree. Run 30067949274 proved that switching the entire installer to ZIP extraction was pathologically slow on both architectures and was cancelled. Packaging now retains fast 7z extraction and uses a custom NSIS post-install repair to copy only a missing top-level executable from the still-materialized extraction staging tree.
- [x] A clean installation includes its Python/PyTorch worker. The uploaded x64 record proves direct, ZIP, and silent-NSIS worker execution with safe-tensor and SQLite persistence.
- [x] State survives a full packaged-app restart and interrupted training. The installed x64 UI record proves create/chat/imagine/close/relaunch persistence; killed-worker and interrupted-promotion fixtures prove recovery and candidate quarantine.

Executed evidence:

- [Windows run 30063378852](https://github.com/Eabusham2/Omni-AGI/actions/runs/30063378852), x64 job: green. Uploaded ZIP SHA-256 `c9935bd703bb8fda9aa7835e5eec13dfe1a2cf15e9f4fc7b6c5de94b6e6fc761`; NSIS SHA-256 `a8218ac9dc2e15be636a61bb68a3008f93a6e8013a199204b2e7e5d80e5f541c`.
- [Windows run 30064377772](https://github.com/Eabusham2/Omni-AGI/actions/runs/30064377772): both source suites, production builds, built-UI tests, self-contained workers, and Windows packages completed. The x64 leg then timed out during a 30-second CDP handshake; the ARM64 leg exercised the ZIP and installed worker before failing a non-recursive desktop-executable lookup. Run 30066558290 exercised both corresponding fixes: x64 passed, while ARM64 exposed the separate NSIS extraction issue described below.
- [Windows run 30066558290](https://github.com/Eabusham2/Omni-AGI/actions/runs/30066558290): x64 passed and uploaded its complete artifact/evidence bundle. ARM64 passed source tests, production build, built UI, self-contained worker, worker smoke, package build, ZIP worker, silent NSIS installation, and the comprehensive installed-worker exercise; its final desktop check proved the installed resources existed but the NSIS 7z copy omitted the native app executable.
- [Windows run 30067949274](https://github.com/Eabusham2/Omni-AGI/actions/runs/30067949274): both architectures again passed source tests, production build, built UI, self-contained worker, worker smoke, and package creation. Both then spent more than twenty minutes in the ZIP-based final package smoke and were cancelled. The replacement fix keeps the previously fast 7z path and repairs only the missing top-level executable through `build/installer.nsh`.

Implementation and fixture sources: `.github/workflows/windows.yml`,
`scripts/smoke-engine.ps1`, `scripts/smoke-windows-package.ps1`,
`tests/e2e/electron.spec.ts`, and
`engine/tests/test_interrupted_recovery.py`.

## Tools, agents, and sharing

- [x] Off, Ask, Auto, and Full Authority grants are enforced by the Electron executor.
- [x] Files, PowerShell, code, web, browser, and media tools have structured protocols.
- [x] Users can invoke tools directly from chat, and bounded model-produced calls use the same visible, permission-checked execution path. Long-running imagination calls wait for the final artifact before their result is learned into chat experience.
- [x] Ask approvals are one-use, expiring, and bound to the exact brain, tool, action, and serialized-argument digest; execution and worker jobs have tested cancellation paths.
- [x] Browser execution produces a guarded, isolated, sanitized inert HTML/text/link/PNG snapshot without exposing remote active content or an interactive signed-in browser. A native Electron retry on the development host after full restart verified HTTP status/title/text, an absolute HTTPS link, visible chat feedback, and a real artifact PNG; Windows packaging remains a separate open gate.
- [x] Source self-modification uses an authorized isolated Git worktree with diff inventory, allowlisted build/test, exact-diff validation, tamper rejection, and optional promotion.
- [x] Active subagent turns use isolated copy-on-write brain forks and cannot mutate the parent directly; merge remains explicit and overlay-based.
- [x] `.omni` import/export enforces architecture/schema, checksum, exact-byte, safe-tensor, secret-redaction-policy, and license-ledger contracts.
- [x] Referenced-local `.omni` exports use verified content-addressed local tensor objects and fail on an installation that lacks them.
- [x] GitHub/catalog URLs import complete `.omni` bundles, strict recipe JSON, and namespace-limited `.omnipack` safe tensors without auto-executing repository code.

Evidence: `engine/tests/test_tool_schemas.py`, `engine/tests/test_brain.py`, `tests/projectIntegrity.test.ts`, `tests/brainRepository.test.ts`, `tests/catalogInstaller.test.ts`, `tests/toolExecutor.test.ts`, `tests/toolWorkflows.test.ts`, `src/main/catalogInstaller.ts`, `src/main/toolExecutor.ts`, and `src/main/brainService.ts`.

## Accuracy, provenance, and licensing

- [x] UI does not claim proven AGI, consciousness, human equivalence, or perfect parametric recall.
- [x] Every currently identified research-derived module has a source and license-boundary ledger.
- [x] Vendored MIT and Apache license texts are configured to ship in the package.
- [x] The pinned imageio-ffmpeg wrapper and its separately licensed GPL FFmpeg executable are disclosed in the packaged third-party notice, including exact source and redistribution-obligation links.
- [x] Original code uses PolyForm Noncommercial plus separate commercial licensing.
- [x] Imported model/dataset licenses remain separate and visible through catalog labels and the required `.omni` per-source provenance/license ledger; undeclared rights remain explicitly undeclared.

Evidence: `RESEARCH.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE.md`, `COMMERCIAL_LICENSE.md`, package resources, `tests/brainRepository.test.ts`, and `tests/projectIntegrity.test.ts`.
