# Completion evidence matrix

This file preserves the requested scope. A checked item has current inspectable implementation evidence and, where practical, an automated fixture. A checked research item is not a claim of model quality. Windows packaging and clean-install items remain unchecked until a green Windows Actions run and a Windows 11 smoke test produce artifacts and logs.

## Custom adaptive brain

- [x] Custom model answers without a hosted or local third-party LLM.
- [x] Blank and compatible starter-checkpoint build paths exist.
- [x] Effective ternary weights are exactly `-1`, `0`, or `+1`.
- [x] New input produces persistent parameter/synapse mutations.
- [x] Slow self-supervised weights can consolidate without RLHF.
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
- [x] Image understanding and generation execute real model paths.
- [x] Audio input and generation execute real model paths for PCM WAV.
- [x] Video input and generation execute real model paths for the tested animated-GIF path.
- [x] Uploading data can change model weights rather than only appending context.
- [x] Dataset catalog, quarantined web-corpus mode, and source provenance are visible.
- [x] Low-end hardware profiles retain every modality at reduced scale.
- [x] Automatic build selection profiles CPU, RAM, and GPU status, then resolves to a scaled Micro, Personal, GPU, or Workstation configuration.
- [x] A compatible starter preserves its imported tensors, and optional local pretraining mutates only the current copy after immutable-origin creation.

Evidence: `engine/tests/test_memory_modalities.py`, `engine/tests/test_brain.py`, `engine/tests/test_release_gates.py`, `tests/brainService.test.ts`, `catalog/catalog.json`, and the creation/ingestion paths in `src/main/brainService.ts`.

## Windows application

- [ ] Windows 11 Build and Run modes are complete and accessible. The renderer builds, but Windows keyboard/screen-reader/manual evidence has not been recorded.
- [ ] Brain Library, Build wizard, one-chat workspace, Data Studio, Brain Map, Trace, Journal, Tools, Agents, and Catalog are usable. The surfaces exist; a packaged-app end-to-end smoke pass is still required.
- [ ] x64 Windows package builds in clean CI. NSIS and ZIP are configured; a successful workflow artifact is still required.
- [ ] ARM64 Windows package builds in clean CI. NSIS and ZIP are configured; a successful workflow artifact is still required.
- [ ] A clean installation can provision or include its Python/PyTorch worker. CI now configures direct, ZIP, and silent-NSIS JSON-RPC worker smoke checks, but a green Windows artifact is still required.
- [ ] State survives a full packaged-app restart and interrupted training. A killed JSON-RPC worker and an interrupted promotion both reload the complete stable safe-tensor checkpoint and quarantine the candidate; the packaged desktop restart scenario remains unverified.

Configured evidence: `.github/workflows/windows.yml`,
`scripts/smoke-engine.ps1`, `scripts/smoke-windows-package.ps1`, and
`engine/tests/test_interrupted_recovery.py`. CI configuration is not counted as
a successful Windows run until its uploaded evidence exists.

## Tools, agents, and sharing

- [x] Off, Ask, Auto, and Full Authority grants are enforced by the Electron executor.
- [x] Files, PowerShell, code, web, browser, and media tools have structured protocols.
- [x] Users can invoke tools directly from chat, and bounded model-produced calls use the same visible, permission-checked execution path.
- [x] Ask approvals are one-use, expiring, and bound to the exact brain, tool, action, and serialized-argument digest; execution and worker jobs have tested cancellation paths.
- [x] Browser execution produces a guarded, isolated, sanitized inert HTML/text/link/PNG snapshot without exposing remote active content or an interactive signed-in browser. A native Electron retry on the development host after full restart verified HTTP status/title/text, an absolute HTTPS link, visible chat feedback, and a real artifact PNG; Windows packaging remains a separate open gate.
- [x] Source self-modification uses an authorized isolated Git worktree with diff inventory, allowlisted build/test, exact-diff validation, tamper rejection, and optional promotion.
- [x] Active subagent turns use isolated copy-on-write brain forks and cannot mutate the parent directly; merge remains explicit and overlay-based.
- [x] `.omni` import/export enforces architecture/schema, checksum, exact-byte, safe-tensor, secret-redaction-policy, and license-ledger contracts.
- [x] Referenced-local `.omni` exports use verified content-addressed local tensor objects and fail on an installation that lacks them.
- [x] GitHub/catalog imports declarative `.omni` packs without auto-executing repository code.

Evidence: `engine/tests/test_tool_schemas.py`, `tests/projectIntegrity.test.ts`, `tests/brainRepository.test.ts`, `tests/toolExecutor.test.ts`, `tests/toolWorkflows.test.ts`, `src/main/toolExecutor.ts`, and `src/main/brainService.ts`.

## Accuracy, provenance, and licensing

- [x] UI does not claim proven AGI, consciousness, human equivalence, or perfect parametric recall.
- [x] Every currently identified research-derived module has a source and license-boundary ledger.
- [x] Vendored MIT and Apache license texts are configured to ship in the package.
- [x] Original code uses PolyForm Noncommercial plus separate commercial licensing.
- [x] Imported model/dataset licenses remain separate and visible through catalog labels and the required `.omni` per-source provenance/license ledger; undeclared rights remain explicitly undeclared.

Evidence: `RESEARCH.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE.md`, `COMMERCIAL_LICENSE.md`, package resources, `tests/brainRepository.test.ts`, and `tests/projectIntegrity.test.ts`.
