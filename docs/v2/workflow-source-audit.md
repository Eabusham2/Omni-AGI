# v2 package-workflow source boundary

Scope: local source and fixed contract checks on October 15. This is not a
remote GitHub Actions result, a package smoke result, or a released v2 artifact.

## Changes and exact purpose

- Windows, macOS and Linux workflows, including each desktop release job, run
  `npm run test:python:ci:contracts`. Its fixed manifest contains three reviewed
  stdlib-only source-contract files, not whole-engine test discovery. The runner
  rejects arbitrary test paths and neural-runtime imports and has a 120-second
  process timeout. It does not create or train an instance. The import guard is
  in-process, not an operating-system sandbox; additions to this fixed lane
  still require review for child processes or neural execution.
- Package worker checks retain their explicit health-only invocation. Neural
  quality, large-dataset training and the deferred new-brain proof are not
  smuggled into package CI through a default smoke command.
- The write-enabled release publication checkout disables credential
  persistence. The tag fetch receives its scoped authentication header through
  Git's command-process configuration environment, after an exact current-repo
  origin check. The checkout commit and fetched tag commit must still match.
  No credential is written into the repository configuration by this fetch.
  This does not imply that a publication job has no privileged environment:
  the explicit GitHub token remains available to its intended release step.
- Internal artifact names are version-neutral `omni-release-*`; this does not
  rename, replace or publish any existing release. Existing immutable-release
  and exact version/tag/lock bindings remain in force.

## Local evidence

- The new bounded runner executed the existing 16 selected source-contract
  cases once: 16 passed in 0.087 seconds; outer command about 0.346 seconds.
  This proves runner integration, not 16 additional neural capabilities.
- Two new `releaseWorkflowBoundaries.test.ts` cases passed in 185 ms. They check
  the fixed selection/health-only calls and use an owned temporary Git repo with
  an inert, noncredential header to prove process-only configuration does not
  persist into the repo. No token or network action was exercised.
- Seven workflow YAML files parsed with PyYAML BaseLoader; publication Bash
  syntax passed `bash -n`. Node syntax checks and whitespace checks passed.
  `actionlint` was initially unavailable on PATH. A later scoped run used the
  official Go module at v1.7.12 (version checked against its release metadata),
  with `GOMAXPROCS=2`, and passed all seven actual workflow paths with exit0.
  Embedded shellcheck/pyflakes integrations were disabled in this run; the
  separate publication Bash syntax check remains its own evidence. An initial
  invocation named nonexistent `codec-runtime.yml` and failed before validation;
  the valid source file is `video-runtime.yml`. That failed invocation is not
  counted as a pass and no workflow was triggered.
- The bundled local repository auditor initially reported five findings. After
  the explicit publish-checkout correction, it reported four: one medium,
  two low and one note, with no high/critical findings. Its generated safe-only
  patch was empty and was not applied. The remaining observations concern
  CODEOWNERS, dependency-update policy, issue forms and branch-protection
  documentation. These are owner-policy observations, not newly invented
  product blockers or proof about live GitHub rulesets.

## Still required for delivery

Final current-revision contract checks, coordinated platform/package audits,
the v2 version/commit/tag, required successful remote jobs, checksummed release
assets and download/provenance verification remain open. No local source check
substitutes for those gates. No commit, push, remote CI or package publication
is claimed by this note.

## October 18 v2 release preparation

- Coordinated application, lock, worker, MCP client/App host and both mobile
  version fields now target `2.0.0`; mobile build numbers are `20000`. The
  existing stable version policy and source verifier accept this coordinated
  state. No v2 tag or remote release is implied by that source check.
- The new CPU kernel builder previously selected the first generic C++
  compiler on PATH and recorded the operating-system machine name. On Windows
  ARM64 that could disagree with the intentionally emulated x64 Python worker.
  It now derives the target from Python's build platform, discovers Visual
  Studio C++ tools, initializes that target, uses the static C runtime and
  checks the resulting library's ABI/exports in a fresh stdlib-only process
  before publishing it. No tensor/model/forward/training operation is used.
- Release and continuous Windows packaging now give the portable-runtime
  receipt the checked source commit. ZIP and installed NSIS smoke checks
  validate that receipt against the expected commit, application version,
  protocol and actual desktop/worker architectures. The final artifact
  verifier requires this evidence for both layouts while retaining the false
  selected-instance/reservation/neural-acceptance fields.
- Current runner labels were checked against GitHub's
  [hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
  The pinned Xcode 16.4/iOS 18.5 path remains listed in the
  [macOS15 image inventory](https://github.com/actions/runner-images/blob/main/images/macos/macos-15-arm64-Readme.md)
  and was retained. Actual runner availability and package behavior still
  require the corresponding remote jobs.

Focused local evidence: two new kernel packaging cases passed in 1.28 seconds,
including an actual small C++ library build into an owned temporary directory
and a simulated ARM64-host/x64-interpreter developer-environment selection.
After environment-key normalization, only the affected Windows-selection case
was repeated and passed in 98 ms. One new inert portable ASAR/PE receipt case
passed in 32 ms. Two release-artifact cases passed in 249 ms: the existing complete
synthetic artifact set and the new missing/foreign portable-commit rejection.
These are four new tests plus one affected existing test, not real Windows
packages. `actionlint` v1.7.12 passed the two changed workflow paths with exit0;
shellcheck/pyflakes integration was disabled. Changed JavaScript syntax,
Python AST and whitespace checks passed. The bounded local auditor reports
four optional governance observations, no high/critical findings, and an empty
safe-only generated patch. No governance automation was added.

No remote CI, package, new brain, Load, neural training, tag, push, release,
credential inspection or hardware benchmark was performed in this preparation
pass. Current full-goal audit and remote package/release gates remain open.

## October 18 Windows contract corrections

Hosted Windows x64 job `117785824818`, run `39164947102` at `9ae0f313`,
failed four portable-receipt cases and five catalog-lineage cases before
packaging. ASAR header traversal requires host separators even though the
validated `package.json` main entry is POSIX-form; the receipt reader now
normalizes that exact entry. Deep catalog-lineage SQLite paths now use the
Windows extended-length namespace after the existing ownership/hash checks.
Read-only opening, extension refusal, schema/count checks and exact archive
closure remain unchanged. Separate plugin lifecycle/timing failures are not
attributed to these two fixes.

Two new selected-file cases cover actual installed ASAR traversal using Win32
path operations, and drive/UNC/already-prefixed catalog paths. The nine affected
existing cases plus the two new cases passed locally. Embedded Electron Node24
needed `ELECTRON_NO_ASAR=1` for the four physical-ASAR cases so its virtual ASAR
filesystem did not replace normal Node filesystem behavior; CI uses Node22.
This is not a Windows rerun, native SQLite-on-Windows proof, package smoke, or
release result. No model, Load, training, commit, push or CI rerun was performed
by this correction lane.

## October 18 Intel macOS DMG detach control

Hosted macOS Intel job `117785824405`, run `39164947011` at `9ae0f313`,
completed worker/application staging and ZIP creation but the pinned
`dmg-builder@1.2.5` helper failed its ordinary detach with `Resource busy`.
The supported `CUSTOM_DMGBUILD_PATH` override now delegates to that same
checksum-pinned helper and requests its built-in `--detach-retries 30` option
after checking its help. The existing builder exposes no corresponding DMG
configuration field. An explicitly supplied custom helper is preserved.

The helper remains responsible for its own temporary image and mount. The
wrapper does not enumerate or unmount disks, force eject, change Spotlight,
rebuild the image after failure, or skip any format/signing/smoke gate. A
nonzero helper status remains fatal. Two new inert build-control cases in the
existing selected release-compatibility file passed in9ms; JavaScript/Bash
syntax and scoped whitespace passed. No binary download, actual helper/image
execution, model, local package or remote rerun occurred in this lane. Actual
DMG build and package smoke success remain CI evidence, not these fixtures.
