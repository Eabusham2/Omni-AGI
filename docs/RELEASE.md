# Stable v1 release

Omni AGI Studio releases are built from an immutable `vMAJOR.MINOR.PATCH` tag
whose value exactly matches `package.json`. The tagged commit must be contained
in `main`; release jobs deliberately reject tags cut from another branch.

## Required native artifacts

| Platform | Architectures | Formats | Worker |
| --- | --- | --- | --- |
| Windows | x64, ARM64 shell | NSIS, ZIP | x64 on x64; x64-emulated on ARM64 |
| macOS | Intel x64, Apple Silicon ARM64 | DMG, ZIP | native |
| Linux | x64, ARM64 | AppImage, DEB, tar.gz | native |
| Android | API 26+ | debug-signed APK, release unsigned APK | same desktop brain over the paired gateway |
| iOS/iPadOS | current Xcode device target | unsigned IPA | same desktop brain over the paired gateway |

Mobile marketing versions exactly match `package.json`. Android `versionCode`
and iOS `CURRENT_PROJECT_VERSION` share the stable numeric mapping
`MAJOR * 10000 + MINOR * 100 + PATCH`; release `1.1.0` therefore uses build
number `10100` on both platforms.

Electron Builder uses target-native x64 suffixes on Linux: `x86_64` for
AppImage, `amd64` for DEB, and `x64` for tar.gz. The package smoke, upload
patterns, and release verifier share one naming helper so these are not
mistaken for missing artifacts.

The final publish gate replaces spaces in package filenames with periods before
writing release metadata or uploading assets. This matches GitHub's public
asset-name rules, so every entry in `SHA256SUMS.txt` and
`RELEASE-MANIFEST.json` is the exact downloadable filename.

Each native-host job checks Python syntax and focused Node/UI contract suites,
builds the Electron application and PyInstaller brain worker, verifies their
machine architectures, checks packaged-worker health without creating a brain,
and opens the packaged desktop shell without training. Neural learning,
retention, generated media, and installed-app restart remain separate live
acceptance gates; CI packaging success does not prove them. The macOS job
validates the DMG and extracts the ZIP. The Linux job independently extracts
the AppImage, DEB, and tarball and checks their desktop and worker machine
types. The Windows job expands the ZIP and silently installs NSIS before
checking both layouts. A workflow file or locally produced archive is not
release evidence by itself.

The full Node suite, Python neural suite, and 10k-row stress fixtures
remain available for manual or dedicated acceptance runs, but are not hosted
package-CI prerequisites. Package CI runs an explicit fast code, security,
release-integrity, and UI-contract selection on each desktop platform. This
is a documented coverage boundary, not a claim that large-scale training,
retention, or every application feature has passed live acceptance.

The Android release job runs unit tests and a host-backed API 35 emulator test
that pairs, streams a visible chat turn, and streams a 2 MiB attachment. It
publishes the verified installable debug-signed APK and a separately named
unsigned release APK for custom signing. The iOS release job runs its native
unit/UI suite against the same gateway protocol and publishes an unsigned IPA
for inspection or later custom signing. The unsigned mobile packages are never
described as directly installable or signed. Both mobile jobs bind their
package hashes, variant, emulator evidence, and observed signing state into
records consumed by the final release verifier.

`windows-11-arm` and `ubuntu-24.04-arm` are GitHub public-preview runner
labels. They are still native architecture gates, but GitHub does not provide
the same service-level guarantee as its stable hosted images. Windows ARM64
uses a native Electron shell and the explicitly labeled x64 PyTorch worker
through Windows 11 emulation. PyTorch and PyInstaller now provide Windows
ARM64 builds, but the complete v1 worker dependency set does not: required
binary wheels including `pyarrow` and `imageio-ffmpeg` remain unavailable.
The worker can move to native ARM64 only after the full binary-only dependency
probe and packaged-runtime smoke pass, not merely when the core tensor wheel
exists.

The Intel macOS runtime remains on PyTorch 2.2.2, the last native Intel wheel.
Transformers, Accelerate, Tokenizers, and the retired external-foundation loader
are not worker dependencies. Every desktop job runs `pip check`, imports the
native OmniCortex runtime against its pinned PyTorch/NumPy/safetensors stack,
and checks the packaged worker's health response without neural training.

Desktop worker packages use a reviewed CPython 3.11 patch and one target lock
from `engine/locks`. Linux x64/ARM64 uses 3.11.16; macOS Intel/Apple Silicon
and the Windows x64 worker (also used by the Windows ARM64 shell) use 3.11.9,
the final macOS/Windows 3.11 binary published in setup-python's manifest. The
installer rejects a different patch release so the interpreter cannot drift.
Every runtime dependency, applicable transitive dependency, platform helper,
and PyInstaller is an exact version bound to the SHA-256 of one reviewed binary
wheel; source distributions are disabled. `scripts/install-engine-lock.py`
validates each target-specific closure, selects the lock from the interpreter's
actual OS and architecture, force-reinstalls only those hashed wheels, checks
the installed versions, and runs `pip check`. Both engine build scripts call
that installer before PyInstaller. `OMNI_SKIP_BUILD_DEPENDENCY_INSTALL=1` is
safe only after the same lock was provisioned: skip mode still rejects Python
versions or installed distributions that differ from it.

macOS Intel additionally pins NumPy 1.26.4 and PyArrow 17.0.0 because newer
reviewed releases no longer publish the required Intel CPython 3.11 wheel.
Linux and Windows locks select the `2.10.0+cpu` Torch wheels from PyTorch's CPU
index; macOS locks select their native Torch wheels from PyPI. Hash checking
applies to both indexes.

## Publication gate

Run `npm run verify:release -- --tag v1.1.0` before creating the tag. Pushing
that tag starts `.github/workflows/release.yml`; it rebuilds every required
artifact rather than reusing an unverified local package. Publication stops if
one expected artifact is missing, duplicated, empty, has a hash that disagrees
with its smoke record, lacks passing smoke evidence, or if any unexpected file
is present. Publication waits for all six desktop architecture jobs plus the
Android and iOS emulator/package gates. Official GitHub actions are pinned to
reviewed commit SHAs and only the final publish job receives `contents: write`.

The publish job writes:

- `SHA256SUMS.txt`, covering every package and smoke record.
- `RELEASE-MANIFEST.json`, recording the product, version, tag, artifact count,
  names, byte sizes, SHA-256 values, native worker architectures, and observed
  signing/notarization state, including explicit mobile debug-signed and
  unsigned-signed-ready labels.

Windows signing uses `WINDOWS_CSC_LINK` and
`WINDOWS_CSC_KEY_PASSWORD` when configured (forwarded to electron-builder as
`WIN_CSC_LINK` and `WIN_CSC_KEY_PASSWORD`). macOS signing uses
`MACOS_CSC_LINK` and `MACOS_CSC_KEY_PASSWORD`; electron-builder notarizes when
either `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, and `APPLE_TEAM_ID` or the
three App Store Connect API-key secrets are also configured. The
`APPLE_API_KEY` GitHub secret contains the raw or base64-encoded `.p8` private
key, not a runner-local path; the release job writes it to a mode-`0600`
temporary file, gives electron-builder that path, and removes the file when
packaging exits. Partial credential sets fail before packaging. Configured
signing is forced rather than silently falling back, and the smoke gate independently inspects
Authenticode signatures, macOS certificate authorities, and stapled
notarization tickets. Without those secrets, keychain discovery is disabled
and the same gates produce artifacts explicitly recorded as
`unsigned-signed-ready`; they must not be described as signed.

After the release is green and published, merge any remaining verified work
into `main` before deleting obsolete branches. Branch deletion is a separate,
explicit repository-maintenance action and is never performed by a package
job.
