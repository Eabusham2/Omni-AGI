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

Electron Builder uses target-native x64 suffixes on Linux: `x86_64` for
AppImage, `amd64` for DEB, and `x64` for tar.gz. The package smoke, upload
patterns, and release verifier share one naming helper so these are not
mistaken for missing artifacts.

Each native-host job runs the Python and Node suites, builds the Electron
application and PyInstaller brain worker, verifies their machine
architectures, creates and reloads a safe-tensor/SQLite brain through the
packaged worker, and runs the packaged desktop restart test. The macOS job
validates the DMG and extracts the ZIP. The Linux job independently extracts
the AppImage, DEB, and tarball and checks their desktop and worker machine
types. The Windows job expands the ZIP and silently installs NSIS before
checking both layouts. A workflow file or locally produced archive is not
release evidence by itself.

`windows-11-arm` and `ubuntu-24.04-arm` are GitHub public-preview runner
labels. They are still native architecture gates, but GitHub does not provide
the same service-level guarantee as its stable hosted images. Windows ARM64
uses a native Electron shell and the explicitly labeled x64 PyTorch worker
through Windows 11 emulation because stable native Windows ARM64 PyTorch wheels
are not available for this build.

## Publication gate

Run `npm run verify:release -- --tag v1.0.0` before creating the tag. Pushing
that tag starts `.github/workflows/release.yml`; it rebuilds every required
artifact rather than reusing an unverified local package. Publication stops if
one expected artifact is missing, duplicated, empty, has a hash that disagrees
with its smoke record, lacks passing smoke evidence, or if any unexpected file
is present. Official GitHub actions are pinned to reviewed commit SHAs and only
the final publish job receives `contents: write`.

The publish job writes:

- `SHA256SUMS.txt`, covering every package and smoke record.
- `RELEASE-MANIFEST.json`, recording the product, version, tag, artifact count,
  names, byte sizes, SHA-256 values, native worker architectures, and observed
  signing/notarization state.

Windows signing uses `WINDOWS_CSC_LINK` and
`WINDOWS_CSC_KEY_PASSWORD` when configured (forwarded to electron-builder as
`WIN_CSC_LINK` and `WIN_CSC_KEY_PASSWORD`). macOS signing uses
`MACOS_CSC_LINK` and `MACOS_CSC_KEY_PASSWORD`; electron-builder notarizes when
either `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, and `APPLE_TEAM_ID` or the
three App Store Connect API-key variables are also configured. Partial
credential sets fail before packaging. Configured signing is forced rather
than silently falling back, and the smoke gate independently inspects
Authenticode signatures, macOS certificate authorities, and stapled
notarization tickets. Without those secrets, keychain discovery is disabled
and the same gates produce artifacts explicitly recorded as
`unsigned-signed-ready`; they must not be described as signed.

After the release is green and published, merge any remaining verified work
into `main` before deleting obsolete branches. Branch deletion is a separate,
explicit repository-maintenance action and is never performed by a package
job.
