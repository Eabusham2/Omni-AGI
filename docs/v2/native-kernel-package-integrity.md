# macOS native-kernel packaging integrity

## Observed failure and causal scope

macOS ARM64 run `39164947011`, job `117785824111`, passed all ten desktop UI
cases and then failed packaged-worker health at `smoke-engine.mjs:107`. That
old error omitted the health payload, so the exact remote failing field was
not preserved. The job logged a compiler-issued kernel hash, PyInstaller
binary/data reclassification and ad-hoc signing, followed by skipped Electron
certificate signing. No new remote job has been dispatched by this source slice.

A real source defect was independently reproduced: compiling only the current
small C++ native kernel and replacing its ad-hoc signature changed the full
file SHA-256 while its ABI remained1 and the stdlib loader still opened it.
The current runtime correctly rejected the old hash. Locked
[PyInstaller6.21 reclassifies collected binaries](https://github.com/pyinstaller/pyinstaller/blob/v6.21.0/PyInstaller/building/build_main.py#L679)
and [rewrites/signs Mach-O files](https://github.com/pyinstaller/pyinstaller/blob/v6.21.0/PyInstaller/building/utils.py#L284),
so a pre-packager raw-file manifest cannot generally describe the final file.
This establishes the packaging/hash defect and its mechanism, not an invented
copy of the missing remote health payload.

## Fix without weakening native runtime verification

`build-native-kernels.py` still verifies the freshly compiled library's real
ABI and all four required exports. On macOS it additionally records a
build-only machine-payload fingerprint in the compiler-issued manifest.
`native_kernel_integrity.py` parses the supported thin64-bit Mach-O format.
It retains machine/data/link metadata bytes and functional load commands,
normalizing only this dylib's own ID path, the exact relative loader rpath,
signature envelope and its `__LINKEDIT` size accounting. Unexpected rpaths,
malformed sections, foreign roots and changed machine payloads fail closed.

The finalizer verifies the compiler-issued source tree's source hash, full raw
library hash, architecture and payload fingerprint. Each packaged copy must
retain that exact compiler-issued manifest except its final whole-file hash.
Its source/architecture/payload must match before an atomic manifest update.
Forging a new payload digest in the packaged manifest does not authorize it.

Unsigned workers finalize after PyInstaller's transformation. The existing
afterPack hook finalizes both the source and bundled-worker copies before
archiving. Signed apps keep the standard signer, identity, entitlements,
hardened-runtime options and existing retry policy. Its final root callback
requires both native children to have completed the normal signing sequence,
then finalizes their manifests **before** the outer app seals Resources.
Notarization still follows the existing Electron-builder path. Unexpected
root/order or skipped native children fails rather than signing stale metadata.

The runtime `packed_native_kernel._library` is unchanged: it still verifies
the exact full final-file SHA-256, current C++ source SHA and native ABI before
loading. There is no signature-neutral runtime bypass, dense fallback,
automatic compilation/download or replacement of the required native kernel.

Package health now emits only bounded known categories and booleans/counts:
for example, `native-library-integrity-mismatch`. Raw errors, paths,
credentials, environment and arbitrary response fields are not emitted.
Health-only smoke also requires `loadedBrains === 0`; it cannot silently
construct a brain to make the probe pass.

## New focused evidence

- Seven new stdlib-only Python cases passed across affected selections. They cover allowed relocation/
  signature changes, exact final hashes, altered/forged machine payloads,
  source/ABI/architecture/compiler custody, contained paths and unexpected
  headers/rpaths and a changed library during final verification. The actual runtime loader function is AST-extracted without
  importing Torch; it refuses a stale hash, accepts the checked final hash,
  and refuses subsequent code tampering before opening the library.
- Five new Node contracts passed across affected runs, including unchanged
  signing-option delegation, actual pinned signer child/root source ordering,
  rejected root/order, bounded stdlib finalizer invocation, diagnostic
  redaction, source hooks and the zero-brain health invariant.
- Final dependency integration review found that the installed signer's
  `optionsForFile` callback is synchronous. The initial asynchronous wrapper
  could lose per-file options and race the outer seal. It now uses synchronous
  path checks and a bounded synchronous stdlib finalizer for that callback;
  the separate afterPack helper remains asynchronous. One additional case
  executed the installed signing loop with inert filesystem/command callbacks
  and passed in41ms. The earlier five cases were not rerun as new evidence.
- A separate real ARM64 fixture compiled the current C++ kernel only, applied
  the same dylib-ID/relative-rpath transform and ad-hoc signing, then used the
  actual loader functions with stdlib ctypes: native-ready=true, ABI1,
  compiled-payload-unchanged=true, exact-final-hash=true, Torch-imported=false.
  A further ad-hoc hardened-runtime/entitlements signature retained the same
  compiled payload and finalized successfully. No certificate identity was used.

No brain/model/decoder construction, neural forward/backward, Load, training,
complete local PyInstaller build, package build or remote rerun occurred.
Whole signed/notarized applications and the next actual macOS package smoke
remain integration evidence to obtain; these tiny fixtures do not claim them.
