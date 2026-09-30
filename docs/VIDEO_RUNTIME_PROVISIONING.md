# Automatic video runtime

Omni now has a compiled-in, hash-pinned catalog for Windows, macOS and Linux on x64 and ARM64. The independent [FFmpeg 9.0.2 runtime release](https://github.com/Eabusham2/Omni-AGI/releases/tag/omni-video-9.0.2-r1) contains native executables, exact corresponding sources, x264 sources, build material, licenses, dependency receipts and checksums. All 43 uploaded asset sizes and server SHA-256 digests were compared with the reviewed files. The application still excludes the imageio wheel's standalone executable.

## Build and source evidence

The codec-only workflow obtains the official signed FFmpeg 9.0.2 archive and checks fingerprint FCF986EA15E6E293A5644F10B4322F04D67658D8. x264 is built from exact commit 0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee. Autodetection and nonfree components are disabled; libx264 matches the encoder; codec libraries are static with ordinary system dependencies.

All six native targets passed real H.264/AAC encode/decode checks. Review covered PE/Mach-O/ELF headers, every recorded size/hash, exact configuration, signed source evidence, source contents and dependency listings. Linux uses system libc/libm/loaders, macOS system libraries/frameworks, and Windows system DLLs. Receipts bind each build commit, including targeted Windows rebuilds.

Hosts were Windows hosted runners, macOS 15 and Ubuntu 24.04. Older OS/library compatibility is not certified by these checks; a fixed version probe still gates activation. Sources and build material remain available for rebuilding another compatible standalone runtime.

## Licenses and distribution

The Python wrapper is BSD-2-Clause. The separate FFmpeg/x264 builds are GPL-2.0-or-later, not BSD or the application's PolyForm/commercial terms. Matching sources and license texts stay alongside the executable. Omni uses standard CLI media formats and does not link the codec libraries into its neural worker; its license does not restrict their component rights. See [FFmpeg licensing](https://ffmpeg.org/legal.html) and the [GNU aggregation discussion](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation). Technical distribution review is not legal advice or patent clearance.

The current libx264/AAC MP4 encoder matches the actual build. It was not silently replaced with an LGPL-only build lacking libx264, a soundless container or a preview advertised as final output.

## Installation and first use

The manifest validates only compiled-in versioned Omni pins, not renderer manifests, latest URLs, arbitrary repositories or package-manager commands. The provisioner streams and verifies source/build/licenses before the executable, checks reserves and cancellation, validates native headers, and runs only a fixed version probe. It never executes downloaded setup scripts. Reuse rechecks pinned payloads and the receipt. User-selected/existing host executables remain explicitly external, not authenticated Omni builds.

Explicit media operations prepare before dispatch. First organic video or media discovered inside a mixed training operation uses the correlated codec gateway and main bridge out-of-band lane. The occupied worker does not queue required setup behind itself. Request/brain/job/turn/action/nonce ownership binds replies; binary changes wait for real codec leases, not unrelated image jobs. The worker rechecks its fixed cache root, receipt and binary before selection.

Setup bytes are separate from neural progress. Cancelling setup or one claimed inline artifact preserves the warm worker and saved text; it does not use process-wide termination or a discarded acknowledgement as proof of completion. Health, creation/loading, ordinary text, images and raw PCM/WAV do not install codecs merely because video is available.

## Verification boundary

Real codec-only checks passed on all six hosts. Source/mock fixtures cover pinned provisioning, package policy, warm configuration, strict gateway ownership, leases, cancellation and speech job ownership. They do not certify trained media quality, real long utterances, every external FFmpeg installation or all hardware/I/O races. No brain was constructed or trained for this distribution work.
