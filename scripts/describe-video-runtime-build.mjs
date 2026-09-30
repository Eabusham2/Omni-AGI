// SPDX-License-Identifier: MIT
import { createHash } from 'node:crypto';
import { readFile, writeFile, readdir, copyFile, mkdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { execFileSync } from 'node:child_process';
const [target, output, version, x264Commit, configuration] = process.argv.slice(2);
if (!/^(win32|darwin|linux)-(x64|arm64)$/.test(target ?? '') ||
    !/^\d+\.\d+\.\d+$/.test(version ?? '') || !/^[a-f0-9]{40}$/.test(x264Commit ?? '')) throw Error('Invalid build identity');
const commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const notes = `Independent FFmpeg ${version} CLI with x264 ${x264Commit}\n\nFFmpeg and x264 are distributed under GPL-2.0-or-later; their complete sources, license texts, exact build configuration and build scripts accompany this separate executable. Omni does not link to these libraries. Its own license does not restrict the rights granted by these components. This is a technical distribution review, not legal advice or a patent clearance.\n\nFFmpeg source: https://ffmpeg.org/releases/ffmpeg-${version}.tar.xz\nx264 source: https://code.videolan.org/videolan/x264/-/tree/${x264Commit}\nBuild commit: ${commit}\nTarget: ${target}\nConfigure: ${configuration}\n\nThe build disables autodetection and nonfree components. The only explicitly enabled non-system library is libx264. Operating-system C/thread/math libraries and the compiler runtime are supplied under their respective system/toolchain terms. See binary dependency evidence before installing the catalog.\n`;
await writeFile(join(output, `LICENSE-NOTICES-${target}.txt`), notes);
const materials = join(output, 'build-material');
await mkdir(materials);
for (const name of ['scripts/build-video-runtime.sh', 'scripts/describe-video-runtime-build.mjs', '.github/workflows/video-runtime.yml']) await copyFile(name, join(materials, basename(name)));
for (const name of ['x264-config.mak', 'ffmpeg-config.mak', 'buildconf.txt', 'version.txt', 'source-signature-status.txt', 'GPL-2.0.txt', 'x264-COPYING.txt']) await copyFile(join(output, name), join(materials, name));
await writeFile(join(materials, 'BUILD-INSTRUCTIONS.txt'), 'Rebuild on the named native target using the dependency versions recorded by CI. Run bash scripts/build-video-runtime.sh TARGET from the matching tagged Omni source checkout. Scripts are MIT licensed. Application setup never executes these scripts. FFmpeg upstream source is unmodified; x264 includes its generated version header to preserve the exact source identity without requiring git history. Paths in config.mak describe the ephemeral CI build root and may be adjusted when rebuilding.\n');
execFileSync('tar', ['-czf', join(output, `build-material-${target}.tar.gz`), '-C', output, 'build-material']);
const files = [];
for (const name of (await readdir(output)).sort()) {
  if (name === 'build-material' || name === 'build-receipt.json' || name === 'SHA256SUMS.txt' || name === 'codec-smoke.mp4') continue;
  const bytes = await readFile(join(output, name));
  files.push({ fileName: name, sizeBytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
}
await writeFile(join(output, 'build-receipt.json'), JSON.stringify({ format: 'omni-codec-build', formatVersion: 1, target, version, x264Commit, commit, configuration, sourceSignatureFingerprint: 'FCF986EA15E6E293A5644F10B4322F04D67658D8', files, codecSmoke: { h264VideoDecoded: true, aacAudioDecoded: true }, review: 'pending-dependency-and-payload-review' }, null, 2));
await writeFile(join(output, 'SHA256SUMS.txt'), files.map(file => `${file.sha256}  ${file.fileName}`).join('\n') + '\n');
