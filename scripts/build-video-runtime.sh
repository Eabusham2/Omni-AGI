#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Builds the independent FFmpeg CLI. Never invoked by the installed application.
set -euo pipefail
target=${1:?explicit runtime target required}
case "$target" in win32-x64|win32-arm64|darwin-x64|darwin-arm64|linux-x64|linux-arm64) ;; *) exit 2 ;; esac
repo_root=$(pwd)
build_root=$(mktemp -d)
output="$repo_root/codec-output/$target"
mkdir -p "$output" "$build_root/prefix" "$build_root/gnupg"
chmod 700 "$build_root/gnupg"
ffmpeg_version=9.0.2
x264_commit=0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee
curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz" -o "$build_root/ffmpeg.tar.xz"
curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz.asc" -o "$build_root/ffmpeg.tar.xz.asc"
curl --fail --location --retry 3 --proto '=https' --tlsv1.2 https://ffmpeg.org/ffmpeg-devel.asc -o "$build_root/ffmpeg-signing-key.asc"
gpg --homedir "$build_root/gnupg" --batch --import "$build_root/ffmpeg-signing-key.asc"
gpg --homedir "$build_root/gnupg" --batch --status-fd 1 --verify "$build_root/ffmpeg.tar.xz.asc" "$build_root/ffmpeg.tar.xz" > "$build_root/signature-status.txt"
grep -q 'VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8 ' "$build_root/signature-status.txt"
tar -xf "$build_root/ffmpeg.tar.xz" -C "$build_root"
git init -q "$build_root/x264"
git -C "$build_root/x264" remote add origin https://code.videolan.org/videolan/x264.git
git -C "$build_root/x264" fetch --depth 1 origin "$x264_commit"
git -C "$build_root/x264" checkout --detach FETCH_HEAD
test "$(git -C "$build_root/x264" rev-parse HEAD)" = "$x264_commit"
jobs=${OMNI_CODEC_BUILD_JOBS:-4}
case "$jobs" in ''|*[!0-9]*) exit 2 ;; esac
test "$jobs" -gt 0
extra_ldflags=()
x264_host=()
ffmpeg_host=()
case "$target" in
  win32-x64) extra_ldflags=(--extra-ldflags=-static); x264_host=(--host=x86_64-w64-mingw32); ffmpeg_host=(--target-os=mingw32 --arch=x86_64) ;;
  win32-arm64) export CC=clang; extra_ldflags=(--extra-ldflags=-static); x264_host=(--host=aarch64-w64-mingw32); ffmpeg_host=(--target-os=mingw32 --arch=aarch64 --cc=clang) ;;
esac
cd "$build_root/x264"
./configure --prefix="$build_root/prefix" --enable-static --disable-cli --disable-asm --disable-opencl "${extra_ldflags[@]}" "${x264_host[@]}"
make -j "$jobs"
make install
# Include the exact source tree and generated version header; no private app files.
git archive --format=tar --prefix=x264/ "$x264_commit" -o "$build_root/x264-source.tar"
mkdir -p "$build_root/source"
tar -xf "$build_root/x264-source.tar" -C "$build_root/source"
# x264 generates x264_config.h, not version.h. Retain the pinned shallow
# public git history so version.sh can reproduce its exact build identity.
cp x264_config.h "$build_root/source/x264/x264_config.h"
cp -R .git "$build_root/source/x264/.git"
cp config.mak "$output/x264-config.mak"
cd "$build_root/ffmpeg-$ffmpeg_version"
export PKG_CONFIG_PATH="$build_root/prefix/lib/pkgconfig"
configure=(--prefix="$build_root/prefix" --disable-autodetect --enable-gpl --enable-libx264 --enable-static --disable-shared --disable-debug --disable-doc --disable-ffplay --disable-ffprobe --disable-network --disable-devices --disable-hwaccels --disable-x86asm "${extra_ldflags[@]}" "${ffmpeg_host[@]}")
./configure "${configure[@]}"
make -j "$jobs" ffmpeg
executable=ffmpeg
case "$target" in win32-*) executable=ffmpeg.exe ;; esac
cp "$executable" "$output/ffmpeg-$target${executable#ffmpeg}"
./"$executable" -version > "$output/version.txt"
./"$executable" -buildconf > "$output/buildconf.txt" 2>&1
# Real codec-only acceptance: H.264 + AAC round-trip, both tracks retained.
./"$executable" -hide_banner -loglevel error -f lavfi -i testsrc=size=64x64:rate=4 -f lavfi -i sine=frequency=440:sample_rate=16000 -t 1 -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart "$output/codec-smoke.mp4"
./"$executable" -hide_banner -loglevel error -i "$output/codec-smoke.mp4" -map 0:v:0 -f null -
./"$executable" -hide_banner -loglevel error -i "$output/codec-smoke.mp4" -map 0:a:0 -f null -
cp ffbuild/config.mak "$output/ffmpeg-config.mak"
cp "$build_root/ffmpeg.tar.xz" "$output/ffmpeg-$ffmpeg_version-source.tar.xz"
cp "$build_root/ffmpeg.tar.xz.asc" "$output/ffmpeg-$ffmpeg_version-source.tar.xz.asc"
cp "$build_root/ffmpeg-signing-key.asc" "$output/ffmpeg-signing-key.asc"
tar -czf "$output/x264-$x264_commit-source.tar.gz" -C "$build_root/source" x264
cp COPYING.GPLv2 "$output/GPL-2.0.txt"
cp "$build_root/x264/COPYING" "$output/x264-COPYING.txt"
cp "$build_root/signature-status.txt" "$output/source-signature-status.txt"
cd "$repo_root"
node scripts/describe-video-runtime-build.mjs "$target" "$output" "$ffmpeg_version" "$x264_commit" "${configure[*]}"
