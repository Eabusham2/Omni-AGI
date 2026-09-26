#!/usr/bin/env bash
set -euo pipefail

EXPECTED_PLATFORM="${1:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKER="${REPO_ROOT}/engine/worker.py"
DIST_ROOT="${REPO_ROOT}/engine-dist"
WORK_ROOT="${REPO_ROOT}/.engine-build"
PYTHON_BIN="${OMNI_BUILD_PYTHON:-python3}"
LOCK_INSTALLER="${REPO_ROOT}/scripts/install-engine-lock.py"

case "$(uname -s)" in
  Darwin) HOST_PLATFORM="mac" ;;
  Linux) HOST_PLATFORM="linux" ;;
  *)
    echo "The POSIX worker builder supports macOS and Linux hosts." >&2
    exit 2
    ;;
esac

if [[ -n "${EXPECTED_PLATFORM}" && "${EXPECTED_PLATFORM}" != "${HOST_PLATFORM}" ]]; then
  echo "Requested ${EXPECTED_PLATFORM} worker build on ${HOST_PLATFORM}." >&2
  exit 2
fi

if [[ ! -f "${WORKER}" ]]; then
  echo "OmniCortex worker not found at ${WORKER}" >&2
  exit 2
fi

if [[ ! -f "${LOCK_INSTALLER}" ]]; then
  echo "Engine dependency lock installer not found at ${LOCK_INSTALLER}" >&2
  exit 2
fi

if [[ "${OMNI_SKIP_BUILD_DEPENDENCY_INSTALL:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${LOCK_INSTALLER}" --verify-only || {
    echo "The existing Python environment does not match the reviewed engine release lock." >&2
    exit 1
  }
else
  "${PYTHON_BIN}" "${LOCK_INSTALLER}"
fi

# These are fixed, repository-local build directories rather than user paths.
rm -rf -- "${DIST_ROOT}" "${WORK_ROOT}"

"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name omni-engine \
  --distpath "${DIST_ROOT}" \
  --workpath "${WORK_ROOT}" \
  --specpath "${WORK_ROOT}" \
  --paths "${REPO_ROOT}/engine" \
  --collect-all torch \
  --collect-all safetensors \
  --collect-all imageio_ffmpeg \
  --collect-all soundfile \
  "${WORKER}"

# Keep imageio-ffmpeg's BSD wrapper, but do not convey the separately licensed
# wheel-provided FFmpeg executable without complete corresponding source and
# build material. Fail closed if PyInstaller placed another FFmpeg executable
# anywhere in the worker distribution.
while IFS= read -r -d '' PACKAGED_FFMPEG; do
  rm -f -- "${PACKAGED_FFMPEG}"
done < <(
  find "${DIST_ROOT}/omni-engine" -type f \
    -path '*/imageio_ffmpeg/binaries/ffmpeg*' \
    ! -name '*.py' ! -name '*.pyc' ! -name '*.pyo' \
    ! -name '*.md' ! -name '*.txt' -print0
)
node "${REPO_ROOT}/scripts/verify-packaged-compliance.mjs" \
  --repo-root "${REPO_ROOT}" \
  --engine-dir "${DIST_ROOT}/omni-engine"

EXECUTABLE="${DIST_ROOT}/omni-engine/omni-engine"
if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Engine packaging completed without producing executable ${EXECUTABLE}" >&2
  exit 1
fi

# PyInstaller's analysis cache can be larger than the worker itself. Native
# GitHub runners only provide 14 GB of storage, so retain the distributable and
# remove the reproducible intermediate directory before electron-builder makes
# three Linux artifacts or two macOS artifacts.
rm -rf -- "${WORK_ROOT}"

echo "Packaged OmniCortex worker: ${EXECUTABLE}"
