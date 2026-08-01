#!/usr/bin/env bash
set -euo pipefail

EXPECTED_PLATFORM="${1:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKER="${REPO_ROOT}/engine/worker.py"
DIST_ROOT="${REPO_ROOT}/engine-dist"
WORK_ROOT="${REPO_ROOT}/.engine-build"
PYTHON_BIN="${OMNI_BUILD_PYTHON:-python3}"

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

if [[ "${OMNI_SKIP_BUILD_DEPENDENCY_INSTALL:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -c \
    "import PyInstaller; major=int(PyInstaller.__version__.split('.')[0]); assert major == 6" \
    || {
      echo "Protected runtime evolution requires an existing PyInstaller 6.x; automatic dependency installation is disabled." >&2
      exit 1
    }
else
  "${PYTHON_BIN}" -m pip install --disable-pip-version-check "pyinstaller>=6.10,<7"
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
