#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <mac|linux> <x64|arm64>" >&2
  exit 2
fi

TARGET_PLATFORM="$1"
TARGET_ARCH="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

case "$(uname -s)" in
  Darwin) HOST_PLATFORM="mac" ;;
  Linux) HOST_PLATFORM="linux" ;;
  *)
    echo "Packaging supports macOS and Linux hosts." >&2
    exit 2
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64) HOST_ARCH="x64" ;;
  arm64|aarch64) HOST_ARCH="arm64" ;;
  *)
    echo "Unsupported host architecture: $(uname -m)" >&2
    exit 2
    ;;
esac

if [[ "${TARGET_PLATFORM}" != "mac" && "${TARGET_PLATFORM}" != "linux" ]]; then
  echo "Target platform must be mac or linux." >&2
  exit 2
fi
if [[ "${TARGET_ARCH}" != "x64" && "${TARGET_ARCH}" != "arm64" ]]; then
  echo "Target architecture must be x64 or arm64." >&2
  exit 2
fi
if [[ "${HOST_PLATFORM}" != "${TARGET_PLATFORM}" || "${HOST_ARCH}" != "${TARGET_ARCH}" ]]; then
  echo "A native ${TARGET_PLATFORM}-${TARGET_ARCH} package requires a matching host; current host is ${HOST_PLATFORM}-${HOST_ARCH}." >&2
  exit 2
fi

cd "${REPO_ROOT}"
npm run build
bash scripts/build-engine-posix.sh "${TARGET_PLATFORM}"

if [[ "${TARGET_PLATFORM}" == "mac" ]]; then
  # GitHub renders an absent secret as an exported empty string. Electron-builder
  # treats a present empty CSC_LINK as the working directory, so normalize empty
  # signing variables back to truly absent before deciding whether to sign.
  [[ -n "${CSC_LINK:-}" ]] || unset CSC_LINK
  [[ -n "${CSC_KEY_PASSWORD:-}" ]] || unset CSC_KEY_PASSWORD
  [[ -n "${APPLE_ID:-}" ]] || unset APPLE_ID
  [[ -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]] || unset APPLE_APP_SPECIFIC_PASSWORD
  [[ -n "${APPLE_TEAM_ID:-}" ]] || unset APPLE_TEAM_ID
  [[ -n "${APPLE_API_KEY:-}" ]] || unset APPLE_API_KEY
  [[ -n "${APPLE_API_KEY_ID:-}" ]] || unset APPLE_API_KEY_ID
  [[ -n "${APPLE_API_ISSUER:-}" ]] || unset APPLE_API_ISSUER

  if [[ -z "${CSC_LINK:-}" && -n "${CSC_KEY_PASSWORD:-}" ]]; then
    echo "CSC_KEY_PASSWORD was provided without CSC_LINK." >&2
    exit 2
  fi
  SIGNING_CONFIGURED=0
  if [[ -n "${CSC_LINK:-}" ]]; then
    SIGNING_CONFIGURED=1
  else
    # Never sign a local/CI artifact merely because an unrelated identity is
    # present in the runner keychain.
    export CSC_IDENTITY_AUTO_DISCOVERY=false
  fi

  APPLE_ID_FIELDS=0
  for value in "${APPLE_ID:-}" "${APPLE_APP_SPECIFIC_PASSWORD:-}" "${APPLE_TEAM_ID:-}"; do
    if [[ -n "${value}" ]]; then
      APPLE_ID_FIELDS=$((APPLE_ID_FIELDS + 1))
    fi
  done
  API_KEY_FIELDS=0
  for value in "${APPLE_API_KEY:-}" "${APPLE_API_KEY_ID:-}" "${APPLE_API_ISSUER:-}"; do
    if [[ -n "${value}" ]]; then
      API_KEY_FIELDS=$((API_KEY_FIELDS + 1))
    fi
  done
  if [[ "${APPLE_ID_FIELDS}" -ne 0 && "${APPLE_ID_FIELDS}" -ne 3 ]]; then
    echo "Apple-ID notarization credentials are incomplete." >&2
    exit 2
  fi
  if [[ "${API_KEY_FIELDS}" -ne 0 && "${API_KEY_FIELDS}" -ne 3 ]]; then
    echo "App Store Connect API-key notarization credentials are incomplete." >&2
    exit 2
  fi

  NOTARIZATION_CONFIGURED=0
  if [[ "${APPLE_ID_FIELDS}" -eq 3 || "${API_KEY_FIELDS}" -eq 3 ]]; then
    NOTARIZATION_CONFIGURED=1
    if [[ "${SIGNING_CONFIGURED}" -ne 1 ]]; then
      echo "Notarization credentials require a macOS signing certificate in CSC_LINK." >&2
      exit 2
    fi
  fi

  BUILDER_ARGS=(--mac "--${TARGET_ARCH}" --publish never)
  if [[ "${SIGNING_CONFIGURED}" -eq 1 ]]; then
    BUILDER_ARGS+=(--config.forceCodeSigning=true)
  else
    BUILDER_ARGS+=(--config.forceCodeSigning=false)
  fi
  if [[ "${NOTARIZATION_CONFIGURED}" -eq 1 ]]; then
    BUILDER_ARGS+=(--config.mac.notarize=true)
  else
    BUILDER_ARGS+=(--config.mac.notarize=false)
  fi
  npx electron-builder "${BUILDER_ARGS[@]}"
else
  npx electron-builder --linux "--${TARGET_ARCH}" --publish never
fi
