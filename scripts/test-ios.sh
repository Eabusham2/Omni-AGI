#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

if [[ -n "${OMNI_IOS_SIMULATOR_ID:-}" ]]; then
  DESTINATION="platform=iOS Simulator,id=$OMNI_IOS_SIMULATOR_ID"
else
  DESTINATION="platform=iOS Simulator,name=${OMNI_IOS_SIMULATOR_NAME:-iPhone 16 Pro}"
fi

xcodebuild \
  -project "$REPO_ROOT/mobile/ios/OmniCompanion.xcodeproj" \
  -scheme OmniCompanion \
  -destination "$DESTINATION" \
  -resultBundlePath "${OMNI_IOS_RESULT_BUNDLE:-$REPO_ROOT/mobile/ios/OmniCompanion.xcresult}" \
  test
