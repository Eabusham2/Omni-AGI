#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-unsigned}"
case "$MODE" in
  simulator|unsigned|signed) ;;
  *) echo "Usage: $0 [simulator|unsigned|signed]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT="$REPO_ROOT/mobile/ios/OmniCompanion.xcodeproj"
SCHEME="OmniCompanion"
RELEASE_DIR="$REPO_ROOT/release/mobile"
VERSION="$(node -p "require('$REPO_ROOT/package.json').version")"
TASK_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/omni-ios-package.XXXXXX")"
trap 'rm -rf "$TASK_TEMP"' EXIT
mkdir -p "$RELEASE_DIR"

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

if [[ "$MODE" == "simulator" ]]; then
  xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration Release \
    -destination "generic/platform=iOS Simulator" \
    -derivedDataPath "$TASK_TEMP/DerivedData" \
    CODE_SIGNING_ALLOWED=NO \
    build
  APP="$TASK_TEMP/DerivedData/Build/Products/Release-iphonesimulator/Omni AGI Companion.app"
  ARTIFACT="$RELEASE_DIR/Omni-AGI-Companion-${VERSION}-iOS-Simulator.zip"
  test -d "$APP"
  ditto -c -k --sequesterRsrc --keepParent "$APP" "$ARTIFACT"
else
  ARCHIVE="$TASK_TEMP/OmniCompanion.xcarchive"
  BUILD_ARGUMENTS=(
    -project "$PROJECT"
    -scheme "$SCHEME"
    -configuration Release
    -destination "generic/platform=iOS"
    -archivePath "$ARCHIVE"
  )
  if [[ "$MODE" == "signed" ]]; then
    : "${OMNI_IOS_TEAM_ID:?Set OMNI_IOS_TEAM_ID to your Apple Developer team identifier.}"
    BUILD_ARGUMENTS+=(DEVELOPMENT_TEAM="$OMNI_IOS_TEAM_ID")
    if [[ -n "${OMNI_IOS_PROVISIONING_PROFILE:-}" ]]; then
      BUILD_ARGUMENTS+=(CODE_SIGN_STYLE=Manual PROVISIONING_PROFILE_SPECIFIER="$OMNI_IOS_PROVISIONING_PROFILE")
    else
      BUILD_ARGUMENTS+=(CODE_SIGN_STYLE=Automatic)
    fi
    if [[ -n "${OMNI_IOS_SIGNING_IDENTITY:-}" ]]; then
      BUILD_ARGUMENTS+=(CODE_SIGN_IDENTITY="$OMNI_IOS_SIGNING_IDENTITY")
    fi
    if [[ -n "${OMNI_IOS_BUNDLE_ID:-}" ]]; then
      BUILD_ARGUMENTS+=(PRODUCT_BUNDLE_IDENTIFIER="$OMNI_IOS_BUNDLE_ID")
    fi
    if [[ "${OMNI_IOS_ALLOW_PROVISIONING_UPDATES:-0}" == "1" ]]; then
      BUILD_ARGUMENTS+=(-allowProvisioningUpdates)
    fi
  else
    BUILD_ARGUMENTS+=(CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO)
  fi
  xcodebuild "${BUILD_ARGUMENTS[@]}" archive
  APP="$ARCHIVE/Products/Applications/Omni AGI Companion.app"
  test -d "$APP"
  if [[ "$MODE" == "signed" ]]; then
    codesign --verify --deep --strict "$APP"
  fi
  mkdir -p "$TASK_TEMP/Payload"
  ditto "$APP" "$TASK_TEMP/Payload/Omni AGI Companion.app"
  SUFFIX=$([[ "$MODE" == "signed" ]] && echo signed || echo unsigned)
  ARTIFACT="$RELEASE_DIR/Omni-AGI-Companion-${VERSION}-iOS-${SUFFIX}.ipa"
  (cd "$TASK_TEMP" && /usr/bin/zip -qry "$ARTIFACT" Payload)
fi

node "$REPO_ROOT/scripts/verify-packaged-compliance.mjs" \
  --repo-root "$REPO_ROOT" \
  --platform ios \
  --artifact "$ARTIFACT"
shasum -a 256 "$ARTIFACT"
echo "Created $ARTIFACT"
