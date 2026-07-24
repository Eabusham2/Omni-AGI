; electron-builder's verified ARM64 unpacked directory contains the native
; Electron executable, but its generated 7z application archive can omit that
; one top-level file while retaining the entire resources tree. Embed a second,
; installer-owned copy directly from electron-builder's known ARM64 unpacked
; output and materialize it only when the normal extraction path did not. x64
; installers remain unchanged.
!macro customInstall
  !ifdef APP_ARM64
    ${IfNot} ${FileExists} "$INSTDIR\${APP_EXECUTABLE_FILENAME}"
      SetOutPath "$INSTDIR"
      File /oname="${APP_EXECUTABLE_FILENAME}" "${PROJECT_DIR}\release\win-arm64-unpacked\${APP_EXECUTABLE_FILENAME}"
    ${EndIf}
  !endif
!macroend
