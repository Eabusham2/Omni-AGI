; electron-builder's normal 7z path first extracts into
; $PLUGINSDIR\7z-out and then copies that tree atomically into $INSTDIR.
; On the native Windows ARM64 runner that directory copy retained the full
; resources tree but omitted only the top-level Electron executable. Repair
; that exact boundary from the still-materialized staging tree. Existing x64
; installs are untouched because their executable is already present.
!macro customInstall
  ${IfNot} ${FileExists} "$INSTDIR\${APP_EXECUTABLE_FILENAME}"
    ClearErrors
    CopyFiles /SILENT "$PLUGINSDIR\7z-out\${APP_EXECUTABLE_FILENAME}" "$INSTDIR"
    ${If} ${Errors}
      DetailPrint "Could not restore ${APP_EXECUTABLE_FILENAME} from the extraction staging directory."
    ${EndIf}
  ${EndIf}
!macroend
