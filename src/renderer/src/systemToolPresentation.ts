export interface NativeSystemToolExamples {
  windows: boolean;
  directory: string;
  shellCommand: string;
  pythonEntryPath: string;
}

/**
 * Resolve visible examples from the renderer host only. Neural schemas remain
 * platform-neutral; this adapter text is never part of the ground-up
 * curriculum or a hidden prompt.
 */
export function nativeSystemToolExamples(
  navigatorPlatform: string
): NativeSystemToolExamples {
  const windows = navigatorPlatform.startsWith("Win");
  return windows
    ? {
        windows: true,
        directory: "C:\\Users\\Public",
        shellCommand: "Get-Date",
        pythonEntryPath: "C:\\path\\to\\script.py"
      }
    : {
        windows: false,
        directory: "/tmp",
        shellCommand: "date",
        pythonEntryPath: "/tmp/script.py"
      };
}
