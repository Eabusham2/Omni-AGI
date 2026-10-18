import { describe, expect, it } from "vitest";
import { clipboardHostHelperPids, clipboardHostLaunch, SystemClipboardHostAdapter } from "../src/main/clipboardHostBridge";

describe("isolated host clipboard bridge", () => {
  it("uses fixed no-shell OS adapter commands and never Electron clipboard APIs", () => {
    const mac = clipboardHostLaunch("macos-pasteboard", "read"); expect(mac.command).toBe("/usr/bin/osascript"); expect(mac.args.join(" ")).toContain("NSPasteboardTypeString");
    const windows = clipboardHostLaunch("windows-powershell", "read"); expect(windows.command).toBe("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe");
    expect(windows.args.join(" ")).toContain("Get-Clipboard"); expect(windows.args.join(" ")).toContain("UnicodeText"); expect(windows.args.join(" ")).not.toContain("electron");
    expect(clipboardHostLaunch("linux-wl-clipboard", "clear").args).toContain("--clear");
  });

  it("streams a bounded page from a tiny synthetic helper child without touching the real clipboard", async () => {
    const script = "process.stdout.write('abcdef🙂ghijklmnopqrstuvwxyz'.repeat(20000))";
    const adapter = new SystemClipboardHostAdapter(process.platform, undefined, { adapter: "macos-pasteboard",
      launch: () => ({ command: process.execPath, args: ["-e", script], environment: { PATH: process.env.PATH ?? "" } }) });
    const pids: number[] = [], page = await adapter.read(undefined, 97, { signal: new AbortController().signal, onSpawn: pid => pids.push(pid) });
    expect(pids).toHaveLength(1); expect(page.complete).toBe(false); expect(page.bytes).toBeLessThanOrEqual(97); expect(Buffer.byteLength(page.text)).toBe(page.bytes); expect(page.next?.afterBytes).toBe(page.bytes);
    expect(clipboardHostHelperPids()).toEqual([]);
  });
});
