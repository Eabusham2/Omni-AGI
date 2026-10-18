import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { clipboardHostHelperPids, clipboardHostLaunch, SystemClipboardHostAdapter,
  type ClipboardAdapterId, type ClipboardHostBridgeTestSeam, type ClipboardProcessLaunch,
  type HostClipboardAdapter, type HostClipboardContext, type HostClipboardCursor, type HostClipboardHolderLease,
  type HostClipboardMutation, type HostClipboardPage } from "../src/main/clipboardHostBridge";
import { ClipboardService } from "../src/main/clipboardService";
import { CLIPBOARD_FORMAT, CLIPBOARD_READ_PAGE_BYTES, type ClipboardCapabilities } from "../src/shared/clipboard";

const roots: string[] = [], adapters: SystemClipboardHostAdapter[] = [];
afterEach(async () => {
  await Promise.allSettled(adapters.splice(0).map(adapter => adapter.shutdown()));
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })));
});

const holderScript = [
  "const fs=require('node:fs');const target=process.argv[1];const chunks=[];",
  "process.stdin.on('data',chunk=>chunks.push(chunk));",
  "process.stdin.on('end',()=>{fs.writeFileSync(target,Buffer.concat(chunks));setInterval(()=>{},1000)});",
  "process.on('SIGTERM',()=>process.exit(0));"
].join("");
const readScript = "const fs=require('node:fs');const target=process.argv[1];if(fs.existsSync(target))process.stdout.write(fs.readFileSync(target));";
const clearScript = "process.stdin.resume();process.stdin.on('end',()=>process.exit(0));";

function seam(adapter: ClipboardAdapterId, target: string, wrongReadback = false): ClipboardHostBridgeTestSeam {
  return { adapter, launch: (_adapter, action): ClipboardProcessLaunch => ({ command: process.execPath,
    args: ["-e", action === "read" && wrongReadback ? "process.stdout.write('wrong')" : action === "read" ? readScript : action === "clear" ? clearScript : holderScript, target],
    environment: { PATH: process.env.PATH ?? "" } }) };
}
function context(holders: HostClipboardHolderLease[], pids: number[]) {
  return { signal: new AbortController().signal, beforeDispatch: async () => undefined,
    onSpawn: (pid: number): void => { pids.push(pid); }, onRetainedHolder: (holder: HostClipboardHolderLease): void => { holders.push(holder); } };
}

class SyntheticRetainedHost implements HostClipboardAdapter {
  stopped = 0; private resolveExit!: () => void; private readonly exit = new Promise<void>(resolve => { this.resolveExit = resolve; });
  async capabilities(): Promise<ClipboardCapabilities> { return { format: CLIPBOARD_FORMAT, scope: "host", implemented: true, adapter: "linux-wl-clipboard",
    contentTypes: ["text/plain"], maxReadPageBytes: CLIPBOARD_READ_PAGE_BYTES, readIsBounded: true, write: true, clear: true,
    authority: "current-host-permission-only", detail: "synthetic retained owner" }; }
  read(_cursor: HostClipboardCursor | undefined, _maximumBytes: number, _context: HostClipboardContext): Promise<HostClipboardPage> { throw new Error("not used"); }
  async write(_text: string, context: HostClipboardContext): Promise<HostClipboardMutation> {
    await context.beforeDispatch?.(); context.signal.throwIfAborted(); context.onSpawn(44001);
    const holder: HostClipboardHolderLease = { id: "synthetic-holder", pid: 44001, adapter: "linux-wl-clipboard", startedAt: new Date().toISOString(), exit: this.exit,
      stop: async () => { if (this.stopped++ === 0) this.resolveExit(); } };
    context.onRetainedHolder?.(holder); return { adapter: "linux-wl-clipboard", completedAt: new Date().toISOString(), holder };
  }
  clear(context: HostClipboardContext): Promise<HostClipboardMutation> { return this.write("", context); }
  async shutdown(): Promise<void> { if (this.stopped++ === 0) this.resolveExit(); await this.exit; }
}

describe("owned Linux text clipboard holders", () => {
  it("uses documented foreground modes and explicit UTF-8 text operations", () => {
    expect(clipboardHostLaunch("linux-wl-clipboard", "write").args).toEqual(["--foreground", "--type", "text/plain;charset=utf-8"]);
    expect(clipboardHostLaunch("linux-wl-clipboard", "read").args).toEqual(["--no-newline", "--type", "text/plain;charset=utf-8"]);
    const xclip = clipboardHostLaunch("linux-xclip", "write").args;
    expect(xclip).toContain("-quiet"); expect(xclip).not.toContain("-silent"); expect(xclip).toEqual(expect.arrayContaining(["-selection", "clipboard", "-loops", "0"]));
  });

  it("requires exact hash-only readback, replaces only its prior owner, and stops its owner at shutdown", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-linux-clipboard-")); roots.push(root); const target = join(root, "selection.bin");
    const adapter = new SystemClipboardHostAdapter("linux", undefined, seam("linux-wl-clipboard", target)); adapters.push(adapter);
    const holders: HostClipboardHolderLease[] = [], pids: number[] = [], first = await adapter.write("first🙂", context(holders, pids));
    expect(first.holder).toBe(holders[0]); expect(pids.length).toBeGreaterThanOrEqual(2); expect(clipboardHostHelperPids()).toContain(first.holder!.pid);
    const second = await adapter.write("second", context(holders, pids));
    await first.holder!.exit; expect(second.holder?.pid).not.toBe(first.holder?.pid); expect(clipboardHostHelperPids()).toContain(second.holder!.pid);
    await adapter.clear(context(holders, pids)); await second.holder!.exit; expect(clipboardHostHelperPids()).not.toContain(second.holder!.pid);
    const third = await adapter.write("third", context(holders, pids)); await adapter.shutdown(); await third.holder!.exit;
    expect(clipboardHostHelperPids()).not.toContain(third.holder!.pid);
  });

  it("keeps a mismatched publication as an owned uncertain holder until explicit shutdown", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-linux-clipboard-mismatch-")); roots.push(root); const target = join(root, "selection.bin");
    const adapter = new SystemClipboardHostAdapter("linux", undefined, seam("linux-xclip", target, true)); adapters.push(adapter);
    const holders: HostClipboardHolderLease[] = [], pids: number[] = [];
    await expect(adapter.write("expected", context(holders, pids))).rejects.toThrow(/readback did not match|more bytes|uncertain/i);
    expect(holders).toHaveLength(1); expect(clipboardHostHelperPids()).toContain(holders[0]!.pid);
    await adapter.shutdown(); await holders[0]!.exit; expect(clipboardHostHelperPids()).not.toContain(holders[0]!.pid);
  });

  it("keeps monitoring a completed owned holder across brain Unload and releases it at Studio shutdown", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-linux-clipboard-service-")); roots.push(root); const host = new SyntheticRetainedHost(); let checks = 0;
    const brainId = "51515151-5151-4515-8515-515151515151", service = new ClipboardService({ brainDirectory: async () => root,
      assertBrainOwner: async () => undefined, requireActiveOwner: async () => undefined, currentPermission: async () => ({ level: "ask", revision: "current" }),
      checkResources: async () => { checks++; }, reserveWrite: async () => undefined, host });
    const receipt = await service.execute(brainId, "system.clipboard", "write", { scope: "host", contentType: "text/plain", text: "retained" },
      "retained-write", new AbortController().signal) as { state: string };
    expect(receipt.state).toBe("complete"); const beforeUnload = checks; await service.forget(brainId);
    await new Promise(resolve => setTimeout(resolve, 300)); expect(checks).toBeGreaterThan(beforeUnload); expect(host.stopped).toBe(0);
    await service.shutdown(); expect(host.stopped).toBe(1);
  });
});
