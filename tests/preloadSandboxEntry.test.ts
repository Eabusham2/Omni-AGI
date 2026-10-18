import { webcrypto } from "node:crypto";
import { resolve } from "node:path";
import { runInNewContext } from "node:vm";
import { build } from "esbuild";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { IPC } from "../src/shared/ipc";
import type { OmniApi } from "../src/shared/types";
import type { SparseSubstrateJobView, SparseSubstrateUiEvent, SparseSubstrateUiEventEnvelope, SparseSubstrateUiSubscriptionAck } from "../src/shared/sparseSubstrateTransfer";

let code: string;
beforeAll(async () => {
  // Bundle the complete shipped entry, not an extracted callback. Browser mode
  // rejects Node-only dependencies; Electron is the only external require.
  const bundled = await build({ entryPoints: [resolve("src/preload/index.ts")], bundle: true, write: false,
    platform: "browser", format: "cjs", target: "es2022", external: ["electron"], metafile: true, logLevel: "silent" });
  const external = Object.values(bundled.metafile!.outputs).flatMap(output => output.imports.filter(value => value.external).map(value => value.path));
  expect([...new Set(external)]).toEqual(["electron"]);
  expect(bundled.outputFiles).toHaveLength(1);
  code = bundled.outputFiles![0]!.text;
});

function harness(isMainFrame = true) {
  const exposed: Record<string, unknown> = {}, listeners = new Map<string, Set<(...values: unknown[]) => void>>();
  const acknowledgements = new Map<string, (value: SparseSubstrateUiSubscriptionAck) => void>();
  const phases: string[] = [];
  const invoke = vi.fn((channel: string, ...args: unknown[]): Promise<unknown> => {
    phases.push(`invoke:${channel}`);
    if (channel === IPC.window.platform) return Promise.resolve("darwin");
    if (channel === IPC.brain.list) return Promise.resolve([]);
    if (channel === IPC.substrate.subscribe) return new Promise(resolveAck => { acknowledgements.set(String(args[1]), resolveAck); });
    return Promise.resolve(undefined);
  });
  const electron = {
    contextBridge: { exposeInMainWorld: vi.fn((name: string, value: unknown) => { exposed[name] = value; }) },
    ipcRenderer: {
      invoke,
      on: vi.fn((channel: string, listener: (...values: unknown[]) => void) => {
        phases.push(`listen:${channel}`); const current = listeners.get(channel) ?? new Set(); current.add(listener); listeners.set(channel, current);
      }),
      removeListener: vi.fn((channel: string, listener: (...values: unknown[]) => void) => { listeners.get(channel)?.delete(listener); })
    },
    webUtils: { getPathForFile: vi.fn(() => "") }
  };
  const requireModule = vi.fn((name: string) => {
    if (name !== "electron") throw new Error(`Sandbox preload cannot require ${name}`);
    return electron;
  });
  // Only the sandbox-supported Electron process flag and browser WebCrypto
  // are supplied. No Buffer, fs, node crypto import, polyfill or Node globals.
  const scope = { require: requireModule, crypto: webcrypto, process: { isMainFrame }, console: { error: vi.fn() } };
  runInNewContext(code, scope, { filename: "sandbox-preload-entry.cjs", timeout: 1000 });
  return { api: exposed.omni as OmniApi | undefined, exposed, electron, invoke, phases, requireModule, acknowledgements,
    emit: (channel: string, value: unknown) => { for (const listener of listeners.get(channel) ?? []) listener({ sender: "not exposed" }, value); }, listeners };
}

const now = "2026-10-18T12:00:00.000Z";
const job = (revision: number, brainId = "current-brain"): SparseSubstrateJobView => ({
  format: "omni-sparse-substrate-job-view", version: 1, scope: "native-sparse-assemblies", id: "fixture-job", brainId,
  originalBrainId: brainId, requestId: "fixture-request", requestSha256: "a".repeat(64), planId: "fixture-plan",
  planSha256: "b".repeat(64), state: "queued", phase: "queued", createdAt: now, updatedAt: now, revision, liveAuthorityAvailable: false
});
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };

describe("complete sandbox-compatible Studio preload entry", () => {
  it("executes the whole entry with electron as its only require and exposes typed IPC without eager work", async () => {
    const value = harness();
    expect(value.requireModule.mock.calls).toEqual([["electron"]]);
    expect(value.electron.contextBridge.exposeInMainWorld).toHaveBeenCalledOnce();
    expect(Object.keys(value.exposed)).toEqual(["omni"]); expect(value.invoke).not.toHaveBeenCalled();
    expect(value.api).toBeDefined();
    await expect(value.api!.window.platform()).resolves.toBe("darwin");
    await expect(value.api!.brain.list()).resolves.toEqual([]);
    expect(value.invoke.mock.calls).toEqual([[IPC.window.platform], [IPC.brain.list]]);
    for (const forbidden of ["require", "ipcRenderer", "contextBridge", "process", "crypto"]) expect(Object.hasOwn(value.api!, forbidden)).toBe(false);
  });

  it("uses browser UUIDs and keeps subscription identity, owner, buffering and disposal at the typed IPC boundary", async () => {
    const value = harness(), received: SparseSubstrateUiEvent[] = [];
    const dispose = value.api!.substrate.onEvent("current-brain", event => received.push(event));
    const subscribe = value.invoke.mock.calls.find(call => call[0] === IPC.substrate.subscribe)!;
    const subscriptionId = subscribe[2] as string;
    expect(subscriptionId).toMatch(/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i);
    expect(value.phases.slice(0, 2)).toEqual([`listen:${IPC.substrate.event}`, `invoke:${IPC.substrate.subscribe}`]);
    const event = (revision: number, owner = "current-brain"): SparseSubstrateUiEventEnvelope => ({ subscriptionId, brainId: owner,
      event: { format: "omni-sparse-substrate-ui-event", version: 1, brainId: owner, kind: "job", job: job(revision, owner), observedAt: now } });
    value.emit(IPC.substrate.event, { ...event(2), subscriptionId: "foreign" });
    value.emit(IPC.substrate.event, event(2, "another-brain"));
    value.emit(IPC.substrate.event, event(2)); expect(received).toEqual([]);
    value.acknowledgements.get(subscriptionId)!({ brainId: "current-brain", subscriptionId, subscribed: true,
      initial: { entries: [job(1)], total: 1 }, observedAt: now });
    await flush();
    expect(received.map(item => item.kind === "job" ? item.job.revision : item.kind)).toEqual([1, 2]);
    value.emit(IPC.substrate.event, event(1)); expect(received).toHaveLength(2);
    dispose(); dispose();
    expect(value.listeners.get(IPC.substrate.event)?.size).toBe(0);
    expect(value.invoke.mock.calls.filter(call => call[0] === IPC.substrate.unsubscribe)).toEqual([[IPC.substrate.unsubscribe, "current-brain", subscriptionId]]);
    value.emit(IPC.substrate.event, event(3)); expect(received).toHaveLength(2);
    const next = value.api!.substrate.onEvent("current-brain", () => undefined);
    const secondId = value.invoke.mock.calls.filter(call => call[0] === IPC.substrate.subscribe)[1]![2];
    expect(secondId).not.toBe(subscriptionId); next();
  });

  it("does not expose the Studio API to a subframe", () => {
    const value = harness(false);
    expect(value.exposed).toEqual({}); expect(value.electron.contextBridge.exposeInMainWorld).not.toHaveBeenCalled();
    expect(value.invoke).not.toHaveBeenCalled(); expect(value.listeners.size).toBe(0);
  });
});
