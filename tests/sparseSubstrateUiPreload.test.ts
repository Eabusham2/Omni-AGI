import { beforeEach, describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
import type { SparseSubstrateJobView, SparseSubstrateUiEventEnvelope,
  SparseSubstrateUiSubscriptionAck } from "../src/shared/sparseSubstrateTransfer";

const bridge = vi.hoisted(() => ({ expose: vi.fn(), invoke: vi.fn(), on: vi.fn(), off: vi.fn() }));
vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: bridge.on, removeListener: bridge.off },
  webUtils: { getPathForFile: vi.fn() }
}));
import "../src/preload/index";

const api = (bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi).substrate;
const job = (revision: number): SparseSubstrateJobView => ({
  format: "omni-sparse-substrate-job-view", version: 1, scope: "native-sparse-assemblies",
  id: "job-one", brainId: "brain-one", originalBrainId: "brain-one", requestId: "request-one",
  requestSha256: "a".repeat(64), planId: "plan-one", planSha256: "b".repeat(64),
  actionEventId: "action-one", chatTurnId: "turn-one", state: "queued", phase: "queued",
  createdAt: "2026-10-15T00:00:00.000Z", updatedAt: "2026-10-15T00:00:00.000Z",
  revision, liveAuthorityAvailable: false
});
const envelope = (subscriptionId: string, revision: number, brainId = "brain-one"): SparseSubstrateUiEventEnvelope => ({
  subscriptionId, brainId, event: { format: "omni-sparse-substrate-ui-event", version: 1, brainId,
    kind: "job", job: { ...job(revision), brainId }, observedAt: "2026-10-15T00:00:01.000Z" }
});
const acknowledgement = (brainId: string, subscriptionId: string,
  entries: SparseSubstrateJobView[] = []): SparseSubstrateUiSubscriptionAck => ({
  brainId, subscriptionId, subscribed: true, initial: { entries, total: entries.length },
  observedAt: "2026-10-15T00:00:02.000Z"
});

beforeEach(() => { bridge.invoke.mockReset(); bridge.on.mockReset(); bridge.off.mockReset(); });

describe("sparse substrate preload UI boundary", () => {
  it("maps read/Stop IPC and buffers owner events until a listener-first subscription acknowledgement", async () => {
    bridge.invoke.mockResolvedValue(undefined);
    await api.jobs("brain-one", { after: "job-zero", limit: 20 });
    await api.job("brain-one", "job-one"); await api.plan("brain-one", "plan-one"); await api.stop("brain-one", "job-one");
    expect(bridge.invoke.mock.calls).toEqual([
      [IPC.substrate.jobs, "brain-one", { after: "job-zero", limit: 20 }],
      [IPC.substrate.job, "brain-one", "job-one"], [IPC.substrate.plan, "brain-one", "plan-one"],
      [IPC.substrate.stop, "brain-one", "job-one"]
    ]);

    let accept!: (value: SparseSubstrateUiSubscriptionAck) => void;
    const subscribeAck = new Promise<SparseSubstrateUiSubscriptionAck>(resolve => { accept = resolve; });
    bridge.invoke.mockImplementation((channel: string) => channel === IPC.substrate.subscribe ? subscribeAck : Promise.resolve(true));
    const listener = vi.fn(), dispose = api.onEvent("brain-one", listener);
    const subscribe = bridge.invoke.mock.calls.find(([channel]) => channel === IPC.substrate.subscribe)!;
    const subscriptionId = String(subscribe[2]);
    const wrapped = bridge.on.mock.calls.find(([channel]) => channel === IPC.substrate.event)![1] as
      (_event: unknown, value: SparseSubstrateUiEventEnvelope) => void;
    expect(bridge.on.mock.invocationCallOrder[0]).toBeLessThan(bridge.invoke.mock.invocationCallOrder.at(-1)!);
    wrapped({}, envelope(subscriptionId, 1));
    expect(listener).not.toHaveBeenCalled();
    accept(acknowledgement("brain-one", subscriptionId));
    await Promise.resolve(); await Promise.resolve();
    expect(listener).toHaveBeenCalledTimes(1);
    wrapped({}, envelope(subscriptionId, 1));
    wrapped({}, envelope("foreign-subscription", 2));
    wrapped({}, envelope(subscriptionId, 2, "other-brain"));
    expect(listener).toHaveBeenCalledTimes(1);
    wrapped({}, envelope(subscriptionId, 2));
    expect(listener).toHaveBeenCalledTimes(2);
    dispose();
    expect(bridge.off).toHaveBeenCalledWith(IPC.substrate.event, wrapped);
    expect(bridge.invoke).toHaveBeenCalledWith(IPC.substrate.unsubscribe, "brain-one", subscriptionId);
  });

  it("surfaces subscribe refusal once and never revives a disposed late acknowledgement", async () => {
    bridge.invoke.mockImplementation((channel: string) => channel === IPC.substrate.subscribe
      ? Promise.reject(new Error("subscription refused")) : Promise.resolve(false));
    const listener = vi.fn(); api.onEvent("brain-one", listener);
    await Promise.resolve(); await Promise.resolve();
    expect(listener).toHaveBeenCalledWith(expect.objectContaining({ kind: "error", brainId: "brain-one", error: "subscription refused" }));
    expect(bridge.off).toHaveBeenCalledWith(IPC.substrate.event, expect.any(Function));

    let accept!: (value: SparseSubstrateUiSubscriptionAck) => void;
    const subscribeAck = new Promise<SparseSubstrateUiSubscriptionAck>(resolve => { accept = resolve; });
    bridge.invoke.mockImplementation((channel: string) => channel === IPC.substrate.subscribe ? subscribeAck : Promise.resolve(true));
    const second = vi.fn(), dispose = api.onEvent("brain-two", second);
    const subscribe = bridge.invoke.mock.calls.filter(([channel]) => channel === IPC.substrate.subscribe).at(-1)!;
    const subscriptionId = String(subscribe[2]); dispose();
    accept(acknowledgement("brain-two", subscriptionId));
    await Promise.resolve(); await Promise.resolve();
    expect(second).not.toHaveBeenCalled();
    expect(bridge.invoke.mock.calls.filter(call => call[0] === IPC.substrate.unsubscribe && call[2] === subscriptionId)).toHaveLength(2);
  });

  it("merges the owner snapshot before coalesced live revisions and surfaces owner validation failure", async () => {
    let accept!: (value: SparseSubstrateUiSubscriptionAck) => void;
    const subscribeAck = new Promise<SparseSubstrateUiSubscriptionAck>(resolve => { accept = resolve; });
    bridge.invoke.mockImplementation((channel: string) => channel === IPC.substrate.subscribe ? subscribeAck : Promise.resolve(true));
    const listener = vi.fn(); api.onEvent("brain-one", listener);
    const subscribe = bridge.invoke.mock.calls.find(([channel]) => channel === IPC.substrate.subscribe)!;
    const subscriptionId = String(subscribe[2]);
    const wrapped = bridge.on.mock.calls.find(([channel]) => channel === IPC.substrate.event)![1] as
      (_event: unknown, value: SparseSubstrateUiEventEnvelope) => void;
    wrapped({}, envelope(subscriptionId, 3)); wrapped({}, envelope(subscriptionId, 4)); wrapped({}, envelope(subscriptionId, 2));
    accept(acknowledgement("brain-one", subscriptionId, [job(2), { ...job(1), id: "job-two", requestId: "request-two" }]));
    await Promise.resolve(); await Promise.resolve();
    expect(listener.mock.calls.map(([event]) => [event.job.id, event.job.revision])).toEqual([
      ["job-one", 2], ["job-two", 1], ["job-one", 4]
    ]);

    bridge.invoke.mockImplementation((channel: string) => channel === IPC.substrate.subscribe
      ? Promise.reject(new Error("foreign owner refused")) : Promise.resolve(false));
    const refused = vi.fn(); api.onEvent("foreign-brain", refused);
    await Promise.resolve(); await Promise.resolve();
    expect(refused).toHaveBeenCalledWith(expect.objectContaining({ kind: "error", brainId: "foreign-brain",
      error: "foreign owner refused" }));
  });
});
