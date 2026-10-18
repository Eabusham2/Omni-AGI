import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor } from "../src/main/engineSupervisor";
import { checkedRouterSpikeActivity, validateRouterSpikeQuery } from "../src/main/routerSpikeProtocol";
import type { RouterSpikeActivity } from "../src/shared/routerActivity";

const epoch = "a".repeat(32);
const query = { enabled: true, captureId: "one", epoch };
const receipt = (changes: Partial<RouterSpikeActivity> = {}): RouterSpikeActivity => ({
  brainId: "b", source: "actual-lif-hard-spikes", owner: "root.router.population", channelIdentity: "brainId:root.router.population:channelIndex",
  available: true, enabled: true, observed: true, captureId: "one", epoch, totalChannels: 10000,
  channelStart: 9998, admittedCount: 2, requestedCount: 2, ringBytes: 65505, frameCapacity: 1985,
  lastSequence: 1, firstAvailableSequence: 1, nextSequence: 1, droppedFrames: 0, missedFrames: 0, hasMore: false,
  frames: [{ sequence: 1, observedAtUnix: 1_792_000_000, monotonicNs: "12345678901234567", channelStart: 9998, spikes: [1, 0] }],
  semanticNeuronMapping: false, synapseFiringObserved: false, ...changes
});

describe("new exact router spike protocol (no worker/model startup)", () => {
  it("checks source owner, epoch, binary channel window and exact ordered sequence", () => {
    expect(checkedRouterSpikeActivity(receipt(), "b", query).frames[0]!.spikes).toEqual([1, 0]);
    for (const value of [receipt({ brainId: "other" }), receipt({ epoch: "b".repeat(32) }),
      receipt({ semanticNeuronMapping: true as false }), receipt({ frames: [{ ...receipt().frames[0]!, spikes: [0.7, 0] as Array<0 | 1> }] }),
      receipt({ nextSequence: 0 }), receipt({ admittedCount: 3 })])
      expect(() => checkedRouterSpikeActivity(value, "b", query)).toThrow();
    expect(() => validateRouterSpikeQuery({ enabled: true, captureId: "one", afterSequence: 10 })).toThrow(/epoch/);
  });

  it("keeps unloaded/unavailable observation explicit with no synthetic frame", () => {
    const unavailable = receipt({ available: false, enabled: false, observed: false, epoch: null, totalChannels: null,
      channelStart: 0, admittedCount: 0, requestedCount: 0, ringBytes: 0, frameCapacity: 0, lastSequence: 0,
      firstAvailableSequence: 0, nextSequence: 0, frames: [], reason: "brain-not-loaded" });
    expect(checkedRouterSpikeActivity(unavailable, "b", { ...query, afterSequence: 1 }).available).toBe(false);
    expect(() => checkedRouterSpikeActivity({ ...unavailable, frames: receipt().frames }, "b", query)).toThrow();
  });

  it("uses only an existing child's non-terminating live control lane for polls/stops", async () => {
    const rawRequest = vi.fn(async () => receipt());
    const session = { workerRole: "neural", child: { killed: false, exitCode: null, stdin: { writable: true } },
      rawRequest, start: vi.fn(() => { throw new Error("must not start"); }) };
    await EngineSupervisor.prototype.routerActivity.call(session as unknown as EngineSupervisor, "b", "/private/b", query);
    expect(rawRequest).toHaveBeenCalledWith("router_activity_control", { brainId: "b", storagePath: "/private/b", query }, 10_000, undefined, true);
    expect(session.start).not.toHaveBeenCalled();
    await expect(EngineSupervisor.prototype.routerActivity.call({ ...session, child: undefined } as unknown as EngineSupervisor, "b", "/private/b", query)).rejects.toThrow(/already explicitly loaded/);
    expect(session.start).not.toHaveBeenCalled();
  });

  it("serializes first allocation and refuses a replaced child without starting another", async () => {
    const rawRequest = vi.fn(async () => receipt());
    const reservation = { controller: new AbortController(), dispatched: false };
    const release = vi.fn(), finish = vi.fn();
    const session = { workerRole: "neural", child: { killed: false, exitCode: null, stdin: { writable: true } }, rawRequest,
      createRequestReservation: vi.fn(() => reservation), reserveRequestTurn: vi.fn(() => ({ ready: Promise.resolve(), release })),
      instanceDispatchHeld: vi.fn(() => false), transitionActivity: vi.fn(), finishRequestReservation: finish };
    const initial = { enabled: true, captureId: "one" };
    await EngineSupervisor.prototype.routerActivity.call(session as unknown as EngineSupervisor, "b", "/private/b", initial);
    expect(rawRequest).toHaveBeenCalledWith("router_activity", { brainId: "b", storagePath: "/private/b", query: initial }, 10_000, undefined, true);
    expect(release).toHaveBeenCalledOnce(); expect(finish).toHaveBeenCalledWith(reservation);
    rawRequest.mockClear();
    session.reserveRequestTurn = vi.fn(() => ({ ready: Promise.resolve().then(() => { session.child = { killed: false, exitCode: null, stdin: { writable: true } }; }), release }));
    await expect(EngineSupervisor.prototype.routerActivity.call(session as unknown as EngineSupervisor, "b", "/private/b", initial)).rejects.toThrow(/owner changed/);
    expect(rawRequest).not.toHaveBeenCalled();
  });
});
