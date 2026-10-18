import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import type { WholeBrainInspectionPage } from "../src/shared/wholeBrainInspection";
import { BrainService } from "../src/main/brainService";
import { IPC } from "../src/shared/ipc";

const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));
const page: WholeBrainInspectionPage = { format: "omni-whole-brain-inspection", version: 1, brainId: "fixture", entity: "catalog",
  snapshot: { brainId: "fixture", metadataSha256: "a".repeat(64), coreGeneration: "b".repeat(64), substrateGeneration: "c".repeat(64),
    attentionOverlaySha256: "d".repeat(64), connectedDescriptorSha256: null, connectedGraphStatesSha256: null, catalogSha256: null },
  scope: { id: "root", sourceBrainId: "fixture", nativeRegionPath: [] },
  catalog: { available: false, reason: "missing-historical-owner-registry", records: 0, shards: 0 },
  records: [], anchors: [], sparseAliases: [], hasMore: false, nextCursor: null, cortex: null, substrate: null,
  evidence: { topology: "registered-ownership-and-explicit-regional-bridges", semanticAxisMapping: false,
    tokenNeuronEquivalence: false, liveActivityJoined: false, snapshotConsistent: true, unavailable: ["No registry was fabricated"] } };

it("reads the whole-brain source without selecting/loading an active neural runtime", async () => {
  const service = Object.create(BrainService.prototype) as BrainService;
  const state = service as unknown as Record<string, unknown>;
  const request = vi.fn(async () => page), get = vi.fn(async () => ({ id: "fixture" }));
  const forbidden = vi.fn(() => { throw new Error("No runtime activation for map inspection"); });
  Object.assign(state, { requireRuntimeAdmissionOpen: vi.fn(), preflightStart: forbidden,
    engine: { request, claimForeground: forbidden, start: forbidden }, repository: { get, brainDirectory: () => "/owned/fixture" } });
  await expect(service.queryWholeBrain("fixture", { entity: "catalog", catalogFacet: "overview", pageSize: 128 })).resolves.toEqual(page);
  expect(get).toHaveBeenCalledWith("fixture", false);
  expect(request).toHaveBeenCalledWith("query_whole_brain", { brainId: "fixture", storagePath: "/owned/fixture",
    query: { entity: "catalog", catalogFacet: "overview", pageSize: 128 } }, 30_000, undefined);
  await expect(service.queryWholeBrain("fixture", { storagePath: "/foreign" } as never)).rejects.toThrow("source paths");
  expect(request).toHaveBeenCalledTimes(1); expect(forbidden).not.toHaveBeenCalled();
});
it("exposes only typed whole-brain queries through the existing preload", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  expect(Object.keys(api.wholeBrainInspection)).toEqual(["query"]);
  await api.wholeBrainInspection.query("fixture", { entity: "catalog", catalogFacet: "overview" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.wholeBrainInspection.query, "fixture", { entity: "catalog", catalogFacet: "overview" });
  expect(api.wholeBrainInspection).not.toHaveProperty("load"); expect(api.wholeBrainInspection).not.toHaveProperty("paths");
});
