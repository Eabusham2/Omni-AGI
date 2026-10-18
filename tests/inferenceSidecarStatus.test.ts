import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { assertSavedInferenceSidecarReady } from "../src/main/inferenceSidecarStatus";

const source = {
  mutable_state: { activeGeneration: "m" }, substrate: { persistence: { activeGeneration: "s" } },
  packed_ternary_manifest: { contentSha256: "p", parameterChecksum: "c" },
  packed_inference_sidecar: { format: "omni-derived-inference-sidecar-status", formatVersion: 1,
    status: "verified-committed-copy", refreshRequired: false, runtimeConsumesSidecar: false,
    existingSidecarFullyVerified: true, binding: { mutableGeneration: "m", substrateGeneration: "s", connectedGraphs: null },
    manifestContentSha256: "p", parameterChecksum: "c" }
};

describe("new known-stale derived inference export guard", () => {
  it("accepts matching checked generation and rejects stale/mismatched descriptors", () => {
    expect(() => assertSavedInferenceSidecarReady(source)).not.toThrow();
    expect(() => assertSavedInferenceSidecarReady({ ...source, packed_inference_sidecar: { ...source.packed_inference_sidecar, refreshRequired: true } })).toThrow(/requires refresh/);
    expect(() => assertSavedInferenceSidecarReady({ ...source, mutable_state: { activeGeneration: "later" } })).toThrow(/not bound/);
    expect(() => assertSavedInferenceSidecarReady({ ...source, packed_ternary_manifest: null })).toThrow(/not bound/);
  });
  it("does not call unmarked historical metadata proof of current freshness", () => {
    const { packed_inference_sidecar: _omitted, ...old } = source;
    expect(() => assertSavedInferenceSidecarReady(old)).not.toThrow();
    expect(() => assertSavedInferenceSidecarReady({ ...old, runtime_card: { packedInferenceSidecar: { ...source.packed_inference_sidecar, refreshRequired: true } } })).toThrow(/requires refresh/);
  });
  it("retains the public export's native checkpoint/repack and file verification before archive", () => {
    const ipc = readFileSync(new URL("../src/main/ipc.ts", import.meta.url), "utf8");
    const handler = ipc.slice(ipc.indexOf("handle(IPC.brain.export,"), ipc.indexOf("handle(IPC.brain.importFile,"));
    expect(handler.indexOf("await service.flushNeuralCheckpoint(")).toBeLessThan(handler.indexOf("await repository.exportBundle("));
    const service = readFileSync(new URL("../src/main/brainService.ts", import.meta.url), "utf8");
    const flush = service.slice(service.indexOf("async flushNeuralCheckpoint("), service.indexOf("async createRecoveryPoint("));
    expect(flush).toContain("requireNeuralCheckpointReceipt(");
    expect(flush).toContain("await verifyNeuralCheckpointFiles(");
  });
});
