import { expect, it } from "vitest";
import { EngineRequestError, BackgroundRequestDeferredError } from "../src/main/engineSupervisor";
import { sparseSubstrateRpcError } from "../src/main/sparseSubstrateRpcErrors";

it("accepts only exact current-owner native pre-mutation cancellation proof", () => {
  const data = { brainId: "owner", nativeSparseTransfer: true, safeBoundary: true, cancelled: true, committed: false, nativeOwnerReleased: false };
  const proof = new EngineRequestError("cancelled before rows", -32800, data);
  expect(sparseSubstrateRpcError(proof, "owner")).toMatchObject({ code: "cancelled-unapplied", committed: false });
  for (const changed of [{ brainId: "foreign" }, { committed: null }, { safeBoundary: false }, { nativeSparseTransfer: false }, { nativeOwnerReleased: true }]) {
    const unknown = new EngineRequestError("unproven", -32800, { ...data, ...changed });
    expect(sparseSubstrateRpcError(unknown, "owner")).toBe(unknown);
  }
  const transport = new Error("worker was cancelled");
  expect(sparseSubstrateRpcError(transport, "owner")).toBe(transport);
});

it("separates genuine waiting/review boundaries from released or uncertain publication", () => {
  expect(sparseSubstrateRpcError(new BackgroundRequestDeferredError("sparse_substrate_apply"), "owner")).toMatchObject({ code: "boundary-required" });
  expect(sparseSubstrateRpcError(new EngineRequestError("selected row changed", -32033, { brainId: "owner", nativeSparseTransfer: true, needsReview: true }), "owner")).toMatchObject({ code: "needs-review" });
  for (const committed of [true, false, null]) {
    expect(sparseSubstrateRpcError(new EngineRequestError("reload", -32031, { brainId: "owner", nativeSparseTransfer: true, reloadRequired: true, nativeOwnerReleased: true, committed }), "owner")).toMatchObject({ code: "reload-required", committed });
  }
});
