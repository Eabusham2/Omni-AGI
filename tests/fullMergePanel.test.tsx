import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { FullMergePanel, FullMergeReceipt, reviewedMergeConfig } from "../src/renderer/src/FullMergePanel";
import { DEFAULT_CONFIG } from "../src/shared/types";
import type { FullMergeProgress } from "../src/shared/fullMerge";

const operation: FullMergeProgress = { operationId: "operation", targetBrainId: "new", name: "Connected",
  state: "paused", filesDiscovered: 8, filesCopied: 3, copiedBytes: 3072, logicalBytes: 8192, uniqueBlobBytes: 2048,
  parents: [{ brainId: "one", name: "Parent One", generationSha256: "a".repeat(64), ancestry: [] },
    { brainId: "two", name: "Parent Two", generationSha256: "b".repeat(64), ancestry: [] }],
  reviewedSettings: { ...DEFAULT_CONFIG, idleCognition: false }, reviewedAccess: { credentials: "none", mcp: "disabled", toolPermissions: [] } };

describe("full merge settings and receipts", () => {
  it("shows safe-boundary controls and exact snapshot counts without calling the file count whole-merge completion", () => {
    const markup = renderToStaticMarkup(<FullMergeReceipt operation={operation} onCommand={() => undefined} />);
    expect(markup).toContain("Resume same merge"); expect(markup).toContain("Stop &amp; discard");
    expect(markup).toContain("3 of 8 snapshot files"); expect(markup).toContain("Reviewed settings and receipt");
    expect(markup).toContain("Parent One + Parent Two"); expect(markup).not.toContain("100%");
    const complete = renderToStaticMarkup(<FullMergeReceipt operation={{ ...operation, state: "complete" }} onCommand={() => undefined} onOpen={() => undefined} />);
    expect(complete).toContain("Open merged instance"); expect(complete).not.toContain("Stop &amp; discard");
    const stopped = renderToStaticMarkup(<FullMergeReceipt operation={{ ...operation, state: "cancelled" }} onCommand={() => undefined} />);
    expect(stopped).toContain("Stopped — incomplete output removed"); expect(stopped).not.toContain("Resume same merge");
    const preparing = renderToStaticMarkup(<FullMergeReceipt operation={{ ...operation, state: "preparing", filesCopied: 0, filesDiscovered: 0 }} onCommand={() => undefined} />);
    expect(preparing).toContain("Parent inventory has not finished yet."); expect(preparing).not.toContain("0 of 0");
  });
  it("refuses preview generation without a live bridge and keeps the parents-preserved contract visible", () => {
    const markup = renderToStaticMarkup(<FullMergePanel summaries={[]} initialOperations={[operation]} />);
    expect(markup).toContain("Full merging requires the desktop native runtime.");
    expect(markup).toContain("Originals stay in your library."); expect(markup).toContain("aria-labelledby=\"full-merge-heading\"");
  });
  it("validates complete capacity review without treating a parent's fixed neural shape as a new merged architecture", () => {
    expect(reviewedMergeConfig({ ...DEFAULT_CONFIG, name: " Reviewed ", idleCognition: false }).name).toBe("Reviewed");
    expect(() => reviewedMergeConfig({ ...DEFAULT_CONFIG, idleCognition: false, contextWindowTokens: 1 })).toThrow("capacities");
    expect(() => reviewedMergeConfig({ ...DEFAULT_CONFIG, idleCognition: false, nativeArchitecture: {} })).toThrow("Parent neural architecture");
    expect(() => reviewedMergeConfig(DEFAULT_CONFIG)).toThrow("Active mode");
  });
});
