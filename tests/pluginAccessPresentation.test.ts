import { describe, expect, it } from "vitest";
import type { PluginInstanceAccess } from "../src/shared/plugins";
import { refreshedPluginAccess } from "../src/renderer/src/pluginAccessPresentation";

describe("plugin learning status and access drafts", () => {
  const saved: PluginInstanceAccess = { brainId: "b", pluginId: "p", contentSha256: "a".repeat(64), enabled: false,
    capabilities: [], toolPermissions: [], revision: 2, updatedAt: "2026-10-11T00:00:00Z" };
  it("updates a genuine receipt without silently clearing unsaved toggles", () => {
    const draft = { ...saved, enabled: true };
    const complete = { ...saved, semanticLearning: { brainId: "b", pluginId: "p", contentSha256: saved.contentSha256,
      experienceSha256: "e".repeat(64), state: "complete" as const, attempt: 1, updatedAt: saved.updatedAt } };
    const result = refreshedPluginAccess(draft, complete, true);
    expect(result).toMatchObject({ dirty: true, conflict: false, access: { enabled: true, semanticLearning: { state: "complete" } } });
    expect(saved).not.toHaveProperty("semanticLearning");
  });
  it("replaces a conflicting draft with the actual revised access and requires review", () => {
    expect(refreshedPluginAccess({ ...saved, enabled: true }, { ...saved, revision: 3 }, true)).toMatchObject({ dirty: false, conflict: true, access: { enabled: false, revision: 3 } });
  });
  it("never carries edits across instance or immutable package ownership", () => {
    for (const other of [{ ...saved, brainId: "other" }, { ...saved, contentSha256: "b".repeat(64) }])
      expect(refreshedPluginAccess({ ...saved, enabled: true }, other, true)).toMatchObject({ dirty: false, conflict: true, access: { enabled: false } });
  });
});
