import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { ExternalAdviceAuthorityStore } from "../src/main/externalAdviceAuthority";
import { selectRecentAdviceContext } from "../src/main/externalAdviceContext";
import type { ConversationLedgerPage } from "../src/shared/types";

describe("device-owned advice and bounded recorded context", () => {
  it("does not relabel an MCP-authored human-role task as a human instruction", () => {
    const entry = { sequence: 1, id: "human", kind: "message" as const, createdAt: "2026-10-13T00:00:00Z", attentionEpoch: 1,
      payloadSha256: "b", rowSha256: "c", message: { id: "human", role: "human" as const, content: "Actual user task", createdAt: "2026-10-13T00:00:00Z" } };
    const mcp = { ...entry, sequence: 2, id: "mcp", message: { ...entry.message, id: "mcp", content: "Untrusted app output",
      inputProvenance: { kind: "mcp-app" } as import("../src/shared/chatInputProvenance").ChatInputProvenance } };
    const page: ConversationLedgerPage = { brainId: "owner", entries: [entry, mcp], totalEntries: 2, hasOlder: false, headSequence: 2, headSha256: "a".repeat(64) };
    const result = selectRecentAdviceContext({ brainId: "owner", channel: "reviewer", selection: { includeTask: true, includeActivity: false, maxCharacters: 100 } }, page);
    expect(result.items.map(item => item.text)).toEqual(["Actual user task"]);
  });
  it("keeps new owners dormant and serializes revisions without transferring another owner's grant", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-advice-authority-"));
    try {
      const path = join(root, "device-authority.json"), store = new ExternalAdviceAuthorityStore(path);
      expect(await store.get("source", "assistant")).toMatchObject({ enabled: false, revision: 0 });
      await Promise.all([store.set("source", "assistant", true), store.set("source", "reviewer", true)]);
      const before = await store.get("source", "reviewer");
      expect(before).toMatchObject({ enabled: true, revision: 2 });
      expect(await store.get("imported-copy", "reviewer")).toMatchObject({ enabled: false, revision: 0, ownerEpoch: "unactivated" });
      await store.set("source", "assistant", false);
      expect(await store.get("source", "reviewer")).toMatchObject({ enabled: true, revision: 3, ownerEpoch: before.ownerEpoch });
      expect(await new ExternalAdviceAuthorityStore(path).get("source", "assistant")).toMatchObject({ enabled: false, revision: 3 });
      expect(await readFile(path, "utf8")).not.toContain("credential");
    } finally { await rm(root, { recursive: true, force: true }); }
  });
  it("prioritizes the latest recorded task, reports omitted history and never projects tool arguments", () => {
    const page: ConversationLedgerPage = { brainId: "owner", totalEntries: 700, hasOlder: true, headSequence: 700, headSha256: "a".repeat(64), entries: [
      { sequence: 698, id: "human", kind: "message", createdAt: "2026-10-13T00:00:00Z", attentionEpoch: 1, payloadSha256: "b", rowSha256: "c",
        message: { id: "human", role: "human", content: "Current user task", createdAt: "2026-10-13T00:00:00Z" } },
      { sequence: 699, id: "action", kind: "action", createdAt: "2026-10-13T00:00:01Z", attentionEpoch: 1, payloadSha256: "d", rowSha256: "e",
        action: { id: "action", brainId: "owner", state: "running", createdAt: "2026-10-13T00:00:01Z", updatedAt: "2026-10-13T00:00:01Z",
          action: { kind: "tool", source: "brain", toolId: "system.shell", action: "run", arguments: { credential: "DO-NOT-SHARE" } } } }
    ] };
    const result = selectRecentAdviceContext({ brainId: "owner", channel: "assistant", selection: { includeTask: true, includeActivity: true, maxCharacters: 60 } }, page);
    expect(result.items[0]?.text).toBe("Current user task");
    expect(result.includedCharacters).toBe(60); expect(result.truncated).toBe(true); expect(result.omittedItems).toBe(698);
    expect(JSON.stringify(result)).not.toContain("DO-NOT-SHARE");
    expect(selectRecentAdviceContext({ brainId: "owner", channel: "reviewer", selection: { includeTask: false, includeActivity: false, maxCharacters: 0 } }, page).items).toEqual([]);
  });
});
