import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ActionReviewerApprovalReview } from "../src/shared/actionReviewer";
import { ExternalAdviceService, externalAdviceSha256, validateExternalAdviceDatabaseFile, validateExternalAdviceStoredReview } from "../src/main/externalAdviceService";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

describe("portable ActionReviewer receipt custody", () => {
  it("persists one complete advisory receipt with every configured credential redacted and no authority", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-action-review-history-")); roots.push(root);
    const brainRoot = join(root, "brain"); await mkdir(brainRoot);
    const values = new Map<string, string>(), secret = "synthetic-review-secret";
    const secrets = { get: vi.fn(async (id: string) => values.get(id)), has: vi.fn(async (id: string) => values.has(id)),
      set: vi.fn(async (id: string, value: string) => { values.set(id, value); return "encrypted" as const; }),
      remove: vi.fn(async (id: string) => values.delete(id)), persistence: vi.fn(() => "encrypted" as const) };
    const service = new ExternalAdviceService({ stream: vi.fn() }, secrets, { deviceDirectory: join(root, "device"), brainRoot: () => brainRoot,
      authority: async (_brainId, channel) => ({ brainId: "brain-one", channel, enabled: false, ownerEpoch: "device-owner", revision: 1 }),
      selectContext: async ({ brainId, channel, selection }) => ({ format: "omni-external-advice-selected-context", version: 1,
        brainId, channel, selection, items: [], includedCharacters: 0, omittedItems: 0, truncated: false }), admitWrite: async () => undefined });
    await service.saveCredential({ provider: "openai", credential: secret });
    const receipt: ActionReviewerApprovalReview = { kind: "action-review", formatVersion: 1, reviewId: "review-one", brainId: "brain-one",
      requestId: "request-one", configurationRevision: "config-one", ownerRevision: "owner-one", reviewerId: "openai:manual-model",
      permissionRevision: "permission-one", proposalSha256: "a".repeat(64),
      proposedInvocation: { brainId: "brain-one", toolId: "fixture.tool", action: "inspect", arguments: { credential: secret, nested: [`value ${secret}`] } },
      resolvedInvocation: { brainId: "brain-one", toolId: "fixture.tool", action: "inspect", arguments: { credential: secret } },
      disposition: "reviewed", startedAt: "2026-10-13T20:00:00.000Z", finishedAt: "2026-10-13T20:00:01.000Z", elapsedMs: 1_000,
      observedContext: [{ id: "task-one", kind: "task", text: `Observed ${secret}` }], requestBytes: 400, responseBytes: 120,
      verdict: "ask", summary: `Review ${secret} before dispatch.`, evidence: [{ sourceId: "task-one", quote: secret }] };
    await service.recordReview(receipt, new AbortController().signal);
    await service.recordReview(receipt, new AbortController().signal);

    const path = join(brainRoot, "external-advice/workspace.sqlite3"), inventory = await validateExternalAdviceDatabaseFile(path, "brain-one");
    expect(inventory.reviews).toBe(1);
    const database = new DatabaseSync(path, { readOnly: true });
    try {
      const row = database.prepare("SELECT id,owner_brain_id,source_brain_id,review_id,finished_at,payload,sha256 FROM reviews").get()!;
      const stored = validateExternalAdviceStoredReview(row, "brain-one"), text = JSON.stringify(stored.receipt);
      expect(stored.receipt).toMatchObject({ reviewId: "review-one", brainId: "brain-one", proposalSha256: "a".repeat(64), disposition: "reviewed",
        proposedInvocation: { arguments: { credential: "[REDACTED]", nested: ["value [REDACTED]"] } },
        observedContext: [{ text: "Observed [REDACTED]" }], summary: "Review [REDACTED] before dispatch.",
        evidence: [{ sourceId: "task-one", quote: "[REDACTED]" }] });
      expect(text).not.toContain(secret); expect(stored.sha256).toBe(externalAdviceSha256(stored.payload));
      expect(row).not.toHaveProperty("approval_token");
    } finally { database.close(); }
    await expect(service.recordReview({ ...receipt, summary: "Different receipt bytes" }, new AbortController().signal)).rejects.toThrow(/collided/);
    await expect(service.recordReview({ ...receipt, approvalToken: "forbidden" } as ActionReviewerApprovalReview, new AbortController().signal)).rejects.toThrow(/unsupported fields/);
  });
});
