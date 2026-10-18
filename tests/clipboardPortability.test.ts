import { cp, mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ClipboardStore } from "../src/main/clipboardStore";
import { clipboardCanonical, clipboardHash } from "../src/main/clipboardFormat";
import { mergeClipboardWorkspace, rekeyClipboardWorkspace, validateClipboardWorkspace } from "../src/main/clipboardPortability";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";

const roots: string[] = [], sourceBrain = "88888888-8888-4888-8888-888888888888", targetBrain = "99999999-9999-4999-8999-999999999999";
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const payload = (scope: "instance" | "host", action: "write" | "clear", text: string) => { const contentSha256 = clipboardHash(text), unsigned = { scope, action, contentSha256, contentBytes: Buffer.byteLength(text) };
  return { ...unsigned, requestSha256: clipboardHash(clipboardCanonical(unsigned)) }; };

describe("complete instance clipboard portability", () => {
  it("rekeys exact unsanitized content and turns a dispatched host mutation into nonreplayable evidence", async () => {
    const source = await mkdtemp(join(tmpdir(), "omni-clipboard-source-")), target = await mkdtemp(join(tmpdir(), "omni-clipboard-copy-")); roots.push(source, target);
    const store = await ClipboardStore.open(source, sourceBrain); try {
      await store.mutateInstance("portable-write", payload("instance", "write", "secret\0unsanitized🙂"), "secret\0unsanitized🙂",
        { kind: "instance-owned", capturedAt: new Date().toISOString(), brainId: sourceBrain });
      store.prepareHostEffect("lost-host-effect", payload("host", "clear", "")); store.dispatched("lost-host-effect");
    } finally { store.close(); }
    await cp(join(source, "clipboard"), join(target, "clipboard"), { recursive: true });
    await rekeyClipboardWorkspace(target, sourceBrain, targetBrain); await validateClipboardWorkspace(target, targetBrain);
    const adopted = await ClipboardStore.open(target, targetBrain, true); try {
      const read = await adopted.read(undefined, 0, 1024); expect(read.text).toBe("secret\0unsanitized🙂");
      expect(adopted.effect("lost-host-effect")?.receipt).toMatchObject({ state: "write-uncertain", writeMayHaveOccurred: true, retrySafe: false });
    } finally { adopted.close(); }
  });

  it("full merge selects the first retained parent clipboard while preserving versions and receipts", async () => {
    const parent = await mkdtemp(join(tmpdir(), "omni-clipboard-parent-")), stage = await mkdtemp(join(tmpdir(), "omni-clipboard-stage-")), target = await mkdtemp(join(tmpdir(), "omni-clipboard-merge-")); roots.push(parent, stage, target);
    const source = await ClipboardStore.open(parent, sourceBrain); try { await source.mutateInstance("parent-write", payload("instance", "write", "parent exact text"), "parent exact text",
      { kind: "instance-owned", capturedAt: new Date().toISOString(), brainId: sourceBrain }); } finally { source.close(); }
    await cp(join(parent, "clipboard"), join(stage, "clipboard"), { recursive: true }); await rekeyClipboardWorkspace(stage, sourceBrain, targetBrain);
    await mkdir(target, { recursive: true }); const initial = await ClipboardStore.open(target, targetBrain); initial.close();
    await mergeClipboardWorkspace(stage, target, targetBrain, sourceBrain, async () => undefined);
    await mergeClipboardWorkspace(stage, target, targetBrain, sourceBrain, async () => undefined); await validateClipboardWorkspace(target, targetBrain);
    const merged = await ClipboardStore.open(target, targetBrain, true); try {
      expect((await merged.read(undefined, 0, 1024)).text).toBe("parent exact text");
      expect(merged.versions("", 50).entries.some(version => version.provenance.kind === "inherited" && version.provenance.parentBrainId === sourceBrain)).toBe(true);
      expect(merged.allEffects().some(effect => effect.receipt?.provenance.kind === "inherited")).toBe(true);
    } finally { merged.close(); }
    const writable = await ClipboardStore.open(target, targetBrain); try { await expect(writable.mutateInstance("post-merge-write", payload("instance", "write", "new root text"), "new root text",
      { kind: "instance-owned", capturedAt: new Date().toISOString(), brainId: targetBrain })).resolves.toMatchObject({ state: "complete" }); }
    finally { writable.close(); }
  });

  it("streams clipboard blobs through the complete workspace archive without carrying a host grant", async () => {
    const source = await mkdtemp(join(tmpdir(), "omni-clipboard-archive-source-")), packaged = await mkdtemp(join(tmpdir(), "omni-clipboard-archive-package-")),
      target = await mkdtemp(join(tmpdir(), "omni-clipboard-archive-target-")); roots.push(source, packaged, target);
    const store = await ClipboardStore.open(source, sourceBrain); try { await store.mutateInstance("archive-write", payload("instance", "write", "archive\0exact🙂"), "archive\0exact🙂",
      { kind: "instance-owned", capturedAt: new Date().toISOString(), brainId: sourceBrain }); } finally { store.close(); }
    const archive = await snapshotLearningWorkspace(source, packaged, sourceBrain); expect(archive).toBeDefined(); await adoptLearningWorkspace(archive!, target, targetBrain);
    const copied = await ClipboardStore.open(target, targetBrain, true); try { expect((await copied.read(undefined, 0, 1024)).text).toBe("archive\0exact🙂"); }
    finally { copied.close(); }
  });
});
