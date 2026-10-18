/** Actual tiny host SQLite/settings fixtures; no model, capture, or speech. */
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../src/shared/types";
import { DEFAULT_INSTANCE_VOICE_SETTINGS } from "../src/shared/voiceRouting";
import { BrainRepository } from "../src/main/brainRepository";
import { InstanceVoiceService } from "../src/main/voiceService";
import { VoicePreferencesStore, instanceVoicePreferences, initializeInstanceVoice, mergeVoiceWorkspace,
  validateVoiceWorkspace, VOICE_PREFERENCES_RELATIVE_PATH } from "../src/main/voicePreferencesStore";
import { adoptLearningWorkspace, cloneLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";

const roots: string[] = [];
const legacy = { schemaVersion: 1 as const, deliveryMode: "buffered" as const, pace: "fast" as const, neuralListening: true, neuralVoice: false };
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-voice-store-")); roots.push(root); return root; }
const hash = (value: Buffer) => createHash("sha256").update(value).digest("hex");
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

describe("independent per-instance voice preference persistence", () => {
  it("initializes fresh host-created instances to Auto and cannot inherit an old global boolean choice", async () => {
    const repository = new BrainRepository(join(await directory(), "brains"));
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Voice host fixture", idleCognition: false });
    const root = repository.brainDirectory(brain.id), prefs = await instanceVoicePreferences(root, brain.id);
    expect(prefs.preferences).toEqual(DEFAULT_INSTANCE_VOICE_SETTINGS); expect(prefs.legacyMigrationAllowed).toBe(false);
    const store = (await VoicePreferencesStore.open(root, brain.id, true))!;
    try { expect(store.migrateLegacy(legacy).preferences).toEqual(DEFAULT_INSTANCE_VOICE_SETTINGS); }
    finally { store.close(); }
  });
  it("migrates only an existing local missing record once, preserving explicit true and false", async () => {
    const root = await directory(); await writeFile(join(root, "brain.json"), "{}");
    const service = new InstanceVoiceService({ brainDirectory: () => root, voiceParameterEvidence: async () => { throw new Error("No native model in fixture"); } });
    expect((await service.preferences("old-brain")).legacyMigrationAllowed).toBe(true);
    const first = await service.migrateLegacy({ brainId: "old-brain", legacy });
    expect(first.preferences).toEqual({ schemaVersion: 1, input: "native", output: "platform", deliveryMode: "buffered", pace: "fast" });
    const next = await service.migrateLegacy({ brainId: "old-brain", legacy: { ...legacy, neuralListening: false, neuralVoice: true } });
    expect(next.revision).toBe(first.revision); expect(next.preferences).toEqual(first.preferences);
    expect(next.parameterEvidence).toBeNull(); expect(next.parameterDetail).toContain("unresolved");
  });
  it("persists independently across reopening and rejects stale CAS from another settings writer", async () => {
    const root = await directory(); await initializeInstanceVoice(root, "brain", "fresh-auto");
    const a = (await VoicePreferencesStore.open(root, "brain", true))!, b = (await VoicePreferencesStore.open(root, "brain", true))!;
    try {
      const original = a.snapshot(); b.update(original.revision, { ...original.preferences, output: "native" });
      expect(() => a.update(original.revision, { ...original.preferences, output: "platform" })).toThrow(/changed/);
    } finally { a.close(); b.close(); }
    expect((await instanceVoicePreferences(root, "brain")).preferences.output).toBe("native");
  });
  it("a later stale neural/host document save cannot overwrite the separately owned voice choice", async () => {
    const repository = new BrainRepository(join(await directory(), "brains"));
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Separate host state", idleCognition: false });
    const store = (await VoicePreferencesStore.open(repository.brainDirectory(brain.id), brain.id, true))!;
    try { store.update(store.snapshot().revision, { ...DEFAULT_INSTANCE_VOICE_SETTINGS, output: "native" }); }
    finally { store.close(); }
    await repository.save(brain);
    expect((await instanceVoicePreferences(repository.brainDirectory(brain.id), brain.id)).preferences.output).toBe("native");
  });
  it("snapshots, copies, and restores exact preferences with isolated owners and unchanged source", async () => {
    const source = await directory(), archiveRoot = await directory(), copy = await directory(), recovery = await directory();
    await initializeInstanceVoice(source, "parent", "fresh-auto");
    const store = (await VoicePreferencesStore.open(source, "parent", true))!;
    try { store.update(1, { ...DEFAULT_INSTANCE_VOICE_SETTINGS, input: "native", output: "platform", pace: "slow" }); }
    finally { store.close(); }
    const sourceBytes = hash(await readFile(join(source, VOICE_PREFERENCES_RELATIVE_PATH)));
    const archive = await snapshotLearningWorkspace(source, archiveRoot, "parent");
    await adoptLearningWorkspace(archive!, copy, "child"); await adoptLearningWorkspace(archive!, recovery, "parent");
    const copied = await instanceVoicePreferences(copy, "child"), restored = await instanceVoicePreferences(recovery, "parent");
    expect(copied.preferences.input).toBe("native"); expect(copied.preferences.output).toBe("platform");
    expect(restored.preferences).toEqual(copied.preferences); expect(restored.revision).toBe(2);
    const childStore = (await VoicePreferencesStore.open(copy, "child", true))!;
    try { childStore.update(2, { ...copied.preferences, output: "native" }); } finally { childStore.close(); }
    expect(hash(await readFile(join(source, VOICE_PREFERENCES_RELATIVE_PATH)))).toBe(sourceBytes);
    expect((await instanceVoicePreferences(source, "parent")).preferences.output).toBe("platform");
  });
  it("copies/imports with no preferences initialize Auto instead of inheriting destination legacy settings", async () => {
    const source = await directory(), target = await directory();
    await cloneLearningWorkspace(source, target, "old-parent", "external-copy");
    const store = (await VoicePreferencesStore.open(target, "external-copy", true))!;
    try { expect(store.migrateLegacy(legacy).preferences).toEqual(DEFAULT_INSTANCE_VOICE_SETTINGS); expect(store.snapshot().legacyMigrationAllowed).toBe(false); }
    finally { store.close(); }
  });
  it("retains conflicting parent preferences as idempotent history without last-parent-wins routing", async () => {
    const a = await directory(), b = await directory(), stagedA = await directory(), stagedB = await directory(), target = await directory();
    for (const [root, owner, route] of [[a, "parent-a", "platform"], [b, "parent-b", "native"]] as const) {
      await initializeInstanceVoice(root, owner, "fresh-auto"); const store = (await VoicePreferencesStore.open(root, owner, true))!;
      try { store.update(1, { ...DEFAULT_INSTANCE_VOICE_SETTINGS, output: route }); } finally { store.close(); }
    }
    await cloneLearningWorkspace(a, stagedA, "parent-a", "joined"); await cloneLearningWorkspace(b, stagedB, "parent-b", "joined");
    await mergeVoiceWorkspace(stagedA, target, "joined", "parent-a"); await mergeVoiceWorkspace(stagedB, target, "joined", "parent-b");
    await mergeVoiceWorkspace(stagedA, target, "joined", "parent-a");
    const prefs = await instanceVoicePreferences(target, "joined"); expect(prefs.preferences.output).toBe("auto"); expect(prefs.inheritedVariants).toBe(2);
    const targetStore = (await VoicePreferencesStore.open(target, "joined", true))!;
    try { targetStore.update(prefs.revision, { ...prefs.preferences, output: "platform" }); } finally { targetStore.close(); }
    await mergeVoiceWorkspace(stagedB, target, "joined", "parent-b");
    expect((await instanceVoicePreferences(target, "joined")).preferences.output).toBe("platform");
    await validateVoiceWorkspace(target, "joined");
  });
  it("refuses wrong ownership, oversized typed rows and unknown portable files", async () => {
    const root = await directory(); await initializeInstanceVoice(root, "brain", "fresh-auto");
    await expect(validateVoiceWorkspace(root, "wrong-owner")).rejects.toThrow(/another instance/);
    const store = (await VoicePreferencesStore.open(root, "brain", true))!;
    try { store.db.prepare("UPDATE preferences SET payload=?").run("x".repeat(4097)); } finally { store.close(); }
    await expect(instanceVoicePreferences(root, "brain")).rejects.toThrow(/envelope/);
    const clean = await directory(); await initializeInstanceVoice(clean, "brain", "fresh-auto");
    await writeFile(join(clean, "voice", "unapproved.js"), "not executed");
    await expect(snapshotLearningWorkspace(clean, await directory(), "brain")).rejects.toThrow(/Unsafe portable/);
  });
});
