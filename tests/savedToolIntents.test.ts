import { randomUUID } from "node:crypto";
import { mkdtemp, mkdir, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { persistAuthorizedToolIntent } from "../src/main/toolIntentJournal";
import { savedToolIntentFiles, validateSavedToolIntent } from "../src/main/savedToolIntents";

describe("inert saved operational evidence", () => {
  it("validates original ownership without rekeying historical actions", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-intent-save-"));
    try {
      const original = join(root, "ancestral");
      await mkdir(original);
      const path = await persistAuthorizedToolIntent(original, { brainId: "ancestral", toolId: "files.read",
        action: "read", arguments: { path: "/user/content" } }, { id: randomUUID(), requestId: "turn", startedAt: new Date().toISOString(), permission: "full", permissionRevision: "revision" });
      await expect(validateSavedToolIntent(path)).resolves.toBeUndefined();
      expect((await savedToolIntentFiles(join(original, "engine"))).size).toBe(1);
      await expect(savedToolIntentFiles(join(root, "absent"))).resolves.toEqual(new Map());
      const bad = join(original, "engine", "operational-tool-intents", `${randomUUID()}.json`);
      await writeFile(bad, JSON.stringify({ format: "omni-authorized-tool-intent", permission: "off" }));
      await expect(validateSavedToolIntent(bad)).rejects.toThrow("invalid");
    } finally { await rm(root, { recursive: true, force: true }); }
  });

  it("rejects filesystem redirects and arbitrary payloads", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-intent-links-"));
    try {
      const engine = join(root, "engine");
      const outside = join(root, "outside");
      await mkdir(engine); await mkdir(outside);
      await symlink(outside, join(engine, "operational-tool-intents"), process.platform === "win32" ? "junction" : "dir");
      await expect(savedToolIntentFiles(engine)).rejects.toThrow("unsafe");
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});
