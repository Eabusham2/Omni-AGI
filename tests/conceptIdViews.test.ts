import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it, expect } from "vitest";
import { normalizeConceptIdView } from "../src/shared/conceptIdView";
import { savedConceptIdViewFiles, validateConceptIdViewFile } from "../src/main/conceptIdViewFiles";

describe("complete structural concept ID views", () => {
  it("binds brain/turn/count/hash/path and rejects text or cross-owner data", () => {
    const sha = "a".repeat(64);
    const descriptor = { format: "omni-structural-concept-id-view", version: 1, brainId: "brain", turnId: "turn", count: 5000000, bytes: 150000000, sha256: sha, path: `state/concept-id-views/${sha}.jsonl` };
    expect(normalizeConceptIdView(descriptor, "brain", "turn").count).toBe(5000000);
    expect(() => normalizeConceptIdView(descriptor, "different", "turn")).toThrow();
    expect(() => normalizeConceptIdView(descriptor, "brain", "different")).toThrow();
    expect(() => normalizeConceptIdView({ ...descriptor, path: "../secret" }, "brain", "turn")).toThrow();
    expect(() => normalizeConceptIdView({ ...descriptor, source_text: "answer text" }, "brain", "turn")).toThrow();
  });

  it("preserves exported structural files and checks every member without model execution", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-concept-view-"));
    try {
      const directory = join(root, "state", "concept-id-views");
      await mkdir(directory, { recursive: true });
      const ids = Array.from({ length: 3000 }, (_, index) => `assembly-${index}`);
      const header = { brainId: "original-owner", count: ids.length, format: "omni-structural-concept-id-view", turnId: "turn", version: 1 };
      const bytes = Buffer.from(`${JSON.stringify(header)}\n${ids.map((id) => `${JSON.stringify(id)}\n`).join("")}`);
      const name = `${createHash("sha256").update(bytes).digest("hex")}.jsonl`;
      const path = join(directory, name);
      await writeFile(path, bytes);
      expect((await savedConceptIdViewFiles(root)).size).toBe(1);
      await validateConceptIdViewFile(path, name);
      await writeFile(path, Buffer.concat([bytes, Buffer.from('"extra"\n')]));
      await expect(validateConceptIdViewFile(path, name)).rejects.toThrow();
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});
