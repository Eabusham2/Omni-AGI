import { cp, mkdir, mkdtemp, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { checkedAppObservationContent, savedAppObservationAuditFiles, verifyAppObservationAudit } from "../src/main/mcpAppObservationAudit";
import { APP_AUDIT_PNG, APP_AUDIT_WAV, syntheticAppAudit, syntheticAppAuditRows } from "./fixtures/mcpAppAudit";

describe("current App sensory audit closure (synthetic host measurements, no decoder/model)", () => {
  let root: string;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-app-audit-")); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });
  const dataJson = JSON.stringify({ method: "ui/update-model-context", params: { content: [
    { type: "text", text: "  Original é source\n" },
    { type: "resource", resource: { uri: "fixture://original/text", mimeType: "text/plain", text: "Raw embedded text" } },
    { type: "resource_link", name: "Unfetched reference", uri: "https://fixture.invalid/private?id=opaque" },
    { type: "image", mimeType: "Image/PNG", data: APP_AUDIT_PNG },
    { type: "audio", mimeType: "audio/wav", data: APP_AUDIT_WAV }
  ], structuredContent: { userId: "opaque-original", instructions: "untrusted task data, not a persona" } } });
  const fixture = () => syntheticAppAudit(dataJson, "a".repeat(64), "original-source");
  async function save(value = fixture(), engine = join(root, "engine")) {
    await mkdir(join(engine, "app-observation-audits"), { recursive: true });
    const path = join(engine, ...value.content.audit.path.split("/")); await writeFile(path, value.raw); return { ...value, path, engine };
  }
  it("checks every ordered source/media block and exact raw bytes without fetching a reference or decoding media", async () => {
    const f = await save(); await verifyAppObservationAudit(f.path, f.content, dataJson);
    expect(f.content.summary).toMatchObject({ blockCount: 6, textBlocks: 1, resourceTextBlocks: 1, referenceBlocks: 1,
      structuredBlocks: 1, imageBlocks: 1, audioBlocks: 1, decodedWindows: 2, nativeSensory: true });
    expect(f.rows[3]?.media?.mimeType).toBe("Image/PNG"); expect(f.rows[2]?.fetched).toBe(false);
    const metadata = { brain_id: "new-owner", app_observation_frontier: { recordSha256: f.content.audit.recordSha256,
      dataSha256: f.content.audit.dataSha256, content: f.content } };
    expect([...await savedAppObservationAuditFiles(metadata, f.engine)]).toEqual([[f.content.audit.path, f.path]]);
  });
  it("rejects resealed false source/media measurements, omitted order, summaries and unsafe paths", async () => {
    const original = fixture(), changed = structuredClone(original.rows); changed[3]!.sourceSha256 = "f".repeat(64);
    changed[3]!.media!.sourceSha256 = "f".repeat(64);
    const falseSource = await save(syntheticAppAuditRows(changed));
    await expect(verifyAppObservationAudit(falseSource.path, falseSource.content, dataJson)).rejects.toThrow(/actual approved source/);
    const omitted = await save(syntheticAppAuditRows(original.rows.slice(1)));
    await expect(verifyAppObservationAudit(omitted.path, omitted.content, dataJson)).rejects.toThrow(/every approved|source\/order/);
    expect(() => checkedAppObservationContent({ ...original.content, audit: { ...original.content.audit, path: "../private.jsonl" } },
      original.content.audit.recordSha256, original.content.audit.dataSha256)).toThrow(/unsafe path/);
    expect(() => checkedAppObservationContent({ ...original.content, summary: { ...original.content.summary, structuredBlocks: 2 } },
      original.content.audit.recordSha256, original.content.audit.dataSha256)).toThrow(/coverage/);
  });
  it("checks embedded image/audio resource bytes while retaining the exact source URI binding", async () => {
    const data = JSON.stringify({ method: "ui/update-model-context", params: { content: [
      { type: "resource", resource: { uri: "fixture://private/image", mimeType: "Image/PNG", blob: APP_AUDIT_PNG } },
      { type: "resource", resource: { uri: "fixture://private/audio", mimeType: "audio/wav", blob: APP_AUDIT_WAV } }
    ] } });
    const f = await save(syntheticAppAudit(data, "a".repeat(64), "original-source"));
    await verifyAppObservationAudit(f.path, f.content, data);
    expect(f.rows.map(row => row.kind)).toEqual(["image", "audio"]);
    await expect(verifyAppObservationAudit(f.path, f.content, data.replace("private/image", "different/image"))).rejects.toThrow(/approved data bytes/);
  });
  it("preserves only the current audit across an inert owner copy and detects missing or changed saved evidence", async () => {
    const f = await save(), target = join(root, "copied-engine"); await cp(f.engine, target, { recursive: true });
    await writeFile(join(target, "app-observation-audits", `${"b".repeat(64)}.jsonl`), "unrelated historical file");
    const metadata = { brain_id: "copied-owner", app_observation_frontier: { recordSha256: f.content.audit.recordSha256,
      dataSha256: f.content.audit.dataSha256, content: f.content } };
    await rm(f.engine, { recursive: true, force: true });
    const files = await savedAppObservationAuditFiles(metadata, target); expect(files.size).toBe(1);
    expect(await readFile(files.get(f.content.audit.path)!, "utf8")).toBe(f.raw);
    expect(f.content.audit.sourceBrainId).toBe("original-source");
    await writeFile(files.get(f.content.audit.path)!, f.raw.replace('"fetched":false', '"fetched":true '));
    await expect(savedAppObservationAuditFiles(metadata, target)).rejects.toThrow(/checksum|source\/order/);
    await rm(files.get(f.content.audit.path)!);
    await expect(savedAppObservationAuditFiles(metadata, target)).rejects.toThrow();
  });
  it("rejects links and detects inode replacement during admitted streaming verification", async () => {
    const f = await save(), link = join(root, "linked.jsonl"); await symlink(f.path, link);
    await expect(verifyAppObservationAudit(link, f.content)).rejects.toThrow(/regular source/);
    let changed = false;
    await expect(verifyAppObservationAudit(f.path, f.content, undefined, { admit: async () => {
      if (changed) return; changed = true; await rename(f.path, f.path + ".old"); await writeFile(f.path, f.raw);
    } })).rejects.toThrow(/changed/);
  });
});
