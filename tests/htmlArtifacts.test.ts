import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DatabaseSync } from "node:sqlite";
import { normalizeHtmlArtifactContent } from "../src/main/htmlArtifactFormat";
import {
  mergeHtmlArtifactWorkspace,
  rekeyHtmlArtifactWorkspace,
  safeHtmlArtifactWorkspacePath,
  validateHtmlArtifactWorkspace
} from "../src/main/htmlArtifactPortability";
import { HtmlArtifactStore } from "../src/main/htmlArtifactStore";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";

function artifact(path = "note.txt") {
  return {
    title: "Interactive note viewer",
    html: "<main><button id=\"read\">Read</button><canvas id=\"chart\"></canvas></main>",
    css: "main { padding: 1rem; }",
    script: `document.querySelector('#read').addEventListener('click',()=>void omniArtifact.call('system.files','read',{path:${JSON.stringify(path)}}));`,
    tools: [{ toolId: "system.files", actions: ["read"] }],
    height: 420
  };
}

describe("durable chat HTML artifacts", () => {
  const roots: string[] = [];
  afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))));
  async function root(name: string): Promise<string> {
    const value = await mkdtemp(join(tmpdir(), `omni-${name}-`)); roots.push(value); return value;
  }

  it("accepts local controls/canvas and rejects active markup, resource CSS, recursive tools, and script breakouts", () => {
    expect(normalizeHtmlArtifactContent(artifact())).toMatchObject({
      format: "omni-chat-html-artifact-content", version: 1, height: 420,
      tools: [{ toolId: "system.files", actions: ["read"] }]
    });
    expect(() => normalizeHtmlArtifactContent({ ...artifact(), html: "<svg/onload=bad()>" })).toThrow(/no scripts, events/i);
    expect(() => normalizeHtmlArtifactContent({ ...artifact(), css: "body{background:url(https://leak.invalid)}" })).toThrow(/may not import/i);
    expect(() => normalizeHtmlArtifactContent({ ...artifact(), script: "const x='</script data-x>';" })).toThrow(/closing script/i);
    expect(() => normalizeHtmlArtifactContent({ ...artifact(), tools: [{ toolId: "studio.artifact", actions: ["create"] }] })).toThrow(/nonrecursive/i);
    const noScript = artifact() as Record<string, unknown>; delete noScript.script;
    expect(() => normalizeHtmlArtifactContent(noScript)).toThrow(/scripted HTML artifact/i);
  });

  it("commits one hash-bound artifact per source action and exposes no raw source in its projection", async () => {
    const directory = await root("html-artifact-store"), store = new HtmlArtifactStore(directory, "brain-one");
    const first = await store.create("action-one", artifact());
    expect(first.artifact).toMatchObject({ brainId: "brain-one", originalBrainId: "brain-one",
      sourceActionId: "action-one", interactive: true, operationalState: "enabled", savedArtifact: true, untrusted: true });
    expect(first.artifact).not.toHaveProperty("html");
    expect(first.artifact).not.toHaveProperty("script");
    await expect(store.create("action-one", artifact())).resolves.toEqual(first);
    await expect(store.create("action-one", artifact("other.txt"))).rejects.toThrow(/different content/i);
    const row = await store.find(first.artifact), content = await store.content(row);
    expect(content.html).toContain("<canvas");
    expect(content.script).toContain("omniArtifact.call");
    expect(await validateHtmlArtifactWorkspace(directory, "brain-one")).toBe(true);
  });

  it("keeps exact bytes on copy but makes a new owner's operational bridge dormant until reviewed", async () => {
    const source = await root("html-artifact-copy-source"), target = await root("html-artifact-copy-target");
    const created = await new HtmlArtifactStore(source, "source-brain").create("source-action", artifact());
    await mkdir(target, { recursive: true });
    await cp(join(source, "html-artifacts"), join(target, "html-artifacts"), { recursive: true });
    await rekeyHtmlArtifactWorkspace(target, "source-brain", "target-brain");
    const targetStore = new HtmlArtifactStore(target, "target-brain");
    const retained = await targetStore.describe({ ...created.artifact, brainId: "target-brain" });
    expect(retained).toMatchObject({ originalBrainId: "source-brain", operationalState: "dormant", revision: 2 });
    const enabled = await targetStore.activate({ ...retained, expectedRevision: retained.revision });
    expect(enabled).toMatchObject({ operationalState: "enabled", revision: 3 });
    expect(await validateHtmlArtifactWorkspace(target, "target-brain")).toBe(true);
  });

  it("merges parent artifacts as distinct dormant provenance without request grants", async () => {
    const parent = await root("html-artifact-parent"), stage = await root("html-artifact-stage"), target = await root("html-artifact-merge");
    const created = await new HtmlArtifactStore(parent, "parent-brain").create("parent-action", artifact());
    await cp(join(parent, "html-artifacts"), join(stage, "html-artifacts"), { recursive: true });
    await rekeyHtmlArtifactWorkspace(stage, "parent-brain", "merged-brain");
    await new HtmlArtifactStore(target, "merged-brain").initialize();
    await mergeHtmlArtifactWorkspace(stage, target, "merged-brain", "parent-brain");
    const merged = await new HtmlArtifactStore(target, "merged-brain").describe({ ...created.artifact, brainId: "merged-brain" });
    expect(merged).toMatchObject({ originalBrainId: "parent-brain", artifactId: created.artifact.artifactId,
      operationalState: "dormant" });
    expect(await validateHtmlArtifactWorkspace(target, "merged-brain")).toBe(true);
    expect(safeHtmlArtifactWorkspacePath("html-artifacts/index.sqlite3")).toBe(true);
    expect(safeHtmlArtifactWorkspacePath(`html-artifacts/blobs/${created.artifact.contentSha256}.json`)).toBe(true);
    expect(safeHtmlArtifactWorkspacePath("html-artifacts/../secret")).toBe(false);
  });

  it("uses the ordinary workspace archive hook and adopts exact content with dormant bridge authority", async () => {
    const source = await root("html-artifact-archive-source"), packaged = await root("html-artifact-archive"),
      target = await root("html-artifact-archive-target");
    const created = await new HtmlArtifactStore(source, "source-brain").create("source-action", artifact());
    const archive = await snapshotLearningWorkspace(source, packaged, "source-brain");
    expect(archive).toBeDefined();
    await adoptLearningWorkspace(archive!, target, "imported-brain");
    const imported = await new HtmlArtifactStore(target, "imported-brain").describe({
      ...created.artifact,
      brainId: "imported-brain"
    });
    expect(imported).toMatchObject({ originalBrainId: "source-brain", contentSha256: created.artifact.contentSha256,
      operationalState: "dormant", revision: 2 });
    expect(await validateHtmlArtifactWorkspace(target, "imported-brain")).toBe(true);
  });

  it("detects changed artifact content bytes", async () => {
    const directory = await root("html-artifact-corrupt"), store = new HtmlArtifactStore(directory, "brain-one");
    const created = await store.create("action-one", artifact());
    const path = join(directory, "html-artifacts", "blobs", `${created.artifact.contentSha256}.json`);
    const original = await readFile(path);
    await writeFile(path, Buffer.concat([original, Buffer.from(" ")]));
    await expect(validateHtmlArtifactWorkspace(directory, "brain-one")).rejects.toThrow(/checksum|canonical|content/i);
  });

  it("rejects undeclared workspace payloads before a portable archive can claim completeness", async () => {
    const directory = await root("html-artifact-extra"), store = new HtmlArtifactStore(directory, "brain-one");
    await store.create("action-one", artifact());
    await writeFile(join(directory, "html-artifacts", "unlisted.html"), "<p>not registered</p>");
    await expect(validateHtmlArtifactWorkspace(directory, "brain-one")).rejects.toThrow(/unsupported payload/i);
  });

  it("rejects executable or altered SQLite schema before importing an artifact registry", async () => {
    const directory = await root("html-artifact-trigger"), store = new HtmlArtifactStore(directory, "brain-one");
    await store.create("action-one", artifact());
    const database = new DatabaseSync(join(directory, "html-artifacts", "index.sqlite3"));
    try {
      database.exec(`CREATE TRIGGER imported_artifact_trigger AFTER UPDATE ON artifacts
        BEGIN DELETE FROM artifacts; END;`);
    } finally { database.close(); }
    await expect(validateHtmlArtifactWorkspace(directory, "brain-one")).rejects.toThrow(/unsupported executable\/schema/i);
  });
});
