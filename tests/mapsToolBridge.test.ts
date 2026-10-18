/** Existing ToolExecutor permission/intent path; no model/native training. */
import { mkdtemp, mkdir, readFile, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import { MapsService } from "../src/main/mapsService";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-map-policy-")); roots.push(root); const directory = join(root, "brain-one"); await mkdir(directory);
  let permission: ToolPermissionRecord = { toolId: "maps.search", label: "Private maps", level: "ask", updatedAt: "2026-10-11T12:00:00Z" };
  const service = { repository: { brainDirectory: () => directory }, listToolPermissions: async () => [permission], authorizeInlineImagination: async () => undefined } as unknown as BrainService;
  const executor = new ToolExecutor(service, {} as RuntimeJobManager); vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
  const fetcher = vi.fn(async () => new Response(JSON.stringify({ type: "FeatureCollection", features: [] })));
  const maps = new MapsService(join(root, "maps.sqlite3"), new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() }), { fetch: fetcher as typeof fetch });
  await maps.initialize(); await maps.save({ id: "private", expectedRevision: 0, label: "Fixture", endpoint: "http://127.0.0.1:2322/api/", attribution: "Fixture data",
    attributionUrl: "https://example.invalid/licence", enabled: true, retainedResultsAccepted: true, minimumIntervalMs: 0 }); executor.setMapsService(maps);
  const invocation: ToolInvocation = { brainId: "brain-one", toolId: "maps.search", action: "search", arguments: { connectionId: "private", expectedRevision: 1, query: "Selected place" } };
  return { executor, maps, fetcher, directory, invocation, permission(level: ToolPermissionRecord["level"]) { permission = { ...permission, level, updatedAt: new Date(Date.parse(permission.updatedAt) + 1000).toISOString() }; } };
}
describe("maps share normal permission and actual-outcome retention", () => {
  it("Ask performs no network read until exact approval; completed source reaches the existing observer and intent journal", async () => {
    const f = await fixture(), completed = vi.fn(async () => undefined), asked = await f.executor.execute(f.invocation, undefined, "map-turn", completed);
    expect(asked.state).toBe("approval-required"); expect(f.fetcher).not.toHaveBeenCalled();
    const actual = await f.executor.execute({ ...f.invocation, approvalToken: asked.approvalToken }, undefined, "approved-map-turn", completed);
    expect(actual.state).toBe("complete"); expect(f.fetcher).toHaveBeenCalledOnce(); expect(completed).toHaveBeenCalledExactlyOnceWith(actual);
    expect(actual.output).toMatchObject({ format: "omni-map-search", geoJson: { features: [] }, untrustedTaskData: true });
    const intents = join(f.directory, "engine", "operational-tool-intents"), files = await readdir(intents); expect(files).toHaveLength(1);
    expect(JSON.parse(await readFile(join(intents, files[0]!), "utf8"))).toMatchObject({ toolId: "maps.search", permission: "ask", state: "authorized-before-side-effects" });
  });
  it("Off and a changed connection revision cannot dispatch an old approved request", async () => {
    const f = await fixture(), completed = vi.fn(async () => undefined), asked = await f.executor.execute(f.invocation, undefined, undefined, completed);
    f.permission("off"); expect((await f.executor.execute({ ...f.invocation, approvalToken: asked.approvalToken }, undefined, undefined, completed)).state).toBe("failed");
    expect(f.fetcher).not.toHaveBeenCalled(); f.permission("ask");
    const fresh = await f.executor.execute(f.invocation, undefined, undefined, completed);
    const prior = (await f.maps.list())[0]!;
    await f.maps.save({ id: prior.id, expectedRevision: prior.revision, label: prior.label, endpoint: "http://127.0.0.1:2323/api/", attribution: prior.attribution,
      attributionUrl: prior.attributionUrl, enabled: true, retainedResultsAccepted: true, minimumIntervalMs: 0 });
    const rejected = await f.executor.execute({ ...f.invocation, approvalToken: fresh.approvalToken }, undefined, undefined, completed);
    expect(rejected.state).toBe("failed"); expect(rejected.error).toContain("current revision"); expect(f.fetcher).not.toHaveBeenCalled();
  });
});
