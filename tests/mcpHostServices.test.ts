import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, readdir, realpath, rename, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { McpRootsService, type McpRootsOptions } from "../src/main/mcpRoots";
import { McpDiagnosticService } from "../src/main/mcpDiagnostics";
import type { McpHostConnection, McpRootDisclosureReview } from "../src/shared/mcpHostServices";

let root: string, connection: McpHostConnection, level: "off" | "ask" | "full", revision: string, allowed: boolean;
const signal = () => new AbortController().signal;
beforeEach(async () => {
  root = await realpath(await mkdtemp(join(tmpdir(), "omni-mcp-host-services-")));
  connection = { brainId: "first", serverId: "fixture", serverLabel: "Selected fixture", endpoint: "https://example.invalid/mcp", connectionEpoch: 1,
    configurationSha256: "a".repeat(64), connectionId: "actual-main-session-one", ready: true, enabled: true };
  level = "ask"; revision = "file-policy-one"; allowed = true;
});
afterEach(async () => { await rm(root, { recursive: true, force: true }); });
function fixture(overrides: Partial<McpRootsOptions> = {}) {
  const reviews: McpRootDisclosureReview[] = [], sent: McpHostConnection[] = [];
  const options: McpRootsOptions = { directory: join(root, "device-private-roots"), connection: () => structuredClone(connection),
    requireConnection: expected => { if (!allowed || !connection.ready || expected.connectionId !== connection.connectionId || expected.endpoint !== connection.endpoint) throw new Error("connection revoked"); },
    filePolicy: async () => ({ level, revision }), approve: async (_source, review) => { reviews.push(review); return true; },
    admit: async (_bytes, brainId) => { expect(brainId).toBe("first"); }, checkDisk: async () => undefined,
    notify: async expected => { if (!connection.ready || expected.connectionId !== connection.connectionId || expected.endpoint !== connection.endpoint) return false; sent.push(expected); return true; }, ...overrides };
  return { service: new McpRootsService(options), reviews, sent, options };
}

it("lists no automatic roots and shares only explicitly selected accessible paths after exact endpoint review", async () => {
  const f = fixture(), selected = join(root, "picked project"); await mkdir(selected);
  expect(await f.service.list(connection, signal())).toEqual({ roots: [] }); expect(f.reviews).toEqual([]);
  expect(await f.service.status("first", "fixture")).toMatchObject({ state: "empty", permissionsPortable: false, filesystemSandbox: false });
  const status = await f.service.select("first", "fixture", [selected, selected], signal());
  expect(status).toMatchObject({ state: "shared", revision: 1, notification: "sent", roots: [{ uri: pathToFileURL(selected).href, accessible: true }] });
  expect(f.reviews).toHaveLength(1); expect(f.reviews[0]).toMatchObject({ disclosureOnly: true, filesystemSandbox: false, policyRevision: revision, roots: [{ uri: pathToFileURL(selected).href }] });
  expect((await f.service.list(connection, signal())).roots).toEqual([{ uri: pathToFileURL(selected).href, name: "picked project" }]);
  expect(f.sent).toHaveLength(2); await f.service.close();
});

it("retains only dormant device choices across restart and never revives grants on epoch or replaced-runtime identity", async () => {
  const f = fixture(), selected = join(root, "picked"); await mkdir(selected); await f.service.select("first", "fixture", [selected], signal()); await f.service.close();
  const raw = await readFile(join(f.options.directory, (await readdir(f.options.directory))[0]!), "utf8");
  expect(raw).not.toContain("actual-main-session-one"); expect(raw).not.toContain("file-policy-one");
  const next = new McpRootsService(f.options);
  expect(await next.status("first", "fixture")).toMatchObject({ state: "review-required" }); expect(await next.list(connection, signal())).toEqual({ roots: [] });
  await next.review("first", "fixture", signal()); expect((await next.list(connection, signal())).roots).toHaveLength(1);
  connection = { ...connection, connectionId: "replacement-runtime-same-integer-epoch" };
  expect(await next.list(connection, signal())).toEqual({ roots: [] });
  await next.review("first", "fixture", signal()); connection = { ...connection, connectionEpoch: 2, connectionId: "new-live-session" };
  expect(await next.status("first", "fixture")).toMatchObject({ state: "review-required" }); expect(await next.list(connection, signal())).toEqual({ roots: [] }); await next.close();
});

it("clears immediately during human review and rechecks current Off policy after awaited allocation", async () => {
  let release!: (value: boolean) => void, entered!: () => void;
  const waiting = new Promise<void>(resolve => { entered = resolve; });
  const f = fixture({ approve: async () => { entered(); return new Promise<boolean>(resolve => { release = resolve; }); } });
  const selected = join(root, "picked"); await mkdir(selected);
  const work = f.service.select("first", "fixture", [selected], signal()).catch(error => error); await waiting;
  expect(await f.service.clear("first", "fixture", signal())).toMatchObject({ state: "empty", revision: 2 });
  release(true); expect(await work).toBeInstanceOf(Error); expect(await f.service.list(connection, signal())).toEqual({ roots: [] }); await f.service.close();
  let revokeDuringAdmit = false;
  const other = fixture({ directory: join(root, "other-device-roots"), admit: async () => { if (revokeDuringAdmit) { level = "off"; revision = "changed-off"; } } });
  await other.service.select("first", "fixture", [selected], signal()); revokeDuringAdmit = true;
  expect(await other.service.list(connection, signal())).toEqual({ roots: [] });
  expect(await other.service.status("first", "fixture")).toMatchObject({ state: "review-required" }); await other.service.close();
});

it("rejects links and changed filesystem ownership, endpoint selection or same-revision metadata substitution", async () => {
  const f = fixture(), selected = join(root, "picked"), link = join(root, "linked"); await mkdir(selected); await symlink(selected, link);
  await expect(f.service.select("first", "fixture", [link], signal())).rejects.toThrow("link");
  await f.service.select("first", "fixture", [selected], signal());
  await rename(selected, join(root, "original-picked")); await mkdir(selected);
  expect(await f.service.list(connection, signal())).toEqual({ roots: [] });
  await expect(f.service.review("first", "fixture", signal())).rejects.toThrow("replaced");
  await f.service.select("first", "fixture", [selected], signal());
  const path = join(f.options.directory, (await readdir(f.options.directory))[0]!), metadata = JSON.parse(await readFile(path, "utf8"));
  metadata.roots[0].name = "unreviewed renamed descriptor"; await writeFile(path, JSON.stringify(metadata));
  expect(await f.service.list(connection, signal())).toEqual({ roots: [] });
  connection = { ...connection, endpoint: "https://different.invalid/mcp", configurationSha256: "b".repeat(64), connectionId: "different-peer" };
  await expect(f.service.review("first", "fixture", signal())).rejects.toThrow("different endpoint"); await f.service.close();
});

it("aborts a pending ungranted disclosure on owner revocation and leaves unready selections dormant", async () => {
  const selected = join(root, "picked"); await mkdir(selected); connection = { ...connection, ready: false };
  const f = fixture(); expect(await f.service.select("first", "fixture", [selected], signal())).toMatchObject({ state: "review-required", notification: "not-ready" }); expect(f.reviews).toEqual([]);
  connection = { ...connection, ready: true }; let release!: (value: boolean) => void, entered!: () => void;
  const enteredPromise = new Promise<void>(resolve => { entered = resolve; });
  const other = new McpRootsService({ ...f.options, approve: async () => { entered(); return new Promise<boolean>(resolve => { release = resolve; }); } });
  const work = other.review("first", "fixture", signal()).catch(error => error); await enteredPromise; await other.revokeBrain("first"); release(true);
  expect(await work).toBeInstanceOf(Error); expect(await other.list(connection, signal())).toEqual({ roots: [] });
  level = "off"; await expect(other.select("first", "fixture", [selected], signal())).rejects.toThrow("Off"); await other.close(); await f.service.close();
});

it("records only bounded attributed redacted diagnostics, never a model message or raw credential", async () => {
  const diagnostic = new McpDiagnosticService({ requireAccess: () => { if (!allowed) throw new Error("revoked"); },
    knownSecrets: async () => ["fixture-bearer-secret", "fixture-environment-secret"], admit: async (_bytes, brainId) => { expect(brainId).toBe("first"); }, now: () => 1000 });
  const observe = vi.fn(); diagnostic.subscribe(observe);
  await diagnostic.receive(connection, { level: "warning", logger: "fixture", data: { Authorization: "Bearer arbitrary-hidden-key", nested: ["fixture-bearer-secret", "fixture-environment-secret", "actual-private-view-token"], public: "<script>inert text</script>" } }, signal(), "app", "actual-view", ["actual-private-view-token"]);
  const page = diagnostic.page("first", "fixture"); expect(page.records[0]).toMatchObject({ source: "app", viewId: "actual-view", level: "warning", redactionApplied: true, untrustedData: true, nativeLearning: false });
  const text = JSON.stringify(page); for (const secret of ["arbitrary-hidden-key", "fixture-bearer-secret", "fixture-environment-secret", "actual-private-view-token"]) expect(text).not.toContain(secret);
  expect(text).toContain("<script>inert text</script>"); expect(observe).toHaveBeenCalledOnce(); expect(diagnostic.page("other", "fixture").records).toEqual([]); diagnostic.close();
});

it("rejects invalid/oversized/accessor logs and isolates revoked async secret reads without accepting effects", async () => {
  const diagnostic = new McpDiagnosticService({ requireAccess: () => { if (!allowed) throw new Error("revoked"); }, knownSecrets: async () => { allowed = false; return []; }, admit: async () => undefined });
  await expect(diagnostic.receive(connection, { level: "trace", data: "bad" }, signal())).rejects.toThrow("level");
  await expect(diagnostic.receive(connection, { level: "info", data: "x".repeat(65537) }, signal())).rejects.toThrow("64 KiB");
  const getter = vi.fn(() => "must not execute"), data = Object.defineProperty({}, "hidden", { enumerable: true, get: getter });
  await expect(diagnostic.receive(connection, { level: "info", data }, signal())).rejects.toThrow("accessors"); expect(getter).not.toHaveBeenCalled();
  await expect(diagnostic.receive(connection, { level: "info", data: "safe before revoke" }, signal())).rejects.toThrow("revoked");
  expect(diagnostic.page("first", "fixture")).toMatchObject({ records: [], rejected: 4 }); diagnostic.close();
});

it("reports rate suppression and bounded retention honestly with paged process-only history", async () => {
  let now = 1000;
  const diagnostic = new McpDiagnosticService({ requireAccess: () => undefined, knownSecrets: async () => [], admit: async () => undefined, now: () => now });
  for (let index = 0; index < 60; index++) await diagnostic.receive(connection, { level: "debug", data: index }, signal());
  expect(await diagnostic.receive(connection, { level: "debug", data: "rate-limited" }, signal())).toBe("rate-limited");
  for (let index = 60; index < 260; index++) { now += 10_001; await diagnostic.receive(connection, { level: "info", data: index }, signal()); }
  const page = diagnostic.page("first", "fixture", 0, 1); expect(page).toMatchObject({ accepted: 260, rateLimited: 1, evicted: 4, hasMore: true, retention: "bounded-current-process-only" });
  expect(page.records).toHaveLength(1); expect(page.records[0]?.data).toBe(4);
  expect(diagnostic.page("first", "fixture", page.nextSequence, 100).records).toHaveLength(100); diagnostic.close();
});

it("retires only the matching pending connection review and fences a status read changed during policy await", async () => {
  let pending = false, entered!: () => void, release!: (value: boolean) => void, raceStatus = false, reviewSignal: AbortSignal | undefined;
  const enteredPromise = new Promise<void>(resolve => { entered = resolve; });
  const f = fixture({ approve: async (_connection, _review, signal) => { if (!pending) return true; reviewSignal = signal; entered(); return new Promise<boolean>(resolve => { release = resolve; }); },
    filePolicy: async () => { if (raceStatus) { raceStatus = false; connection = { ...connection, connectionId: "status-replaced-session", ready: false }; } return { level, revision }; } });
  const selected = join(root, "picked"); await mkdir(selected); await f.service.select("first", "fixture", [selected], signal());
  raceStatus = true; expect(await f.service.status("first", "fixture")).toMatchObject({ state: "review-required", connectionReady: false });
  connection = { ...connection, connectionId: "next-reviewed-session", ready: true }; pending = true;
  const work = f.service.review("first", "fixture", signal()).catch(error => error); await enteredPromise;
  await f.service.revokeConnection("first", "unrelated-server"); // does not cancel this server's review
  expect(reviewSignal?.aborted).toBe(false);
  await f.service.revokeConnection("first", "fixture"); release(true);
  expect(await work).toBeInstanceOf(Error); expect(await f.service.list(connection, signal())).toEqual({ roots: [] }); await f.service.close();
});
