import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExternalAdviceClientRequest, ExternalAdviceDeltaHandler } from "../src/main/externalAdviceClient";
import { ExternalAdviceService, externalAdviceSha256, validateExternalAdviceWorkspace, type ExternalAdviceDeviceAuthority } from "../src/main/externalAdviceService";
import type { ExternalAdviceChannel, ExternalAdviceSelectedContext } from "../src/shared/externalAdvice";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-external-advice-")); roots.push(root);
  const brainRoot = join(root, "brain"); await mkdir(brainRoot);
  const secrets = new Map<string, string>();
  const secretStore = {
    get: vi.fn(async (id: string) => secrets.get(id)), has: vi.fn(async (id: string) => secrets.has(id)),
    set: vi.fn(async (id: string, value: string) => { secrets.set(id, value); return "encrypted" as const; }),
    remove: vi.fn(async (id: string) => secrets.delete(id)), persistence: vi.fn(() => "encrypted" as const)
  };
  const grants: Record<ExternalAdviceChannel, ExternalAdviceDeviceAuthority> = {
    assistant: { brainId: "brain-one", channel: "assistant", enabled: false, ownerEpoch: "assistant-owner", revision: 1 },
    reviewer: { brainId: "brain-one", channel: "reviewer", enabled: false, ownerEpoch: "reviewer-owner", revision: 1 }
  };
  const credential = "synthetic-configured-secret";
  const selectContext = vi.fn(async ({ brainId, channel, selection }: { brainId: string; channel: ExternalAdviceChannel; selection: ExternalAdviceSelectedContext["selection"] }) => {
    const text = `Recorded task contains ${credential}`;
    return { format: "omni-external-advice-selected-context" as const, version: 1 as const, brainId, channel, selection,
      items: [{ id: "task-one", kind: "task" as const, label: "Current task", text, contentSha256: externalAdviceSha256(text) }],
      includedCharacters: text.length, omittedItems: 2, truncated: true };
  });
  let block = false, mutateAuthority = false, echoedSecret: string | undefined;
  const requests: ExternalAdviceClientRequest[] = [];
  const client = { stream: vi.fn(async (request: ExternalAdviceClientRequest, onDelta: ExternalAdviceDeltaHandler, signal: AbortSignal) => {
    requests.push(request);
    if (echoedSecret) {
      const at = Math.max(1, Math.floor(echoedSecret.length / 2));
      await onDelta(`before ${echoedSecret.slice(0, at)}`); await onDelta(`${echoedSecret.slice(at)} after`);
      return { text: `before ${echoedSecret} after`, providerRequestId: "provider-one", finishReason: "stop" };
    }
    await onDelta("advice ");
    if (mutateAuthority) { grants.assistant.revision += 1; await onDelta("stale"); }
    if (block) await new Promise<never>((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    await onDelta("complete"); return { text: "advice complete", providerRequestId: "provider-one", finishReason: "stop" };
  }) };
  const service = new ExternalAdviceService(client, secretStore, { deviceDirectory: join(root, "device"), brainRoot: () => brainRoot,
    authority: async (_brainId, channel) => ({ ...grants[channel] }), selectContext, admitWrite: async () => undefined });
  return { root, brainRoot, service, secrets, secretStore, grants, credential, selectContext, client, requests,
    setBlock(value: boolean) { block = value; }, setMutateAuthority(value: boolean) { mutateAuthority = value; },
    setEchoedSecret(value: string | undefined) { echoedSecret = value; } };
}

async function enable(f: Awaited<ReturnType<typeof fixture>>, channel: ExternalAdviceChannel, provider: "openai" | "openai-compatible-local" = "openai") {
  f.grants[channel].enabled = true;
  const current = await f.service.settings("brain-one");
  return f.service.configure("brain-one", { expectedRevision: current.revision, channel, configuration: { enabled: true, provider,
    model: provider === "openai" ? "manual-model" : "local-model", ...(provider === "openai-compatible-local" ? { endpoint: "http://127.0.0.1:9988/v1" } : {}),
    maxOutputTokens: 512, context: { includeTask: true, includeActivity: false, maxCharacters: 20_000 } } });
}

describe("persistent per-brain external advice", () => {
  it("defaults both channels off and requires separate current device authority", async () => {
    const f = await fixture(), settings = await f.service.settings("brain-one");
    expect(settings).toMatchObject({ revision: 1, assistant: { enabled: false }, reviewer: { enabled: false },
      deviceAuthority: { assistant: false, reviewer: false }, credentialsStoredInBrain: false });
    await expect(f.service.preview("brain-one", { channel: "assistant", prompt: "help" })).rejects.toThrow(/disabled in this brain/);
    const inventory = await validateExternalAdviceWorkspace(f.brainRoot, "brain-one");
    expect(inventory).toMatchObject({ brainId: "brain-one", settingsRevision: 1, requests: 0, inheritedRequests: 0, inheritedSettings: 0, inheritedMeta: 0, reviews: 0 });
  });

  it("keeps opaque credentials outside the brain and redacts every configured secret from previews and history", async () => {
    const f = await fixture(); await f.service.saveCredential({ provider: "openai", credential: f.credential }); await enable(f, "assistant");
    const preview = await f.service.preview("brain-one", { channel: "assistant", prompt: `Please inspect ${f.credential}` });
    expect(preview.requestText).toContain("[REDACTED]"); expect(JSON.stringify(preview)).not.toContain(f.credential);
    expect(preview.context.items[0]).toMatchObject({ sourceContentSha256: externalAdviceSha256(`Recorded task contains ${f.credential}`) });
    expect(preview.context.redactedCredentials).toBeGreaterThanOrEqual(2);
    const complete = new Promise<void>((resolve) => f.service.on("event", (event) => { if (event.request.state === "complete") resolve(); }));
    const started = await f.service.start("brain-one", { channel: "assistant", prompt: `Please inspect ${f.credential}`,
      requestId: "request-one", expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 });
    expect(started.state).toBe("queued"); await complete;
    const history = await f.service.history("brain-one");
    expect(history.entries[0]).toMatchObject({ state: "complete", response: "advice complete", automaticLearning: false, hostActionAuthority: false });
    expect(JSON.stringify(history)).not.toContain(f.credential);
    expect(f.requests[0]).toMatchObject({ provider: "openai", model: "manual-model", maxOutputTokens: 512 });
    const brainBytes = await readFile(join(f.brainRoot, "external-advice/workspace.sqlite3")); expect(brainBytes.includes(Buffer.from(f.credential))).toBe(false);
    const deviceBytes = await readFile(join(f.root, "device/external-advice-device.json"), "utf8"); expect(deviceBytes).not.toContain(f.credential);
    expect([...f.secrets.values()]).toContain(f.credential);
    const reopened = new ExternalAdviceService(f.client, f.secretStore, { deviceDirectory: join(f.root, "device"), brainRoot: () => f.brainRoot,
      authority: async (_brainId, channel) => ({ ...f.grants[channel] }), selectContext: f.selectContext, admitWrite: async () => undefined });
    expect((await reopened.settings("brain-one")).assistant).toMatchObject({ enabled: true, provider: "openai", model: "manual-model" });
    expect((await reopened.history("brain-one")).entries[0]).toMatchObject({ id: "request-one", state: "complete", response: "advice complete" });
  });

  it("fences settings/authority revisions, rejects preview reuse, and never silently falls back", async () => {
    const f = await fixture(); await f.service.saveCredential({ provider: "openai", credential: f.credential }); await enable(f, "assistant");
    const preview = await f.service.preview("brain-one", { channel: "assistant", prompt: "check" });
    await f.service.configure("brain-one", { expectedRevision: preview.settingsRevision, channel: "assistant", configuration: { enabled: true,
      provider: "openai", model: "another-manual-model", maxOutputTokens: 512, context: preview.context.selection } });
    await expect(f.service.start("brain-one", { channel: "assistant", prompt: "check", requestId: "stale-preview",
      expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 })).rejects.toThrow(/preview changed|settings changed/);
    expect(f.client.stream).not.toHaveBeenCalled();

    const current = await f.service.preview("brain-one", { channel: "assistant", prompt: "fence" }); f.setMutateAuthority(true);
    const failed = new Promise<void>((resolve) => f.service.on("event", (event) => { if (event.request.id === "authority-fence" && event.request.state === "failed") resolve(); }));
    await f.service.start("brain-one", { channel: "assistant", prompt: "fence", requestId: "authority-fence",
      expectedSettingsRevision: current.settingsRevision, previewSha256: current.sha256 }); await failed;
    expect((await f.service.history("brain-one")).entries[0]).toMatchObject({ id: "authority-fence", state: "failed" });
    expect(f.requests).toHaveLength(1);
  });

  it("redacts a credential from another configured provider even when the response splits it across deltas", async () => {
    const f = await fixture(), echoed = "synthetic-other-provider-secret";
    await f.service.saveCredential({ provider: "openai", credential: f.credential });
    await f.service.saveCredential({ provider: "anthropic", credential: echoed });
    await enable(f, "assistant"); f.setEchoedSecret(echoed);
    const observed: string[] = []; f.service.on("event", (event) => observed.push(JSON.stringify(event)));
    const preview = await f.service.preview("brain-one", { channel: "assistant", prompt: "answer" });
    const complete = new Promise<void>((resolve) => f.service.on("event", (event) => { if (event.request.id === "redacted-response" && event.request.state === "complete") resolve(); }));
    await f.service.start("brain-one", { channel: "assistant", prompt: "answer", requestId: "redacted-response",
      expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 }); await complete;
    const request = (await f.service.history("brain-one")).entries[0]!;
    expect(request).toMatchObject({ response: "before [REDACTED] after", responseCredentialRedactions: 1, state: "complete" });
    expect(JSON.stringify(request)).not.toContain(echoed); expect(observed.join("\n")).not.toContain(echoed);
  });

  it("suspends and awaits active work without replay, then retains a terminal cancelled receipt", async () => {
    const f = await fixture(); await enable(f, "assistant", "openai-compatible-local"); f.setBlock(true);
    const preview = await f.service.preview("brain-one", { channel: "assistant", prompt: "wait" });
    const streaming = new Promise<void>((resolve) => f.service.on("event", (event) => { if (event.request.id === "wait-one" && event.request.state === "streaming") resolve(); }));
    await f.service.start("brain-one", { channel: "assistant", prompt: "wait", requestId: "wait-one",
      expectedSettingsRevision: preview.settingsRevision, previewSha256: preview.sha256 }); await streaming; await f.service.suspend("brain-one");
    const history = await f.service.history("brain-one"); expect(history.entries[0]).toMatchObject({ id: "wait-one", state: "cancelled", response: "advice " });
    expect(f.client.stream).toHaveBeenCalledTimes(1); await f.service.suspend("brain-one"); expect(f.client.stream).toHaveBeenCalledTimes(1);
  });

  it("keeps reviewer execution internal, advisory-only, and separate from assistant history", async () => {
    const f = await fixture(); await f.service.saveCredential({ provider: "openai", credential: f.credential }); await enable(f, "reviewer");
    const settings = await f.service.settings("brain-one");
    await expect(f.service.start("brain-one", { channel: "reviewer", prompt: "proposal", requestId: "not-public",
      expectedSettingsRevision: settings.revision, previewSha256: "0".repeat(64) })).rejects.toThrow(/assistant-only/);
    const receipt = await f.service.completeReview("brain-one", "Review selected proposal", new AbortController().signal);
    expect(receipt).toMatchObject({ text: "advice complete", advisoryOnly: true, hostActionAuthority: false,
      request: { channel: "reviewer", state: "complete", automaticLearning: false } });
    expect((await f.service.history("brain-one", { channel: "reviewer" })).entries).toHaveLength(1);
  });
});
