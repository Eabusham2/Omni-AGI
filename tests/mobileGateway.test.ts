import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { request as httpsRequest } from "node:https";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type {
  BrainDocument,
  BrainSummary,
  ChatResult,
  ChatStreamEvent,
  IngestResult
} from "../src/shared/types";
import { MobileGateway } from "../src/main/mobileGateway";

const temporaryDirectories: string[] = [];
const MOBILE_GATEWAY_TEST_TIMEOUT_MS = 60_000;
const GIB = 1024 ** 3;

async function fetchStage(
  stage: string,
  input: string,
  init?: RequestInit
): Promise<Response> {
  try {
    return await fetch(input, {
      ...init,
      headers: { connection: "close", ...init?.headers }
    });
  } catch (error) {
    throw new Error(`Mobile gateway ${stage} request failed.`, { cause: error });
  }
}

async function pinnedHttpsJson(
  input: string,
  certificatePem: string,
  method = "GET",
  value?: unknown
): Promise<{ status: number; value: Record<string, unknown> }> {
  const url = new URL(input);
  const body = value === undefined ? undefined : Buffer.from(JSON.stringify(value));
  return new Promise((resolve, reject) => {
    const request = httpsRequest({
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method,
      rejectUnauthorized: true,
      ca: certificatePem,
      headers: body
        ? {
            "content-type": "application/json",
            "content-length": body.length,
            connection: "close"
          }
        : { connection: "close" }
    }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
      response.on("end", () => {
        resolve({
          status: response.statusCode ?? 0,
          value: JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>
        });
      });
    });
    request.once("error", reject);
    if (body) request.write(body);
    request.end();
  });
}

function brainDocument(): BrainDocument {
  return {
    schemaVersion: 1,
    releaseFormat: "stable-1.0",
    id: "brain-1",
    name: "Same persistent mind",
    createdAt: "2026-08-23T00:00:00.000Z",
    updatedAt: "2026-08-23T00:00:00.000Z",
    readiness: {
      state: "ready",
      startedAt: "2026-08-23T00:00:00.000Z",
      completedAt: "2026-08-23T00:00:00.000Z"
    },
    lineage: { rootId: "brain-1", generation: 0 },
    config: {} as BrainDocument["config"],
    concepts: {},
    synapses: {},
    ideas: [],
    workingMemory: [],
    liquidState: { values: [], timeConstants: [], lastUpdatedAt: "2026-08-23T00:00:00.000Z" },
    messages: [
      {
        id: "message-existing",
        role: "brain",
        content: "I remember this conversation.",
        createdAt: "2026-08-23T00:00:00.000Z",
        status: "complete"
      }
    ],
    traces: [],
    trainingSources: [],
    counters: { plasticityEvents: 0, inferenceCount: 0, consolidationCycles: 0 }
  };
}

function summary(): BrainSummary {
  return {
    id: "brain-1",
    name: "Same persistent mind",
    preset: "whole-brain",
    runtime: "adaptive-core",
    updatedAt: "2026-08-23T00:00:00.000Z",
    concepts: 0,
  synapses: 0,
  activeMode: true,
  generation: 0
  };
}

class FakeActions extends EventEmitter {
  sent: Array<{ brainId: string; input: string; turnId?: string }> = [];
  cancelled: Array<{ brainId: string; turnId?: string }> = [];

  async send(
    brainId: string,
    input: string,
    signal?: AbortSignal,
    turnId?: string
  ): Promise<ChatResult> {
    signal?.throwIfAborted();
    this.sent.push({ brainId, input, turnId });
    this.emit("stream", {
      id: "stream-1",
      brainId,
      turnId: turnId!,
      sequence: 0,
      createdAt: "2026-08-23T00:00:01.000Z",
      type: "chat-token",
      delta: "same brain"
    } satisfies ChatStreamEvent);
    const brain = brainDocument();
    const humanMessage = {
      id: "human-1",
      role: "human" as const,
      content: input,
      createdAt: "2026-08-23T00:00:01.000Z",
      status: "complete" as const
    };
    const brainMessage = {
      id: "brain-message-1",
      role: "brain" as const,
      content: "same brain",
      createdAt: "2026-08-23T00:00:02.000Z",
      status: "complete" as const
    };
    return {
      brain,
      humanMessage,
      brainMessage,
      trace: {
        id: "trace-1",
        createdAt: "2026-08-23T00:00:02.000Z",
        input,
        seed: 1,
        runtime: "adaptive-core",
        activatedConcepts: [],
        recalledIdeas: [],
        driveScores: { novelty: 0, coherence: 1, curiosity: 0 },
        steps: [],
        branches: 1,
        selectedBranch: 0,
        note: "test"
      },
      actionEvents: []
    };
  }

  cancel(brainId: string, turnId?: string): number {
    this.cancelled.push({ brainId, turnId });
    return 1;
  }
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) =>
    rm(directory, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 50
    })
  ));
}, MOBILE_GATEWAY_TEST_TIMEOUT_MS);

describe("mobile companion gateway", () => {
  it("advertises only certificate-pinned HTTPS when LAN access is enabled", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mobile-gateway-tls-"));
    temporaryDirectories.push(root);
    const gateway = new MobileGateway(
      {
        root,
        list: async () => [summary()],
        get: async () => brainDocument()
      },
      { ingestPaths: async () => [] },
      new FakeActions()
    );
    const pairing = await gateway.startPairing({ allowLan: true });
    try {
      expect(pairing.protocolVersion).toBe(2);
      expect(pairing.transportSecurity).toBe("certificate-pinned-tls");
      expect(pairing.certificateSha256).toMatch(/^[a-f0-9]{64}$/u);
      expect(pairing.baseUrls.length).toBeGreaterThan(0);
      expect(pairing.baseUrls.every((url) =>
        url.startsWith("https://") &&
        url.endsWith(`#sha256=${pairing.certificateSha256}`)
      )).toBe(true);
      const identity = JSON.parse(
        await readFile(join(root, "mobile-tls-identity.json"), "utf8")
      ) as { certificatePem: string; certificateSha256: string; privateKeyPem: string };
      expect(identity.certificateSha256).toBe(pairing.certificateSha256);
      expect(JSON.stringify(pairing)).not.toContain(identity.privateKeyPem);
      const base = `https://127.0.0.1:${pairing.port}`;
      const health = await pinnedHttpsJson(
        `${base}/v1/health`,
        identity.certificatePem
      );
      expect(health).toMatchObject({
        status: 200,
        value: {
          protocolVersion: 2,
          transportSecurity: "certificate-pinned-tls",
          certificateSha256: pairing.certificateSha256
        }
      });
      const paired = await pinnedHttpsJson(
        `${base}/v1/pair`,
        identity.certificatePem,
        "POST",
        { code: pairing.code, deviceName: "Pinned phone" }
      );
      expect(paired).toMatchObject({
        status: 201,
        value: {
          certificateSha256: pairing.certificateSha256,
          token: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/u)
        }
      });
      await expect(fetchStage(
        "cleartext LAN downgrade",
        `http://127.0.0.1:${pairing.port}/v1/health`
      )).rejects.toThrow(/request failed/i);
    } finally {
      await gateway.stop();
    }
  }, MOBILE_GATEWAY_TEST_TIMEOUT_MS);

  it("pairs once, uses the existing streamed action path, and persists only a token hash", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mobile-gateway-"));
    temporaryDirectories.push(root);
    const brain = brainDocument();
    const actions = new FakeActions();
    const uploaded: Buffer[] = [];
    let diskFreeBytes = 80 * GIB;
    const gateway = new MobileGateway(
      {
        root,
        list: async () => [summary()],
        get: async (id) => {
          if (id !== brain.id) throw new Error("not found");
          return structuredClone(brain);
        }
      },
      {
        ingestPaths: async (_brainId, paths): Promise<IngestResult[]> => {
          uploaded.push(await readFile(paths[0]!));
          return [{
            brain,
            source: {
              id: "source-mobile",
              name: "phone-photo.jpg",
              kind: "image",
              policy: "pretrain",
              bytes: uploaded[0]!.length,
              learnedIdeas: 1,
              learnedConcepts: 1,
              learnedSynapses: 3,
              importedAt: "2026-08-23T00:00:03.000Z",
              rawTextRetained: false,
              contentHash: "a".repeat(64),
              provenanceUrl: "mobile://pixel-test"
            },
            warnings: []
          }];
        }
      },
      actions,
      () => false,
      async () => ({
        diskTotalBytes: 100 * GIB,
        diskFreeBytes
      })
    );

    const pairing = await gateway.startPairing({ allowLan: false });
    expect(pairing.state).toBe("listening");
    expect(pairing.protocolVersion).toBe(2);
    expect(pairing.transportSecurity).toBe("loopback-http");
    expect(pairing.certificateSha256).toBeUndefined();
    expect(pairing.code).toMatch(/^\d{6}$/u);
    expect(pairing.baseUrls).toContain(`http://10.0.2.2:${pairing.port}`);
    const base = `http://127.0.0.1:${pairing.port}`;

    const rejected = await fetchStage("rejected pairing", `${base}/v1/pair`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code: "000000", deviceName: "wrong" })
    });
    expect(rejected.status).toBe(401);
    expect(await rejected.json()).toMatchObject({ error: expect.any(String) });

    const paired = await fetchStage("pairing", `${base}/v1/pair`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code: pairing.code, deviceName: "Pixel test" })
    });
    expect(paired.status).toBe(201);
    const pairBody = await paired.json() as {
      token: string;
      device: { id: string; name: string };
    };
    expect(pairBody.device.name).toBe("Pixel test");
    expect(pairBody.token).toMatch(/^[A-Za-z0-9_-]{43}$/u);
    const store = await readFile(join(root, "mobile-pairing.json"), "utf8");
    expect(store).not.toContain(pairBody.token);
    expect(store).toContain("tokenSha256");

    const reused = await fetchStage("one-time-code reuse", `${base}/v1/pair`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code: pairing.code, deviceName: "second" })
    });
    expect(reused.status).toBe(401);
    expect(await reused.json()).toMatchObject({ error: expect.any(String) });

    const authorization = { authorization: `Bearer ${pairBody.token}` };
    const brains = await fetchStage("brain list", `${base}/v1/brains`, { headers: authorization });
    expect(await brains.json()).toMatchObject({
      brains: [{ id: "brain-1", readiness: "ready" }]
    });
    const history = await fetchStage("history", `${base}/v1/brains/brain-1/messages`, {
      headers: authorization
    });
    expect(await history.json()).toMatchObject({
      messages: [{ content: "I remember this conversation." }]
    });

    const turnId = "mobile-turn-1";
    const chat = await fetchStage("streamed chat", `${base}/v1/brains/brain-1/chat`, {
      method: "POST",
      headers: { ...authorization, "content-type": "application/json" },
      body: JSON.stringify({ input: "Continue our chat", turnId })
    });
    expect(chat.headers.get("content-type")).toContain("application/x-ndjson");
    const frames = (await chat.text()).trim().split("\n").map((line) => JSON.parse(line));
    expect(frames).toEqual(expect.arrayContaining([
      expect.objectContaining({
        type: "stream",
        event: expect.objectContaining({ type: "chat-token", delta: "same brain", turnId })
      }),
      expect.objectContaining({
        type: "result",
        turnId,
        brainMessage: expect.objectContaining({ content: "same brain" })
      })
    ]));
    expect(actions.sent).toEqual([{ brainId: "brain-1", input: "Continue our chat", turnId }]);

    const cancelled = await fetchStage("cancellation", `${base}/v1/brains/brain-1/chat/${turnId}/cancel`, {
      method: "POST",
      headers: authorization
    });
    expect(await cancelled.json()).toMatchObject({ cancelled: 1 });
    expect(actions.cancelled).toContainEqual({ brainId: "brain-1", turnId });

    const experience = Buffer.alloc(2 * 1024 * 1024, 0x5a);
    const learned = await fetchStage("experience upload", `${base}/v1/brains/brain-1/experience`, {
      method: "POST",
      headers: {
        ...authorization,
        "content-type": "image/jpeg",
        "x-omni-filename": encodeURIComponent("phone photo.jpg")
      },
      body: experience
    });
    expect(learned.status).toBe(201);
    expect(await learned.json()).toMatchObject({
      fileName: "phone photo.jpg",
      bytes: experience.length,
      results: [{ sourceId: "source-mobile" }]
    });
    expect(uploaded).toHaveLength(1);
    expect(uploaded[0]).toEqual(experience);

    // The happy path must not depend on how full the CI host happens to be,
    // while the production reserve remains a fail-closed admission boundary.
    diskFreeBytes = GIB;
    const pressure = await fetchStage("low-disk experience upload", `${base}/v1/brains/brain-1/experience`, {
      method: "POST",
      headers: {
        ...authorization,
        "content-type": "image/jpeg",
        "x-omni-filename": encodeURIComponent("must-not-stage.jpg")
      },
      body: Buffer.from([0x5a])
    });
    expect(pressure.status).toBe(507);
    expect(await pressure.json()).toMatchObject({
      error: expect.stringMatching(/storage reserve/i)
    });
    expect(uploaded).toHaveLength(1);

    const browserOrigin = await fetchStage("browser-origin rejection", `${base}/v1/brains`, {
      headers: { ...authorization, origin: "https://attacker.example" }
    });
    expect(browserOrigin.status).toBe(403);
    expect(await browserOrigin.json()).toMatchObject({ error: expect.any(String) });

    const revoked = await gateway.revoke(pairBody.device.id);
    expect(revoked.devices).toHaveLength(0);
    const afterRevoke = await fetchStage("revoked-token rejection", `${base}/v1/brains`, {
      headers: authorization
    });
    expect(afterRevoke.status).toBe(401);
    expect(await afterRevoke.json()).toMatchObject({ error: expect.any(String) });
    await gateway.stop();
  }, MOBILE_GATEWAY_TEST_TIMEOUT_MS);
});
