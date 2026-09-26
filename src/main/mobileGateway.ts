import { createHash, randomBytes, randomInt, randomUUID } from "node:crypto";
import { createServer as createHttpServer, type IncomingMessage, type Server as HttpServer, type ServerResponse } from "node:http";
import { createServer as createHttpsServer, type Server as HttpsServer } from "node:https";
import { isIP } from "node:net";
import { networkInterfaces } from "node:os";
import { basename, join } from "node:path";
import { once } from "node:events";
import type { TLSSocket } from "node:tls";
import {
  mkdir,
  open,
  readFile,
  rename,
  rm,
  writeFile
} from "node:fs/promises";
import type { EventEmitter } from "node:events";
import type {
  BrainDocument,
  BrainSummary,
  ChatResult,
  ChatStreamEvent,
  ConversationLedgerPage,
  DataIngestionPolicy,
  IngestResult,
  MobileGatewayDevice,
  MobileGatewayStartRequest,
  MobileGatewayStatus,
  MobilePairingSession
} from "../shared/types";
import { actionEventForTurnStream } from "./chatActionController";
import {
  loadOrCreateMobileTlsIdentity,
  type MobileTlsIdentity
} from "./mobileTlsIdentity";
import {
  DiskReservePauseError,
  requireDiskWrite,
  type DiskSpaceReader
} from "./diskSpace";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const JSON_BODY_LIMIT = 1024 * 1024;
const STREAM_BUFFER_LIMIT = 64 * 1024 * 1024;
const PAIRING_LIFETIME_MS = 5 * 60_000;

interface MobileRepository {
  readonly root: string;
  list(): Promise<BrainSummary[]>;
  get(id: string): Promise<BrainDocument>;
  conversationPage?(
    id: string,
    beforeSequence?: number,
    limit?: number
  ): Promise<ConversationLedgerPage>;
}

interface MobileBrainService {
  ingestPaths(
    brainId: string,
    paths: string[],
    policy?: DataIngestionPolicy
  ): Promise<IngestResult[]>;
}

interface MobileActions extends EventEmitter {
  send(
    brainId: string,
    input: string,
    signal?: AbortSignal,
    requestedTurnId?: string
  ): Promise<ChatResult>;
  cancel(brainId: string, turnId?: string): number;
}

interface StoredMobileDevice extends MobileGatewayDevice {
  tokenSha256: string;
}

interface MobileGatewayStore {
  schemaVersion: 1;
  devices: StoredMobileDevice[];
}

interface AuthenticatedRequest {
  device: StoredMobileDevice;
}

interface PairAttempt {
  count: number;
  resetAt: number;
}

function asErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function tokenHash(token: string): string {
  return createHash("sha256").update(token, "utf8").digest("hex");
}

function safeDeviceName(value: unknown): string {
  if (typeof value !== "string") return "Mobile device";
  return value.replace(/[\u0000-\u001f\u007f]/gu, " ").replace(/\s+/gu, " ").trim().slice(0, 80) ||
    "Mobile device";
}

function safeFileName(value: string | undefined): string {
  const decoded = (() => {
    try {
      return decodeURIComponent(value ?? "");
    } catch {
      return value ?? "";
    }
  })();
  const normalized = basename(decoded)
    .replace(/[<>:"/\\|?*\u0000-\u001f]/gu, "-")
    .replace(/[. ]+$/gu, "")
    .slice(0, 180);
  return normalized || `mobile-experience-${Date.now()}.bin`;
}

function requireId(value: string | undefined, label: string): string {
  if (!value || !SAFE_ID.test(value)) throw new HttpError(400, `Invalid ${label}.`);
  return value;
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message);
  }
}

class NdjsonWriter {
  private pending = Promise.resolve();
  private queuedBytes = 0;
  private failed: Error | undefined;

  constructor(
    private readonly response: ServerResponse,
    private readonly abort: () => void
  ) {}

  write(value: unknown): Promise<void> {
    const line = `${JSON.stringify(value)}\n`;
    const bytes = Buffer.byteLength(line);
    this.queuedBytes += bytes;
    if (this.queuedBytes > STREAM_BUFFER_LIMIT) {
      this.failed = new Error("The mobile receiver could not keep up with the neural stream.");
      this.abort();
      return Promise.reject(this.failed);
    }
    this.pending = this.pending.then(async () => {
      if (this.failed) throw this.failed;
      if (this.response.destroyed || this.response.writableEnded) {
        throw new Error("The mobile stream closed.");
      }
      if (!this.response.write(line)) await once(this.response, "drain");
      this.queuedBytes -= bytes;
    }).catch((error: unknown) => {
      this.failed = error instanceof Error ? error : new Error(String(error));
      this.abort();
      throw this.failed;
    });
    return this.pending;
  }

  async end(): Promise<void> {
    await this.pending;
    if (!this.response.destroyed && !this.response.writableEnded) this.response.end();
  }
}

function jsonResponse(response: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
    "x-content-type-options": "nosniff"
  });
  response.end(body);
}

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const rawChunk of request) {
    const chunk = Buffer.isBuffer(rawChunk) ? rawChunk : Buffer.from(rawChunk);
    bytes += chunk.length;
    if (bytes > JSON_BODY_LIMIT) throw new HttpError(413, "The JSON request is too large.");
    chunks.push(chunk);
  }
  let value: unknown;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "The request body must be valid JSON.");
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new HttpError(400, "The request body must be a JSON object.");
  }
  return value as Record<string, unknown>;
}

function requestHost(request: IncomingMessage): string {
  const raw = request.headers.host;
  if (!raw || raw.length > 300) throw new HttpError(400, "A valid Host header is required.");
  try {
    return new URL(`http://${raw}`).hostname.replace(/^\[|\]$/gu, "").toLocaleLowerCase();
  } catch {
    throw new HttpError(400, "The Host header is invalid.");
  }
}

function lanIpv4Addresses(): string[] {
  const values = new Set<string>();
  for (const entries of Object.values(networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (!entry.internal && entry.family === "IPv4") values.add(entry.address);
    }
  }
  return [...values];
}

function localBaseUrls(
  port: number,
  allowLan: boolean,
  certificateSha256?: string,
  lanAddresses: string[] = lanIpv4Addresses()
): string[] {
  if (!allowLan) {
    return [`http://127.0.0.1:${port}`, `http://10.0.2.2:${port}`];
  }
  if (!certificateSha256) throw new Error("LAN mobile access requires a TLS identity.");
  return [...new Set(["127.0.0.1", "10.0.2.2", ...lanAddresses])].map(
    (address) => `https://${address}:${port}/#sha256=${certificateSha256}`
  );
}

/**
 * Opt-in companion transport. It exposes the existing BrainService and
 * ChatActionController; it never instantiates or copies a model.
 */
export class MobileGateway {
  private server: HttpServer | HttpsServer | undefined;
  private port: number | undefined;
  private allowLan = false;
  private pairingCode: string | undefined;
  private pairingExpiresAt = 0;
  private loaded = false;
  private devices: StoredMobileDevice[] = [];
  private readonly pairAttempts = new Map<string, PairAttempt>();
  private tlsIdentity: MobileTlsIdentity | undefined;
  private allowedHosts = new Set<string>();
  private lanAddresses: string[] = [];

  constructor(
    private readonly repository: MobileRepository,
    private readonly service: MobileBrainService,
    private readonly actions: MobileActions,
    private readonly isInitializing: (brainId: string) => boolean = () => false,
    private readonly diskSpaceReader?: DiskSpaceReader
  ) {}

  private get storePath(): string {
    return join(this.repository.root, "mobile-pairing.json");
  }

  private async loadStore(): Promise<void> {
    if (this.loaded) return;
    this.loaded = true;
    try {
      const parsed = JSON.parse(await readFile(this.storePath, "utf8")) as MobileGatewayStore;
      if (parsed.schemaVersion !== 1 || !Array.isArray(parsed.devices)) return;
      this.devices = parsed.devices.filter((device) =>
        typeof device?.id === "string" &&
        SAFE_ID.test(device.id) &&
        typeof device.name === "string" &&
        typeof device.createdAt === "string" &&
        typeof device.lastSeenAt === "string" &&
        typeof device.tokenSha256 === "string" &&
        /^[a-f0-9]{64}$/u.test(device.tokenSha256)
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        console.warn("Mobile pairing store could not be read:", asErrorMessage(error));
      }
    }
  }

  private async saveStore(): Promise<void> {
    await mkdir(this.repository.root, { recursive: true });
    const temporary = `${this.storePath}.${randomUUID()}.tmp`;
    await writeFile(
      temporary,
      JSON.stringify({ schemaVersion: 1, devices: this.devices } satisfies MobileGatewayStore, null, 2),
      { encoding: "utf8", mode: 0o600, flag: "wx" }
    );
    await rename(temporary, this.storePath);
  }

  private publicDevices(): MobileGatewayDevice[] {
    return this.devices.map(({ tokenSha256: _secret, ...device }) => ({ ...device }));
  }

  async status(): Promise<MobileGatewayStatus> {
    await this.loadStore();
    return {
      schemaVersion: 1,
      protocolVersion: 2,
      state: this.server ? "listening" : "stopped",
      allowLan: this.server ? this.allowLan : false,
      transportSecurity: !this.server
        ? "stopped"
        : this.allowLan
          ? "certificate-pinned-tls"
          : "loopback-http",
      ...(this.server && this.port
        ? {
            port: this.port,
            baseUrls: localBaseUrls(
              this.port,
              this.allowLan,
              this.tlsIdentity?.certificateSha256,
              this.lanAddresses
            ),
            ...(this.tlsIdentity
              ? { certificateSha256: this.tlsIdentity.certificateSha256 }
              : {})
          }
        : { baseUrls: [] }),
      devices: this.publicDevices()
    };
  }

  async startPairing(request: MobileGatewayStartRequest): Promise<MobilePairingSession> {
    if (typeof request !== "object" || request === null || typeof request.allowLan !== "boolean") {
      throw new Error("Invalid mobile gateway request.");
    }
    await this.loadStore();
    if (this.server && request.allowLan !== this.allowLan) await this.stop();
    if (!this.server) await this.listen(request.allowLan);
    this.pairingCode = String(randomInt(0, 1_000_000)).padStart(6, "0");
    this.pairingExpiresAt = Date.now() + PAIRING_LIFETIME_MS;
    const status = await this.status();
    if (status.state !== "listening") throw new Error("Mobile gateway did not start.");
    return {
      ...status,
      state: "listening",
      code: this.pairingCode,
      expiresAt: new Date(this.pairingExpiresAt).toISOString()
    };
  }

  private async listen(allowLan: boolean): Promise<void> {
    const handler = (request: IncomingMessage, response: ServerResponse): void => {
      void this.handle(request, response).catch((error: unknown) => {
        if (response.destroyed || response.writableEnded) return;
        const status = error instanceof HttpError ? error.status : 500;
        jsonResponse(response, status, {
          error: status === 500 ? "The mobile gateway could not complete this request." : asErrorMessage(error)
        });
        if (status === 500) console.error("Mobile gateway request failed:", error);
      });
    };
    const lanAddresses = allowLan ? lanIpv4Addresses() : [];
    const tlsIdentity = allowLan
      ? await loadOrCreateMobileTlsIdentity(this.repository.root, lanAddresses)
      : undefined;
    const server = tlsIdentity
      ? createHttpsServer({
          key: tlsIdentity.privateKeyPem,
          cert: tlsIdentity.certificatePem,
          minVersion: "TLSv1.2"
        }, handler)
      : createHttpServer(handler);
    server.on("clientError", (_error, socket) => socket.destroy());
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, allowLan ? "0.0.0.0" : "127.0.0.1", () => {
        server.off("error", reject);
        resolve();
      });
    });
    const address = server.address();
    if (!address || typeof address === "string") {
      server.close();
      throw new Error("Mobile gateway did not receive a TCP port.");
    }
    this.server = server;
    this.port = address.port;
    this.allowLan = allowLan;
    this.tlsIdentity = tlsIdentity;
    this.lanAddresses = lanAddresses;
    this.allowedHosts = new Set([
      "localhost",
      "127.0.0.1",
      "::1",
      "10.0.2.2",
      ...lanAddresses
    ]);
  }

  async stop(): Promise<MobileGatewayStatus> {
    this.pairingCode = undefined;
    this.pairingExpiresAt = 0;
    const server = this.server;
    this.server = undefined;
    this.port = undefined;
    this.allowLan = false;
    this.tlsIdentity = undefined;
    this.lanAddresses = [];
    this.allowedHosts.clear();
    if (server) {
      server.closeAllConnections?.();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    return this.status();
  }

  async revoke(deviceId: string): Promise<MobileGatewayStatus> {
    requireId(deviceId, "device id");
    await this.loadStore();
    const next = this.devices.filter((device) => device.id !== deviceId);
    if (next.length === this.devices.length) throw new Error("The paired device was not found.");
    this.devices = next;
    await this.saveStore();
    return this.status();
  }

  private validateHost(request: IncomingMessage): void {
    const host = requestHost(request);
    if (host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "10.0.2.2") {
      return;
    }
    if (
      !this.allowLan ||
      isIP(host) === 0 ||
      !this.allowedHosts.has(host)
    ) throw new HttpError(421, "This host is not permitted.");
    if (!(request.socket as TLSSocket).encrypted || !this.tlsIdentity) {
      throw new HttpError(426, "LAN companion access requires certificate-pinned TLS.");
    }
  }

  private async authenticate(request: IncomingMessage): Promise<AuthenticatedRequest> {
    const authorization = request.headers.authorization;
    if (!authorization?.startsWith("Bearer ")) throw new HttpError(401, "Pair this device first.");
    const token = authorization.slice(7).trim();
    if (!/^[A-Za-z0-9_-]{40,100}$/u.test(token)) throw new HttpError(401, "Invalid mobile token.");
    const hash = tokenHash(token);
    const device = this.devices.find((entry) => entry.tokenSha256 === hash);
    if (!device) throw new HttpError(401, "This mobile pairing is no longer valid.");
    device.lastSeenAt = new Date().toISOString();
    await this.saveStore();
    return { device };
  }

  private checkPairRate(request: IncomingMessage): void {
    const key = request.socket.remoteAddress ?? "unknown";
    const now = Date.now();
    const current = this.pairAttempts.get(key);
    const attempt = !current || current.resetAt <= now
      ? { count: 0, resetAt: now + PAIRING_LIFETIME_MS }
      : current;
    attempt.count += 1;
    this.pairAttempts.set(key, attempt);
    if (attempt.count > 10) throw new HttpError(429, "Too many pairing attempts. Start a new pairing session.");
  }

  private async pair(request: IncomingMessage, response: ServerResponse): Promise<void> {
    this.checkPairRate(request);
    const value = await readJson(request);
    if (
      !this.pairingCode ||
      this.pairingExpiresAt <= Date.now() ||
      typeof value.code !== "string" ||
      value.code !== this.pairingCode
    ) {
      throw new HttpError(401, "The pairing code is invalid or expired.");
    }
    const now = new Date().toISOString();
    const token = randomBytes(32).toString("base64url");
    const device: StoredMobileDevice = {
      id: randomUUID(),
      name: safeDeviceName(value.deviceName),
      createdAt: now,
      lastSeenAt: now,
      tokenSha256: tokenHash(token)
    };
    this.devices.push(device);
    this.pairingCode = undefined;
    this.pairingExpiresAt = 0;
    await this.saveStore();
    const { tokenSha256: _secret, ...publicDevice } = device;
    jsonResponse(response, 201, {
      schemaVersion: 1,
      protocolVersion: 2,
      token,
      device: publicDevice,
      ...(this.tlsIdentity
        ? { certificateSha256: this.tlsIdentity.certificateSha256 }
        : {})
    });
  }

  private async listBrains(response: ServerResponse): Promise<void> {
    const summaries = await this.repository.list();
    const brains = await Promise.all(summaries.map(async (summary) => {
      const brain = await this.repository.get(summary.id);
      return {
        id: summary.id,
        name: summary.name,
        updatedAt: summary.updatedAt,
        preset: summary.preset,
        generation: summary.generation,
        readiness: brain.readiness.state
      };
    }));
    jsonResponse(response, 200, { schemaVersion: 1, brains });
  }

  private async streamChat(
    request: IncomingMessage,
    response: ServerResponse,
    brainId: string
  ): Promise<void> {
    const brain = await this.repository.get(brainId);
    if (brain.readiness.state !== "ready" || this.isInitializing(brainId)) {
      throw new HttpError(409, "This mind is still completing its initial learning.");
    }
    const value = await readJson(request);
    const input = typeof value.input === "string" ? value.input.trim() : "";
    if (!input) throw new HttpError(400, "A non-empty chat message is required.");
    const turnId = value.turnId === undefined
      ? randomUUID()
      : requireId(typeof value.turnId === "string" ? value.turnId : undefined, "turn id");
    const controller = new AbortController();
    let completed = false;
    response.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
      connection: "keep-alive"
    });
    const writer = new NdjsonWriter(response, () => controller.abort());
    const streamListener = (event: ChatStreamEvent): void => {
      if (event.brainId !== brainId || event.turnId !== turnId) return;
      void writer.write({ type: "stream", event }).catch(() => undefined);
    };
    const closeListener = (): void => {
      if (!completed) controller.abort();
    };
    this.actions.on("stream", streamListener);
    response.once("close", closeListener);
    try {
      const result = await this.actions.send(brainId, input, controller.signal, turnId);
      await writer.write({
        type: "result",
        turnId,
        humanMessage: result.humanMessage,
        brainMessage: result.brainMessage,
        actionEvents: result.actionEvents?.map(actionEventForTurnStream) ?? []
      });
      completed = true;
      await writer.end();
    } catch (error) {
      if (!response.destroyed && !response.writableEnded) {
        await writer.write({
          type: "error",
          turnId,
          cancelled: controller.signal.aborted,
          error: controller.signal.aborted ? "The mobile chat turn was cancelled." : asErrorMessage(error)
        }).catch(() => undefined);
        completed = true;
        await writer.end().catch(() => response.destroy());
      }
    } finally {
      this.actions.off("stream", streamListener);
      response.off("close", closeListener);
    }
  }

  private async ingestExperience(
    request: IncomingMessage,
    response: ServerResponse,
    brainId: string
  ): Promise<void> {
    const brain = await this.repository.get(brainId);
    if (brain.readiness.state !== "ready" || this.isInitializing(brainId)) {
      throw new HttpError(409, "This mind is still completing its initial learning.");
    }
    const directory = join(this.repository.root, ".mobile-staging");
    await mkdir(directory, { recursive: true });
    const name = safeFileName(
      Array.isArray(request.headers["x-omni-filename"])
        ? request.headers["x-omni-filename"][0]
        : request.headers["x-omni-filename"]
    );
    const path = join(directory, `${randomUUID()}-${name}`);
    const hash = createHash("sha256");
    let bytes = 0;
    let nextDiskCheck = 0;
    const declaredLength = Number.parseInt(request.headers["content-length"] ?? "0", 10);
    try {
      const handle = await open(path, "wx", 0o600);
      try {
        for await (const rawChunk of request) {
          const chunk = Buffer.isBuffer(rawChunk) ? rawChunk : Buffer.from(rawChunk);
          if (bytes >= nextDiskCheck) {
            const remainingDeclared = Number.isFinite(declaredLength) && declaredLength > 0
              ? Math.max(0, declaredLength - bytes)
              : 0;
            // A chunked request has no reliable declared remainder. Price at
            // least the bytes that are about to be written, and use the same
            // device-adaptive reserve policy as every other desktop write.
            const pendingWriteBytes = Math.max(chunk.length, remainingDeclared);
            try {
              await requireDiskWrite(
                directory,
                { operationWriteBytes: pendingWriteBytes },
                this.diskSpaceReader ? { read: this.diskSpaceReader } : {}
              );
            } catch (error) {
              if (!(error instanceof DiskReservePauseError)) throw error;
              throw new HttpError(507, "The upload paused before consuming the storage reserve.");
            }
            nextDiskCheck = bytes + 64 * 1024 * 1024;
          }
          await handle.write(chunk);
          hash.update(chunk);
          bytes += chunk.length;
        }
        await handle.sync();
        if (declaredLength > 0 && bytes !== declaredLength) {
          throw new HttpError(400, "The mobile upload ended before every declared byte arrived.");
        }
        if (bytes === 0) throw new HttpError(400, "The uploaded experience is empty.");
      } finally {
        await handle.close();
      }
      const results = await this.service.ingestPaths(brainId, [path], "pretrain");
      jsonResponse(response, 201, {
        schemaVersion: 1,
        fileName: name,
        bytes,
        sha256: hash.digest("hex"),
        results: results.map((result) => ({
          sourceId: result.source.id,
          sourceName: result.source.name,
          warnings: result.warnings,
          coverage: result.coverage
        }))
      });
    } finally {
      await rm(path, { force: true });
    }
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    this.validateHost(request);
    if (request.headers.origin) throw new HttpError(403, "Browser-origin requests are not accepted.");
    const url = new URL(request.url ?? "/", "http://mobile.local");
    const path = url.pathname.replace(/\/+$/u, "") || "/";
    if (request.method === "GET" && path === "/v1/health") {
      jsonResponse(response, 200, {
        schemaVersion: 1,
        protocolVersion: 2,
        service: "OmniCortex companion",
        samePersistentBrain: true,
        transportSecurity: this.allowLan
          ? "certificate-pinned-tls"
          : "loopback-http",
        ...(this.tlsIdentity
          ? { certificateSha256: this.tlsIdentity.certificateSha256 }
          : {})
      });
      return;
    }
    if (request.method === "POST" && path === "/v1/pair") {
      await this.pair(request, response);
      return;
    }
    await this.authenticate(request);
    if (request.method === "GET" && path === "/v1/brains") {
      await this.listBrains(response);
      return;
    }
    const messageMatch = /^\/v1\/brains\/([^/]+)\/messages$/u.exec(path);
    if (request.method === "GET" && messageMatch) {
      const brainId = requireId(messageMatch[1], "brain id");
      const page = this.repository.conversationPage
        ? await this.repository.conversationPage(brainId, undefined, 200)
        : undefined;
      const legacyMessages = page
        ? []
        : (await this.repository.get(brainId)).messages;
      jsonResponse(response, 200, {
        schemaVersion: 1,
        messages: page
          ? page.entries.flatMap((entry) => entry.message ? [entry.message] : [])
          : legacyMessages,
        hasOlder: page?.hasOlder ?? false,
        nextBeforeSequence: page?.nextBeforeSequence
      });
      return;
    }
    const chatMatch = /^\/v1\/brains\/([^/]+)\/chat$/u.exec(path);
    if (request.method === "POST" && chatMatch) {
      await this.streamChat(request, response, requireId(chatMatch[1], "brain id"));
      return;
    }
    const cancelMatch = /^\/v1\/brains\/([^/]+)\/chat\/([^/]+)\/cancel$/u.exec(path);
    if (request.method === "POST" && cancelMatch) {
      const cancelled = this.actions.cancel(
        requireId(cancelMatch[1], "brain id"),
        requireId(cancelMatch[2], "turn id")
      );
      jsonResponse(response, 200, { schemaVersion: 1, cancelled });
      return;
    }
    const experienceMatch = /^\/v1\/brains\/([^/]+)\/experience$/u.exec(path);
    if (request.method === "POST" && experienceMatch) {
      await this.ingestExperience(request, response, requireId(experienceMatch[1], "brain id"));
      return;
    }
    throw new HttpError(404, "Mobile endpoint not found.");
  }
}
